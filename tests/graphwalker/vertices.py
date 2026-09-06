# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker vertices and independently observable state conditions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tests.graphwalker.model import (
    STATE_TIMEOUTS,
    FailureDetails,
    FailureKind,
    RuntimeObservation,
    Vertex,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.graphwalker.graph_utils import GraphContext, StateFunction


StateAssertion = Callable[["GraphContext"], None]
ObservationAssertion = Callable[["GraphContext", RuntimeObservation], None]


def _assert_idle(context: GraphContext) -> None:
    state = context.state
    if state.wiseman.active_question is not None:
        context.state.reject(
            FailureDetails(
                FailureKind.INVARIANT,
                "idle",
                "no active question",
                state.wiseman.active_question,
                f"idle state retained active question {state.wiseman.active_question}",
            )
        )


def _assert_active_turn(context: GraphContext) -> None:
    state = context.state
    if state.wiseman.active_question is None:
        context.state.reject(
            FailureDetails(FailureKind.INVARIANT, str(state.vertex), "active question", "none", "active lifecycle state has no active question")
        )


def _assert_retired(context: GraphContext) -> None:
    state = context.state
    if state.wiseman.active_question is not None or state.wiseman.pending_questions:
        context.state.reject(
            FailureDetails(
                FailureKind.INVARIANT,
                "retired",
                "no active or queued work",
                f"active={state.wiseman.active_question}, pending={state.wiseman.pending_questions}",
                "retired state retained active or queued work",
            )
        )


STATE_ASSERTIONS: Mapping[Vertex, StateAssertion] = {
    Vertex.IDLE: _assert_idle,
    Vertex.PREPARING: _assert_active_turn,
    Vertex.RUNNING: _assert_active_turn,
    Vertex.RECOVERING: _assert_active_turn,
    Vertex.CANCELLING: _assert_active_turn,
    Vertex.DELIVERING: _assert_active_turn,
    Vertex.OUTCOME_UNKNOWN: _assert_active_turn,
    Vertex.ERROR: _assert_active_turn,
    Vertex.RETIRED: _assert_retired,
}


def _reject_observation(context: GraphContext, location: str, expected: str, observed: str, message: str) -> None:
    context.state.reject(FailureDetails(FailureKind.OBSERVATION, location, expected, observed, message, context.message_id))


def _assert_subsequence(expected: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    position = 0
    for message_id in observed:
        if position < len(expected) and message_id == expected[position]:
            position += 1
    return position == len(expected)


def _assert_observation_context(context: GraphContext, observation: RuntimeObservation) -> None:
    state = context.state
    expected_messages = tuple(message.id for message in state.chat.messages)
    if not _assert_subsequence(expected_messages, observation.chat.message_ids):
        _reject_observation(
            context,
            "chat.messages",
            str(expected_messages),
            str(observation.chat.message_ids),
            "runtime observation lost or reordered chat messages",
        )
    expected_background = tuple(state.chat.background_context)
    if observation.chat.background_context_ids != expected_background:
        _reject_observation(
            context,
            "chat.background_context",
            str(expected_background),
            str(observation.chat.background_context_ids),
            "runtime observation changed unconsumed background context",
        )
    expected_consumed = tuple(state.chat.consumed_context)
    if observation.chat.consumed_context_ids != expected_consumed:
        _reject_observation(
            context,
            "chat.consumed_context",
            str(expected_consumed),
            str(observation.chat.consumed_context_ids),
            "runtime observation changed the consumed-context cursor",
        )
    expected_pending = tuple(state.wiseman.pending_questions)
    if observation.wiseman.pending_question_ids != expected_pending:
        _reject_observation(
            context,
            "wiseman.pending_questions",
            str(expected_pending),
            str(observation.wiseman.pending_question_ids),
            "runtime observation lost or reordered queued questions",
        )
    if observation.wiseman.active_question != state.wiseman.active_question:
        _reject_observation(
            context,
            "wiseman.active_question",
            str(state.wiseman.active_question),
            str(observation.wiseman.active_question),
            "runtime observation changed the active question",
        )
    if observation.wiseman.session_id != state.wiseman.session_id:
        _reject_observation(
            context,
            "wiseman.session_id",
            str(state.wiseman.session_id),
            str(observation.wiseman.session_id),
            "runtime observation changed the Codex session",
        )
    if observation.wiseman.turn != state.wiseman.turns:
        _reject_observation(
            context,
            "wiseman.turn",
            str(state.wiseman.turns),
            str(observation.wiseman.turn),
            "runtime observation changed the settled-turn count",
        )
    expected_steering = tuple(state.chat.steering_messages)
    if observation.chat.steering_ids != expected_steering:
        _reject_observation(
            context,
            "chat.steering",
            str(expected_steering),
            str(observation.chat.steering_ids),
            "runtime observation lost or duplicated steering messages",
        )
    expected_stops = tuple(sorted(state.chat.stop_commands))
    if tuple(sorted(observation.chat.stop_command_ids)) != expected_stops:
        _reject_observation(
            context,
            "chat.stop_commands",
            str(expected_stops),
            str(tuple(sorted(observation.chat.stop_command_ids))),
            "runtime observation lost or duplicated stop commands",
        )


def _assert_phase(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    if observation.wiseman.phase is not vertex:
        _reject_observation(
            context,
            f"phase:{vertex}",
            str(vertex),
            str(observation.wiseman.phase),
            f"runtime Wiseman phase is not {vertex}",
        )
    _assert_observation_context(context, observation)


def _assert_idle_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.IDLE)
    if observation.wiseman.active_question is not None:
        _reject_observation(context, "idle.active_question", "none", str(observation.wiseman.active_question), "idle runtime retains active work")
    if observation.wiseman.result_known or observation.wiseman.error:
        _reject_observation(context, "idle.result", "no terminal result", "terminal result", "idle runtime exposes a terminal result")


def _assert_preparing_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.PREPARING)
    if observation.wiseman.active_question is None:
        _reject_observation(context, "preparing.active_question", "question id", "none", "preparing runtime has no admitted question")
    if len(observation.chat.progress_message_ids) > 1:
        _reject_observation(
            context,
            "preparing.progress",
            "at most one progress message",
            str(observation.chat.progress_message_ids),
            "preparing created duplicate progress messages",
        )
    if observation.wiseman.result_known or observation.wiseman.error:
        _reject_observation(context, "preparing.result", "no terminal result", "terminal result", "preparing runtime already exposes a result")


def _assert_running_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.RUNNING)
    if observation.wiseman.active_question is None:
        _reject_observation(context, "running.active_question", "question id", "none", "running runtime has no active question")
    if len(observation.chat.progress_message_ids) > 1:
        _reject_observation(
            context,
            "running.progress",
            "at most one progress message",
            str(observation.chat.progress_message_ids),
            "running created duplicate progress messages",
        )
    if observation.wiseman.result_known or observation.wiseman.error:
        _reject_observation(context, "running.result", "no terminal result", "terminal result", "running runtime exposes a terminal result")


def _assert_recovering_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.RECOVERING)
    if observation.wiseman.active_question is None or observation.wiseman.session_id is None:
        _reject_observation(context, "recovering.session", "active question and session", "missing value", "recovery lost the active Codex session")
    if observation.wiseman.result_known or observation.wiseman.error:
        _reject_observation(context, "recovering.result", "no terminal result", "terminal result", "recovering runtime exposes a terminal result")


def _assert_cancelling_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.CANCELLING)
    active_question = context.state.wiseman.active_question
    if active_question is None or not observation.chat.stop_command_ids:
        _reject_observation(context, "cancelling.target", "active question and stop command", "missing value", "cancellation has no active target")
    if observation.wiseman.stop_target_question_id != active_question:
        _reject_observation(
            context,
            "cancelling.target",
            str(active_question),
            str(observation.wiseman.stop_target_question_id),
            "stop command targets a different question",
        )


def _assert_delivering_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.DELIVERING)
    if not observation.wiseman.result_known or len(observation.chat.answer_message_ids) != 1:
        _reject_observation(context, "delivering.answer", "one known answer", str(observation), "delivery lacks exactly one answer result")


def _assert_outcome_unknown_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.OUTCOME_UNKNOWN)
    if observation.wiseman.result_known or observation.chat.answer_message_ids:
        _reject_observation(context, "outcome_unknown.result", "unknown result without answer", str(observation), "unknown execution produced an answer")


def _assert_error_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.ERROR)
    if not observation.wiseman.error:
        _reject_observation(context, "error.result", "error result", str(observation.wiseman.error), "error state has no recorded error")


def _assert_retired_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.RETIRED)
    if not observation.chat.archived:
        _reject_observation(context, "retired.archive", "archived conversation", "active conversation", "retired runtime is not archived")


async def _wait_for_vertex(context: GraphContext, vertex: Vertex, assertion: ObservationAssertion) -> None:
    state = context.state
    if state.vertex is not vertex:
        context.state.reject(
            FailureDetails(
                FailureKind.INVARIANT,
                f"vertex:{vertex}",
                str(vertex),
                str(state.vertex),
                f"{vertex} condition requires model state {vertex}, got {state.vertex}",
            )
        )
    STATE_ASSERTIONS[vertex](context)
    observation = await context.harness.wait_for_state(vertex, context, deadline_seconds=STATE_TIMEOUTS[vertex])
    assertion(context, observation)


async def idle(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.IDLE, _assert_idle_observation)


async def preparing(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.PREPARING, _assert_preparing_observation)


async def running(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.RUNNING, _assert_running_observation)


async def recovering(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.RECOVERING, _assert_recovering_observation)


async def cancelling(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.CANCELLING, _assert_cancelling_observation)


async def delivering(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.DELIVERING, _assert_delivering_observation)


async def outcome_unknown(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.OUTCOME_UNKNOWN, _assert_outcome_unknown_observation)


async def error(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.ERROR, _assert_error_observation)


async def retired(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.RETIRED, _assert_retired_observation)


STATE_FUNCTIONS: Mapping[Vertex, StateFunction] = {
    Vertex.IDLE: idle,
    Vertex.PREPARING: preparing,
    Vertex.RUNNING: running,
    Vertex.RECOVERING: recovering,
    Vertex.CANCELLING: cancelling,
    Vertex.DELIVERING: delivering,
    Vertex.OUTCOME_UNKNOWN: outcome_unknown,
    Vertex.ERROR: error,
    Vertex.RETIRED: retired,
}
