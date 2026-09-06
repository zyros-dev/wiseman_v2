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
ObservationValue = tuple[str, ...] | str | int | Vertex | None
ObservationComparison = tuple[str, ObservationValue, ObservationValue, str]


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


def _assert_equal(context: GraphContext, location: str, expected: ObservationValue, observed: ObservationValue, message: str) -> None:
    if expected != observed:
        _reject_observation(context, location, str(expected), str(observed), message)


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
    comparisons: tuple[ObservationComparison, ...] = (
        ("chat.background_context", tuple(state.chat.background_context), observation.chat.background_context_ids, "unconsumed context changed"),
        ("chat.consumed_context", tuple(state.chat.consumed_context), observation.chat.consumed_context_ids, "consumed cursor changed"),
        ("wiseman.pending_questions", tuple(state.wiseman.pending_questions), observation.wiseman.pending_question_ids, "queued questions lost or reordered"),
        ("wiseman.active_question", state.wiseman.active_question, observation.wiseman.active_question, "active question changed"),
        ("wiseman.session_id", state.wiseman.session_id, observation.wiseman.session_id, "Codex session changed"),
        ("wiseman.turn", state.wiseman.turns, observation.wiseman.turn, "settled turn count changed"),
        ("chat.steering", tuple(state.chat.steering_messages), observation.chat.steering_ids, "steering lost or duplicated"),
        ("chat.stop_commands", tuple(sorted(state.chat.stop_commands)), tuple(sorted(observation.chat.stop_command_ids)), "stops lost or duplicated"),
    )
    for location, expected, observed, message in comparisons:
        _assert_equal(context, location, expected, observed, message)


def _assert_phase(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    _assert_equal(context, f"phase:{vertex}", vertex, observation.wiseman.phase, f"runtime Wiseman phase is not {vertex}")
    _assert_observation_context(context, observation)


def _assert_no_terminal_result(context: GraphContext, observation: RuntimeObservation, location: str) -> None:
    if observation.wiseman.result_known or observation.wiseman.error:
        _reject_observation(context, location, "no terminal result", "terminal result", f"{location} runtime exposes a terminal result")


def _assert_active_phase(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    _assert_phase(context, observation, vertex)
    _assert_active_turn_number(context, observation, str(vertex))
    _assert_no_terminal_result(context, observation, str(vertex))


def _assert_active_turn_number(context: GraphContext, observation: RuntimeObservation, location: str) -> None:
    if observation.wiseman.active_question is None:
        _reject_observation(context, f"{location}.active_question", "question id", "none", f"{location} runtime has no active question")
    _assert_equal(context, f"{location}.active_turn", context.state.wiseman.turns + 1, observation.wiseman.active_turn, "active turn number changed")


def _assert_progress_bound(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    if len(observation.chat.progress_message_ids) > 1:
        _reject_observation(
            context,
            f"{vertex}.progress",
            "at most one progress message",
            str(observation.chat.progress_message_ids),
            f"{vertex} created duplicate progress messages",
        )


def _assert_idle_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.IDLE)
    _assert_equal(context, "idle.active_question", None, observation.wiseman.active_question, "idle runtime retains active work")
    _assert_no_terminal_result(context, observation, "idle")


def _assert_preparing_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_active_phase(context, observation, Vertex.PREPARING)
    _assert_progress_bound(context, observation, Vertex.PREPARING)


def _assert_running_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_active_phase(context, observation, Vertex.RUNNING)
    _assert_equal(context, "running.progress", 1, len(observation.chat.progress_message_ids), "running must have one progress message")
    if observation.chat.progress_edit_count < 1:
        _reject_observation(
            context, "running.progress_edits", "at least one update", str(observation.chat.progress_edit_count), "running progress was never updated"
        )
    if not observation.chat.typing:
        _reject_observation(context, "running.typing", "typing signal", "absent", "running has no Discord typing signal")


def _assert_recovering_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_active_phase(context, observation, Vertex.RECOVERING)
    if observation.wiseman.active_question is None or observation.wiseman.session_id is None:
        _reject_observation(context, "recovering.session", "active question and session", "missing value", "recovery lost the active Codex session")


def _assert_cancelling_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.CANCELLING)
    _assert_active_turn_number(context, observation, "cancelling")
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
    _assert_active_turn_number(context, observation, "delivering")
    _assert_equal(context, "delivering.progress", 0, len(observation.chat.progress_message_ids), "delivery retained progress buildup")
    if not observation.wiseman.result_known or len(observation.chat.answer_message_ids) != 1 or observation.chat.answer_edit_count < 1:
        _reject_observation(context, "delivering.answer", "one known answer", str(observation), "delivery lacks exactly one answer result")


def _assert_outcome_unknown_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.OUTCOME_UNKNOWN)
    _assert_active_turn_number(context, observation, "outcome_unknown")
    if observation.wiseman.result_known or observation.chat.answer_message_ids:
        _reject_observation(context, "outcome_unknown.result", "unknown result without answer", str(observation), "unknown execution produced an answer")


def _assert_error_observation(context: GraphContext, observation: RuntimeObservation) -> None:
    _assert_phase(context, observation, Vertex.ERROR)
    _assert_active_turn_number(context, observation, "error")
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
