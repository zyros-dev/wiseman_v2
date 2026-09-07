# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker vertices and independently observable state conditions."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING

from app.presentation import DEFAULT_REACTION_EMOJIS
from tests.graphwalker.model import (
    STATE_TIMEOUTS,
    FailureDetails,
    FailureKind,
    GraphState,
    RuntimeObservation,
    Vertex,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.graphwalker.graph_utils import GraphContext, StateFunction


StateAssertion = Callable[["GraphContext"], None]
ObservationAssertion = Callable[["GraphContext", RuntimeObservation], None]
ObservationValue = tuple[str, ...] | str | int | bool | Vertex | None
ObservationComparison = tuple[str, ObservationValue, ObservationValue, str]


def _assert_idle(context: GraphContext) -> None:
    state = context.state
    if state.wiseman.active_question is not None and not state.handoff_pending:
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
    if state.wiseman.active_question is not None or state.wiseman.pending_questions or state.handoff_pending:
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


def _reject_invariant(context: GraphContext, location: str, expected: str, observed: str, message: str) -> None:
    context.state.reject(FailureDetails(FailureKind.INVARIANT, location, expected, observed, message, context.message_id))


def _assert_subsequence(expected: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    position = 0
    for message_id in observed:
        if position < len(expected) and message_id == expected[position]:
            position += 1
    return position == len(expected)


def _assert_equal(context: GraphContext, location: str, expected: ObservationValue, observed: ObservationValue, message: str) -> None:
    if expected != observed:
        _reject_observation(context, location, str(expected), str(observed), message)


def _assert_state_equal(context: GraphContext, location: str, expected: ObservationValue, observed: ObservationValue, message: str) -> None:
    if expected != observed:
        _reject_invariant(context, location, str(expected), str(observed), message)


def _assert_model_invariants(context: GraphContext) -> None:
    state = context.state
    if len(state.wiseman.pending_questions) != len(set(state.wiseman.pending_questions)):
        _reject_invariant(
            context, "graph-state", "unique ordered pending questions", str(state.wiseman.pending_questions), "pending questions must be ordered and unique"
        )
    if len(state.chat.messages) != len(state.chat.message_ids):
        _reject_invariant(
            context, "chat-state", "deduplicated message history", str([message.id for message in state.chat.messages]), "chat history must be deduplicated"
        )
    if state.vertex is Vertex.RETIRED and (state.wiseman.active_question or state.wiseman.pending_questions or state.handoff_pending):
        _reject_invariant(
            context,
            "retired",
            "no active or queued work",
            f"active={state.wiseman.active_question}, pending={state.wiseman.pending_questions}",
            "retired conversations cannot retain active work",
        )
    if state.wiseman.active_question and state.wiseman.active_question not in state.wiseman.pending_questions:
        _reject_invariant(
            context,
            "wiseman-state",
            "active question remains pending",
            state.wiseman.active_question,
            "active question must remain pending until terminal delivery",
        )
    if not state.wiseman.settled_questions <= state.wiseman.seen_questions:
        _reject_invariant(
            context,
            "wiseman-state.settled_questions",
            "settled questions were admitted",
            str(state.wiseman.settled_questions),
            "a terminal result was recorded for an unknown question",
        )


def _assert_question_transition(context: GraphContext, previous: GraphState) -> None:
    _assert_state_equal(
        context,
        "transition.pending_questions",
        (*previous.wiseman.pending_questions, context.last_message_id),
        tuple(context.state.wiseman.pending_questions),
        "question edge did not append the new question",
    )


def _assert_dispatch_transition(context: GraphContext, previous: GraphState) -> None:
    expected = (
        previous.wiseman.pending_questions[1]
        if previous.handoff_pending and len(previous.wiseman.pending_questions) > 1
        else previous.wiseman.pending_questions[0]
        if previous.wiseman.pending_questions
        else None
    )
    _assert_state_equal(
        context,
        "transition.active_question",
        expected,
        context.state.wiseman.active_question,
        "dispatch-queued did not activate the oldest pending question",
    )


def _assert_background_transition(context: GraphContext, previous: GraphState) -> None:
    if previous.vertex is Vertex.IDLE and context.last_response and context.last_response.body.get("status") == "ignored":
        return
    _assert_state_equal(
        context,
        "transition.background_context",
        (*previous.chat.background_context, context.last_message_id),
        tuple(context.state.chat.background_context),
        "background edge did not append context",
    )


def _assert_context_transition(context: GraphContext, previous: GraphState) -> None:
    _assert_state_equal(context, "transition.consumed_context", (), tuple(context.state.chat.background_context), "context-ready retained background context")
    _assert_state_equal(
        context,
        "transition.consumed_context",
        (*previous.chat.consumed_context, *previous.chat.background_context),
        tuple(context.state.chat.consumed_context),
        "context-ready did not consume background context",
    )


def _assert_resume_transition(context: GraphContext, previous: GraphState) -> None:
    _assert_state_equal(
        context,
        "transition.resume_consumed_context",
        tuple(previous.chat.consumed_context),
        tuple(context.state.chat.consumed_context),
        "resume-session changed the consumed context cursor",
    )
    _assert_state_equal(
        context,
        "transition.resume_background_context",
        tuple(previous.chat.background_context),
        tuple(context.state.chat.background_context),
        "resume-session discarded unconsumed background context",
    )


def _assert_steering_transition(context: GraphContext, previous: GraphState) -> None:
    _assert_state_equal(
        context,
        "transition.steering",
        (*previous.chat.steering_messages, context.last_message_id),
        tuple(context.state.chat.steering_messages),
        "steering edge did not append the steering message",
    )


def _assert_stop_transition(context: GraphContext, previous: GraphState) -> None:
    _assert_state_equal(
        context,
        "transition.stop_commands",
        tuple(sorted((*previous.chat.stop_commands, context.last_message_id))),
        tuple(sorted(context.state.chat.stop_commands)),
        "stop edge did not record its command",
    )


def _assert_terminal_transition(context: GraphContext, previous: GraphState) -> None:
    active = previous.wiseman.active_question
    queued_handoff = bool(active and previous.wiseman.pending_questions[:1] == [active] and len(previous.wiseman.pending_questions) > 1)
    expected_active = active if queued_handoff else None
    expected_pending = (
        previous.wiseman.pending_questions
        if queued_handoff
        else previous.wiseman.pending_questions[1:]
        if active and previous.wiseman.pending_questions[:1] == [active]
        else previous.wiseman.pending_questions
    )
    _assert_state_equal(
        context, "transition.active_question", expected_active, context.state.wiseman.active_question, "terminal edge changed active work incorrectly"
    )
    _assert_state_equal(
        context,
        "transition.pending_questions",
        tuple(expected_pending),
        tuple(context.state.wiseman.pending_questions),
        "terminal edge lost queued questions",
    )
    expected_turn = previous.wiseman.turns if queued_handoff else previous.wiseman.turns + bool(active)
    _assert_state_equal(context, "transition.turn", expected_turn, context.state.wiseman.turns, "terminal edge did not settle one turn")


TRANSITION_ASSERTIONS: Mapping[str, Callable[[GraphContext, GraphState], None]] = {
    "admit-question": _assert_question_transition,
    "dispatch-queued": _assert_dispatch_transition,
    "queue-question": _assert_question_transition,
    "background-chatter": _assert_background_transition,
    "running-background-chatter": _assert_background_transition,
    "context-ready": _assert_context_transition,
    "resume-session": _assert_resume_transition,
    "steer-active-turn": _assert_steering_transition,
    "repeat-steer": _assert_steering_transition,
    "stop-preparing": _assert_stop_transition,
    "stop-running": _assert_stop_transition,
    "stop-recovering": _assert_stop_transition,
    "stop-delivering": _assert_stop_transition,
    "stop-confirmed": _assert_terminal_transition,
    "answer-finalized": _assert_terminal_transition,
    "error-finalized": _assert_terminal_transition,
}


def _assert_transition(context: GraphContext) -> None:
    previous = context.previous_state
    if previous is None:
        return
    _assert_state_equal(context, "transition.vertex", context.last_target, context.state.vertex, "edge did not reach its declared target")
    _assert_state_equal(context, "transition.step", previous.step + 1, context.state.step, "edge did not advance the model step")
    if context.last_edge == "fixture-reset":
        return
    if not {message.id for message in previous.chat.messages}.issubset(context.state.chat.message_ids):
        _reject_invariant(
            context,
            "transition.messages",
            "previous messages retained",
            str(context.state.chat.message_ids),
            "edge discarded chat history",
        )
    if not set(previous.wiseman.seen_questions).issubset(context.state.wiseman.seen_questions):
        _reject_invariant(
            context,
            "transition.questions",
            "previous questions retained",
            str(context.state.wiseman.seen_questions),
            "edge discarded question history",
        )
    assertion = TRANSITION_ASSERTIONS.get(context.last_edge or "")
    if assertion is not None:
        assertion(context, previous)


def _assert_observation_context(context: GraphContext, observation: RuntimeObservation) -> None:
    state = context.state
    state.reconcile_processed(observation.wiseman.processed_question_ids, observation.wiseman.active_question)
    state.reconcile_session(observation.wiseman.session_id)
    expected_messages = tuple(message.id for message in state.chat.messages)
    if not _assert_subsequence(expected_messages, observation.chat.message_ids):
        _reject_observation(
            context,
            "chat.messages",
            str(expected_messages),
            str(observation.chat.message_ids),
            "runtime observation lost or reordered chat messages",
        )
    expected_pending = tuple(state.wiseman.pending_questions)
    if state.handoff_pending and expected_pending[1:] == observation.wiseman.pending_question_ids:
        expected_pending = expected_pending[1:]
    comparisons: tuple[ObservationComparison, ...] = (
        ("chat.background_context", tuple(state.chat.background_context), observation.chat.background_context_ids, "unconsumed context changed"),
        ("chat.consumed_context", tuple(state.chat.consumed_context), observation.chat.consumed_context_ids, "consumed cursor changed"),
        ("wiseman.pending_questions", expected_pending, observation.wiseman.pending_question_ids, "queued questions lost or reordered"),
        (
            "wiseman.active_question",
            None if state.handoff_pending else state.wiseman.active_question,
            observation.wiseman.active_question,
            "active question changed",
        ),
        ("wiseman.session_id", state.wiseman.session_id, observation.wiseman.session_id, "Codex session changed"),
        ("wiseman.turn", state.wiseman.turns, observation.wiseman.turn, "settled turn count changed"),
        ("chat.steering", tuple(state.chat.steering_messages), observation.chat.steering_ids, "steering lost or duplicated"),
        ("chat.stop_commands", tuple(sorted(state.chat.stop_commands)), tuple(sorted(observation.chat.stop_command_ids)), "stops lost or duplicated"),
    )
    for location, expected, observed, message in comparisons:
        _assert_equal(context, location, expected, observed, message)
    previous = context.previous_observation
    if previous and previous.wiseman.session_id and observation.wiseman.session_id:
        _assert_equal(
            context,
            "wiseman.session_continuity",
            previous.wiseman.session_id,
            observation.wiseman.session_id,
            "a follow-up or lifecycle transition changed the Codex session",
        )
    processed = set(observation.wiseman.processed_question_ids)
    missing = state.wiseman.settled_questions - processed
    if missing:
        _reject_observation(
            context,
            "wiseman.processed_questions",
            str(sorted(state.wiseman.settled_questions)),
            str(sorted(processed)),
            f"Temporal has not processed settled questions {sorted(missing)}",
        )


def _assert_id_delta(
    context: GraphContext,
    observation: RuntimeObservation,
    ids: tuple[str, ...],
    location: str,
    message: str,
) -> None:
    if context.last_message_id not in ids:
        _reject_observation(context, location, context.last_message_id, str(ids), message)


def _assert_question_delta(context: GraphContext, _previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    if context.last_message_id not in observation.chat.message_ids or context.last_message_id not in observation.wiseman.pending_question_ids:
        _reject_observation(context, "delta.question", "message admitted and queued", str(observation), "question edge produced no queued message")


def _assert_background_delta(context: GraphContext, previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    if previous.wiseman.phase is not Vertex.IDLE:
        _assert_id_delta(context, observation, observation.chat.background_context_ids, "delta.background", "background message was not retained")


def _assert_steering_delta(context: GraphContext, _previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    _assert_id_delta(context, observation, observation.chat.steering_ids, "delta.steering", "steering message was not retained")


def _assert_stop_delta(context: GraphContext, _previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    _assert_id_delta(context, observation, observation.chat.stop_command_ids, "delta.stop", "stop command was not retained")


def _assert_context_delta(context: GraphContext, previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    expected = (*previous.chat.consumed_context_ids, *previous.chat.background_context_ids)
    _assert_equal(context, "delta.consumed_context", expected, observation.chat.consumed_context_ids, "context edge did not advance the consumed cursor")


def _assert_progress_delta(context: GraphContext, previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    if observation.chat.progress_edit_count <= previous.chat.progress_edit_count:
        _reject_observation(
            context,
            "delta.progress_edits",
            f"> {previous.chat.progress_edit_count}",
            str(observation.chat.progress_edit_count),
            "progress-preview did not edit the existing progress message",
        )
    if not observation.chat.typing:
        _reject_observation(context, "delta.typing", "typing signal", "absent", "progress-preview stopped Discord typing")


def _assert_resume_delta(context: GraphContext, previous: RuntimeObservation, observation: RuntimeObservation) -> None:
    _assert_equal(
        context,
        "delta.resume_consumed_context",
        previous.chat.consumed_context_ids,
        observation.chat.consumed_context_ids,
        "resume-session changed the consumed context cursor",
    )
    _assert_equal(
        context,
        "delta.resume_background_context",
        previous.chat.background_context_ids,
        observation.chat.background_context_ids,
        "resume-session changed unconsumed background context",
    )


DELTA_ASSERTIONS: Mapping[str, Callable[[GraphContext, RuntimeObservation, RuntimeObservation], None]] = {
    "admit-question": _assert_question_delta,
    "queue-question": _assert_question_delta,
    "background-chatter": _assert_background_delta,
    "running-background-chatter": _assert_background_delta,
    "steer-active-turn": _assert_steering_delta,
    "repeat-steer": _assert_steering_delta,
    "stop-preparing": _assert_stop_delta,
    "stop-running": _assert_stop_delta,
    "stop-recovering": _assert_stop_delta,
    "stop-delivering": _assert_stop_delta,
    "context-ready": _assert_context_delta,
    "progress-preview": _assert_progress_delta,
    "resume-session": _assert_resume_delta,
}


def _assert_observation_delta(context: GraphContext, observation: RuntimeObservation) -> None:
    previous = context.previous_observation
    if previous is None:
        return
    assertion = DELTA_ASSERTIONS.get(context.last_edge or "")
    if assertion is not None:
        assertion(context, previous, observation)


def _assert_phase(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    _assert_equal(context, f"phase:{vertex}", vertex, observation.wiseman.phase, f"runtime Wiseman phase is not {vertex}")
    _assert_observation_context(context, observation)
    _assert_temporal_state(context, observation, vertex)
    _assert_evidence(context, observation, vertex)


def _assert_temporal_state(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    wiseman = observation.wiseman
    expected_active = vertex not in {Vertex.IDLE, Vertex.RETIRED}
    _assert_equal(
        context,
        f"{vertex}.active_question",
        expected_active,
        wiseman.active_question is not None,
        "Temporal active message disagrees with lifecycle state",
    )
    if vertex is Vertex.RUNNING:
        expected_inferencing = vertex is Vertex.RUNNING
        _assert_equal(context, "running.inferencing", expected_inferencing, wiseman.inferencing, "running state is not executing inference")
        _assert_equal(context, "running.delivery_phase", "progress", wiseman.delivery_phase, "running state lost progress delivery")
    elif vertex is Vertex.DELIVERING:
        expected_inferencing = vertex is Vertex.RUNNING
        _assert_equal(context, "delivering.inferencing", expected_inferencing, wiseman.inferencing, "delivery still reports inference active")
        _assert_equal(context, "delivering.delivery_phase", "answer", wiseman.delivery_phase, "delivery state has no answer phase")
        if wiseman.reaction_phase not in {"success", "failure"}:
            _reject_observation(
                context,
                "delivering.reaction_phase",
                "success or failure",
                wiseman.reaction_phase,
                "delivery has not reconciled its terminal reaction",
            )
    elif vertex is Vertex.CANCELLING:
        expected_stop = vertex is Vertex.CANCELLING
        _assert_equal(context, "cancelling.stop_requested", expected_stop, wiseman.stop_requested, "cancelling state lost its stop request")
    elif vertex is Vertex.OUTCOME_UNKNOWN:
        expected_unknown = vertex is Vertex.OUTCOME_UNKNOWN
        _assert_equal(context, "outcome_unknown.flag", expected_unknown, wiseman.outcome_unknown, "unknown outcome state is not recorded in Temporal")
    elif vertex is Vertex.RETIRED:
        expected_closed = vertex is Vertex.RETIRED
        _assert_equal(context, "retired.closed", expected_closed, observation.chat.archived, "retired state is not closed in Temporal")


def _assert_evidence(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    required: set[str] = set()
    if vertex in {Vertex.RUNNING, Vertex.DELIVERING, Vertex.ERROR}:
        required.update({"admission", "turn"})
    if vertex in {Vertex.RUNNING, Vertex.DELIVERING}:
        required.update({"context", "grammar", "prompt"})
    if vertex is Vertex.DELIVERING:
        required.add("completed")
    if vertex is Vertex.ERROR:
        required.add("failure")
    if vertex in {Vertex.DELIVERING, Vertex.ERROR}:
        required.add("reaction")
    missing = tuple(sorted(required - set(observation.phoenix_nodes)))
    if missing:
        _reject_observation(context, f"phoenix:{vertex}", str(sorted(required)), str(observation.phoenix_nodes), f"Phoenix evidence missing {missing}")
    if vertex in {Vertex.RUNNING, Vertex.DELIVERING, Vertex.ERROR}:
        _assert_phoenix_order(context, observation, vertex)
        _assert_phoenix_payload(context, observation, vertex)
    if vertex in {Vertex.RUNNING, Vertex.DELIVERING, Vertex.ERROR}:
        _assert_discord_delivery(context, observation, vertex)


def _assert_phoenix_payload(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    evidence = observation.phoenix
    active = observation.wiseman.active_question
    expected_trace = f"discord-{active}" if active else ""
    _assert_equal(context, f"phoenix:{vertex}.trace", expected_trace, evidence.trace, "Phoenix trace is not correlated to the active Discord message")
    _assert_equal(context, f"phoenix:{vertex}.audit_id", expected_trace, evidence.audit_id, "admission audit is not keyed by the Phoenix trace")
    if not evidence.audit_present or not evidence.audit_matches_admission:
        _reject_observation(
            context,
            f"phoenix:{vertex}.audit",
            "persisted audit matches admission record",
            f"present={evidence.audit_present}, matches={evidence.audit_matches_admission}",
            "Phoenix audit evidence cannot replay the exact admitted request",
        )
    normalized = evidence.normalized_request
    trigger = normalized.get("trigger") if isinstance(normalized, dict) else None
    trigger = trigger if isinstance(trigger, dict) else {}
    trigger_id = _string_value(trigger.get("id"))
    author_id = _string_value(trigger.get("author_id"))
    _assert_equal(context, f"phoenix:{vertex}.trigger_id", active, trigger_id, "normalized admission changed the trigger id")
    _assert_equal(context, f"phoenix:{vertex}.author", "human", author_id, "normalized admission changed the author")
    mentions = trigger.get("mentions")
    if not isinstance(mentions, list) or "bot" not in mentions:
        _reject_observation(context, f"phoenix:{vertex}.mentions", "bot mention", str(mentions), "admitted request lost its bot mention")
    if active not in evidence.context_message_ids:
        _reject_observation(
            context,
            f"phoenix:{vertex}.context",
            active or "active message",
            str(evidence.context_message_ids),
            "context evidence does not contain the admitted message",
        )
    if not evidence.context_author_ids or any(author != "human" for author in evidence.context_author_ids):
        _reject_observation(
            context,
            f"phoenix:{vertex}.context_authors",
            "ordered human authors",
            str(evidence.context_author_ids),
            "context evidence changed message authors",
        )
    if vertex in {Vertex.RUNNING, Vertex.DELIVERING}:
        if not evidence.grammar_rendered or not evidence.prompt_final_input:
            _reject_observation(context, f"phoenix:{vertex}.prompt", "rendered grammar and final input", "missing", "prompt evidence is incomplete")
        if not evidence.context_attachment_urls or evidence.prompt_final_input.count(evidence.context_attachment_urls[-1]) != 1:
            _reject_observation(
                context,
                f"phoenix:{vertex}.images",
                "one selected attachment reference",
                str(evidence.context_attachment_urls),
                "prompt evidence did not carry exactly one selected image reference",
            )
    if vertex is Vertex.DELIVERING:
        _assert_equal(
            context, "phoenix.codex_thread", observation.wiseman.session_id, evidence.codex_thread_id, "Phoenix result used a different Codex session"
        )
        if evidence.served_model != "mock" or not isinstance(evidence.usage, dict) or not isinstance(evidence.cost, dict):
            _reject_observation(
                context,
                "phoenix.billing",
                "served model, usage and cost",
                f"model={evidence.served_model}, usage={evidence.usage}, cost={evidence.cost}",
                "Phoenix result is missing model billing evidence",
            )
        total = evidence.cost.get("total") if isinstance(evidence.cost, dict) else None
        if not isinstance(total, (int, float)) or total < 0:
            _reject_observation(context, "phoenix.billing.total", "non-negative numeric cost", str(total), "Phoenix cost estimate is invalid")


def _assert_discord_delivery(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    if vertex is Vertex.RUNNING:
        if not observation.chat.typing_operations or observation.chat.typing_operations[-1].startswith("stop:"):
            _reject_observation(
                context, "discord.typing", "active typing signal", str(observation.chat.typing_operations), "running turn is not typing in Discord"
            )
        return
    if vertex not in {Vertex.DELIVERING, Vertex.ERROR}:
        return
    if observation.chat.typing or not observation.chat.typing_operations or not observation.chat.typing_operations[-1].startswith("stop:"):
        _reject_observation(
            context, "discord.typing", "typing stopped after inference", str(observation.chat.typing_operations), "terminal delivery left Discord typing active"
        )
    if observation.chat.answer_message_ids and observation.chat.answer_message_ids[0] not in observation.chat.edited_message_ids:
        _reject_observation(
            context,
            "discord.answer_message",
            str(observation.chat.answer_message_ids),
            str(observation.chat.edited_message_ids),
            "final answer was not an edit of the existing delivery message",
        )
    operations = observation.phoenix.reaction_operations
    processing = DEFAULT_REACTION_EMOJIS["processing"]
    terminal_add = next(
        (
            index
            for index, value in enumerate(operations)
            if value in {f"add:{DEFAULT_REACTION_EMOJIS['success']}", f"add:{DEFAULT_REACTION_EMOJIS['failure']}"}
        ),
        None,
    )
    processing_add = next((index for index, value in enumerate(operations) if value == f"add:{processing}"), None)
    processing_remove = next((index for index, value in enumerate(operations) if value == f"remove:{processing}"), None)
    if processing_add is None or terminal_add is None or processing_remove is None or not processing_add < terminal_add < processing_remove:
        _reject_observation(
            context,
            "discord.reactions",
            "processing add < terminal add < processing remove",
            str(operations),
            "terminal reactions were reconciled in the wrong order",
        )


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _assert_phoenix_order(context: GraphContext, observation: RuntimeObservation, vertex: Vertex) -> None:
    sequence = observation.phoenix_sequence
    positions = {node: sequence.index(node) for node in set(sequence)}
    required_order = ("admission", "turn", "context", "grammar", "prompt")
    missing = tuple(node for node in required_order if node not in positions)
    if missing:
        return
    if any(positions[left] >= positions[right] for left, right in pairwise(required_order)):
        _reject_observation(
            context,
            f"phoenix:{vertex}.order",
            "admission < turn < context < grammar < prompt",
            str(sequence),
            "Phoenix nodes are out of lifecycle order",
        )


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
    expected = context.state.wiseman.turns if location == "delivering" else context.state.wiseman.turns + 1
    _assert_equal(context, f"{location}.active_turn", expected, observation.wiseman.active_turn, "active turn number changed")


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
    _assert_model_invariants(context)
    _assert_transition(context)
    STATE_ASSERTIONS[vertex](context)
    observation = await context.harness.wait_for_state(vertex, context, deadline_seconds=STATE_TIMEOUTS[vertex])
    context.observation = observation
    _assert_observation_delta(context, observation)
    assertion(context, observation)


OBSERVATION_ASSERTIONS: Mapping[Vertex, ObservationAssertion] = {
    Vertex.IDLE: _assert_idle_observation,
    Vertex.PREPARING: _assert_preparing_observation,
    Vertex.RUNNING: _assert_running_observation,
    Vertex.RECOVERING: _assert_recovering_observation,
    Vertex.CANCELLING: _assert_cancelling_observation,
    Vertex.DELIVERING: _assert_delivering_observation,
    Vertex.OUTCOME_UNKNOWN: _assert_outcome_unknown_observation,
    Vertex.ERROR: _assert_error_observation,
    Vertex.RETIRED: _assert_retired_observation,
}


def _vertex_function(vertex: Vertex) -> StateFunction:
    async def state(context: GraphContext) -> None:
        await _wait_for_vertex(context, vertex, OBSERVATION_ASSERTIONS[vertex])

    state.__name__ = vertex.value.replace("-", "_")
    return state


STATE_FUNCTIONS: Mapping[Vertex, StateFunction] = {vertex: _vertex_function(vertex) for vertex in Vertex}
