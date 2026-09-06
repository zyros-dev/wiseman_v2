# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker edges and their executable boundary actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.graphwalker.graph_utils import EdgeFunction, GraphContext, apply_edge
from tests.graphwalker.model import EDGE_TIMEOUTS, EDGES_BY_NAME, EdgeName
from tests.graphwalker.wiseman_client import WisemanResponse

if TYPE_CHECKING:
    from collections.abc import Mapping


ACTION_EDGES = frozenset(
    {
        EdgeName.ADMIT_QUESTION,
        EdgeName.BACKGROUND_CHATTER,
        EdgeName.DUPLICATE_QUESTION,
        EdgeName.IDLE_STOP,
        EdgeName.RUNNING_BACKGROUND_CHATTER,
        EdgeName.STEER_ACTIVE_TURN,
        EdgeName.REPEAT_STEER,
        EdgeName.QUEUE_QUESTION,
        EdgeName.STOP_PREPARING,
        EdgeName.STOP_RUNNING,
        EdgeName.STOP_RECOVERING,
        EdgeName.STOP_DELIVERING,
        EdgeName.DUPLICATE_STOP,
    }
)


async def _wiseman_action(context: GraphContext, edge: EdgeName, message_id: str) -> WisemanResponse:
    thread_id = context.state.chat.thread_id
    if edge is EdgeName.ADMIT_QUESTION:
        return await context.wiseman.ask(message_id, thread_id=thread_id, startup=True)
    if edge is EdgeName.QUEUE_QUESTION:
        return await context.wiseman.ask(message_id, thread_id=thread_id)
    if edge is EdgeName.DUPLICATE_QUESTION:
        return await context.wiseman.ask(message_id, thread_id=thread_id, duplicate=True)
    if edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER}:
        return await context.wiseman.background(message_id, thread_id=thread_id)
    if edge in {EdgeName.STEER_ACTIVE_TURN, EdgeName.REPEAT_STEER}:
        reply_to = context.observation.chat.progress_message_ids[-1] if context.observation and context.observation.chat.progress_message_ids else "answer"
        return await context.wiseman.steer(message_id, thread_id=thread_id, reply_to=reply_to)
    return await context.wiseman.stop(message_id, thread_id=thread_id)


async def _edge(context: GraphContext, edge: EdgeName, message_id: str = "") -> None:
    definition = EDGES_BY_NAME[edge]
    resolved_message_id = message_id or context.message_id
    if edge is EdgeName.DUPLICATE_QUESTION and context.state.wiseman.seen_questions:
        resolved_message_id = next(iter(context.state.wiseman.seen_questions))
    if edge is EdgeName.DUPLICATE_STOP and context.state.chat.stop_commands:
        resolved_message_id = next(iter(context.state.chat.stop_commands))
    context.begin_edge(definition, resolved_message_id)
    await context.harness.prepare_edge(definition, context, deadline_seconds=EDGE_TIMEOUTS[edge])
    if edge is EdgeName.DUPLICATE_QUESTION and not context.state.wiseman.seen_questions:
        context.last_response = WisemanResponse(200, {"status": "ignored", "message_id": resolved_message_id})
        await apply_edge(context, definition, EDGE_TIMEOUTS[edge], resolved_message_id, execute_boundary=False)
        return
    if edge in ACTION_EDGES:
        context.last_response = await _wiseman_action(context, edge, resolved_message_id)
    transition_message_id = resolved_message_id
    if edge is EdgeName.BACKGROUND_CHATTER and definition.source.value == "idle":
        transition_message_id = ""
    await apply_edge(
        context,
        definition,
        EDGE_TIMEOUTS[edge],
        transition_message_id,
        execute_boundary=True,
    )


async def admit_question(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.ADMIT_QUESTION, message_id)


async def background_chatter(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.BACKGROUND_CHATTER, message_id)


async def duplicate_question(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.DUPLICATE_QUESTION, message_id)


async def idle_stop(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.IDLE_STOP, message_id)


async def context_ready(context: GraphContext) -> None:
    await _edge(context, EdgeName.CONTEXT_READY)


async def preparation_failed(context: GraphContext) -> None:
    await _edge(context, EdgeName.PREPARATION_FAILED)


async def stop_preparing(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.STOP_PREPARING, message_id)


async def running_background_chatter(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.RUNNING_BACKGROUND_CHATTER, message_id)


async def steer_active_turn(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.STEER_ACTIVE_TURN, message_id)


async def repeat_steer(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.REPEAT_STEER, message_id)


async def queue_question(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.QUEUE_QUESTION, message_id)


async def progress_preview(context: GraphContext) -> None:
    await _edge(context, EdgeName.PROGRESS_PREVIEW)


async def worker_restart_running(context: GraphContext) -> None:
    await _edge(context, EdgeName.WORKER_RESTART_RUNNING)


async def inference_complete(context: GraphContext) -> None:
    await _edge(context, EdgeName.INFERENCE_COMPLETE)


async def transient_failure(context: GraphContext) -> None:
    await _edge(context, EdgeName.TRANSIENT_FAILURE)


async def permanent_failure(context: GraphContext) -> None:
    await _edge(context, EdgeName.PERMANENT_FAILURE)


async def execution_uncertain(context: GraphContext) -> None:
    await _edge(context, EdgeName.EXECUTION_UNCERTAIN)


async def resume_session(context: GraphContext) -> None:
    await _edge(context, EdgeName.RESUME_SESSION)


async def retry_exhausted(context: GraphContext) -> None:
    await _edge(context, EdgeName.RETRY_EXHAUSTED)


async def stop_running(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.STOP_RUNNING, message_id)


async def stop_recovering(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.STOP_RECOVERING, message_id)


async def stop_delivering(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.STOP_DELIVERING, message_id)


async def duplicate_stop(context: GraphContext, message_id: str = "") -> None:
    await _edge(context, EdgeName.DUPLICATE_STOP, message_id)


async def worker_restart_cancelling(context: GraphContext) -> None:
    await _edge(context, EdgeName.WORKER_RESTART_CANCELLING)


async def completion_race(context: GraphContext) -> None:
    await _edge(context, EdgeName.COMPLETION_RACE)


async def stop_confirmed(context: GraphContext) -> None:
    await _edge(context, EdgeName.STOP_CONFIRMED)


async def cancellation_unknown(context: GraphContext) -> None:
    await _edge(context, EdgeName.CANCELLATION_UNKNOWN)


async def delivery_retry(context: GraphContext) -> None:
    await _edge(context, EdgeName.DELIVERY_RETRY)


async def worker_restart_delivering(context: GraphContext) -> None:
    await _edge(context, EdgeName.WORKER_RESTART_DELIVERING)


async def answer_finalized(context: GraphContext) -> None:
    await _edge(context, EdgeName.ANSWER_FINALIZED)


async def worker_restart_error(context: GraphContext) -> None:
    await _edge(context, EdgeName.WORKER_RESTART_ERROR)


async def error_finalized(context: GraphContext) -> None:
    await _edge(context, EdgeName.ERROR_FINALIZED)


async def outcome_established(context: GraphContext) -> None:
    await _edge(context, EdgeName.OUTCOME_ESTABLISHED)


async def idle_retirement(context: GraphContext) -> None:
    await _edge(context, EdgeName.IDLE_RETIREMENT)


async def fixture_reset(context: GraphContext) -> None:
    await _edge(context, EdgeName.FIXTURE_RESET)


EDGE_FUNCTIONS: Mapping[EdgeName, EdgeFunction] = {
    EdgeName.ADMIT_QUESTION: admit_question,
    EdgeName.BACKGROUND_CHATTER: background_chatter,
    EdgeName.DUPLICATE_QUESTION: duplicate_question,
    EdgeName.IDLE_STOP: idle_stop,
    EdgeName.CONTEXT_READY: context_ready,
    EdgeName.PREPARATION_FAILED: preparation_failed,
    EdgeName.STOP_PREPARING: stop_preparing,
    EdgeName.RUNNING_BACKGROUND_CHATTER: running_background_chatter,
    EdgeName.STEER_ACTIVE_TURN: steer_active_turn,
    EdgeName.REPEAT_STEER: repeat_steer,
    EdgeName.QUEUE_QUESTION: queue_question,
    EdgeName.PROGRESS_PREVIEW: progress_preview,
    EdgeName.WORKER_RESTART_RUNNING: worker_restart_running,
    EdgeName.INFERENCE_COMPLETE: inference_complete,
    EdgeName.TRANSIENT_FAILURE: transient_failure,
    EdgeName.PERMANENT_FAILURE: permanent_failure,
    EdgeName.EXECUTION_UNCERTAIN: execution_uncertain,
    EdgeName.RESUME_SESSION: resume_session,
    EdgeName.RETRY_EXHAUSTED: retry_exhausted,
    EdgeName.STOP_RUNNING: stop_running,
    EdgeName.STOP_RECOVERING: stop_recovering,
    EdgeName.STOP_DELIVERING: stop_delivering,
    EdgeName.DUPLICATE_STOP: duplicate_stop,
    EdgeName.WORKER_RESTART_CANCELLING: worker_restart_cancelling,
    EdgeName.COMPLETION_RACE: completion_race,
    EdgeName.STOP_CONFIRMED: stop_confirmed,
    EdgeName.CANCELLATION_UNKNOWN: cancellation_unknown,
    EdgeName.DELIVERY_RETRY: delivery_retry,
    EdgeName.WORKER_RESTART_DELIVERING: worker_restart_delivering,
    EdgeName.ANSWER_FINALIZED: answer_finalized,
    EdgeName.WORKER_RESTART_ERROR: worker_restart_error,
    EdgeName.ERROR_FINALIZED: error_finalized,
    EdgeName.OUTCOME_ESTABLISHED: outcome_established,
    EdgeName.IDLE_RETIREMENT: idle_retirement,
    EdgeName.FIXTURE_RESET: fixture_reset,
}
