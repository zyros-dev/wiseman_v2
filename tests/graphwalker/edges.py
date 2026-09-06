# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker edges and their executable boundary actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.graphwalker.graph_utils import EdgeFunction, GraphHarness, ModelState, apply_edge
from tests.graphwalker.model import EDGE_TIMEOUTS, EDGES_BY_NAME, EdgeName

if TYPE_CHECKING:
    from collections.abc import Mapping


async def _edge(harness: GraphHarness, state: ModelState, edge: EdgeName, message_id: str = "") -> None:
    definition = EDGES_BY_NAME[edge]
    await apply_edge(harness, state, definition, EDGE_TIMEOUTS[edge], message_id)


async def admit_question(harness: GraphHarness, state: ModelState, message_id: str = "q") -> None:
    await _edge(harness, state, EdgeName.ADMIT_QUESTION, message_id)


async def background_chatter(harness: GraphHarness, state: ModelState, message_id: str = "background") -> None:
    await _edge(harness, state, EdgeName.BACKGROUND_CHATTER, message_id)


async def duplicate_question(harness: GraphHarness, state: ModelState, message_id: str = "q") -> None:
    await _edge(harness, state, EdgeName.DUPLICATE_QUESTION, message_id)


async def idle_stop(harness: GraphHarness, state: ModelState, message_id: str = "stop") -> None:
    await _edge(harness, state, EdgeName.IDLE_STOP, message_id)


async def context_ready(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.CONTEXT_READY)


async def preparation_failed(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.PREPARATION_FAILED)


async def stop_preparing(harness: GraphHarness, state: ModelState, message_id: str = "stop") -> None:
    await _edge(harness, state, EdgeName.STOP_PREPARING, message_id)


async def running_background_chatter(harness: GraphHarness, state: ModelState, message_id: str = "background") -> None:
    await _edge(harness, state, EdgeName.RUNNING_BACKGROUND_CHATTER, message_id)


async def steer_active_turn(harness: GraphHarness, state: ModelState, message_id: str = "steer") -> None:
    await _edge(harness, state, EdgeName.STEER_ACTIVE_TURN, message_id)


async def repeat_steer(harness: GraphHarness, state: ModelState, message_id: str = "steer-2") -> None:
    await _edge(harness, state, EdgeName.REPEAT_STEER, message_id)


async def queue_question(harness: GraphHarness, state: ModelState, message_id: str = "q-next") -> None:
    await _edge(harness, state, EdgeName.QUEUE_QUESTION, message_id)


async def progress_preview(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.PROGRESS_PREVIEW)


async def worker_restart_running(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.WORKER_RESTART_RUNNING)


async def inference_complete(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.INFERENCE_COMPLETE)


async def transient_failure(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.TRANSIENT_FAILURE)


async def permanent_failure(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.PERMANENT_FAILURE)


async def execution_uncertain(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.EXECUTION_UNCERTAIN)


async def resume_session(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.RESUME_SESSION)


async def retry_exhausted(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.RETRY_EXHAUSTED)


async def stop_running(harness: GraphHarness, state: ModelState, message_id: str = "stop") -> None:
    await _edge(harness, state, EdgeName.STOP_RUNNING, message_id)


async def stop_recovering(harness: GraphHarness, state: ModelState, message_id: str = "stop") -> None:
    await _edge(harness, state, EdgeName.STOP_RECOVERING, message_id)


async def stop_delivering(harness: GraphHarness, state: ModelState, message_id: str = "stop") -> None:
    await _edge(harness, state, EdgeName.STOP_DELIVERING, message_id)


async def duplicate_stop(harness: GraphHarness, state: ModelState, message_id: str = "stop") -> None:
    await _edge(harness, state, EdgeName.DUPLICATE_STOP, message_id)


async def worker_restart_cancelling(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.WORKER_RESTART_CANCELLING)


async def completion_race(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.COMPLETION_RACE)


async def stop_confirmed(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.STOP_CONFIRMED)


async def cancellation_unknown(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.CANCELLATION_UNKNOWN)


async def delivery_retry(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.DELIVERY_RETRY)


async def worker_restart_delivering(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.WORKER_RESTART_DELIVERING)


async def answer_finalized(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.ANSWER_FINALIZED)


async def worker_restart_error(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.WORKER_RESTART_ERROR)


async def error_finalized(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.ERROR_FINALIZED)


async def outcome_established(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.OUTCOME_ESTABLISHED)


async def idle_retirement(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.IDLE_RETIREMENT)


async def fixture_reset(harness: GraphHarness, state: ModelState) -> None:
    await _edge(harness, state, EdgeName.FIXTURE_RESET)


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
