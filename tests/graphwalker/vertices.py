# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker vertices and independently observable state conditions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tests.graphwalker.model import STATE_TIMEOUTS, GraphState, Vertex

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.graphwalker.graph_utils import GraphContext, StateFunction


StateAssertion = Callable[[GraphState], None]


def _assert_idle(state: GraphState) -> None:
    if state.wiseman.active_question is not None:
        message = f"idle state retained active question {state.wiseman.active_question}"
        raise AssertionError(message)


def _assert_active_turn(state: GraphState) -> None:
    if state.wiseman.active_question is None:
        raise AssertionError("active lifecycle state has no active question")


def _assert_retired(state: GraphState) -> None:
    if state.wiseman.active_question is not None or state.wiseman.pending_questions:
        raise AssertionError("retired state retained active or queued work")


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


async def _wait_for_vertex(context: GraphContext, vertex: Vertex) -> None:
    state = context.state
    if state.vertex is not vertex:
        message = f"{vertex} condition requires model state {vertex}, got {state.vertex}"
        raise AssertionError(message)
    STATE_ASSERTIONS[vertex](state)
    await context.harness.wait_for_state(vertex, context, deadline_seconds=STATE_TIMEOUTS[vertex])


async def idle(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.IDLE)


async def preparing(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.PREPARING)


async def running(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.RUNNING)


async def recovering(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.RECOVERING)


async def cancelling(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.CANCELLING)


async def delivering(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.DELIVERING)


async def outcome_unknown(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.OUTCOME_UNKNOWN)


async def error(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.ERROR)


async def retired(context: GraphContext) -> None:
    await _wait_for_vertex(context, Vertex.RETIRED)


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
