# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker vertices and independently observable state conditions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tests.graphwalker.graph_utils import ModelState
from tests.graphwalker.model import STATE_TIMEOUTS, Vertex

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.graphwalker.graph_utils import GraphHarness, StateFunction


StateAssertion = Callable[[ModelState], None]


def _assert_idle(state: ModelState) -> None:
    if state.active_question is not None:
        message = f"idle state retained active question {state.active_question}"
        raise AssertionError(message)


def _assert_active_turn(state: ModelState) -> None:
    if state.active_question is None:
        raise AssertionError("active lifecycle state has no active question")


def _assert_retired(state: ModelState) -> None:
    if state.active_question is not None or state.pending_questions:
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


async def _wait_for_vertex(harness: GraphHarness, state: ModelState, vertex: Vertex) -> None:
    if state.vertex is not vertex:
        message = f"{vertex} condition requires model state {vertex}, got {state.vertex}"
        raise AssertionError(message)
    STATE_ASSERTIONS[vertex](state)
    await harness.wait_for_state(vertex, state, deadline_seconds=STATE_TIMEOUTS[vertex])


async def idle(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.IDLE)


async def preparing(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.PREPARING)


async def running(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.RUNNING)


async def recovering(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.RECOVERING)


async def cancelling(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.CANCELLING)


async def delivering(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.DELIVERING)


async def outcome_unknown(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.OUTCOME_UNKNOWN)


async def error(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.ERROR)


async def retired(harness: GraphHarness, state: ModelState) -> None:
    await _wait_for_vertex(harness, state, Vertex.RETIRED)


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
