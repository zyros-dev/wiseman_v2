# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker vertices and independently observable state conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.graphwalker.graph_utils import GraphHarness, ModelState, StateFunction

from tests.graphwalker.model import STATE_TIMEOUTS, Vertex


async def idle(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.IDLE, state, deadline_seconds=STATE_TIMEOUTS[Vertex.IDLE])


async def preparing(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.PREPARING, state, deadline_seconds=STATE_TIMEOUTS[Vertex.PREPARING])


async def running(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.RUNNING, state, deadline_seconds=STATE_TIMEOUTS[Vertex.RUNNING])


async def recovering(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.RECOVERING, state, deadline_seconds=STATE_TIMEOUTS[Vertex.RECOVERING])


async def cancelling(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.CANCELLING, state, deadline_seconds=STATE_TIMEOUTS[Vertex.CANCELLING])


async def delivering(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.DELIVERING, state, deadline_seconds=STATE_TIMEOUTS[Vertex.DELIVERING])


async def outcome_unknown(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.OUTCOME_UNKNOWN, state, deadline_seconds=STATE_TIMEOUTS[Vertex.OUTCOME_UNKNOWN])


async def error(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.ERROR, state, deadline_seconds=STATE_TIMEOUTS[Vertex.ERROR])


async def retired(harness: GraphHarness, state: ModelState) -> None:
    await harness.wait_for_state(Vertex.RETIRED, state, deadline_seconds=STATE_TIMEOUTS[Vertex.RETIRED])


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
