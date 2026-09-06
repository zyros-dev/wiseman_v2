# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker vertices and independently observable state conditions."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

if TYPE_CHECKING:
    from tests.graphwalker.graph_utils import GraphHarness, ModelState, StateFunction


class Vertex(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    RECOVERING = "recovering"
    CANCELLING = "cancelling"
    DELIVERING = "delivering"
    OUTCOME_UNKNOWN = "outcome-unknown"
    ERROR = "error"
    RETIRED = "retired"


STATE_TIMEOUTS: Mapping[Vertex, int] = {
    Vertex.IDLE: 10,
    Vertex.PREPARING: 60,
    Vertex.RUNNING: 60,
    Vertex.RECOVERING: 60,
    Vertex.CANCELLING: 30,
    Vertex.DELIVERING: 60,
    Vertex.OUTCOME_UNKNOWN: 60,
    Vertex.ERROR: 30,
    Vertex.RETIRED: 10,
}


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
