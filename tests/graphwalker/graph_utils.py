# Copyright (c) 2026 Nick van der Merwe
"""Shared types and state transitions for the GraphWalker model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import NoReturn, Protocol

from tests.graphwalker.model import FailureDetails, GraphState, RuntimeObservation, Vertex


class GraphElement(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def source(self) -> Vertex: ...

    @property
    def target(self) -> Vertex: ...


class GraphHarness(Protocol):
    async def wait_for_state(self, vertex: Vertex, context: GraphContext, *, deadline_seconds: int) -> RuntimeObservation: ...

    async def execute_edge(self, edge: GraphElement, context: GraphContext, *, deadline_seconds: int) -> None: ...


@dataclass(slots=True)
class GraphContext:
    harness: GraphHarness
    state: GraphState = field(default_factory=GraphState)
    message_id: str = ""

    def reject(self, details: FailureDetails) -> NoReturn:
        self.state.reject(details)


StateFunction = Callable[[GraphContext], Awaitable[None]]
EdgeFunction = Callable[..., Awaitable[None]]


async def apply_edge(context: GraphContext, edge: GraphElement, deadline_seconds: int, message_id: str = "") -> None:
    await context.harness.execute_edge(edge, context, deadline_seconds=deadline_seconds)
    context.state.advance(edge.name, edge.source, edge.target, message_id)
