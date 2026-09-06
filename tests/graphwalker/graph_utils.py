# Copyright (c) 2026 Nick van der Merwe
"""Shared types and state transitions for the GraphWalker model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, Protocol

from tests.graphwalker.model import FailureDetails, GraphState, RuntimeObservation, Vertex

if TYPE_CHECKING:
    from app.clients.client_interfaces import ClientContainer


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
    clients: ClientContainer
    state: GraphState = field(default_factory=GraphState)
    previous_state: GraphState | None = None
    last_edge: str | None = None
    last_target: Vertex | None = None
    last_message_id: str = ""
    message_id: str = ""

    def reject(self, details: FailureDetails) -> NoReturn:
        self.state.reject(details)


StateFunction = Callable[[GraphContext], Awaitable[None]]
EdgeFunction = Callable[..., Awaitable[None]]


async def apply_edge(context: GraphContext, edge: GraphElement, deadline_seconds: int, message_id: str = "") -> None:
    context.previous_state = deepcopy(context.state)
    context.last_edge = edge.name
    context.last_target = edge.target
    context.last_message_id = message_id or context.message_id
    await context.harness.execute_edge(edge, context, deadline_seconds=deadline_seconds)
    context.state.advance(edge.name, edge.source, edge.target, context.last_message_id)
