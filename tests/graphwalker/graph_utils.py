# Copyright (c) 2026 Nick van der Merwe
"""Shared types and state transitions for the GraphWalker model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, Protocol

from tests.graphwalker.model import EdgeName, FailureDetails, GraphState, RuntimeObservation, Vertex

if TYPE_CHECKING:
    from app.clients.client_interfaces import ClientContainer
    from tests.graphwalker.wiseman_client import WisemanClient, WisemanResponse


class GraphElement(Protocol):
    @property
    def name(self) -> EdgeName: ...

    @property
    def source(self) -> Vertex: ...

    @property
    def target(self) -> Vertex: ...


class GraphHarness(Protocol):
    async def wait_for_state(self, vertex: Vertex, context: GraphContext, *, deadline_seconds: int) -> RuntimeObservation: ...

    async def prepare_edge(self, edge: GraphElement, context: GraphContext, *, deadline_seconds: int) -> None: ...

    async def execute_edge(self, edge: GraphElement, context: GraphContext, *, deadline_seconds: int) -> None: ...


@dataclass(slots=True)
class GraphContext:
    harness: GraphHarness
    clients: ClientContainer
    wiseman: WisemanClient
    state: GraphState = field(default_factory=GraphState)
    previous_state: GraphState | None = None
    last_edge: str | None = None
    last_target: Vertex | None = None
    last_message_id: str = ""
    message_id: str = ""
    last_response: WisemanResponse | None = None
    previous_observation: RuntimeObservation | None = None
    observation: RuntimeObservation | None = None

    def reject(self, details: FailureDetails) -> NoReturn:
        self.state.reject(details)

    def begin_edge(self, edge: GraphElement, message_id: str) -> None:
        self.previous_state = deepcopy(self.state)
        self.previous_observation = self.observation
        self.last_edge = edge.name
        self.last_target = edge.target
        self.last_message_id = message_id or self.message_id


StateFunction = Callable[[GraphContext], Awaitable[None]]
EdgeFunction = Callable[..., Awaitable[None]]


async def apply_edge(
    context: GraphContext,
    edge: GraphElement,
    deadline_seconds: int,
    message_id: str = "",
    *,
    execute_boundary: bool = True,
) -> None:
    if context.last_edge != edge.name or context.previous_state is None:
        context.begin_edge(edge, message_id)
    if execute_boundary:
        await context.harness.execute_edge(edge, context, deadline_seconds=deadline_seconds)
    context.state.advance(edge.name, edge.source, edge.target, context.last_message_id)
