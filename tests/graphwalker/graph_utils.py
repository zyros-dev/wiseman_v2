# Copyright (c) 2026 Nick van der Merwe
"""Shared types and state transitions for the GraphWalker model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from tests.graphwalker.model import Vertex


@dataclass(slots=True)
class ModelState:
    vertex: Vertex = Vertex.IDLE
    pending_questions: list[str] = field(default_factory=list)
    active_question: str | None = None
    seen_questions: set[str] = field(default_factory=set)
    background_context: list[str] = field(default_factory=list)
    consumed_context: list[str] = field(default_factory=list)
    steering_messages: list[str] = field(default_factory=list)
    stop_commands: set[str] = field(default_factory=set)
    settled_questions: set[str] = field(default_factory=set)
    turns: int = 0

    @property
    def pending(self) -> int:
        return len(self.pending_questions)

    def advance(self, edge: str, source: Vertex, target: Vertex, message_id: str = "") -> None:
        if self.vertex != source:
            message = f"{edge} requires {source}, got {self.vertex}"
            raise AssertionError(message)
        self.vertex = target
        if edge in {"admit-question", "queue-question"}:
            self._remember_question(message_id)
        elif edge in {"background-chatter", "running-background-chatter"}:
            self._remember_background(message_id)
        elif edge in {"context-ready", "resume-session"}:
            self._consume_background()
        elif edge in {"steer-active-turn", "repeat-steer"}:
            self._remember_steering(message_id)
        elif edge in {"stop-preparing", "stop-running", "stop-recovering", "stop-delivering"}:
            self._remember_stop(message_id)
        elif edge in {"stop-confirmed", "answer-finalized", "error-finalized"}:
            self._settle_active()
        self._assert_invariants()

    def _remember_question(self, message_id: str) -> None:
        if message_id and message_id not in self.seen_questions:
            self.pending_questions.append(message_id)
            self.seen_questions.add(message_id)

    def _remember_background(self, message_id: str) -> None:
        if message_id:
            self.background_context.append(message_id)

    def _consume_background(self) -> None:
        if self.active_question is None and self.pending_questions:
            self.active_question = self.pending_questions[0]
        self.consumed_context.extend(self.background_context)
        self.background_context.clear()

    def _remember_steering(self, message_id: str) -> None:
        if message_id:
            self.steering_messages.append(message_id)

    def _remember_stop(self, message_id: str) -> None:
        if message_id:
            self.stop_commands.add(message_id)

    def _settle_active(self) -> None:
        if self.active_question is None:
            return
        self.settled_questions.add(self.active_question)
        if self.pending_questions and self.pending_questions[0] == self.active_question:
            self.pending_questions.pop(0)
        self.active_question = None
        self.turns += 1

    def _assert_invariants(self) -> None:
        if len(self.pending_questions) != len(set(self.pending_questions)):
            raise AssertionError("pending questions must be ordered and unique")
        if self.vertex is Vertex.RETIRED and (self.active_question or self.pending_questions):
            raise AssertionError("retired conversations cannot retain active work")
        if self.active_question and self.active_question not in self.pending_questions:
            raise AssertionError("active question must remain pending until terminal delivery")


class GraphElement(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def source(self) -> Vertex: ...

    @property
    def target(self) -> Vertex: ...


class GraphHarness(Protocol):
    async def wait_for_state(self, vertex: Vertex, state: ModelState, *, deadline_seconds: int) -> None: ...

    async def execute_edge(self, edge: GraphElement, state: ModelState, *, deadline_seconds: int) -> None: ...


StateFunction = Callable[[GraphHarness, ModelState], Awaitable[None]]
EdgeFunction = Callable[..., Awaitable[None]]


async def apply_edge(harness: GraphHarness, state: ModelState, edge: GraphElement, deadline_seconds: int, message_id: str = "") -> None:
    await harness.execute_edge(edge, state, deadline_seconds=deadline_seconds)
    state.advance(edge.name, edge.source, edge.target, message_id)
