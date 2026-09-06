# Copyright (c) 2026 Nick van der Merwe
"""The executable GraphWalker model for the Discord/Temporal boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypedDict

GraphValue = str | int | bool


class GraphVertex(TypedDict):
    id: str
    name: str
    properties: dict[str, GraphValue]


class GraphEdge(TypedDict):
    id: str
    name: str
    sourceVertexId: str
    targetVertexId: str
    properties: dict[str, GraphValue]


class GraphDocumentModel(TypedDict):
    id: str
    name: str
    generator: str
    startElementId: str
    properties: dict[str, GraphValue]
    vertices: list[GraphVertex]
    edges: list[GraphEdge]


class GraphDocument(TypedDict):
    name: str
    models: list[GraphDocumentModel]


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


class EdgeName(StrEnum):
    ADMIT_QUESTION = "admit-question"
    BACKGROUND_CHATTER = "background-chatter"
    DUPLICATE_QUESTION = "duplicate-question"
    IDLE_STOP = "idle-stop"
    CONTEXT_READY = "context-ready"
    PREPARATION_FAILED = "preparation-failed"
    STOP_PREPARING = "stop-preparing"
    RUNNING_BACKGROUND_CHATTER = "running-background-chatter"
    STEER_ACTIVE_TURN = "steer-active-turn"
    REPEAT_STEER = "repeat-steer"
    QUEUE_QUESTION = "queue-question"
    PROGRESS_PREVIEW = "progress-preview"
    WORKER_RESTART_RUNNING = "worker-restart-running"
    INFERENCE_COMPLETE = "inference-complete"
    TRANSIENT_FAILURE = "transient-failure"
    PERMANENT_FAILURE = "permanent-failure"
    EXECUTION_UNCERTAIN = "execution-uncertain"
    RESUME_SESSION = "resume-session"
    RETRY_EXHAUSTED = "retry-exhausted"
    STOP_RUNNING = "stop-running"
    STOP_RECOVERING = "stop-recovering"
    STOP_DELIVERING = "stop-delivering"
    DUPLICATE_STOP = "duplicate-stop"
    WORKER_RESTART_CANCELLING = "worker-restart-cancelling"
    COMPLETION_RACE = "completion-race"
    STOP_CONFIRMED = "stop-confirmed"
    CANCELLATION_UNKNOWN = "cancellation-unknown"
    DELIVERY_RETRY = "delivery-retry"
    WORKER_RESTART_DELIVERING = "worker-restart-delivering"
    ANSWER_FINALIZED = "answer-finalized"
    WORKER_RESTART_ERROR = "worker-restart-error"
    ERROR_FINALIZED = "error-finalized"
    OUTCOME_ESTABLISHED = "outcome-established"
    IDLE_RETIREMENT = "idle-retirement"
    FIXTURE_RESET = "fixture-reset"


EDGES: tuple[tuple[str, Vertex, Vertex], ...] = (
    (EdgeName.ADMIT_QUESTION, Vertex.IDLE, Vertex.PREPARING),
    (EdgeName.BACKGROUND_CHATTER, Vertex.IDLE, Vertex.IDLE),
    (EdgeName.DUPLICATE_QUESTION, Vertex.IDLE, Vertex.IDLE),
    (EdgeName.IDLE_STOP, Vertex.IDLE, Vertex.IDLE),
    (EdgeName.CONTEXT_READY, Vertex.PREPARING, Vertex.RUNNING),
    (EdgeName.PREPARATION_FAILED, Vertex.PREPARING, Vertex.ERROR),
    (EdgeName.STOP_PREPARING, Vertex.PREPARING, Vertex.CANCELLING),
    (EdgeName.RUNNING_BACKGROUND_CHATTER, Vertex.RUNNING, Vertex.RUNNING),
    (EdgeName.STEER_ACTIVE_TURN, Vertex.RUNNING, Vertex.RUNNING),
    (EdgeName.REPEAT_STEER, Vertex.RUNNING, Vertex.RUNNING),
    (EdgeName.QUEUE_QUESTION, Vertex.RUNNING, Vertex.RUNNING),
    (EdgeName.PROGRESS_PREVIEW, Vertex.RUNNING, Vertex.RUNNING),
    (EdgeName.WORKER_RESTART_RUNNING, Vertex.RUNNING, Vertex.RUNNING),
    (EdgeName.INFERENCE_COMPLETE, Vertex.RUNNING, Vertex.DELIVERING),
    (EdgeName.TRANSIENT_FAILURE, Vertex.RUNNING, Vertex.RECOVERING),
    (EdgeName.PERMANENT_FAILURE, Vertex.RUNNING, Vertex.ERROR),
    (EdgeName.EXECUTION_UNCERTAIN, Vertex.RUNNING, Vertex.OUTCOME_UNKNOWN),
    (EdgeName.RESUME_SESSION, Vertex.RECOVERING, Vertex.RUNNING),
    (EdgeName.RETRY_EXHAUSTED, Vertex.RECOVERING, Vertex.ERROR),
    (EdgeName.STOP_RUNNING, Vertex.RUNNING, Vertex.CANCELLING),
    (EdgeName.STOP_RECOVERING, Vertex.RECOVERING, Vertex.CANCELLING),
    (EdgeName.STOP_DELIVERING, Vertex.DELIVERING, Vertex.CANCELLING),
    (EdgeName.DUPLICATE_STOP, Vertex.CANCELLING, Vertex.CANCELLING),
    (EdgeName.WORKER_RESTART_CANCELLING, Vertex.CANCELLING, Vertex.CANCELLING),
    (EdgeName.COMPLETION_RACE, Vertex.CANCELLING, Vertex.DELIVERING),
    (EdgeName.STOP_CONFIRMED, Vertex.CANCELLING, Vertex.IDLE),
    (EdgeName.CANCELLATION_UNKNOWN, Vertex.CANCELLING, Vertex.OUTCOME_UNKNOWN),
    (EdgeName.DELIVERY_RETRY, Vertex.DELIVERING, Vertex.DELIVERING),
    (EdgeName.WORKER_RESTART_DELIVERING, Vertex.DELIVERING, Vertex.DELIVERING),
    (EdgeName.ANSWER_FINALIZED, Vertex.DELIVERING, Vertex.IDLE),
    (EdgeName.WORKER_RESTART_ERROR, Vertex.ERROR, Vertex.ERROR),
    (EdgeName.ERROR_FINALIZED, Vertex.ERROR, Vertex.IDLE),
    (EdgeName.OUTCOME_ESTABLISHED, Vertex.OUTCOME_UNKNOWN, Vertex.ERROR),
    (EdgeName.IDLE_RETIREMENT, Vertex.IDLE, Vertex.RETIRED),
    (EdgeName.FIXTURE_RESET, Vertex.RETIRED, Vertex.IDLE),
)


@dataclass(frozen=True, slots=True)
class Edge:
    name: EdgeName
    source: Vertex
    target: Vertex

    @property
    def id(self) -> str:
        return f"e-{self.name}"


GRAPH_EDGES: tuple[Edge, ...] = tuple(Edge(EdgeName(name), source, target) for name, source, target in EDGES)
EDGES_BY_NAME: Mapping[str, Edge] = {edge.name: edge for edge in GRAPH_EDGES}
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
EDGE_TIMEOUTS: Mapping[EdgeName, int] = {edge.name: 60 for edge in GRAPH_EDGES}


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

    def advance(self, edge: EdgeName | str, source: Vertex | None = None, target: Vertex | None = None, message_id: str = "") -> None:
        definition = EDGES_BY_NAME[str(edge)]
        if source is not None and source != definition.source:
            message = f"source mismatch for {edge}: {source} != {definition.source}"
            raise AssertionError(message)
        if target is not None and target != definition.target:
            message = f"target mismatch for {edge}: {target} != {definition.target}"
            raise AssertionError(message)
        if self.vertex != definition.source:
            message = f"{edge} requires {definition.source}, got {self.vertex}"
            raise AssertionError(message)
        self.vertex = definition.target
        self._apply_transition(definition.name, message_id)
        self._assert_invariants()

    def _apply_transition(self, edge: EdgeName, message_id: str) -> None:
        if edge in {EdgeName.ADMIT_QUESTION, EdgeName.QUEUE_QUESTION}:
            self._remember_question(message_id)
        elif edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER}:
            self._remember_background(message_id)
        elif edge in {EdgeName.CONTEXT_READY, EdgeName.RESUME_SESSION}:
            self._consume_background()
        elif edge in {EdgeName.STEER_ACTIVE_TURN, EdgeName.REPEAT_STEER}:
            self._remember_steering(message_id)
        elif edge in {EdgeName.STOP_PREPARING, EdgeName.STOP_RUNNING, EdgeName.STOP_RECOVERING, EdgeName.STOP_DELIVERING}:
            self._remember_stop(message_id)
        elif edge in {EdgeName.STOP_CONFIRMED, EdgeName.ANSWER_FINALIZED, EdgeName.ERROR_FINALIZED}:
            self._settle_active()

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


class GraphHarness(Protocol):
    async def wait_for_state(self, vertex: Vertex, state: ModelState, *, deadline_seconds: int) -> None: ...

    async def execute_edge(self, edge: Edge, state: ModelState, *, deadline_seconds: int) -> None: ...


StateFunction = Callable[[GraphHarness, ModelState], Awaitable[None]]
EdgeFunction = Callable[..., Awaitable[None]]


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


async def _edge(harness: GraphHarness, state: ModelState, edge: EdgeName, message_id: str = "") -> None:
    await harness.execute_edge(EDGES_BY_NAME[edge], state, deadline_seconds=EDGE_TIMEOUTS[edge])
    state.advance(edge, message_id=message_id)


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


def model() -> GraphDocument:
    """Build the complete native GraphWalker document from the registries."""

    vertices: list[GraphVertex] = [
        {
            "id": f"v-{vertex}",
            "name": vertex,
            "properties": {
                "state": vertex,
                "condition": STATE_FUNCTIONS[vertex].__name__,
                "deadline_seconds": STATE_TIMEOUTS[vertex],
            },
        }
        for vertex in Vertex
    ]
    edges: list[GraphEdge] = [
        {
            "id": edge.id,
            "name": edge.name,
            "sourceVertexId": f"v-{edge.source}",
            "targetVertexId": f"v-{edge.target}",
            "properties": {
                "action": EDGE_FUNCTIONS[edge.name].__name__,
                "source_state": edge.source,
                "target_state": edge.target,
                "deadline_seconds": EDGE_TIMEOUTS[edge.name],
            },
        }
        for edge in GRAPH_EDGES
    ]
    return {
        "name": "wiseman-v2",
        "models": [
            {
                "id": "wiseman-v2-lifecycle",
                "name": "Wiseman V2 lifecycle",
                "generator": "random(edge_coverage(100))",
                "startElementId": "v-idle",
                "properties": {
                    "model_version": 1,
                    "initial_state": Vertex.IDLE,
                    "owner": "temporal-thread-workflow",
                    "harness": "authenticated-discord-http-replay",
                    "state_data": "ids,owners,pending,histories,cursors,cancellation_targets",
                },
                "vertices": vertices,
                "edges": edges,
            }
        ],
    }
