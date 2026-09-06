# Copyright (c) 2026 Nick van der Merwe
"""Graph data definitions and native GraphWalker document serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

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


class ChatMessageKind(StrEnum):
    QUESTION = "question"
    BACKGROUND = "background"
    STEERING = "steering"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    id: str
    kind: ChatMessageKind
    content: str
    mentions_bot: bool
    reply_to: str | None = None


@dataclass(slots=True)
class ChatState:
    thread_id: str = "thread"
    messages: list[ChatMessage] = field(default_factory=list)
    message_ids: set[str] = field(default_factory=set)
    background_context: list[str] = field(default_factory=list)
    consumed_context: list[str] = field(default_factory=list)
    steering_messages: list[str] = field(default_factory=list)
    stop_commands: set[str] = field(default_factory=set)
    archived: bool = False

    def record(self, message_id: str, kind: ChatMessageKind) -> None:
        if not message_id or message_id in self.message_ids:
            return
        content = {
            ChatMessageKind.QUESTION: "question",
            ChatMessageKind.BACKGROUND: "background",
            ChatMessageKind.STEERING: "steer",
            ChatMessageKind.STOP: "/stop",
        }[kind]
        mentions_bot = kind in {ChatMessageKind.QUESTION, ChatMessageKind.STEERING}
        reply_to = "answer" if kind is ChatMessageKind.STEERING else None
        self.messages.append(ChatMessage(message_id, kind, content, mentions_bot, reply_to))
        self.message_ids.add(message_id)


@dataclass(slots=True)
class WisemanState:
    pending_questions: list[str] = field(default_factory=list)
    active_question: str | None = None
    seen_questions: set[str] = field(default_factory=set)
    settled_questions: set[str] = field(default_factory=set)
    session_id: str | None = None
    turns: int = 0


@dataclass(slots=True)
class GraphState:
    vertex: Vertex = Vertex.IDLE
    chat: ChatState = field(default_factory=ChatState)
    wiseman: WisemanState = field(default_factory=WisemanState)

    @property
    def pending(self) -> int:
        return len(self.wiseman.pending_questions)

    def advance(self, edge: str, source: Vertex, target: Vertex, message_id: str = "") -> None:
        if self.vertex != source:
            message = f"{edge} requires {source}, got {self.vertex}"
            raise AssertionError(message)
        self.vertex = target
        self._record_chat_message(edge, message_id)
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
        if message_id and message_id not in self.wiseman.seen_questions:
            self.wiseman.pending_questions.append(message_id)
            if self.wiseman.active_question is None:
                self.wiseman.active_question = message_id
            self.wiseman.seen_questions.add(message_id)

    def _remember_background(self, message_id: str) -> None:
        if message_id:
            self.chat.background_context.append(message_id)

    def _consume_background(self) -> None:
        if self.wiseman.active_question is None and self.wiseman.pending_questions:
            self.wiseman.active_question = self.wiseman.pending_questions[0]
        self.chat.consumed_context.extend(self.chat.background_context)
        self.chat.background_context.clear()

    def _remember_steering(self, message_id: str) -> None:
        if message_id:
            self.chat.steering_messages.append(message_id)

    def _remember_stop(self, message_id: str) -> None:
        if message_id:
            self.chat.stop_commands.add(message_id)

    def _settle_active(self) -> None:
        active_question = self.wiseman.active_question
        if active_question is None:
            return
        self.wiseman.settled_questions.add(active_question)
        if self.wiseman.pending_questions and self.wiseman.pending_questions[0] == active_question:
            self.wiseman.pending_questions.pop(0)
        self.wiseman.active_question = None
        self.wiseman.turns += 1

    def _assert_invariants(self) -> None:
        if len(self.wiseman.pending_questions) != len(set(self.wiseman.pending_questions)):
            raise AssertionError("pending questions must be ordered and unique")
        if len(self.chat.messages) != len(self.chat.message_ids):
            raise AssertionError("chat message history must be deduplicated")
        if self.vertex is Vertex.RETIRED and (self.wiseman.active_question or self.wiseman.pending_questions):
            raise AssertionError("retired conversations cannot retain active work")
        if self.wiseman.active_question and self.wiseman.active_question not in self.wiseman.pending_questions:
            raise AssertionError("active question must remain pending until terminal delivery")

    def _record_chat_message(self, edge: str, message_id: str) -> None:
        kind = {
            "admit-question": ChatMessageKind.QUESTION,
            "queue-question": ChatMessageKind.QUESTION,
            "duplicate-question": ChatMessageKind.QUESTION,
            "background-chatter": ChatMessageKind.BACKGROUND,
            "running-background-chatter": ChatMessageKind.BACKGROUND,
            "steer-active-turn": ChatMessageKind.STEERING,
            "repeat-steer": ChatMessageKind.STEERING,
            "idle-stop": ChatMessageKind.STOP,
            "stop-preparing": ChatMessageKind.STOP,
            "stop-running": ChatMessageKind.STOP,
            "stop-recovering": ChatMessageKind.STOP,
            "stop-delivering": ChatMessageKind.STOP,
            "duplicate-stop": ChatMessageKind.STOP,
        }.get(edge)
        if kind is not None:
            self.chat.record(message_id, kind)


@dataclass(frozen=True, slots=True)
class Edge:
    name: EdgeName
    source: Vertex
    target: Vertex

    @property
    def id(self) -> str:
        return f"e-{self.name}"


EDGES: tuple[tuple[EdgeName, Vertex, Vertex], ...] = (
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


def model() -> GraphDocument:
    """Build the complete GraphWalker JSON-shaped document."""

    from tests.graphwalker.edges import EDGE_FUNCTIONS  # noqa: PLC0415
    from tests.graphwalker.vertices import STATE_FUNCTIONS  # noqa: PLC0415

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
