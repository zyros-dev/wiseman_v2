# Copyright (c) 2026 Nick van der Merwe
from dataclasses import dataclass
from enum import StrEnum


class Vertex(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    RECOVERING = "recovering"
    CANCELLING = "cancelling"
    DELIVERING = "delivering"
    ERROR = "error"
    RETIRED = "retired"


EDGES = (
    ("admit-question", Vertex.IDLE, Vertex.PREPARING),
    ("background-chatter", Vertex.IDLE, Vertex.IDLE),
    ("duplicate-question", Vertex.IDLE, Vertex.IDLE),
    ("idle-stop", Vertex.IDLE, Vertex.IDLE),
    ("context-ready", Vertex.PREPARING, Vertex.RUNNING),
    ("preparation-failed", Vertex.PREPARING, Vertex.ERROR),
    ("steer-active-turn", Vertex.RUNNING, Vertex.RUNNING),
    ("repeat-steer", Vertex.RUNNING, Vertex.RUNNING),
    ("queue-question", Vertex.RUNNING, Vertex.RUNNING),
    ("worker-restart", Vertex.RUNNING, Vertex.RUNNING),
    ("inference-complete", Vertex.RUNNING, Vertex.DELIVERING),
    ("transient-failure", Vertex.RUNNING, Vertex.RECOVERING),
    ("resume-session", Vertex.RECOVERING, Vertex.RUNNING),
    ("stop-active-turn", Vertex.RUNNING, Vertex.CANCELLING),
    ("stop-active-recovery", Vertex.RECOVERING, Vertex.CANCELLING),
    ("stop-active-delivery", Vertex.DELIVERING, Vertex.CANCELLING),
    ("duplicate-stop", Vertex.CANCELLING, Vertex.CANCELLING),
    ("completion-race", Vertex.CANCELLING, Vertex.DELIVERING),
    ("stop-active-turn", Vertex.PREPARING, Vertex.CANCELLING),
    ("stop-confirmed", Vertex.CANCELLING, Vertex.IDLE),
    ("unknown-outcome", Vertex.CANCELLING, Vertex.ERROR),
    ("error-finalized", Vertex.ERROR, Vertex.IDLE),
    ("delivery-finalized", Vertex.DELIVERING, Vertex.IDLE),
    ("delivery-retry", Vertex.DELIVERING, Vertex.DELIVERING),
    ("idle-retirement", Vertex.IDLE, Vertex.RETIRED),
    ("fixture-reset", Vertex.RETIRED, Vertex.IDLE),
)


@dataclass
class ModelState:
    vertex: Vertex = Vertex.IDLE
    pending: int = 0
    turns: int = 0

    def advance(self, name: str, source: Vertex, target: Vertex, message_id: str = "") -> None:
        assert self.vertex == source
        assert (name, source, target) in EDGES
        self.vertex = target
        if name in {"admit-question", "queue-question"} and message_id:
            self.pending += 1
        if name in {"stop-confirmed", "error-finalized", "delivery-finalized"} and self.pending:
            self.pending -= 1
            self.turns += 1


def model() -> dict[str, object]:
    return {
        "models": [
            {
                "id": "wiseman-v2",
                "name": "wiseman-v2",
                "generator": "random(edge_coverage(100))",
                "startElementId": Vertex.IDLE,
                "vertices": [{"id": vertex, "name": vertex} for vertex in Vertex],
                "edges": [
                    {
                        "id": f"{name}-{source}-{target}",
                        "name": name,
                        "sourceVertexId": source,
                        "targetVertexId": target,
                    }
                    for name, source, target in EDGES
                ],
            }
        ]
    }
