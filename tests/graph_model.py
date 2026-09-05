# Copyright (c) 2026 Nick van der Merwe
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
    ("context-ready", Vertex.PREPARING, Vertex.RUNNING),
    ("preparation-failed", Vertex.PREPARING, Vertex.ERROR),
    ("steer-active-turn", Vertex.RUNNING, Vertex.RUNNING),
    ("queue-question", Vertex.RUNNING, Vertex.RUNNING),
    ("worker-restart", Vertex.RUNNING, Vertex.RUNNING),
    ("inference-complete", Vertex.RUNNING, Vertex.DELIVERING),
    ("transient-failure", Vertex.RUNNING, Vertex.RECOVERING),
    ("resume-session", Vertex.RECOVERING, Vertex.RUNNING),
    ("stop-active-turn", Vertex.RUNNING, Vertex.CANCELLING),
    ("stop-active-turn", Vertex.PREPARING, Vertex.CANCELLING),
    ("stop-confirmed", Vertex.CANCELLING, Vertex.IDLE),
    ("unknown-outcome", Vertex.CANCELLING, Vertex.ERROR),
    ("error-finalized", Vertex.ERROR, Vertex.IDLE),
    ("delivery-finalized", Vertex.DELIVERING, Vertex.IDLE),
    ("delivery-retry", Vertex.DELIVERING, Vertex.DELIVERING),
    ("idle-retirement", Vertex.IDLE, Vertex.RETIRED),
    ("fixture-reset", Vertex.RETIRED, Vertex.IDLE),
)


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
