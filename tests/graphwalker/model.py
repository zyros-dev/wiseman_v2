# Copyright (c) 2026 Nick van der Merwe
"""Native GraphWalker document assembled from the vertex and edge modules."""

from tests.graphwalker.edges import EDGE_FUNCTIONS, EDGE_TIMEOUTS, GRAPH_EDGES
from tests.graphwalker.graph_utils import GraphDocument, GraphEdge, GraphVertex
from tests.graphwalker.vertices import STATE_FUNCTIONS, STATE_TIMEOUTS, Vertex


def model() -> GraphDocument:
    """Build the complete GraphWalker JSON-shaped document."""

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
