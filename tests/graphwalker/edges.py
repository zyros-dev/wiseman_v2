# Copyright (c) 2026 Nick van der Merwe
"""GraphWalker edges and their executable boundary actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.graphwalker.graph_utils import EdgeFunction, GraphContext, apply_edge
from tests.graphwalker.model import EDGE_TIMEOUTS, EDGES_BY_NAME, EdgeName
from tests.graphwalker.wiseman_client import WisemanResponse

if TYPE_CHECKING:
    from collections.abc import Mapping


ACTION_EDGES = frozenset(
    {
        EdgeName.ADMIT_QUESTION,
        EdgeName.BACKGROUND_CHATTER,
        EdgeName.DUPLICATE_QUESTION,
        EdgeName.IDLE_STOP,
        EdgeName.RUNNING_BACKGROUND_CHATTER,
        EdgeName.STEER_ACTIVE_TURN,
        EdgeName.REPEAT_STEER,
        EdgeName.QUEUE_QUESTION,
        EdgeName.STOP_PREPARING,
        EdgeName.STOP_RUNNING,
        EdgeName.STOP_RECOVERING,
        EdgeName.STOP_DELIVERING,
        EdgeName.DUPLICATE_STOP,
    }
)


async def _wiseman_action(context: GraphContext, edge: EdgeName, message_id: str) -> WisemanResponse:
    thread_id = context.state.chat.thread_id
    if edge is EdgeName.ADMIT_QUESTION:
        return await context.wiseman.ask(message_id, thread_id=thread_id, startup=True)
    if edge is EdgeName.QUEUE_QUESTION:
        return await context.wiseman.ask(message_id, thread_id=thread_id)
    if edge is EdgeName.DUPLICATE_QUESTION:
        return await context.wiseman.ask(message_id, thread_id=thread_id, duplicate=True)
    if edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER}:
        return await context.wiseman.background(message_id, thread_id=thread_id)
    if edge in {EdgeName.STEER_ACTIVE_TURN, EdgeName.REPEAT_STEER}:
        reply_to = context.observation.chat.progress_message_ids[-1] if context.observation and context.observation.chat.progress_message_ids else "answer"
        return await context.wiseman.steer(message_id, thread_id=thread_id, reply_to=reply_to)
    return await context.wiseman.stop(message_id, thread_id=thread_id)


async def _edge(context: GraphContext, edge: EdgeName, message_id: str = "") -> None:
    definition = EDGES_BY_NAME[edge]
    resolved_message_id = message_id or context.message_id
    if edge is EdgeName.DUPLICATE_QUESTION and context.state.wiseman.seen_questions:
        resolved_message_id = next(iter(context.state.wiseman.seen_questions))
    if edge is EdgeName.DUPLICATE_STOP and context.state.chat.stop_commands:
        resolved_message_id = next(iter(context.state.chat.stop_commands))
    context.begin_edge(definition, resolved_message_id)
    await context.harness.prepare_edge(definition, context, deadline_seconds=EDGE_TIMEOUTS[edge])
    if edge is EdgeName.DUPLICATE_QUESTION and not context.state.wiseman.seen_questions:
        context.last_response = WisemanResponse(200, {"status": "ignored", "message_id": resolved_message_id})
        await apply_edge(context, definition, EDGE_TIMEOUTS[edge], resolved_message_id, execute_boundary=False)
        return
    if edge in ACTION_EDGES:
        context.last_response = await _wiseman_action(context, edge, resolved_message_id)
    transition_message_id = resolved_message_id
    if edge is EdgeName.BACKGROUND_CHATTER and definition.source.value == "idle":
        transition_message_id = ""
    await apply_edge(
        context,
        definition,
        EDGE_TIMEOUTS[edge],
        transition_message_id,
        execute_boundary=True,
    )


def _edge_action(edge: EdgeName) -> EdgeFunction:
    if edge in ACTION_EDGES:

        async def action(context: GraphContext, message_id: str = "") -> None:
            await _edge(context, edge, message_id)
    else:

        async def action(context: GraphContext) -> None:
            await _edge(context, edge)

    action.__name__ = edge.value.replace("-", "_")
    return action


EDGE_FUNCTIONS: Mapping[EdgeName, EdgeFunction] = {edge: _edge_action(edge) for edge in EdgeName}
