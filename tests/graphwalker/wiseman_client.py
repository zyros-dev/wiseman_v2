# Copyright (c) 2026 Nick van der Merwe
"""Public Wiseman actions used by the GraphWalker harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from tests.graphwalker.model import EdgeName

if TYPE_CHECKING:
    import httpx

    from app.types import JsonObject

ACTION_EDGES = frozenset(
    {
        EdgeName.BACKGROUND_CHATTER,
        EdgeName.RUNNING_BACKGROUND_CHATTER,
        EdgeName.DUPLICATE_QUESTION,
        EdgeName.IDLE_STOP,
        EdgeName.STEER_ACTIVE_TURN,
        EdgeName.REPEAT_STEER,
        EdgeName.STOP_PREPARING,
        EdgeName.STOP_RUNNING,
        EdgeName.STOP_RECOVERING,
        EdgeName.STOP_DELIVERING,
        EdgeName.DUPLICATE_STOP,
        EdgeName.ADMIT_QUESTION,
        EdgeName.QUEUE_QUESTION,
    }
)


@dataclass(frozen=True, slots=True)
class WisemanResponse:
    status_code: int
    body: JsonObject


class WisemanClient(Protocol):
    async def send(self, edge: EdgeName, message_id: str, *, thread_id: str) -> WisemanResponse: ...


class HttpWisemanClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def send(self, edge: EdgeName, message_id: str, *, thread_id: str) -> WisemanResponse:
        response = await self.client.post("/v1/replay/discord", json=self._payload(edge, message_id, thread_id))
        if edge is EdgeName.DUPLICATE_QUESTION:
            response = await self.client.post("/v1/replay/discord", json=self._payload(edge, message_id, thread_id))
        return WisemanResponse(response.status_code, _json_object(response))

    def _payload(self, edge: EdgeName, message_id: str, thread_id: str) -> JsonObject:
        if edge is EdgeName.IDLE_STOP or edge in {
            EdgeName.STOP_PREPARING,
            EdgeName.STOP_RUNNING,
            EdgeName.STOP_RECOVERING,
            EdgeName.STOP_DELIVERING,
            EdgeName.DUPLICATE_STOP,
        }:
            return {"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread_id), "kind": "stop"}
        if edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER}:
            return {"t": "MESSAGE_CREATE", "d": _message(message_id, "background", thread_id, mention=False)}
        if edge in {EdgeName.STEER_ACTIVE_TURN, EdgeName.REPEAT_STEER}:
            return {
                "t": "MESSAGE_CREATE",
                "d": _message(message_id, "steer", thread_id, reply_to="answer"),
                "kind": "steer",
            }
        return {
            "t": "MESSAGE_CREATE",
            "d": _message(message_id, "question", None if edge is EdgeName.ADMIT_QUESTION else thread_id),
            "kind": "startup" if edge is EdgeName.ADMIT_QUESTION else "followup",
        }


def _message(
    message_id: str,
    content: str,
    thread_id: str | None,
    *,
    mention: bool = True,
    reply_to: str | None = None,
) -> JsonObject:
    return {
        "id": message_id,
        "author": {"id": "human", "username": "human"},
        "content": content,
        "channel_id": "home",
        "thread_id": thread_id,
        "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z",
        "mentions": [{"id": "bot"}] if mention else [],
        "message_reference": {"message_id": reply_to} if reply_to else {},
    }


def _json_object(response: httpx.Response) -> JsonObject:
    value: object = response.json()
    return cast("JsonObject", value) if isinstance(value, dict) else {}
