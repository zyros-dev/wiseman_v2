# Copyright (c) 2026 Nick van der Merwe
"""Public Wiseman actions used by the GraphWalker harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import httpx

    from app.types import JsonObject


@dataclass(frozen=True, slots=True)
class WisemanResponse:
    status_code: int
    body: JsonObject


class WisemanClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def ask(self, message_id: str, *, thread_id: str | None, duplicate: bool = False) -> WisemanResponse:
        response = await self.client.post("/v1/replay/discord", json=self._question_payload(message_id, thread_id))
        if duplicate:
            response = await self.client.post("/v1/replay/discord", json=self._question_payload(message_id, thread_id))
        return WisemanResponse(response.status_code, _json_object(response))

    async def background(self, message_id: str, *, thread_id: str) -> WisemanResponse:
        return await self._post(_message(message_id, "background", thread_id, mention=False))

    async def steer(self, message_id: str, *, thread_id: str) -> WisemanResponse:
        return await self._post(_message(message_id, "steer", thread_id, reply_to="answer"), kind="steer")

    async def stop(self, message_id: str, *, thread_id: str) -> WisemanResponse:
        return await self._post(_message(message_id, "/stop", thread_id), kind="stop")

    async def _post(self, message: JsonObject, *, kind: str | None = None) -> WisemanResponse:
        payload: JsonObject = {"t": "MESSAGE_CREATE", "d": message}
        if kind is not None:
            payload["kind"] = kind
        response = await self.client.post("/v1/replay/discord", json=payload)
        return WisemanResponse(response.status_code, _json_object(response))

    def _question_payload(self, message_id: str, thread_id: str | None) -> JsonObject:
        return {"t": "MESSAGE_CREATE", "d": _message(message_id, "question", thread_id), "kind": "startup" if thread_id is None else "followup"}


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
