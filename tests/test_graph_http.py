# Copyright (c) 2026 Nick van der Merwe
import json
import os
from pathlib import Path
from typing import cast

import httpx

from app.clients.mock_clients import MockDiscord, mock_container
from app.http_api import create_app
from tests.graph_model import EDGES


def _message(
    message_id: str,
    content: str,
    *,
    thread: str | None = None,
    reply_to: str | None = None,
    mention: bool = True,
) -> dict[str, object]:
    return {
        "id": message_id,
        "author": {"id": "human", "username": "human"},
        "content": content,
        "channel_id": "home",
        "thread_id": thread,
        "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z",
        "mentions": [{"id": "bot"}] if mention else [],
        "message_reference": {"message_id": reply_to} if reply_to else {},
    }


def _elements() -> list[str]:
    path = os.getenv("GRAPHWALKER_PATH")
    if path:
        return [str(json.loads(line)["currentElementName"]) for line in Path(path).read_text().splitlines() if line.strip()]
    return [str(item) for item in "idle background-chatter idle admit-question preparing context-ready running queue-question running".split()]


async def test_graphwalker_edges_use_the_discord_admission_boundary(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    clients = mock_container()
    app = create_app(clients=clients)
    elements = _elements()
    executed: set[str] = set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        for index in range(0, len(elements) - 2, 2):
            edge, message_id = elements[index + 1], f"graph-{index}"
            executed.add(edge)
            if edge not in {"background-chatter", "steer-active-turn", "stop-active-turn", "admit-question", "queue-question"}:
                continue
            if edge == "background-chatter":
                response = await client.post("/v1/discord/events", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "background", mention=False)})
            elif edge == "steer-active-turn":
                response = await client.post(
                    "/v1/replay/discord",
                    json={"t": "MESSAGE_CREATE", "d": _message(message_id, "steer", thread="thread", reply_to="answer"), "kind": "steer"},
                )
            elif edge == "stop-active-turn":
                response = await client.post(
                    "/v1/discord/events", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread="thread"), "kind": "stop"}
                )
            else:
                thread = None if edge == "admit-question" else "thread"
                kind = "startup" if edge == "admit-question" else "followup"
                response = await client.post(
                    "/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "question", thread=thread), "kind": kind}
                )
            assert response.status_code == 200 and (edge != "background-chatter" or response.json()["status"] == "ignored")
            assert response.json()["message_id"] == message_id
    assert isinstance(clients.discord, MockDiscord)
    assert [str(cast("dict[str, object]", item["trigger"])["id"]) for item in clients.discord.state.admitted] == [
        f"graph-{index}" for index in range(0, len(elements) - 2, 2) if elements[index + 1] in {"admit-question", "queue-question"}
    ]
    if os.getenv("GRAPHWALKER_PATH"):
        assert {edge for edge, _, _ in EDGES} <= executed
