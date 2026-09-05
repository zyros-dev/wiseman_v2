# Copyright (c) 2026 Nick van der Merwe
from typing import cast

import httpx

from app.clients.mock_clients import MockDiscord, mock_container
from app.http_api import create_app
from tests.graph_model import EDGES, Vertex


def _message(message_id: str, content: str, *, thread: str | None = None) -> dict[str, object]:
    return {
        "id": message_id,
        "author": {"id": "human", "username": "human"},
        "content": content,
        "channel_id": "home",
        "thread_id": thread,
        "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z",
        "mentions": [{"id": "bot"}],
    }


async def test_graph_http_walk_preserves_background_context_and_turn_history(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    clients = mock_container()
    app = create_app(clients=clients)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        first = await client.post(
            "/v1/replay/discord",
            json={"trigger": _message("q1", "first question", thread="thread"), "kind": "startup"},
        )
        assert first.json()["status"] == "queued"
        second = await client.post(
            "/v1/replay/discord",
            json={
                "trigger": _message("q2", "second question", thread="thread"),
                "kind": "followup",
                "parent_messages": [_message("chat", "background discussion", thread="thread")],
                "thread_messages": [_message("chat", "background discussion", thread="thread")],
            },
        )
        third = await client.post(
            "/v1/replay/discord",
            json={
                "trigger": _message("q3", "third question", thread="thread"),
                "kind": "followup",
                "parent_messages": [_message("chat", "background discussion", thread="thread")],
                "thread_messages": [],
            },
        )
        stopped = await client.post(
            "/v1/discord/events",
            json={"trigger": _message("stop", "/stop", thread="thread"), "kind": "stop"},
        )
        ignored = await client.post(
            "/v1/discord/events",
            json={"trigger": {**_message("chat2", "ordinary chatter"), "mentions": []}},
        )
        assert second.json()["status"] == third.json()["status"] == "queued"
        assert stopped.json() == {"status": "ignored", "message_id": "stop"}
        assert ignored.json() == {"status": "ignored", "message_id": "chat2"}
    assert isinstance(clients.discord, MockDiscord)
    assert [str(cast("dict[str, object]", item["trigger"])["id"]) for item in clients.discord.state.admitted] == [
        "q1",
        "q2",
        "q3",
    ]
    assert "background discussion" in str(clients.discord.state.admitted[1])
    assert [item[1] for item in clients.discord.state.calls if item[0] == "temporal"] == [
        "submit",
        "submit",
        "submit",
        "stop",
    ]
    assert {state for _, source, target in EDGES for state in (source, target)} == set(Vertex)
