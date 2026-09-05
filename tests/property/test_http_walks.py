# Copyright (c) 2026 Nick van der Merwe
import asyncio
from typing import cast

import httpx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.clients.mock_clients import MockTemporal, mock_container
from app.http_api import create_app


@given(actions=st.lists(st.sampled_from(("background", "question", "duplicate", "stop")), min_size=1, max_size=20))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_generated_admission_walk_is_idempotent(monkeypatch, actions: list[str]) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    asyncio.run(_walk(actions))


async def _walk(actions: list[str]) -> None:
    clients = mock_container()
    app = create_app(clients=clients)
    accepted: set[str] = set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        for index, action in enumerate(actions):
            message_id = "question" if action == "duplicate" else f"message-{index}"
            mention = action not in {"background", "stop"}
            content = "/stop" if action == "stop" else action
            payload = {
                "t": "MESSAGE_CREATE",
                "d": {
                    "id": message_id,
                    "author": {"id": "human"},
                    "content": content,
                    "channel_id": "home",
                    "thread_id": "thread",
                    "timestamp": str(index),
                    "mentions": [{"id": "bot"}] if mention else [],
                },
            }
            response = await client.post("/v1/discord/events", json=payload)
            assert response.status_code == 200
            if mention:
                accepted.add(message_id)
    admitted = {str(cast("dict[str, object]", item["trigger"])["id"]) for item in cast("MockTemporal", clients.temporal).state.admitted}
    assert admitted == accepted
