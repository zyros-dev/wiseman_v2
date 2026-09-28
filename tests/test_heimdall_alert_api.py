# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import httpx
import pytest

from app.clients.mock_clients import mock_container
from app.gateway import Gateway
from app.http_api import create_app


def _payload() -> dict[str, object]:
    return {
        "content": "<@42>",
        "embeds": [{"title": "CRITICAL", "description": "Odin is down", "color": 15_158_332}],
        "components": [
            {
                "type": 1,
                "components": [{"type": 2, "style": 2, "label": "Mute 1 hour", "custom_id": "heimdall:mute:abc123:1h"}],
            }
        ],
    }


@pytest.mark.asyncio
async def test_heimdall_alert_endpoint_requires_token_and_delivers(monkeypatch: pytest.MonkeyPatch) -> None:
    delivered: list[int] = []

    async def send_alert(_gateway: Gateway, channel_id: int, _message: object) -> str:
        delivered.append(channel_id)
        return "999"

    monkeypatch.setenv("WISEMAN_REPLAY_TOKEN", "secret")
    monkeypatch.setenv("WISEMAN_HEIMDALL_CHANNEL_ID", "1479893275044479177")
    monkeypatch.setattr(Gateway, "send_alert", send_alert)
    app = create_app(clients=mock_container())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        rejected = await client.post("/v1/alerts/heimdall", json=_payload())
        accepted = await client.post("/v1/alerts/heimdall", headers={"x-heimdall-token": "secret"}, json=_payload())

    assert rejected.status_code == 401
    assert accepted.json() == {"status": "delivered", "message_id": "999"}
    assert delivered == [1479893275044479177]
