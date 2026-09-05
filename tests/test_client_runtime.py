# Copyright (c) 2026 Nick van der Merwe
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from fastapi.testclient import TestClient

from app.clients import ClientMode, ClientSettings
from app.clients.mock_clients import MockDiscord, mock_container
from app.engine import Engine
from app.gateway import Gateway
from app.http_api import create_app
from app.models import Event, Message


def settings():
    return ClientSettings(
        discord_token="discord-fixture",
        temporal_address="temporal.invalid:7233",
        runner_url="http://runner.invalid",
        runner_token="runner-fixture",
        phoenix_endpoint="http://phoenix.invalid/v1/traces",
        prompt_hub_url="http://phoenix.invalid",
        phoenix_key="phoenix-fixture",
        provider_key="provider-fixture",
    )


def test_mock_lifecycle_never_constructs_or_starts_real_network_clients(monkeypatch):
    monkeypatch.setattr("app.http_api.TemporalRuntime", Mock(side_effect=AssertionError("real Temporal in mock mode")))
    monkeypatch.setattr(Gateway, "run_forever", AsyncMock(side_effect=AssertionError("real Discord in mock mode")))
    clients = mock_container(settings())
    with TestClient(create_app(clients=clients)) as api:
        assert api.get("/readyz").status_code == 200
        for mid in ("first", "second"):
            response = api.post(
                "/v1/replay/discord",
                headers={"x-replay-token": clients.settings.runner_token},
                json=Event(
                    trigger=Message(id=mid, author_id="alice", channel_id="thread", thread_id="thread")
                ).model_dump(),
            )
            assert response.status_code == 200
        assert response.json()["state"]["turn"] == 2
    assert isinstance(clients.discord, MockDiscord)
    assert len(clients.discord.state.messages) == 3


def test_real_lifecycle_uses_the_injected_temporal_client(monkeypatch):
    monkeypatch.setattr("app.http_api.TemporalRuntime", Mock(side_effect=AssertionError("bypassed injected Temporal")))
    discord = AsyncMock()
    monkeypatch.setattr(Gateway, "run_forever", discord)
    clients = mock_container(settings())
    clients.mode = ClientMode.REAL
    with TestClient(create_app(clients=clients)) as api:
        response = api.post(
            "/v1/replay/discord",
            headers={"x-replay-token": clients.settings.runner_token},
            json=Event(trigger=Message(id="first", author_id="alice", channel_id="thread")).model_dump(),
        )
        assert response.json()["status"] == "queued"
    discord.assert_awaited_once_with("discord-fixture")
    assert isinstance(clients.discord, MockDiscord)
    assert [call.operation for call in clients.discord.state.calls if call.client == "temporal"] == [
        "start",
        "submit",
        "close",
    ]
    assert not clients.discord.state.messages


def test_explicit_tokens_reach_the_real_adapters(monkeypatch):
    configured = replace(settings(), discord_token="", runner_token="")
    monkeypatch.setattr(ClientSettings, "from_env", lambda: configured)
    clients = mock_container()
    services = Mock(return_value=(clients.phoenix, clients.prompts, clients.runner))
    monkeypatch.setattr("app.http_api.real_services", services)
    app = create_app(token="runner-fixture", discord_token="discord-fixture")
    services.assert_called_once_with(settings())
    assert app.state.clients.settings == settings()


@pytest.mark.parametrize(
    "missing",
    [
        "discord_token",
        "temporal_address",
        "runner_url",
        "runner_token",
        "phoenix_endpoint",
        "prompt_hub_url",
        "provider_key",
    ],
)
def test_real_mode_rejects_incomplete_configuration(missing):
    clients = mock_container(replace(settings(), **{missing: ""}))
    clients.mode = ClientMode.REAL
    with pytest.raises(ValueError, match="real mode"):
        create_app(clients=clients)


def test_internal_phoenix_without_authentication_does_not_need_a_fabricated_key():
    clients = mock_container(replace(settings(), phoenix_key=""))
    clients.mode = ClientMode.REAL
    app = create_app(clients=clients)
    assert app.state.clients.settings.phoenix_key == ""


async def test_gateway_never_creates_a_thread_or_runs_inference_without_temporal(monkeypatch):
    engine = Engine(clients=mock_container())
    gateway = Gateway(engine, set())
    monkeypatch.setattr(gateway, "_eligible", AsyncMock(return_value=True))
    incoming = AsyncMock(return_value=Event(trigger=Message(id="1", author_id="alice", channel_id="parent")))
    monkeypatch.setattr(gateway, "_incoming", incoming)
    with pytest.raises(RuntimeError, match="Temporal"):
        await gateway.on_message(Mock(spec=discord.Message, id="1", content="", raw_mentions=[], mentions=[]))
    incoming.assert_not_awaited()
