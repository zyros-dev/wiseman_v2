# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
import pytest

from app.alert_buttons import AlertButton, MuteDuration, parse_custom_id, view_from_components
from app.gateway import Gateway

if TYPE_CHECKING:
    import discord


def test_parse_custom_id_accepts_only_bounded_heimdall_mutes() -> None:
    assert parse_custom_id("heimdall:mute:abc123:6h") == AlertButton("abc123", MuteDuration.SIX_HOURS)
    assert parse_custom_id("heimdall:mute:not-a-fingerprint:6h") is None
    assert parse_custom_id("heimdall:mute:abc123:forever") is None
    assert parse_custom_id("other:mute:abc123:6h") is None


def test_view_preserves_button_labels_and_custom_ids() -> None:
    components = [
        {
            "type": 1,
            "components": [
                {"type": 2, "style": 2, "label": "Mute 1 hour", "custom_id": "heimdall:mute:abc123:1h"},
                {"type": 2, "style": 2, "label": "Mute 7 days", "custom_id": "heimdall:mute:abc123:7d"},
            ],
        }
    ]

    view = view_from_components(components)

    buttons = [cast("discord.ui.Button[discord.ui.View]", child) for child in view.children]
    assert [button.label for button in buttons] == ["Mute 1 hour", "Mute 7 days"]
    assert [button.custom_id for button in buttons] == ["heimdall:mute:abc123:1h", "heimdall:mute:abc123:7d"]


def test_view_rejects_link_buttons() -> None:
    with pytest.raises(ValueError, match="interactive button"):
        view_from_components([{"type": 1, "components": [{"type": 2, "style": 5, "label": "Mute", "url": "https://example.test"}]}])


class _InteractionResponse:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []
        self.deferred = False

    async def send_message(self, content: str, *, ephemeral: bool) -> None:
        self.messages.append((content, ephemeral))

    async def defer(self, *, ephemeral: bool) -> None:
        self.deferred = ephemeral


class _InteractionFollowup:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    async def send(self, content: str, *, ephemeral: bool) -> None:
        self.messages.append((content, ephemeral))


@pytest.mark.asyncio
async def test_button_click_mutes_through_heimdall_without_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _InteractionResponse()
    followup = _InteractionFollowup()
    interaction = SimpleNamespace(
        data={"custom_id": "heimdall:mute:abc123:6h"},
        user=SimpleNamespace(id=42),
        response=response,
        followup=followup,
    )
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    monkeypatch.setenv("WISEMAN_HEIMDALL_MUTE_USER_ID", "42")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: real_client(transport=transport))

    await Gateway.on_interaction(cast("Gateway", object()), cast("discord.Interaction", interaction))

    assert response.deferred
    assert followup.messages == [("Muted for 6h.", True)]


@pytest.mark.asyncio
async def test_button_click_rejects_other_users_before_muting(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _InteractionResponse()
    followup = _InteractionFollowup()
    interaction = SimpleNamespace(
        data={"custom_id": "heimdall:mute:abc123:6h"},
        user=SimpleNamespace(id=7),
        response=response,
        followup=followup,
    )
    monkeypatch.setenv("WISEMAN_HEIMDALL_MUTE_USER_ID", "42")

    await Gateway.on_interaction(cast("Gateway", object()), cast("discord.Interaction", interaction))

    assert response.messages == [("Only the alert owner can mute this alert.", True)]
    assert followup.messages == []
