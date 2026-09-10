# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from app.clients import discord_client
from app.clients.discord_client import RealDiscord
from app.models import MessageRef, Upload

if TYPE_CHECKING:
    import discord

    from app.gateway import Gateway


class _User:
    id = 42
    name = "global-name"

    def __init__(self) -> None:
        self.edits: list[dict[str, object]] = []

    async def edit(self, **changes: object) -> _User:
        self.edits.append(changes)
        return self


class _Member:
    def __init__(self) -> None:
        self.edits: list[dict[str, object]] = []

    async def edit(self, **changes: object) -> _Member:
        self.edits.append(changes)
        return self


class _Guild:
    id = 7

    def __init__(self, member: _Member) -> None:
        self.member = member

    def get_member(self, user_id: int) -> _Member | None:
        assert user_id == 42
        return self.member


class _SentMessage:
    id = 123
    jump_url = "https://discord.example/messages/123"


class _PartialMessage:
    def __init__(self) -> None:
        self.edits: list[dict[str, object]] = []

    async def edit(self, **changes: object) -> None:
        self.edits.append(changes)


class _Channel:
    def __init__(self) -> None:
        self.sends: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.partial = _PartialMessage()

    async def send(self, *args: object, **kwargs: object) -> _SentMessage:
        self.sends.append((args, kwargs))
        return _SentMessage()

    def get_partial_message(self, message_id: int) -> _PartialMessage:
        assert message_id == 123
        return self.partial


def _client(channel: _Channel, monkeypatch: pytest.MonkeyPatch) -> RealDiscord:
    client = RealDiscord()

    async def resolve_channel(channel_id: str) -> discord.TextChannel | discord.Thread:
        assert channel_id == "456"
        return cast("discord.TextChannel | discord.Thread", channel)

    monkeypatch.setattr(client, "channel", resolve_channel)
    return client


def _assert_no_mentions(changes: dict[str, object]) -> None:
    allowed = changes["allowed_mentions"]
    assert isinstance(allowed, discord_client.discord.AllowedMentions)
    assert allowed.to_dict() == {"parse": []}


async def test_outbound_messages_disable_all_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel()

    await _client(channel, monkeypatch).send("456", "<@42> @everyone @here")

    _assert_no_mentions(channel.sends[0][1])


async def test_message_edits_disable_all_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel()
    client = _client(channel, monkeypatch)

    await client.edit(MessageRef("456", "123"), "<@42>")
    await client.edit(MessageRef("456", "123"), "<@42>", upload=Upload("answer.txt", b"answer"))

    _assert_no_mentions(channel.partial.edits[0])
    _assert_no_mentions(channel.partial.edits[1])


async def test_file_captions_disable_all_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel()
    monkeypatch.setattr(discord_client.discord, "Thread", _Channel)

    await _client(channel, monkeypatch).send_file("456", Upload("answer.txt", b"answer"), "<@42>")

    _assert_no_mentions(channel.sends[0][1])


class _Gateway:
    def __init__(self, user: _User, guild: _Guild) -> None:
        self.user = user
        self.guilds = [guild]
        self.allowlist = {7}


@pytest.mark.asyncio
async def test_set_profile_updates_server_nickname_not_global_username() -> None:
    user = _User()
    member = _Member()
    client = RealDiscord(cast("Gateway", _Gateway(user, _Guild(member))))

    result = await client.set_profile("Wiseman", b"avatar")

    assert result == "Wiseman"
    assert member.edits == [{"nick": "Wiseman"}]
    assert user.edits == [{"avatar": b"avatar"}]
