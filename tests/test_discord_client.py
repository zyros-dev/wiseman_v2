# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from app.clients.discord_client import RealDiscord

if TYPE_CHECKING:
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
