# Copyright (c) 2026 Nick van der Merwe
import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

from app.clients.mock_clients import MockRunner
from app.engine import Engine, EngineConfig
from app.gateway import Gateway
from app.models import Event, Message
from app.phoenix import Phoenix, PromptHub

if TYPE_CHECKING:
    import discord


async def test_dispatch_audits_match_interleaved_messages_before_any_await(monkeypatch):
    bot = Gateway(Engine(EngineConfig(Phoenix(), MockRunner(), PromptHub())), set())
    assert bot._enable_debug_events
    paused, release, second, first = (asyncio.Event() for _ in range(4))
    received = {}

    async def eligible(message):
        if message.id == "1":
            paused.set()
            await release.wait()
        return True

    async def incoming(message, raw):
        return Event(trigger=Message(id=message.id, author_id="u", channel_id="c"), raw_payload=raw)

    async def submit(value):
        received[value["trigger"]["id"]] = value["raw_payload"]
        (first if value["trigger"]["id"] == "1" else second).set()

    monkeypatch.setattr(bot, "_eligible", eligible)
    monkeypatch.setattr(bot, "_incoming", incoming)
    monkeypatch.setattr(bot, "temporal", SimpleNamespace(submit=submit))
    async with bot, asyncio.timeout(5):
        raw1 = {"op": 0, "t": "MESSAGE_CREATE", "s": 1, "d": {"id": "1", "content": "first"}}
        raw2 = {"op": 0, "t": "MESSAGE_CREATE", "s": 2, "d": {"id": "2", "content": "second"}}
        bot.dispatch("socket_raw_receive", json.dumps(raw1))
        bot.dispatch("message", SimpleNamespace(id="1"))
        await paused.wait()
        bot.dispatch("socket_raw_receive", json.dumps(raw2))
        bot.dispatch("message", SimpleNamespace(id="2"))
        await second.wait()
        release.set()
        await first.wait()
    assert received == {"1": raw1, "2": raw2}
    assert not bot.raw_gateway_payloads


async def test_raw_capture_is_bounded_and_ignored_messages_are_removed(monkeypatch):
    bot = Gateway(Engine(EngineConfig(Phoenix(), MockRunner(), PromptHub())), set())
    for number in range(1100):
        await bot.on_socket_raw_receive(json.dumps({"t": "MESSAGE_CREATE", "d": {"id": str(number)}}))
    assert len(bot.raw_gateway_payloads) == 1024
    monkeypatch.setattr(bot, "_eligible", AsyncMock(return_value=False))
    await bot.on_message(cast("discord.Message", SimpleNamespace(id="1099")))
    assert "1099" not in bot.raw_gateway_payloads
