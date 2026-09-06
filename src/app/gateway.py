# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

import discord
from prometheus_client import Counter, Gauge

from app.admission import admitted
from app.clients.discord_client import normalize_message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from app.clients.client_interfaces import TemporalClient
    from app.engine import Engine
    from app.types import JsonObject
from app.models import THREAD_AUTO_ARCHIVE_MINUTES, Event, Message
from app.presentation import thread_name

LOGGER = logging.getLogger("wiseman")
DISCORD_CONNECTED = Gauge("wiseman_discord_connected", "Discord gateway connection state")
DISCORD_MESSAGES = Counter("wiseman_discord_messages_received", "Discord messages received")
RAW_CAPTURE_LIMIT = 1024


class Gateway(discord.Client):
    def __init__(
        self,
        engine: Engine,
        allowlist: set[int],
        activity_path: str | Path | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, enable_debug_events=True)
        self.engine, self.allowlist = engine, allowlist
        self.raw_gateway_payloads: dict[str, JsonObject] = {}
        self.temporal: TemporalClient | None = None
        self.activity_path = Path(activity_path) if activity_path else None
        self.profile_path = Path(profile_path) if profile_path else None
        self._load_profile()

    async def on_ready(self) -> None:
        DISCORD_CONNECTED.set(1)
        LOGGER.info("Discord gateway ready as %s", self.user)
        await self._discover_managed_threads()

    async def on_disconnect(self) -> None:
        DISCORD_CONNECTED.set(0)

    async def on_socket_raw_receive(self, payload: str) -> None:
        value: object = json.loads(payload)
        if isinstance(value, dict) and value.get("t") == "MESSAGE_CREATE":
            data = value.get("d")
            if isinstance(data, dict) and isinstance(data.get("id"), str):
                self.raw_gateway_payloads[data["id"]] = cast("JsonObject", value)
                if len(self.raw_gateway_payloads) > RAW_CAPTURE_LIMIT:
                    self.raw_gateway_payloads.pop(next(iter(self.raw_gateway_payloads)))

    def _load_profile(self) -> None:
        if self.profile_path is None or not self.profile_path.exists():
            return
        try:
            value = json.loads(self.profile_path.read_text(encoding="utf-8"))
            reactions = value.get("reaction_emojis", {})
            if isinstance(reactions, dict):
                self.engine.set_reaction_emojis({str(key): str(item) for key, item in reactions.items()})
        except (OSError, TypeError, ValueError, AttributeError):
            LOGGER.warning("Ignoring invalid Wiseman profile state")

    def persist_profile(self) -> None:
        if self.profile_path is None:
            return
        try:
            _atomic_write(self.profile_path, json.dumps({"reaction_emojis": self.engine.reaction_emojis}))
        except OSError:
            LOGGER.exception("Could not persist Wiseman profile state")

    async def run_forever(self, token: str) -> None:
        delay = 1.0
        while not self.is_closed():
            try:
                await self.start(token, reconnect=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                DISCORD_CONNECTED.set(0)
                LOGGER.exception("Discord gateway session failed; retrying")
            if self.is_closed():
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

    async def _discover_managed_threads(self) -> None:
        if self.user is None:
            return
        for guild in self.guilds:
            try:
                threads = await guild.active_threads()
            except discord.DiscordException:
                LOGGER.warning("Could not discover active threads in guild %s", guild.id)
                continue
            for thread in threads:
                if thread.owner_id != self.user.id or thread.auto_archive_duration == THREAD_AUTO_ARCHIVE_MINUTES:
                    continue
                try:
                    await thread.edit(auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES)
                except discord.DiscordException:
                    LOGGER.warning("Could not set one-hour archive on thread %s", thread.id)

    async def close(self) -> None:
        DISCORD_CONNECTED.set(0)
        await super().close()

    async def on_message(self, message: discord.Message) -> None:
        raw = self.raw_gateway_payloads.pop(str(message.id), {})
        DISCORD_MESSAGES.inc()
        if not await self._eligible(message):
            return
        if self.temporal is None:
            raise RuntimeError("Temporal client is not configured")
        event = await self._incoming(message, raw)
        if event is None:
            return
        try:
            await self.temporal.submit(event.model_dump(mode="json"))
        except Exception:
            LOGGER.exception("Could not submit Discord message %s to Temporal", message.id)

    async def _eligible(self, message: discord.Message) -> bool:
        channel = message.channel
        reply_to_self = await self._replies_to_self(message)
        if isinstance(channel, discord.Thread) and not getattr(message.author, "bot", False) and self._guild_allowed(message):
            if message.content.strip() == "/stop" and self.temporal is not None:
                await self.temporal.stop(Event(trigger=normalize_message(message, str(channel.id), str(channel.id)), kind="stop"))
                return False
            reference_id = getattr(message.reference, "message_id", None)
            if (
                reference_id is not None
                and self.temporal is not None
                and await self.temporal.steer(Event(trigger=normalize_message(message, str(channel.id), str(channel.id))))
            ):
                return False
        eligible = (
            not self._is_self(message)
            and self._guild_allowed(message)
            and admitted(
                normalize_message(message, str(getattr(channel, "id", "")), None),
                str(getattr(self.user, "id", "")),
                reply_to_bot=reply_to_self,
            )
        )
        if not eligible:
            if (
                self.temporal
                and not getattr(message.author, "bot", False)
                and isinstance(channel, discord.Thread)
                and channel.owner_id == getattr(self.user, "id", None)
                and self._guild_allowed(message)
            ):
                await self.temporal.touch(Event(trigger=normalize_message(message, str(channel.id), str(channel.id))))
            LOGGER.debug("Ignoring non-admitted Discord message %s", message.id)
        return eligible

    def _guild_allowed(self, message: discord.Message) -> bool:
        guild_id = getattr(getattr(message.channel, "guild", None), "id", None)
        return not self.allowlist or guild_id in self.allowlist

    async def _replies_to_self(self, message: discord.Message) -> bool:
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None)
        reference_id = getattr(reference, "message_id", None)
        if resolved is None and reference_id and callable(fetch := getattr(message.channel, "fetch_message", None)):
            with suppress(discord.DiscordException, ValueError):
                resolved = await cast("Callable[[int], Awaitable[object]]", fetch)(int(reference_id))
        return resolved is not None and self._is_self(cast("discord.Message", resolved))

    def _is_self(self, message: discord.Message) -> bool:
        return message.author.bot and str(message.author.id) == str(getattr(self.user, "id", ""))

    async def _incoming(self, message: discord.Message, raw: JsonObject) -> Event | None:
        channel = message.channel
        if isinstance(channel, discord.Thread):
            thread_id, kind = str(channel.id), "followup"
            thread_messages = await _history(channel, 100)
            parent_messages = await _history(channel.parent, 100) if channel.parent else []
        else:
            try:
                thread = await message.create_thread(
                    name=thread_name(message.content, str(getattr(self.user, "id", ""))),
                    auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES,
                )
            except discord.DiscordException:
                LOGGER.exception("Could not create a thread for Discord message %s", message.id)
                return None
            thread_id, kind = str(thread.id), "startup"
            thread_messages, parent_messages = [], await _history(channel, 100, before=message)
        trigger = normalize_message(message, str(channel.id), thread_id)
        return Event(
            trigger=trigger,
            kind=kind,
            parent_messages=parent_messages,
            thread_messages=thread_messages,
            raw_payload=raw or trigger.model_dump(mode="json"),
        )


async def _history(
    channel: object | None,
    limit: int,
    before: discord.Message | None = None,
) -> list[Message]:
    if channel is None:
        return []
    method = getattr(channel, "history", None)
    if not callable(method):
        return []
    history_method = cast("Callable[..., AsyncIterator[discord.Message]]", method)
    thread_id, channel_id = (
        (str(channel.id) if isinstance(channel, discord.Thread) else None),
        str(getattr(channel, "id", "")),
    )
    history = history_method(limit=limit, before=before) if before is not None else history_method(limit=limit)
    return [normalize_message(item, channel_id, thread_id) async for item in history]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
