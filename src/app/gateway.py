# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import discord
from prometheus_client import Counter, Gauge

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from app.engine import Engine
    from app.temporal_runtime import TemporalRuntime
    from app.types import JsonObject
from app.models import THREAD_AUTO_ARCHIVE_MINUTES, Event, Message, Messageable
from app.presentation import thread_name

LOGGER = logging.getLogger("wiseman")
DISCORD_CONNECTED = Gauge("wiseman_discord_connected", "Discord gateway connection state")
DISCORD_MESSAGES = Counter("wiseman_discord_messages_received", "Discord messages received")
RAW_CAPTURE_LIMIT = 1024


@dataclass(slots=True)
class _Incoming:
    event: Event
    delivery_channel: Messageable
    live: discord.Message


class Gateway(discord.Client):
    def __init__(
        self,
        engine: Engine,
        allowlist: set[int],
        activity_path: str | Path | None = None,
        profile_path: str | Path | None = None,
        sequence_path: str | Path | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, enable_debug_events=True)
        self.engine, self.allowlist = engine, allowlist
        self.fallback_state: dict[str, JsonObject] = {}
        self.raw_gateway_payloads: dict[str, JsonObject] = {}
        self.temporal: TemporalRuntime | None = None
        self.activity_path = Path(activity_path) if activity_path else None
        self.profile_path = Path(profile_path) if profile_path else None
        self.sequence_path = (
            Path(sequence_path)
            if sequence_path
            else self.activity_path and self.activity_path.with_name("thread-sequence.json")
        )
        self.thread_sequence = self._load_thread_sequence()
        self._load_profile()
        engine.lookup = self.resolve
        engine.lookup_channel = self.resolve_channel
        engine.lookup_delivery = self.resolve_delivery

    async def on_ready(self) -> None:
        DISCORD_CONNECTED.set(1)
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

    def _load_thread_sequence(self) -> int:
        if self.sequence_path is None or not self.sequence_path.exists():
            return 0
        try:
            return max(0, int(json.loads(self.sequence_path.read_text(encoding="utf-8"))))
        except (OSError, TypeError, ValueError):
            LOGGER.warning("Ignoring invalid Wiseman thread sequence state")
            return 0

    def _next_thread_name(self) -> str:
        self.thread_sequence += 1
        if self.sequence_path is not None:
            _atomic_write(self.sequence_path, str(self.thread_sequence))
        return thread_name(self.thread_sequence)

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
        for guild in self.guilds:
            fetch = getattr(guild, "fetch_active_threads", None)
            if not callable(fetch):
                continue
            try:
                result = await cast("Callable[[], Awaitable[object]]", fetch)()
            except discord.DiscordException:
                LOGGER.warning("Could not discover active threads in guild %s", guild.id)
                continue
            threads = getattr(result, "threads", result)
            for thread in cast("list[object]", threads) if isinstance(threads, (list, tuple)) else []:
                thread_id = str(getattr(thread, "id", ""))
                if not thread_id or str(getattr(thread, "owner_id", "")) != str(getattr(self.user, "id", "")):
                    continue
                edit = getattr(thread, "edit", None)
                try:
                    if callable(edit) and getattr(thread, "auto_archive_duration", None) != THREAD_AUTO_ARCHIVE_MINUTES:
                        await cast("Callable[..., Awaitable[object]]", edit)(
                            auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES
                        )
                except discord.DiscordException:
                    LOGGER.warning("Could not set one-hour archive on thread %s", thread_id)

    async def close(self) -> None:
        DISCORD_CONNECTED.set(0)
        await super().close()

    async def resolve(self, event: Event) -> discord.Message | None:
        channel_id = (
            event.trigger.channel_id if event.kind == "startup" else event.trigger.thread_id or event.trigger.channel_id
        )
        try:
            channel = await self.fetch_channel(int(channel_id))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                self.engine.reaction_user = self.user
                return await channel.fetch_message(int(event.trigger.id))
        except (discord.DiscordException, ValueError):
            pass

    async def resolve_channel(self, event: Event) -> Messageable | None:
        channel_id = event.trigger.thread_id or event.trigger.channel_id
        try:
            channel = await self.fetch_channel(int(channel_id))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                return channel
        except (discord.DiscordException, ValueError):
            pass

    async def resolve_delivery(self, event: Event, delivery_id: str) -> discord.Message | None:
        channel = await self.resolve_channel(event)
        fetch = getattr(channel, "fetch_message", None)
        if not callable(fetch):
            return None
        try:
            value = await cast("Callable[[int], Awaitable[object]]", fetch)(int(delivery_id))
        except (discord.DiscordException, ValueError):
            return None
        return value if isinstance(value, discord.Message) else None

    async def on_message(self, message: discord.Message) -> None:
        raw = self.raw_gateway_payloads.pop(str(message.id), {})
        DISCORD_MESSAGES.inc()
        LOGGER.warning("Discord message received id=%s mentions=%s", message.id, mention_ids(message))
        if not await self._eligible(message):
            return
        incoming = await self._incoming(message, raw)
        if incoming is None:
            return
        event = incoming.event
        if self.temporal is not None:
            try:
                await self.temporal.submit(event.model_dump(mode="json"))
            except Exception:
                LOGGER.exception("Could not submit Discord message %s to Temporal", message.id)
        else:
            self.engine.reaction_user = self.user
            result = await self.engine.handle(
                event,
                incoming.live,
                delivery_channel=incoming.delivery_channel,
                state_data=self.fallback_state.get(event.trigger.thread_id or event.trigger.channel_id, {}),
            )
            if isinstance(result.get("state"), dict):
                self.fallback_state[event.trigger.thread_id or event.trigger.channel_id] = cast(
                    "JsonObject", result["state"]
                )

    async def _eligible(self, message: discord.Message) -> bool:
        channel = message.channel
        reply_to_self = await self._replies_to_self(message)
        if isinstance(channel, discord.Thread) and not self._is_self(message):
            reference_id = getattr(message.reference, "message_id", None)
            if (
                reference_id is not None
                and self.temporal is not None
                and await self.temporal.steer(Event(trigger=_message(message, str(channel.parent_id), str(channel.id))))
            ):
                return False
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        eligible = (
            not self._is_self(message)
            and (not self.allowlist or guild_id in self.allowlist)
            and (str(getattr(self.user, "id", "")) in mention_ids(message) or reply_to_self)
        )
        if not eligible:
            LOGGER.warning(
                "Discord message ignored id=%s reason=not-admitted mentions=%s guild=%s allowlisted=%s",
                message.id,
                mention_ids(message),
                guild_id,
                not self.allowlist or guild_id in self.allowlist,
            )
        return eligible

    async def _replies_to_self(self, message: discord.Message) -> bool:
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None)
        if (
            resolved is None
            and getattr(reference, "message_id", None)
            and callable(fetch := getattr(message.channel, "fetch_message", None))
        ):
            with suppress(discord.DiscordException, ValueError):
                resolved = await cast("Callable[[int], Awaitable[object]]", fetch)(int(reference.message_id))
        return resolved is not None and self._is_self(cast("discord.Message", resolved))

    def _is_self(self, message: discord.Message) -> bool:
        return message.author.bot and str(message.author.id) == str(getattr(self.user, "id", ""))

    async def _incoming(self, message: discord.Message, raw: JsonObject) -> _Incoming | None:
        channel = message.channel
        if isinstance(channel, discord.Thread):
            thread_id, parent_id, kind = str(channel.id), str(channel.parent_id), "followup"
            delivery_channel = channel
            thread_messages = await _history(channel, 100)
            parent_messages = await _history(channel.parent, 100) if channel.parent else []
        else:
            try:
                thread = await message.create_thread(
                    name=self._next_thread_name(),
                    auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES,
                )
            except discord.DiscordException:
                LOGGER.exception("Could not create a thread for Discord message %s", message.id)
                return None
            thread_id, parent_id, kind = str(thread.id), str(channel.id), "startup"
            delivery_channel = thread
            thread_messages, parent_messages = [], await _history(channel, 100, before=message)
        trigger = _message(message, parent_id, thread_id)
        event = Event(
            trigger=trigger,
            kind=kind,
            parent_messages=parent_messages,
            thread_messages=thread_messages,
            raw_payload=raw or trigger.model_dump(mode="json"),
        )
        return _Incoming(event, delivery_channel, message)


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
        str(getattr(channel, "parent_id", getattr(channel, "id", ""))),
    )
    history = history_method(limit=limit, before=before) if before is not None else history_method(limit=limit)
    return [_message(item, channel_id, thread_id) async for item in history]


def _message(item: discord.Message, channel_id: str, thread_id: str | None) -> Message:
    author = item.author
    reference = getattr(item, "reference", None)
    return Message(
        id=str(item.id),
        author_id=str(author.id),
        author_name=author.name,
        bot=author.bot,
        content=item.content,
        channel_id=channel_id,
        thread_id=thread_id,
        timestamp=item.created_at.isoformat(),
        reply_to=str(reference.message_id) if reference else None,
        mentions=mention_ids(item),
        attachments=[
            {
                "id": str(attachment.id),
                "filename": attachment.filename,
                "url": attachment.url,
                "content_type": attachment.content_type,
                "size": attachment.size,
            }
            for attachment in item.attachments
        ],
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def mention_ids(message: object) -> list[str]:
    values = getattr(message, "raw_mentions", ()) or getattr(message, "mentions", ())
    return [str(getattr(value, "id", value)) for value in values] + re.findall(
        r"<@!?(\d+)>", str(getattr(message, "content", ""))
    )
