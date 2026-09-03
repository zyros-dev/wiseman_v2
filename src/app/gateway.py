# Copyright (c) 2026 Nick van der Merwe
"""Discord gateway adapter and history normalization."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord
from prometheus_client import Gauge

if TYPE_CHECKING:
    from app.engine import Engine
    from app.temporal_runtime import TemporalRuntime
from app.models import Event, Message, Messageable
from app.presentation import THREAD_AUTO_ARCHIVE_MINUTES
from app.presentation import thread_name as _thread_name

LOGGER = logging.getLogger("wiseman")
DISCORD_CONNECTED = Gauge("wiseman_discord_connected", "Discord gateway connection state")


class Gateway(discord.Client):
    """Direct Discord gateway adapter; it delegates to the same admission used by raw replay."""

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
        super().__init__(intents=intents)
        self.engine, self.allowlist = engine, allowlist
        self.temporal: TemporalRuntime | None = None
        self.activity_path = Path(activity_path) if activity_path else None
        self.profile_path = Path(profile_path) if profile_path else None
        self.sequence_path = Path(sequence_path) if sequence_path else self._default_sequence_path()
        self.thread_sequence = self._load_thread_sequence()
        self.thread_activity = self._load_thread_activity()
        self._load_profile()
        self.expiry_task: asyncio.Task[None] | None = None
        engine.lookup = self.resolve
        engine.lookup_channel = self.resolve_channel

    async def setup_hook(self) -> None:
        self.expiry_task = asyncio.create_task(self._expire_threads())

    async def on_ready(self) -> None:
        DISCORD_CONNECTED.set(1)
        LOGGER.info("Discord gateway ready as %s", self.user)

    async def on_disconnect(self) -> None:
        DISCORD_CONNECTED.set(0)
        LOGGER.warning("Discord gateway disconnected; discord.py will reconnect")

    async def on_resumed(self) -> None:
        DISCORD_CONNECTED.set(1)
        LOGGER.info("Discord gateway session resumed")

    def _load_thread_activity(self) -> dict[str, float]:
        if self.activity_path is None or not self.activity_path.exists():
            return {}
        try:
            value = json.loads(self.activity_path.read_text(encoding="utf-8"))
            return {str(key): float(timestamp) for key, timestamp in value.items()}
        except (OSError, TypeError, ValueError, AttributeError):
            LOGGER.warning("Ignoring invalid Wiseman thread activity state")
            return {}

    def _persist_thread_activity(self) -> None:
        if self.activity_path is None:
            return
        try:
            self.activity_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.activity_path.with_name(f".{self.activity_path.name}.tmp")
            temporary.write_text(json.dumps(self.thread_activity, sort_keys=True), encoding="utf-8")
            temporary.replace(self.activity_path)
        except OSError:
            LOGGER.exception("Could not persist Wiseman thread activity state")

    def _default_sequence_path(self) -> Path | None:
        if self.activity_path is None:
            return None
        return self.activity_path.with_name("thread-sequence.json")

    def _load_thread_sequence(self) -> int:
        if self.sequence_path is None or not self.sequence_path.exists():
            return 0
        try:
            value = json.loads(self.sequence_path.read_text(encoding="utf-8"))
            return max(0, int(value))
        except (OSError, TypeError, ValueError):
            LOGGER.warning("Ignoring invalid Wiseman thread sequence state")
            return 0

    def _next_thread_name(self) -> str:
        self.thread_sequence += 1
        if self.sequence_path is not None:
            self.sequence_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.sequence_path.with_name(f".{self.sequence_path.name}.tmp")
            temporary.write_text(str(self.thread_sequence), encoding="utf-8")
            temporary.replace(self.sequence_path)
        return _thread_name(self.thread_sequence)

    def _touch_thread(self, thread_id: str, timestamp: float | None = None) -> None:
        self.thread_activity[thread_id] = timestamp if timestamp is not None else time.time()
        self._persist_thread_activity()

    def _forget_thread(self, thread_id: str) -> None:
        if thread_id in self.thread_activity:
            self.thread_activity.pop(thread_id)
            self._persist_thread_activity()

    def _load_profile(self) -> None:
        if self.profile_path is None or not self.profile_path.exists():
            return
        try:
            value = json.loads(self.profile_path.read_text(encoding="utf-8"))
            reactions = value.get("reaction_emojis", {})
            if isinstance(reactions, dict):
                self.engine.set_reaction_emojis(
                    {str(key): str(item) for key, item in reactions.items()}
                )
        except (OSError, TypeError, ValueError, AttributeError):
            LOGGER.warning("Ignoring invalid Wiseman profile state")

    def persist_profile(self) -> None:
        if self.profile_path is None:
            return
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.profile_path.with_name(f".{self.profile_path.name}.tmp")
            temporary.write_text(
                json.dumps({"reaction_emojis": self.engine.reaction_emojis}, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.profile_path)
        except OSError:
            LOGGER.exception("Could not persist Wiseman profile state")

    async def run_forever(self, token: str) -> None:
        """Keep the gateway supervised when Discord returns a fatal session error."""
        delay = 1.0
        while not self.is_closed():
            try:
                await self.start(token, reconnect=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                DISCORD_CONNECTED.set(0)
                LOGGER.exception("Discord gateway session failed; retrying")
            else:
                if not self.is_closed():
                    LOGGER.warning("Discord gateway stopped unexpectedly; retrying")
            if self.is_closed():
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

    async def _expire_threads(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self._expire_once()

    async def _expire_once(self, now: float | None = None) -> None:
        """Forget local tracking after Discord's native archive window expires."""
        cutoff = (time.time() if now is None else now) - (THREAD_AUTO_ARCHIVE_MINUTES * 60)
        for thread_id, last_activity in list(self.thread_activity.items()):
            if last_activity > cutoff:
                continue
            self._forget_thread(thread_id)

    async def close(self) -> None:
        DISCORD_CONNECTED.set(0)
        if self.expiry_task is not None:
            self.expiry_task.cancel()
            await asyncio.gather(self.expiry_task, return_exceptions=True)
        await super().close()

    async def resolve(self, event: Event) -> discord.Message | None:
        """Recover a live Discord message for a Temporal activity delivery."""
        channel_id = (
            event.trigger.channel_id
            if event.kind == "startup"
            else event.trigger.thread_id or event.trigger.channel_id
        )
        try:
            channel = await self.fetch_channel(int(channel_id))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                self.engine.reaction_user = self.user
                return await channel.fetch_message(int(event.trigger.id))
            return None  # noqa: TRY300
        except (discord.DiscordException, ValueError):
            return None

    async def resolve_channel(self, event: Event) -> Messageable | None:
        """Resolve the managed thread used for progress and answer delivery."""
        channel_id = event.trigger.thread_id or event.trigger.channel_id
        try:
            channel = await self.fetch_channel(int(channel_id))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                return channel
            return None  # noqa: TRY300
        except (discord.DiscordException, ValueError):
            return None

    async def on_message(self, message: discord.Message) -> None:
        if (
            isinstance(message.channel, discord.Thread)
            and not message.author.bot
            and str(message.channel.id) in self.thread_activity
        ):
            self._touch_thread(str(message.channel.id))
        if message.author.bot:
            return
        channel = message.channel
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        if self.allowlist and guild_id not in self.allowlist:
            return
        if isinstance(channel, discord.Thread):
            reference_id = getattr(message.reference, "message_id", None)
            if reference_id is not None and await self.engine.steer_if_active(
                str(channel.id), str(reference_id), message.content, str(message.author.id)
            ):
                return
        if self.user not in message.mentions:
            return
        if isinstance(channel, discord.Thread):
            thread_id, parent_id, kind = str(channel.id), str(channel.parent_id), "followup"
            delivery_channel = channel
            self._touch_thread(thread_id)
            old = self.engine.states[thread_id]
            thread_messages = await _history(channel, 100)
            parent_messages = await _history(channel.parent, 100) if channel.parent else []
        else:
            thread = await message.create_thread(
                name=self._next_thread_name(),
                auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES,
            )
            thread_id, parent_id, kind = str(thread.id), str(channel.id), "startup"
            delivery_channel = thread
            self._touch_thread(thread_id)
            old = self.engine.states[thread_id]
            thread_messages, parent_messages = [], await _history(channel, 100, before=message)
        trigger = Message(
            id=str(message.id),
            author_id=str(message.author.id),
            author_name=message.author.name,
            content=message.content,
            channel_id=parent_id,
            thread_id=thread_id,
            timestamp=message.created_at.isoformat(),
            reply_to=str(message.reference.message_id) if message.reference else None,
            mentions=[str(user.id) for user in message.mentions],
            attachments=[
                {
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "url": attachment.url,
                    "content_type": attachment.content_type,
                    "size": attachment.size,
                }
                for attachment in message.attachments
            ],
        )
        event = Event(
            trigger=trigger,
            kind=kind,
            parent_messages=parent_messages,
            thread_messages=thread_messages,
            seen_ids=list(old.seen),
            raw_payload={
                "trigger": trigger.model_dump(mode="json"),
                "kind": kind,
                "parent_messages": [item.model_dump(mode="json") for item in parent_messages],
                "thread_messages": [item.model_dump(mode="json") for item in thread_messages],
            },
        )
        if self.temporal is not None:
            await self.temporal.submit(event.model_dump(mode="json"))
        else:
            self.engine.reaction_user = self.user
            await self.engine.handle(event, message, delivery_channel=delivery_channel)


async def _history(
    channel: Any,  # noqa: ANN401 - Discord's channel union shares the history protocol
    limit: int,
    before: discord.Message | None = None,
) -> list[Message]:
    """Normalize bounded Discord history for the same grammar path as replay."""
    if channel is None:
        return []
    kwargs: dict[str, Any] = {"limit": limit}
    if before is not None:
        kwargs["before"] = before
    thread_id = str(channel.id) if isinstance(channel, discord.Thread) else None
    channel_id = str(getattr(channel, "parent_id", channel.id))
    return [
        Message(
            id=str(item.id),
            author_id=str(item.author.id),
            author_name=item.author.name,
            bot=item.author.bot,
            content=item.content,
            channel_id=channel_id,
            thread_id=thread_id,
            timestamp=item.created_at.isoformat(),
            reply_to=str(item.reference.message_id) if item.reference else None,
            mentions=[str(user.id) for user in item.mentions],
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
        async for item in channel.history(**kwargs)
    ]
