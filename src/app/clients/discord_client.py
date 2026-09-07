# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import io
import re
import secrets
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import discord

from app.models import DeliveryReceipt, Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.gateway import Gateway
    from app.models import MessageRef, Upload
    from app.types import JsonObject


class RealDiscord:
    def __init__(self, gateway: Gateway | None = None) -> None:
        self.gateway = cast("Gateway", gateway)
        self.typing_tasks: dict[str, asyncio.Task[None]] = {}

    def attach(self, gateway: Gateway) -> None:
        self.gateway = gateway

    async def channel(self, channel_id: str) -> discord.TextChannel | discord.Thread:
        value = self.gateway.get_channel(int(channel_id)) or await self.gateway.fetch_channel(int(channel_id))
        if not isinstance(value, (discord.TextChannel, discord.Thread)):
            raise TypeError("Discord channel is not a text channel or thread")
        return value

    async def send(self, channel_id: str, content: str = "", *, embed: JsonObject | None = None, nonce: str = "") -> str:
        channel = await self.channel(channel_id)
        message = await channel.send(
            content or None,
            embeds=[discord.Embed.from_dict(embed)] if embed else [],
            nonce=nonce or secrets.token_hex(8),
        )
        return str(message.id)

    async def start_typing(self, channel_id: str) -> None:
        task = self.typing_tasks.get(channel_id)
        if task is not None and not task.done():
            return

        async def keep_typing() -> None:
            channel = await self.channel(channel_id)
            while True:
                async with channel.typing():
                    await asyncio.sleep(8)

        self.typing_tasks[channel_id] = asyncio.create_task(keep_typing())

    async def stop_typing(self, channel_id: str) -> None:
        task = self.typing_tasks.pop(channel_id, None)
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def edit(self, ref: MessageRef, content: str, *, upload: Upload | None = None) -> None:
        message = (await self.channel(ref.channel_id)).get_partial_message(int(ref.message_id))
        if upload is None:
            await message.edit(content=content)
        else:
            await message.edit(content=content, attachments=[discord.File(io.BytesIO(upload.data), filename=upload.name)])

    async def add_reaction(self, ref: MessageRef, emoji: str) -> None:
        await (await self.channel(ref.channel_id)).get_partial_message(int(ref.message_id)).add_reaction(emoji)

    async def remove_reaction(self, ref: MessageRef, emoji: str) -> None:
        if self.gateway.user is None:
            raise RuntimeError("Discord bot identity is unavailable")
        await (await self.channel(ref.channel_id)).get_partial_message(int(ref.message_id)).remove_reaction(emoji, self.gateway.user)

    async def send_file(self, channel_id: str, upload: Upload, caption: str = "") -> DeliveryReceipt:
        channel = await self.channel(channel_id)
        if not isinstance(channel, discord.Thread):
            raise TypeError("file delivery requires a Discord thread")
        message = await channel.send(caption or None, file=discord.File(io.BytesIO(upload.data), filename=upload.name))
        return DeliveryReceipt(str(message.id), message.jump_url)

    async def set_profile(self, nickname: str | None, avatar: bytes | None) -> str:
        if self.gateway.user is None:
            raise RuntimeError("Discord bot identity is unavailable")
        if avatar is not None:
            await self.gateway.user.edit(avatar=avatar)
        if nickname is None:
            return self.gateway.user.name

        updated = False
        managed_guilds = (guild for guild in self.gateway.guilds if not self.gateway.allowlist or guild.id in self.gateway.allowlist)
        for guild in managed_guilds:
            member = guild.get_member(self.gateway.user.id) or getattr(guild, "me", None)
            fetch_member = cast("Callable[[int], Awaitable[discord.Member]] | None", getattr(guild, "fetch_member", None))
            if member is None and callable(fetch_member):
                with suppress(discord.DiscordException):
                    member = await fetch_member(self.gateway.user.id)
            if member is not None:
                await member.edit(nick=nickname)
                updated = True
        if not updated:
            raise RuntimeError("Discord bot member is unavailable in a managed guild")
        return nickname


def normalize_message(item: discord.Message, channel_id: str, thread_id: str | None) -> Message:
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


def mention_ids(message: object) -> list[str]:
    values = getattr(message, "raw_mentions", ()) or getattr(message, "mentions", ())
    return [str(getattr(value, "id", value)) for value in values] + re.findall(r"<@!?(\d+)>", str(getattr(message, "content", "")))
