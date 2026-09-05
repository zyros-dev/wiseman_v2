# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import io
import re
import secrets
from typing import TYPE_CHECKING, cast

import discord

from app.models import DeliveryReceipt, Message

if TYPE_CHECKING:
    from app.gateway import Gateway
    from app.models import MessageRef, Upload
    from app.types import JsonObject


class RealDiscord:
    def __init__(self, gateway: Gateway | None = None) -> None:
        self.gateway = cast("Gateway", gateway)

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

    async def set_profile(self, username: str | None, avatar: bytes | None) -> str:
        if self.gateway.user is None:
            raise RuntimeError("Discord bot identity is unavailable")
        if username is not None and avatar is not None:
            updated = await self.gateway.user.edit(username=username, avatar=avatar)
        elif username is not None:
            updated = await self.gateway.user.edit(username=username)
        else:
            updated = await self.gateway.user.edit(avatar=avatar)
        return updated.name


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
