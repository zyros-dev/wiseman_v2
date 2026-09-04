# Copyright (c) 2026 Nick van der Merwe
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.types import JsonObject

THREAD_AUTO_ARCHIVE_MINUTES = 60
IMAGE_SUFFIXES = (".avif", ".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")  # fmt: skip  # noqa: E501


def is_image_attachment(value: object) -> bool:
    return isinstance(value, dict) and (str(value.get("content_type") or "").startswith("image/") or str(value.get("filename") or "").lower().endswith(IMAGE_SUFFIXES))  # fmt: skip  # noqa: E501


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    author_id: str
    author_name: str = "unknown"
    bot: bool = False
    content: str = Field(default="", max_length=4000)
    timestamp: str = ""
    channel_id: str
    thread_id: str | None = None
    reply_to: str | None = None
    mentions: list[str] = Field(default_factory=list)
    attachments: list[JsonObject] = Field(default_factory=list)


class Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trigger: Message
    kind: str | None = None
    parent_messages: list[Message] = Field(default_factory=list)
    thread_messages: list[Message] = Field(default_factory=list)
    seen_ids: list[str] = Field(default_factory=list)
    anchor_id: str | None = None
    raw_payload: JsonObject = Field(default_factory=dict)


class Messageable(Protocol):
    async def send(self, content: str = "") -> object: ...


class EmbedMessageable(Protocol):
    async def send(self, content: str = "", *, embed: object | None = None) -> object: ...


@dataclass
class State:
    codex_thread: str | None = None
    seen: set[str] = field(default_factory=set)
    processed: set[str] = field(default_factory=set)
    turn: int = 0
    closed: bool = False


@dataclass
class ActiveTurn:
    trigger_id: str
    delivery_id: str | None = None
