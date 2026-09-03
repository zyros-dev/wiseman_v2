# Copyright (c) 2026 Nick van der Merwe
"""Shared data models and small protocol boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """The bounded message shape shared by Discord and replay admission."""

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
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class Event(BaseModel):
    """Raw Discord-shaped event accepted by the HTTP replay harness."""

    model_config = ConfigDict(extra="ignore")
    trigger: Message
    kind: str | None = None
    parent_messages: list[Message] = Field(default_factory=list)
    thread_messages: list[Message] = Field(default_factory=list)
    seen_ids: list[str] = Field(default_factory=list)
    anchor_id: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class Messageable(Protocol):
    """Minimal Discord channel surface needed for turn delivery."""

    async def send(self, content: str) -> object: ...


@dataclass
class State:
    """The small durable state that Temporal would persist between activities."""

    codex_thread: str | None = None
    seen: set[str] = field(default_factory=set)
    processed: set[str] = field(default_factory=set)
    turn: int = 0
    last_activity: float = 0.0
    closed: bool = False


@dataclass
class ActiveTurn:
    """The Discord delivery that can receive steering while Codex is running."""

    trigger_id: str
    delivery_id: str | None = None
