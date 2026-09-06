# Copyright (c) 2026 Nick van der Merwe
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.types import JsonObject

THREAD_AUTO_ARCHIVE_MINUTES = 60
IMAGE_SUFFIXES = (
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass(frozen=True, slots=True)
class MessageRef:
    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class Upload:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    message_id: str
    url: str


class DeliveryPhase(StrEnum):
    IDLE = "idle"
    PROGRESS = "progress"
    ANSWER = "answer"


class DeliveryState(BaseModel):
    phase: DeliveryPhase = DeliveryPhase.IDLE
    typing: bool = False
    progress_message_id: str | None = None
    progress_edit_count: int = 0
    answer_message_id: str | None = None
    answer_edit_count: int = 0
    reaction_phase: Literal["none", "processing", "success", "failure"] = "none"


def is_image_attachment(value: object) -> bool:
    return isinstance(value, dict) and (
        str(value.get("content_type") or "").startswith("image/") or str(value.get("filename") or "").lower().endswith(IMAGE_SUFFIXES)
    )


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


class State(BaseModel):
    owner_id: str = ""
    codex_thread: str | None = None
    seen: set[str] = Field(default_factory=set)
    processed: set[str] = Field(default_factory=set)
    turn: int = 0
    closed: bool = False
    delivery_id: str | None = None
    banner_sent: bool = False
    progress: list[str] = Field(default_factory=list)
    delivery: DeliveryState = Field(default_factory=DeliveryState)
    message_ids: list[str] = Field(default_factory=list)
    background_context_ids: list[str] = Field(default_factory=list)
    consumed_context_ids: list[str] = Field(default_factory=list)
    steering_ids: list[str] = Field(default_factory=list)
    stop_command_ids: list[str] = Field(default_factory=list)


class TurnWork(BaseModel):
    event: Event
    state: State
    current: dict[str, object] = Field(default_factory=dict)
    grammar: dict[str, object] = Field(default_factory=dict)
    prompt: str = ""
    output: str = ""
    billing: dict[str, object] = Field(default_factory=dict)
    error: str = ""
    stopped: bool = False
    processing_emoji: str = ""
    terminal_emoji: str = ""

    @property
    def trace(self) -> str:
        return f"discord-{self.event.trigger.id}"

    @property
    def kind(self) -> str:
        return "followup" if self.state.turn else "startup"
