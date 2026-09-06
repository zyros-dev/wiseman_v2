# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from httpx import Response

    from app.models import DeliveryReceipt, Event, MessageRef, Upload
    from app.types import JsonObject


class ClientMode(StrEnum):
    MOCK = "mock"
    REAL = "real"


@dataclass(frozen=True, slots=True)
class ClientSettings:
    discord_token: str = ""
    temporal_address: str = ""
    temporal_queue: str = "wiseman"
    phoenix_endpoint: str = ""
    prompt_hub_url: str = ""
    phoenix_key: str = ""
    phoenix_project: str = "wiseman-v2"
    runner_url: str = ""
    runner_token: str = ""
    provider_url: str = "https://openrouter.ai"
    provider_key: str = ""
    vision_model: str = "z-ai/glm-5.3-flash"

    @classmethod
    def from_env(cls) -> ClientSettings:
        return cls(
            discord_token=os.getenv("DISCORD_BOT_TOKEN", ""),
            temporal_address=os.getenv("TEMPORAL_ADDRESS", ""),
            temporal_queue=os.getenv("TEMPORAL_TASK_QUEUE", "wiseman"),
            phoenix_endpoint=os.getenv("PHOENIX_OTLP_ENDPOINT", ""),
            prompt_hub_url=os.getenv("PHOENIX_PROMPT_HUB_URL", ""),
            phoenix_key=os.getenv("PHOENIX_API_KEY", ""),
            phoenix_project=os.getenv("PHOENIX_PROJECT", "wiseman-v2"),
            runner_url=os.getenv("WISEMAN_RUNNER_URL", ""),
            runner_token=os.getenv("WISEMAN_RUNNER_API_TOKEN", ""),
            provider_url=os.getenv("OPENROUTER_URL", "https://openrouter.ai"),
            provider_key=os.getenv("OPENROUTER_API_KEY", ""),
            vision_model=os.getenv("WISEMAN_VISION_MODEL", "z-ai/glm-5.3-flash"),
        )


class DiscordClient(Protocol):
    async def send(self, channel_id: str, content: str = "", *, embed: JsonObject | None = None, nonce: str = "") -> str: ...

    async def start_typing(self, channel_id: str) -> None: ...

    async def stop_typing(self, channel_id: str) -> None: ...

    async def edit(self, ref: MessageRef, content: str, *, upload: Upload | None = None) -> None: ...

    async def add_reaction(self, ref: MessageRef, emoji: str) -> None: ...

    async def remove_reaction(self, ref: MessageRef, emoji: str) -> None: ...

    async def send_file(self, channel_id: str, upload: Upload, caption: str = "") -> DeliveryReceipt: ...

    async def set_profile(self, username: str | None, avatar: bytes | None) -> str: ...


class TemporalClient(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def submit(self, event: JsonObject) -> dict[str, object] | None: ...

    async def touch(self, event: Event) -> None: ...

    async def steer(self, event: Event) -> bool: ...

    async def stop(self, event: Event) -> bool: ...


class PhoenixClient(Protocol):
    records: list[dict[str, object]]

    async def record(self, trace: str, node: str, **data: object) -> None: ...

    def audit(self, audit_id: str) -> dict[str, object] | None: ...


class PromptClient(Protocol):
    async def source(self, kind: str) -> str: ...


class HarnessRunner(Protocol):
    async def acquire(self, user: str, workspace: str) -> None: ...

    async def release(self, user: str, workspace: str) -> None: ...

    async def start(self, thread: str, user: str, workspace: str = "") -> str: ...

    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]: ...

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool: ...

    async def stop(self, thread: str, user: str, workspace: str, target_message_id: str, command_id: str) -> bool: ...


class OpenRouter(Protocol):
    async def responses(self, payload: JsonObject) -> Response: ...

    async def describe(self, url: str, question: str) -> dict[str, object]: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class ClientContainer:
    mode: ClientMode
    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    prompts: PromptClient
    runner: HarnessRunner
    provider: OpenRouter
    settings: ClientSettings = field(default_factory=ClientSettings)

    _installed: ClassVar[ClientContainer | None] = None

    @classmethod
    def install(cls, container: ClientContainer) -> ClientContainer:
        cls._installed = container
        return container

    @classmethod
    def current(cls) -> ClientContainer:
        if cls._installed is None:
            raise RuntimeError("client container has not been configured")
        return cls._installed
