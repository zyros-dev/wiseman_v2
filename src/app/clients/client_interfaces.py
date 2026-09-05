# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.models import MessageRef, Upload
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
        )


class DiscordClient(Protocol):
    async def history(self, channel_id: str, limit: int) -> list[JsonObject]: ...

    async def create_thread(self, channel_id: str, name: str, auto_archive_minutes: Literal[60]) -> str: ...

    async def send(
        self, channel_id: str, content: str = "", *, embed: JsonObject | None = None, nonce: str = ""
    ) -> str: ...

    async def edit(self, ref: MessageRef, content: str, *, upload: Upload | None = None) -> None: ...

    async def add_reaction(self, ref: MessageRef, emoji: str) -> None: ...

    async def remove_reaction(self, ref: MessageRef, emoji: str) -> None: ...

    async def archive_thread(self, thread_id: str) -> None: ...

    async def lock_thread(self, thread_id: str) -> None: ...

    async def send_file(self, channel_id: str, path: str, caption: str = "") -> str: ...

    async def set_profile(self, username: str | None, avatar: str | None) -> None: ...

    async def set_reactions(self, values: dict[str, str]) -> None: ...


class TemporalClient(Protocol):
    async def submit(self, event: JsonObject) -> None: ...


class PhoenixClient(Protocol):
    records: list[dict[str, object]]

    async def record(self, trace: str, node: str, **data: object) -> None: ...

    def audit(self, audit_id: str) -> dict[str, object] | None: ...


class PromptClient(Protocol):
    async def source(self, kind: str) -> str: ...


class RunnerClient(Protocol):
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


@dataclass(slots=True)
class ClientContainer:
    mode: ClientMode
    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    prompts: PromptClient
    runner: RunnerClient
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

    @classmethod
    def reset(cls) -> None:
        cls._installed = None
