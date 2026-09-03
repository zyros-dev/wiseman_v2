# Copyright (c) 2026 Nick van der Merwe
"""Typed protocols and the process-scoped external-client container."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from app.types import JsonObject, JsonValue


class ClientMode(StrEnum):
    """Select a complete real or deterministic mock dependency graph."""

    MOCK = "mock"
    REAL = "real"


class ClientContainerError(RuntimeError):
    """Raised when the process client container is not configured."""

    def __init__(self) -> None:
        super().__init__("client container has not been configured")


@dataclass(frozen=True, slots=True)
class ClientSettings:
    """Values needed to construct clients; credentials stay at the adapter edge."""

    discord_token: str = ""
    temporal_address: str = ""
    temporal_queue: str = "wiseman"
    phoenix_endpoint: str = ""
    phoenix_key: str = ""
    phoenix_project: str = "wiseman-v2"
    runner_url: str = ""
    runner_token: str = ""
    provider_url: str = ""
    provider_token: str = ""


class DiscordClient(Protocol):
    async def history(self, channel_id: str, limit: int) -> list[JsonObject]: ...

    async def create_thread(self, channel_id: str, name: str, auto_archive_minutes: int) -> str: ...

    async def send(
        self, channel_id: str, content: str = "", *, embed: JsonObject | None = None
    ) -> str: ...

    async def edit(self, message_id: str, content: str) -> None: ...

    async def add_reaction(self, message_id: str, emoji: str) -> None: ...

    async def remove_reaction(self, message_id: str, emoji: str) -> None: ...

    async def archive_thread(self, thread_id: str) -> None: ...

    async def lock_thread(self, thread_id: str) -> None: ...

    async def send_file(self, channel_id: str, path: str, caption: str = "") -> str: ...

    async def set_profile(self, username: str | None, avatar: str | None) -> None: ...

    async def set_reactions(self, values: dict[str, str]) -> None: ...


class TemporalClient(Protocol):
    async def submit(self, event: JsonObject) -> None: ...

    async def signal(self, workflow_id: str, event: JsonObject) -> None: ...


class PhoenixClient(Protocol):
    async def record(self, trace: str, node: str, **data: JsonValue) -> None: ...

    def audit(self, audit_id: str) -> JsonObject | None: ...


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Normalized result returned by a runner turn."""

    thread_id: str
    output: str
    billing: JsonObject


class RunnerClient(Protocol):
    async def acquire(self, user: str, workspace: str) -> None: ...

    async def start(self, thread: str, user: str, workspace: str = "") -> str: ...

    async def run(
        self, thread: str, prompt: str, user: str, workspace: str = ""
    ) -> RunnerResult: ...

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool: ...


class ProviderClient(Protocol):
    async def response(self, payload: JsonObject) -> JsonObject: ...


@dataclass(slots=True)
class ClientContainer:
    """One application-scoped dependency graph; tests install isolated instances."""

    mode: ClientMode
    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    runner: RunnerClient
    provider: ProviderClient

    _installed: ClassVar[ClientContainer | None] = None

    @classmethod
    def install(cls, container: ClientContainer) -> ClientContainer:
        """Install the one process container and return it for dependency injection."""
        cls._installed = container
        return container

    @classmethod
    def current(cls) -> ClientContainer:
        if cls._installed is None:
            raise ClientContainerError
        return cls._installed

    @classmethod
    def reset(cls) -> None:
        """Clear the process container between isolated tests or application shutdown."""
        cls._installed = None


def current_clients() -> ClientContainer:
    """Return the configured process container."""
    return ClientContainer.current()
