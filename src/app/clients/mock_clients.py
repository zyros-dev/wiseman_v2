# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    DiscordClient,
    HarnessRunner,
    PromptClient,
    TemporalClient,
)
from app.clients.provider import MockOpenRouter
from app.models import DeliveryReceipt
from app.phoenix import Phoenix
from app.runner import STOPPED_STATUS, RunnerError
from app.temporal_runtime import TemporalRuntime

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.models import MessageRef, Upload
    from app.types import JsonObject


@dataclass(slots=True)
class MockState:
    calls: list[tuple[str, str, tuple[str, ...]]] = field(default_factory=list)
    messages: dict[str, str] = field(default_factory=dict)
    embeds: dict[str, JsonObject] = field(default_factory=dict)
    reactions: dict[str, list[str]] = field(default_factory=dict)
    profile: dict[str, str | bytes] = field(default_factory=dict)
    uploads: dict[str, Upload] = field(default_factory=dict)
    nonces: dict[str, str] = field(default_factory=dict)
    typing_channels: set[str] = field(default_factory=set)

    def call(self, client: str, operation: str, *values: str) -> None:
        self.calls.append((client, operation, tuple(values)))


class MockDiscord(DiscordClient):
    def __init__(self, state: MockState) -> None:
        self.state = state
        self.next_id = 1

    async def send(self, channel_id: str, content: str = "", *, embed: JsonObject | None = None, nonce: str = "") -> str:
        self.state.call("discord", "send", channel_id, content)
        if nonce and nonce in self.state.nonces:
            return self.state.nonces[nonce]
        message_id = f"message-{self.next_id}"
        self.next_id += 1
        self.state.messages[message_id] = content
        if embed is not None:
            self.state.embeds[message_id] = embed
        if nonce:
            self.state.nonces[nonce] = message_id
        return message_id

    async def start_typing(self, channel_id: str) -> None:
        self.state.call("discord", "start_typing", channel_id)
        self.state.typing_channels.add(channel_id)

    async def stop_typing(self, channel_id: str) -> None:
        self.state.call("discord", "stop_typing", channel_id)
        self.state.typing_channels.discard(channel_id)

    async def edit(self, ref: MessageRef, content: str, *, upload: Upload | None = None) -> None:
        message_id = ref.message_id
        self.state.call("discord", "edit", message_id, content)
        self.state.messages[message_id] = content
        if upload is not None:
            self.state.uploads[message_id] = upload

    async def add_reaction(self, ref: MessageRef, emoji: str) -> None:
        message_id = ref.message_id
        self.state.call("discord", "add_reaction", message_id, emoji)
        reactions = self.state.reactions.setdefault(message_id, [])
        if emoji not in reactions:
            reactions.append(emoji)

    async def remove_reaction(self, ref: MessageRef, emoji: str) -> None:
        message_id = ref.message_id
        self.state.call("discord", "remove_reaction", message_id, emoji)
        reactions = self.state.reactions.setdefault(message_id, [])
        if emoji in reactions:
            reactions.remove(emoji)

    async def send_file(self, channel_id: str, upload: Upload, caption: str = "") -> DeliveryReceipt:
        self.state.call("discord", "send_file", channel_id, upload.name, caption)
        message_id = f"file-{self.next_id}"
        self.next_id += 1
        self.state.uploads[message_id] = upload
        return DeliveryReceipt(message_id, f"https://discord.test/{channel_id}/{message_id}")

    async def set_profile(self, nickname: str | None, avatar: bytes | None) -> str:
        self.state.call("discord", "set_profile", nickname or "", str(len(avatar or b"")))
        if nickname is not None:
            self.state.profile["nickname"] = nickname
        if avatar is not None:
            self.state.profile["avatar"] = avatar
        return str(self.state.profile.get("nickname", "Wiseman"))


class MockPrompts(PromptClient):
    async def source(self, kind: str) -> str:
        if (self_kind := {"startup": "startup-context", "followup": "followup-context"}.get(kind, kind)) in {"startup-context", "followup-context"}:
            return '{"schema":"mock","mode":"{{ mode }}","messages":{{ messages | tojson }}}'
        return f"mock prompt: {self_kind}"


class MockHarnessRunner(HarnessRunner):
    def __init__(self, state: MockState | None = None) -> None:
        self.state = state or MockState()
        self.run_gate: asyncio.Event | None = None
        self.run_started = asyncio.Event()
        self.stop_requested: set[str] = set()

    async def acquire(self, user: str, workspace: str) -> None:
        self.state.call("runner", "acquire", user, workspace)

    async def release(self, user: str, workspace: str) -> None:
        self.state.call("runner", "release", user, workspace)

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        self.state.call("runner", "start", thread, user, workspace)
        return thread or f"codex-{workspace or user}"

    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        self.state.call("runner", "run", thread, user, workspace, prompt)
        self.run_started.set()
        if self.run_gate is not None:
            await self.run_gate.wait()
        if thread in self.stop_requested:
            self.stop_requested.remove(thread)
            raise RunnerError(STOPPED_STATUS, "turn stopped by user")
        if progress is not None:
            for message in ("🤖 Codex turn started...", "✍️ Writing response..."):
                await progress(message)
        return thread or f"codex-{workspace or user}", f"mock response: {prompt[:80]}", {"model": "mock"}

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        self.state.call("runner", "steer", thread, user, workspace)
        return bool(prompt)

    async def stop(self, thread: str, user: str, workspace: str, target_message_id: str, command_id: str) -> bool:
        self.state.call("runner", "stop", thread, user, workspace, target_message_id, command_id)
        self.stop_requested.add(thread)
        if self.run_gate is not None:
            self.run_gate.set()
        return True


def mock_container(settings: ClientSettings | None = None, *, temporal: TemporalClient | None = None) -> ClientContainer:
    state = MockState()
    settings = settings or ClientSettings()

    return ClientContainer(
        mode=ClientMode.MOCK,
        discord=MockDiscord(state),
        temporal=temporal or TemporalRuntime(settings.temporal_address, settings.temporal_queue),
        phoenix=Phoenix(settings.phoenix_endpoint, settings.phoenix_key, settings.phoenix_project),
        prompts=MockPrompts(),
        runner=MockHarnessRunner(state),
        provider=MockOpenRouter(),
        settings=settings,
    )
