# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    DiscordClient,
    PhoenixClient,
    PromptClient,
    RunnerClient,
    TemporalClient,
)
from app.clients.provider import OpenRouter
from app.models import DeliveryReceipt

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.models import Event, MessageRef, Upload
    from app.types import JsonObject


@dataclass(slots=True)
class MockState:
    calls: list[tuple[str, str, tuple[str, ...]]] = field(default_factory=list)
    messages: dict[str, str] = field(default_factory=dict)
    embeds: dict[str, JsonObject] = field(default_factory=dict)
    reactions: dict[str, list[str]] = field(default_factory=dict)
    records: list[dict[str, object]] = field(default_factory=list)
    audits: dict[str, dict[str, object]] = field(default_factory=dict)
    profile: dict[str, str | bytes] = field(default_factory=dict)
    uploads: dict[str, Upload] = field(default_factory=dict)
    nonces: dict[str, str] = field(default_factory=dict)
    admitted: list[JsonObject] = field(default_factory=list)

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

    async def set_profile(self, username: str | None, avatar: bytes | None) -> str:
        self.state.call("discord", "set_profile", username or "", str(len(avatar or b"")))
        if username is not None:
            self.state.profile["username"] = username
        if avatar is not None:
            self.state.profile["avatar"] = avatar
        return str(self.state.profile.get("username", "Wiseman"))


class MockTemporal(TemporalClient):
    def __init__(self, state: MockState) -> None:
        self.state = state

    async def submit(self, event: JsonObject) -> dict[str, object] | None:
        self.state.call("temporal", "submit", str(event.get("trigger", "")))
        self.state.admitted.append(event)
        return None

    async def start(self) -> None:
        self.state.call("temporal", "start")

    async def close(self) -> None:
        self.state.call("temporal", "close")

    async def steer(self, event: Event) -> bool:
        self.state.call("temporal", "steer", event.trigger.id)
        return False

    async def stop(self, event: Event) -> bool:
        self.state.call("temporal", "stop", event.trigger.id)
        return False


class MockPhoenix(PhoenixClient):
    def __init__(self, state: MockState) -> None:
        self.state = state
        self.records = state.records

    async def record(self, trace: str, node: str, **data: object) -> None:
        self.state.call("phoenix", "record", trace, node)
        self.state.records.append({"trace": trace, "node": node, **data})
        if node == "admission":
            self.state.audits[trace] = {"trace": trace, "node": node, **data}

    def audit(self, audit_id: str) -> dict[str, object] | None:
        return self.state.call("phoenix", "audit", audit_id) or self.state.audits.get(audit_id)


class MockPrompts(PromptClient):
    async def source(self, kind: str) -> str:
        self_kind = {"startup": "startup-context", "followup": "followup-context"}.get(kind, kind)
        if self_kind in {"startup-context", "followup-context"}:
            return '{"schema":"mock","mode":"{{ mode }}","messages":{{ messages | tojson }}}'
        return f"mock prompt: {self_kind}"


class MockRunner(RunnerClient):
    def __init__(self, state: MockState | None = None) -> None:
        self.state = state or MockState()

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
        if progress is not None:
            for message in ("🤖 Codex turn started...", "✍️ Writing response..."):
                await progress(message)
        return thread or f"codex-{workspace or user}", f"mock response: {prompt[:80]}", {"model": "mock"}

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        self.state.call("runner", "steer", thread, user, workspace)
        return bool(prompt)

    async def stop(self, thread: str, user: str, workspace: str, target_message_id: str, command_id: str) -> bool:
        self.state.call("runner", "stop", thread, user, workspace, target_message_id, command_id)
        return True


def mock_container(settings: ClientSettings | None = None) -> ClientContainer:
    state = MockState()

    def provider(request: httpx.Request) -> httpx.Response:
        state.call("provider", request.url.path)
        if request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                text='data: {"type":"response.completed","response":{"model":"mock"}}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"model": "mock-vision", "choices": [{"message": {"content": "mock image description"}}]})

    return ClientContainer(
        mode=ClientMode.MOCK,
        discord=MockDiscord(state),
        temporal=MockTemporal(state),
        phoenix=MockPhoenix(state),
        prompts=MockPrompts(),
        runner=MockRunner(state),
        provider=OpenRouter(ClientSettings(provider_key="mock"), MockPrompts(), httpx.MockTransport(provider)),
        settings=settings or ClientSettings(),
    )
