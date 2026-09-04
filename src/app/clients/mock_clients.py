# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.types import JsonObject

type Failure = str


@dataclass(frozen=True, slots=True)
class MockCall:
    client: str
    operation: str
    values: tuple[str, ...] = ()


@dataclass(slots=True)
class MockState:
    calls: list[MockCall] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    messages: dict[str, str] = field(default_factory=dict)
    reactions: dict[str, list[str]] = field(default_factory=dict)
    threads: dict[str, str] = field(default_factory=dict)
    thread_activity: dict[str, float] = field(default_factory=dict)
    archived: set[str] = field(default_factory=set)
    locked: set[str] = field(default_factory=set)
    channel_history: dict[str, list[JsonObject]] = field(default_factory=dict)
    fake_time: float = 0
    records: list[dict[str, object]] = field(default_factory=list)
    audits: dict[str, dict[str, object]] = field(default_factory=dict)
    turn_ids: dict[str, str] = field(default_factory=dict)
    profile: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    def call(self, client: str, operation: str, *values: str) -> None:
        self.calls.append(MockCall(client, operation, tuple(values)))
        if self.failures and self.failures[0] == f"{client}.{operation}":
            self.failures.pop(0)
            failed_operation = f"{client}.{operation}"
            raise RuntimeError(failed_operation)


class MockDiscord(DiscordClient):
    def __init__(self, state: MockState) -> None:
        self.state = state
        self.next_id = 1

    async def history(self, channel_id: str, limit: int) -> list[JsonObject]:
        self.state.call("discord", "history", channel_id, str(limit))
        return list(self.state.channel_history.get(channel_id, []))[-limit:]

    async def create_thread(self, channel_id: str, name: str, auto_archive_minutes: int) -> str:
        self.state.call("discord", "create_thread", channel_id, name, str(auto_archive_minutes))
        thread_id = f"thread-{self.next_id}"
        self.next_id += 1
        self.state.threads[thread_id] = name
        self.state.thread_activity[thread_id] = self.state.fake_time
        return thread_id

    async def send(self, channel_id: str, content: str = "", *, embed: JsonObject | None = None) -> str:
        del embed
        self.state.call("discord", "send", channel_id, content)
        message_id = f"message-{self.next_id}"
        self.next_id += 1
        self.state.messages[message_id] = content
        self.state.channel_history.setdefault(channel_id, []).append(
            {"id": message_id, "channel_id": channel_id, "content": content}
        )
        return message_id

    async def edit(self, message_id: str, content: str) -> None:
        self.state.call("discord", "edit", message_id)
        self.state.messages[message_id] = content

    async def add_reaction(self, message_id: str, emoji: str) -> None:
        self.state.call("discord", "add_reaction", message_id, emoji)
        reactions = self.state.reactions.setdefault(message_id, [])
        if emoji not in reactions:
            reactions.append(emoji)

    async def remove_reaction(self, message_id: str, emoji: str) -> None:
        self.state.call("discord", "remove_reaction", message_id, emoji)
        reactions = self.state.reactions.setdefault(message_id, [])
        if emoji in reactions:
            reactions.remove(emoji)

    async def archive_thread(self, thread_id: str) -> None:
        self.state.call("discord", "archive_thread", thread_id)
        self.state.archived.add(thread_id)

    async def lock_thread(self, thread_id: str) -> None:
        self.state.call("discord", "lock_thread", thread_id)
        self.state.locked.add(thread_id)

    async def send_file(self, channel_id: str, path: str, caption: str = "") -> str:
        self.state.call("discord", "send_file", channel_id, path, caption)
        message_id = f"file-{self.next_id}"
        self.next_id += 1
        self.state.files[message_id] = path
        return message_id

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("fake time cannot move backwards")
        self.state.fake_time += seconds
        for thread_id, touched in self.state.thread_activity.items():
            if self.state.fake_time - touched >= 60 * 60:
                self.state.archived.add(thread_id)
                self.state.locked.add(thread_id)

    async def set_profile(self, username: str | None, avatar: str | None) -> None:
        self.state.call("discord", "set_profile", username or "", avatar or "")
        if username is not None:
            self.state.profile["username"] = username
        if avatar is not None:
            self.state.profile["avatar"] = avatar

    async def set_reactions(self, values: dict[str, str]) -> None:
        self.state.call("discord", "set_reactions", *sorted(values.values()))


class MockTemporal(TemporalClient):
    def __init__(self, state: MockState) -> None:
        self.state = state

    async def submit(self, event: JsonObject) -> None:
        self.state.call("temporal", "submit", str(event.get("trigger", "")))


class MockPhoenix(PhoenixClient):
    def __init__(self, state: MockState) -> None:
        self.state = state
        self.records = state.records

    async def record(self, trace: str, node: str, **data: object) -> None:
        del data
        self.state.call("phoenix", "record", trace, node)
        self.state.records.append({"trace": trace, "node": node})
        if node == "admission":
            self.state.audits[trace] = {"trace": trace, "node": node}

    def audit(self, audit_id: str) -> dict[str, object] | None:
        return self.state.call("phoenix", "audit", audit_id) or self.state.audits.get(audit_id)


class MockPrompts(PromptClient):
    async def source(self, kind: str) -> str:
        self_kind = {"startup": "startup-context", "followup": "followup-context"}.get(kind, kind)
        if self_kind in {"startup-context", "followup-context"}:
            return '{"schema":"mock","mode":"{{ mode }}","messages":{{ messages | tojson }}}'
        return f"mock prompt: {self_kind}"


class MockRunner(RunnerClient):
    def __init__(self, state: MockState) -> None:
        self.state = state

    async def acquire(self, user: str, workspace: str) -> None:
        self.state.call("runner", "acquire", user, workspace)

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        self.state.call("runner", "start", thread, user, workspace)
        thread_id = thread or f"codex-{user}"
        self.state.turn_ids[workspace or thread_id] = thread_id
        return thread_id

    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        self.state.call("runner", "run", thread, user, workspace)
        thread_id = thread or self.state.turn_ids.get(workspace, f"codex-{user}")
        if progress is not None:
            await progress("🤖 Codex turn started...")
            await progress("✍️ Writing response...")
        return thread_id, f"mock response: {prompt[:80]}", {"model": "mock"}

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        self.state.call("runner", "steer", thread, user, workspace)
        return bool(prompt)


def mock_container(settings: ClientSettings | None = None) -> ClientContainer:
    state = MockState()
    return ClientContainer(
        mode=ClientMode.MOCK,
        discord=MockDiscord(state),
        temporal=MockTemporal(state),
        phoenix=MockPhoenix(state),
        prompts=MockPrompts(),
        runner=MockRunner(state),
        settings=settings or ClientSettings(),
    )
