# Copyright (c) 2026 Nick van der Merwe
"""Deterministic complete client graph used by tests and stateful fuzzing."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    DiscordClient,
    JsonObject,
    JsonValue,
    PhoenixClient,
    ProviderClient,
    RunnerClient,
    RunnerResult,
    TemporalClient,
)

type Failure = str


class MockClientError(RuntimeError):
    """Raised when a configured mock operation fails."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"mock failure: {operation}")


@dataclass(frozen=True, slots=True)
class MockCall:
    """One ordered observable external-client operation."""

    client: str
    operation: str
    values: tuple[str, ...] = ()


@dataclass(slots=True)
class MockState:
    """Shared fake state and deterministic failure queue."""

    calls: list[MockCall] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    messages: dict[str, str] = field(default_factory=dict)
    reactions: dict[str, list[str]] = field(default_factory=dict)
    threads: dict[str, str] = field(default_factory=dict)
    records: list[JsonObject] = field(default_factory=list)
    audits: dict[str, JsonObject] = field(default_factory=dict)
    turn_ids: dict[str, str] = field(default_factory=dict)

    def call(self, client: str, operation: str, *values: str) -> None:
        self.calls.append(MockCall(client, operation, tuple(values)))
        if self.failures and self.failures[0] == f"{client}.{operation}":
            self.failures.pop(0)
            failed_operation = f"{client}.{operation}"
            raise MockClientError(failed_operation)


class MockDiscord(DiscordClient):
    def __init__(self, state: MockState) -> None:
        self.state = state
        self.next_id = 1

    async def history(self, channel_id: str, limit: int) -> list[JsonObject]:
        self.state.call("discord", "history", channel_id, str(limit))
        return []

    async def create_thread(self, channel_id: str, name: str, auto_archive_minutes: int) -> str:
        self.state.call("discord", "create_thread", channel_id, name, str(auto_archive_minutes))
        thread_id = f"thread-{self.next_id}"
        self.next_id += 1
        self.state.threads[thread_id] = name
        return thread_id

    async def send(
        self, channel_id: str, content: str = "", *, embed: JsonObject | None = None
    ) -> str:
        del embed
        self.state.call("discord", "send", channel_id, content)
        message_id = f"message-{self.next_id}"
        self.next_id += 1
        self.state.messages[message_id] = content
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

    async def lock_thread(self, thread_id: str) -> None:
        self.state.call("discord", "lock_thread", thread_id)

    async def send_file(self, channel_id: str, path: str, caption: str = "") -> str:
        self.state.call("discord", "send_file", channel_id, path, caption)
        return f"file-{self.next_id}"

    async def set_profile(self, username: str | None, avatar: str | None) -> None:
        self.state.call("discord", "set_profile", username or "", avatar or "")

    async def set_reactions(self, values: dict[str, str]) -> None:
        self.state.call("discord", "set_reactions", *sorted(values.values()))


class MockTemporal(TemporalClient):
    def __init__(self, state: MockState) -> None:
        self.state = state

    async def submit(self, event: JsonObject) -> None:
        self.state.call("temporal", "submit", str(event.get("trigger", "")))

    async def signal(self, workflow_id: str, event: JsonObject) -> None:
        self.state.call("temporal", "signal", workflow_id, str(event.get("trigger", "")))


class MockPhoenix(PhoenixClient):
    def __init__(self, state: MockState) -> None:
        self.state = state

    async def record(self, trace: str, node: str, **data: JsonValue) -> None:
        del data
        self.state.call("phoenix", "record", trace, node)
        self.state.records.append({"trace": trace, "node": node})
        if node == "admission":
            self.state.audits[trace] = {"trace": trace, "node": node}

    def audit(self, audit_id: str) -> JsonObject | None:
        self.state.call("phoenix", "audit", audit_id)
        return self.state.audits.get(audit_id)


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

    async def run(self, thread: str, prompt: str, user: str, workspace: str = "") -> RunnerResult:
        self.state.call("runner", "run", thread, user, workspace)
        thread_id = thread or self.state.turn_ids.get(workspace, f"codex-{user}")
        return RunnerResult(thread_id, f"mock response: {prompt[:80]}", {"model": "mock"})

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        self.state.call("runner", "steer", thread, user, workspace)
        return bool(prompt)


class MockProvider(ProviderClient):
    def __init__(self, state: MockState) -> None:
        self.state = state

    async def response(self, payload: JsonObject) -> JsonObject:
        self.state.call("provider", "response", str(payload.get("model", "")))
        return {"model": "mock", "output": "mock provider response"}


def mock_container() -> ClientContainer:
    """Create a fresh, fully isolated mock dependency graph."""
    state = MockState()
    return ClientContainer(
        mode=ClientMode.MOCK,
        discord=MockDiscord(state),
        temporal=MockTemporal(state),
        phoenix=MockPhoenix(state),
        runner=MockRunner(state),
        provider=MockProvider(state),
    )
