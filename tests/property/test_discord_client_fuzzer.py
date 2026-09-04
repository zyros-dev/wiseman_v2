# Copyright (c) 2026 Nick van der Merwe

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar, cast

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from app.clients.mock_clients import (
    MockClientError,
    MockDiscord,
    MockPhoenix,
    MockRunner,
    MockState,
)
from app.engine import Engine, EngineConfig
from app.models import Event, Message
from app.phoenix import PromptHub

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import discord

    from app.phoenix import Phoenix
    from app.runner import Runner
    from app.types import JsonObject


async def _resolve[Result](awaitable: Awaitable[Result]) -> Result:
    return await awaitable


ResultT = TypeVar("ResultT")


def run[ResultT](awaitable: Awaitable[ResultT]) -> ResultT:
    return asyncio.run(_resolve(awaitable))


def run_safely[ResultT](awaitable: Awaitable[ResultT]) -> ResultT | None:
    try:
        return run(awaitable)
    except MockClientError:
        return None


class DiscordClientMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.state = MockState()
        self.client = MockDiscord(self.state)
        self.thread_ids: list[str] = []
        self.message_ids: list[str] = []

    @initialize()
    def starts_empty(self) -> None:
        assert self.state.calls == []
        assert self.state.threads == {}
        assert self.state.messages == {}

    @rule(
        operation=st.sampled_from(
            (
                "discord.create_thread",
                "discord.send",
                "discord.edit",
                "discord.add_reaction",
                "discord.remove_reaction",
                "discord.archive_thread",
                "discord.lock_thread",
                "discord.send_file",
                "discord.set_profile",
                "discord.set_reactions",
            )
        )
    )
    def inject_one_shot_failure(self, operation: str) -> None:
        self.state.failures.append(operation)

    @rule(name=st.text(min_size=1, max_size=30))
    def creates_threads(self, name: str) -> None:
        before = set(self.state.threads)
        thread = run_safely(self.client.create_thread("channel", name, 60))
        if thread is not None:
            self.thread_ids.append(thread)
        else:
            assert set(self.state.threads) == before

    @rule()
    def sends_messages(self) -> None:
        if not self.thread_ids:
            return
        before = dict(self.state.messages)
        message = run_safely(self.client.send(self.thread_ids[-1], "working"))
        if message is not None:
            self.message_ids.append(message)
        else:
            assert self.state.messages == before

    @rule()
    def edits_messages(self) -> None:
        if self.message_ids:
            message_id = self.message_ids[-1]
            before = self.state.messages[message_id]
            expected_failure = bool(
                self.state.failures and self.state.failures[0] == "discord.edit"
            )
            run_safely(self.client.edit(message_id, "updated"))
            if expected_failure:
                assert self.state.messages[message_id] == before

    @rule(emoji=st.sampled_from(("👀", "✅", "❌")))
    def adds_reactions(self, emoji: str) -> None:
        if self.message_ids:
            run_safely(self.client.add_reaction(self.message_ids[-1], emoji))

    @rule(emoji=st.sampled_from(("👀", "✅", "❌")))
    def removes_reactions_idempotently(self, emoji: str) -> None:
        if self.message_ids:
            run_safely(self.client.remove_reaction(self.message_ids[-1], emoji))

    @rule()
    def archives_threads(self) -> None:
        if self.thread_ids:
            run_safely(self.client.archive_thread(self.thread_ids[-1]))

    @rule()
    def locks_threads(self) -> None:
        if self.thread_ids:
            run_safely(self.client.lock_thread(self.thread_ids[-1]))

    @rule()
    def sends_files(self) -> None:
        if self.thread_ids:
            run_safely(self.client.send_file(self.thread_ids[-1], "artifact.bin"))

    @rule(
        username=st.one_of(st.none(), st.text(min_size=2, max_size=12)),
        avatar=st.one_of(st.none(), st.text(max_size=12)),
    )
    def updates_profile(self, username: str | None, avatar: str | None) -> None:
        run_safely(self.client.set_profile(username, avatar))

    @rule()
    def updates_reactions(self) -> None:
        run_safely(self.client.set_reactions({"processing": "👀", "success": "✅"}))

    @rule(seconds=st.integers(min_value=0, max_value=7_200))
    def advances_fake_time(self, seconds: int) -> None:
        self.client.advance(seconds)

    @invariant()
    def reactions_are_sets(self) -> None:
        assert all(len(values) == len(set(values)) for values in self.state.reactions.values())

    @invariant()
    def every_call_is_recorded_as_discord(self) -> None:
        assert all(call.client == "discord" for call in self.state.calls)

    @invariant()
    def idle_threads_are_closed(self) -> None:
        for thread_id, created in self.state.thread_activity.items():
            if self.state.fake_time - created >= 3_600:
                assert thread_id in self.state.archived
                assert thread_id in self.state.locked


TestDiscordClientMachine = DiscordClientMachine.TestCase


class _LiveMessage:
    def __init__(self, client: MockDiscord, message_id: str) -> None:
        self.client, self.id = client, message_id

    async def edit(self, *, content: str) -> None:
        await self.client.edit(self.id, content)

    async def add_reaction(self, emoji: str) -> None:
        await self.client.add_reaction(self.id, emoji)

    async def remove_reaction(self, emoji: str, _user: object) -> None:
        await self.client.remove_reaction(self.id, emoji)


class _DeliveryChannel:
    def __init__(self, client: MockDiscord, channel_id: str) -> None:
        self.client, self.channel_id = client, channel_id

    async def send(self, content: str = "", *, embed: object | None = None) -> _LiveMessage:
        message_id = await self.client.send(
            self.channel_id, content, embed=cast("JsonObject | None", embed)
        )
        return _LiveMessage(self.client, message_id)


class _EngineRunner:
    def __init__(self, state: MockState) -> None:
        self.client = MockRunner(state)

    async def acquire(self, user: str, workspace: str) -> None:
        await self.client.acquire(user, workspace)

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        return await self.client.start(thread, user, workspace)

    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        return await self.client.run(thread, prompt, user, workspace, progress=progress)

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        return await self.client.steer(thread, prompt, user, workspace)


class EngineLifecycleMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.state = MockState()
        self.discord = MockDiscord(self.state)
        self.runner = _EngineRunner(self.state)
        self.channel = _DeliveryChannel(self.discord, "thread")
        self.engine = self._new_engine()
        self.started = False
        self.next_message = 0
        self.accepted: list[str] = []
        self.durable: JsonObject = {}

    def _event(self, kind: str, message_id: str) -> Event:
        return Event(
            trigger=Message(
                id=message_id,
                author_id="user",
                author_name="user",
                content=f"request {message_id}",
                channel_id="parent",
                thread_id="thread",
                timestamp=message_id,
            ),
            kind=kind,
            parent_messages=[],
            thread_messages=[],
        )

    def _new_engine(self) -> Engine:
        engine = Engine(
            EngineConfig(
                cast("Phoenix", MockPhoenix(self.state)),
                cast("Runner", self.runner),
                PromptHub(),
            )
        )
        engine.reaction_user = cast("discord.User", object())
        return engine

    def _submit(self, event: Event) -> dict[str, object]:
        result = cast(
            "dict[str, object]",
            run(
                self.engine.handle(
                    event,
                    cast("discord.Message", _LiveMessage(self.discord, event.trigger.id)),
                    self.channel,
                    self.durable,
                )
            ),
        )
        if isinstance(result.get("state"), dict):
            self.durable = cast("JsonObject", result["state"])
        return result

    @initialize()
    def starts_without_turns(self) -> None:
        assert self.durable == {}
        assert self.accepted == []

    @rule()
    def starts_thread_once(self) -> None:
        if self.started:
            return
        message_id = self._new_message()
        result = self._submit(self._event("startup", message_id))
        assert result.get("output")
        self.started = True
        self.accepted.append(message_id)
        self._assert_terminal(message_id, "✅")
        assert self.durable.get("turn") == 1

    @rule()
    def submits_followup(self) -> None:
        if not self.started:
            return
        message_id = self._new_message()
        result = self._submit(self._event("followup", message_id))
        if "output" in result:
            self.accepted.append(message_id)
            self._assert_terminal(message_id, "✅")
            assert self.durable.get("turn") == len(self.accepted)

    @rule()
    def handles_runner_failure_without_advancing_turn(self) -> None:
        if not self.started:
            return
        message_id = self._new_message()
        self.state.failures.append("runner.run")
        result = self._submit(self._event("followup", message_id))
        assert "error" in result
        self._assert_terminal(message_id, "❌")
        assert self.durable.get("turn") == len(self.accepted)

    @rule()
    def repeats_a_delivery(self) -> None:
        if not self.accepted:
            return
        message_id = self.accepted[-1]
        calls_before = len(self.state.calls)
        result = self._submit(self._event("followup", message_id))
        assert result.get("status") == "duplicate"
        assert len(self.state.calls) == calls_before

    @rule()
    def restarts_from_durable_state(self) -> None:
        self.engine = self._new_engine()

    @invariant()
    def accepted_turns_have_one_terminal_reaction(self) -> None:
        assert all(len(self.state.reactions[mid]) == 1 for mid in self.accepted)

    @invariant()
    def durable_turn_matches_successes(self) -> None:
        assert self.durable.get("turn", 0) == len(self.accepted)

    def _new_message(self) -> str:
        self.next_message += 1
        return f"turn-{self.next_message}"

    def _assert_terminal(self, message_id: str, emoji: str) -> None:
        assert self.state.reactions.get(message_id) == [emoji]


TestEngineLifecycleMachine = EngineLifecycleMachine.TestCase
