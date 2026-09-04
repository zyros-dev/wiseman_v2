# Copyright (c) 2026 Nick van der Merwe
"""Stateful fuzzing for the deterministic Discord client boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from app.clients.mock_clients import MockClientError, MockDiscord, MockState

if TYPE_CHECKING:
    from collections.abc import Awaitable


async def _resolve[Result](awaitable: Awaitable[Result]) -> Result:
    return await awaitable


ResultT = TypeVar("ResultT")


def run[ResultT](awaitable: Awaitable[ResultT]) -> ResultT:
    """Execute one async client operation from a synchronous state-machine rule."""
    return asyncio.run(_resolve(awaitable))


def run_safely[ResultT](awaitable: Awaitable[ResultT]) -> ResultT | None:
    try:
        return run(awaitable)
    except MockClientError:
        return None


class DiscordClientMachine(RuleBasedStateMachine):
    """Generate valid and repeated Discord lifecycle calls against one fake API."""

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
