# Copyright (c) 2026 Nick van der Merwe
"""Stateful fuzzing for the deterministic Discord client boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from app.clients.mock_clients import MockDiscord, MockState

if TYPE_CHECKING:
    from collections.abc import Awaitable


async def _resolve[Result](awaitable: Awaitable[Result]) -> Result:
    return await awaitable


def run[Result](awaitable: Awaitable[Result]) -> Result:
    """Execute one async client operation from a synchronous state-machine rule."""
    return asyncio.run(_resolve(awaitable))


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

    @rule(name=st.text(min_size=1, max_size=30))
    def creates_threads(self, name: str) -> None:
        thread = run(self.client.create_thread("channel", name, 60))
        self.thread_ids.append(thread)

    @rule()
    def sends_messages(self) -> None:
        if not self.thread_ids:
            return
        message = run(self.client.send(self.thread_ids[-1], "working"))
        self.message_ids.append(message)

    @rule(emoji=st.sampled_from(("👀", "✅", "❌")))
    def adds_reactions(self, emoji: str) -> None:
        if self.message_ids:
            run(self.client.add_reaction(self.message_ids[-1], emoji))

    @rule(emoji=st.sampled_from(("👀", "✅", "❌")))
    def removes_reactions_idempotently(self, emoji: str) -> None:
        if self.message_ids:
            run(self.client.remove_reaction(self.message_ids[-1], emoji))

    @rule()
    def archives_threads(self) -> None:
        if self.thread_ids:
            run(self.client.archive_thread(self.thread_ids[-1]))

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
