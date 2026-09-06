# Copyright (c) 2026 Nick van der Merwe
"""Graph-only runner controls for deterministic lifecycle edge stimuli."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.clients.mock_clients import MockHarnessRunner
from app.runner import RunnerError
from app.runner_status import STOPPED_STATUS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class GraphRunner(MockHarnessRunner):
    def __init__(self, state) -> None:
        super().__init__(state)
        self.completion_gate: asyncio.Event | None = None
        self.progress_sent = asyncio.Event()
        self.next_error: RunnerError | None = None
        self.start_error: RunnerError | None = None

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        self.state.call("runner", "start", thread, user, workspace)
        if self.start_error is not None:
            raise self.start_error
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
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        if thread in self.stop_requested:
            self.stop_requested.remove(thread)
            raise RunnerError(STOPPED_STATUS, "turn stopped by user")
        if progress is not None:
            for message in ("🤖 Codex turn started...", "✍️ Writing response..."):
                await progress(message)
        self.progress_sent.set()
        if self.completion_gate is not None:
            await self.completion_gate.wait()
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        if thread in self.stop_requested:
            self.stop_requested.remove(thread)
            raise RunnerError(STOPPED_STATUS, "turn stopped by user")
        return thread or f"codex-{workspace or user}", f"mock response: {prompt[:80]}", {"model": "mock"}

    async def stop(self, thread: str, user: str, workspace: str, target_message_id: str, command_id: str) -> bool:
        self.state.call("runner", "stop", thread, user, workspace, target_message_id, command_id)
        self.stop_requested.add(thread)
        if self.run_gate is not None:
            self.run_gate.set()
        if self.completion_gate is not None:
            self.completion_gate.set()
        return True
