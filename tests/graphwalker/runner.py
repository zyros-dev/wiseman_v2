# Copyright (c) 2026 Nick van der Merwe
"""Graph-only runner controls for deterministic lifecycle edge stimuli."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.clients.mock_clients import MockHarnessRunner
from app.runner import MESSAGE_ID, RunnerError
from app.runner_status import STOPPED_STATUS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class GraphRunner(MockHarnessRunner):
    def __init__(self, state) -> None:
        super().__init__(state)
        self.completion_gate: asyncio.Event | None = None
        self.hold_stop = False
        self.active_threads: set[str] = set()
        self.progress_sent = asyncio.Event()
        self._pending_errors: dict[str, tuple[int, RunnerError]] = {}
        self._next_error: tuple[int, RunnerError] | None = None
        self._attempts = 0
        self._error_epochs = 0
        self._consumed_error_epoch = 0
        self._active_attempts: dict[str, int] = {}
        self._active_message_ids: dict[str, str] = {}
        self._attempt_gates: dict[int, tuple[asyncio.Event | None, asyncio.Event | None]] = {}
        self._stop_attempts: dict[str, int] = {}
        self._attempt_errors: dict[int, tuple[str, int, RunnerError]] = {}
        self._failed_threads: set[str] = set()
        self._held_messages: set[str] = set()
        self._released_messages: set[str] = set()
        self._message_gates: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}
        self.start_error: RunnerError | None = None
        self.history: list[str] = []

    def _record(self, event: str) -> None:
        self.history.append(event)
        del self.history[:-64]

    def active_message_id(self) -> str:
        return next(iter(self._active_message_ids.values()), "")

    def inject_error(self, error: RunnerError, message_id: str = "") -> None:
        self._error_epochs += 1
        entry = (self._error_epochs, error)
        self._record(f"inject status={error.status} message={message_id}")
        if not message_id:
            self._next_error = entry
            return
        if self._active_attempts:
            target = max(self._active_attempts.values())
            target_thread = next(thread for thread, attempt in self._active_attempts.items() if attempt == target)
            if self._active_message_ids.get(target_thread) == message_id:
                self._attempt_errors[target] = (message_id, *entry)
                self._pending_errors[message_id] = entry
                self._failed_threads.discard(target_thread)
            else:
                self._pending_errors[message_id] = entry
        else:
            self._pending_errors[message_id] = entry

    def clear_error(self) -> None:
        self._record(f"clear-error pending={len(self._pending_errors)} attempts={len(self._attempt_errors)}")
        self._pending_errors.clear()
        self._attempt_errors.clear()
        self._next_error = None
        self.start_error = None

    def clear_stops(self) -> None:
        self.stop_requested.clear()
        self._stop_attempts.clear()

    def reset_fixture(self) -> None:
        self.run_gate = None
        self.completion_gate = None
        self.hold_stop = False
        self.active_threads.clear()
        self._active_attempts.clear()
        self._active_message_ids.clear()
        self._attempt_gates.clear()
        self._failed_threads.clear()
        self._held_messages.clear()
        self._released_messages.clear()
        self._message_gates.clear()
        self.clear_error()
        self.clear_stops()

    def hold_message(self, message_id: str) -> None:
        if message_id:
            self._held_messages.add(message_id)
            self._message_gates.setdefault(message_id, (asyncio.Event(), asyncio.Event()))
            self._record(f"hold-message {message_id}")

    def release_message(self, message_id: str) -> None:
        self._held_messages.discard(message_id)
        self._released_messages.add(message_id)
        gates = self._message_gates.setdefault(message_id, (asyncio.Event(), asyncio.Event()))
        gates[0].set()
        self._record(f"release-message {message_id}")

    def release_message_attempt(self, message_id: str) -> bool:
        gates = self._message_gates.get(message_id)
        if gates is None:
            return False
        self.release_message(message_id)
        gates[1].set()
        self._record(f"release-message-attempt {message_id}")
        return True

    def reset_message_attempt(self, message_id: str) -> None:
        run_gate, completion_gate = asyncio.Event(), asyncio.Event()
        run_gate.set()
        self._message_gates[message_id] = (run_gate, completion_gate)
        self._released_messages.discard(message_id)
        self._record(f"reset-message-attempt {message_id}")

    def release_attempt(self, message_id: str) -> bool:
        self._record(f"release-attempt message={message_id}")
        released = False
        for thread, attempt in self._active_attempts.items():
            if self._active_message_ids.get(thread) != message_id:
                continue
            for gate in self._attempt_gates.get(attempt, ()):
                if gate is not None:
                    gate.set()
            released = True
        return released

    def confirm_cancellation(self, message_id: str) -> bool:
        confirmed = False
        for thread, attempt in self._active_attempts.items():
            if self._active_message_ids.get(thread) != message_id:
                continue
            self._stop_attempts[thread] = attempt
            self.stop_requested.add(thread)
            confirmed = True
        self._record(f"confirm-cancellation message={message_id} confirmed={confirmed}")
        return confirmed

    def release_attempt_start(self, message_id: str) -> bool:
        for thread, attempt in self._active_attempts.items():
            if self._active_message_ids.get(thread) != message_id:
                continue
            run_gate, _ = self._attempt_gates.get(attempt, (None, None))
            if run_gate is not None:
                run_gate.set()
            return True
        return False

    def release_attempt_starts(self, message_id: str = "") -> bool:
        released = False
        for thread, attempt in self._active_attempts.items():
            if message_id and self._active_message_ids.get(thread) != message_id:
                continue
            run_gate, _ = self._attempt_gates.get(attempt, (None, None))
            if run_gate is not None:
                run_gate.set()
                released = True
        if self.run_gate is not None:
            self.run_gate.set()
            released = True
        if released:
            self._record(f"release-attempt-starts message={message_id}")
        return released

    async def wait_until_idle(self, thread: str) -> None:
        async with asyncio.timeout(5):
            while thread in self.active_threads:
                await asyncio.sleep(0)

    def _take_error(self, attempt: int, message_id: str) -> RunnerError | None:
        entry = self._attempt_errors.get(attempt)
        if entry is not None and entry[0] in {"", message_id}:
            self._attempt_errors.pop(attempt)
            _, epoch, error = entry
            self._pending_errors.pop(message_id, None)
        else:
            entry = self._pending_errors.pop(message_id, None)
            if entry is None:
                entry, self._next_error = self._next_error, None
            epoch, error = entry or (0, None)
        if error is None:
            self._record(f"take-empty attempt={attempt} message={message_id}")
            return None
        if epoch <= self._consumed_error_epoch:
            self._record(f"take-stale attempt={attempt} epoch={epoch} message={message_id}")
            return None
        self._consumed_error_epoch = epoch
        self._record(f"take-error attempt={attempt} status={error.status} message={message_id}")
        return error

    def _mark_failure(self, thread: str) -> None:
        self._failed_threads.add(thread)

    def _failure_replayed(self, thread: str) -> bool:
        return thread in self._failed_threads

    def _gates_for(self, message_id: str) -> tuple[asyncio.Event | None, asyncio.Event | None]:
        message_gates = self._message_gates.get(message_id)
        if message_id not in self._held_messages and message_id not in self._released_messages and message_gates is None:
            return self.run_gate, self.completion_gate
        run_gate, completion_gate = message_gates or (asyncio.Event(), asyncio.Event())
        self._message_gates[message_id] = (run_gate, completion_gate)
        self._released_messages.discard(message_id)
        return run_gate, completion_gate

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        self.state.call("runner", "start", thread, user, workspace)
        if self.start_error is not None:
            raise self.start_error
        return _session_id(thread, user, workspace)

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
        self._attempts += 1
        attempt = self._attempts
        self._record(f"run attempt={attempt} thread={thread} message={MESSAGE_ID.get()}")
        self.active_threads.add(thread)
        self._active_attempts[thread] = attempt
        message_id = MESSAGE_ID.get()
        self._active_message_ids[thread] = message_id
        try:
            run_gate, completion_gate = self._gates_for(message_id)
            self._attempt_gates[attempt] = (run_gate, completion_gate)
            if run_gate is not None:
                await run_gate.wait()
            self._record(f"run-gate-open attempt={attempt} message={message_id}")
            if error := self._take_error(attempt, message_id):
                self._mark_failure(thread)
                raise error
            if self._stop_attempts.get(thread) == attempt:
                self._stop_attempts.pop(thread, None)
                self.stop_requested.discard(thread)
                raise RunnerError(STOPPED_STATUS, "turn stopped by user")
            if progress is not None:
                for message in ("🤖 Codex turn started...", "✍️ Writing response..."):
                    await progress(message)
            self.progress_sent.set()
            self._record(f"progress attempt={attempt}")
            if completion_gate is not None:
                await completion_gate.wait()
            if error := self._take_error(attempt, message_id):
                if self._failure_replayed(thread):
                    return _session_id(thread, user, workspace), f"mock response: {prompt[:80]}", _billing()
                self._mark_failure(thread)
                self._record(f"raise-error attempt={attempt} status={error.status}")
                raise error
            if self._stop_attempts.get(thread) == attempt:
                self._stop_attempts.pop(thread, None)
                self.stop_requested.discard(thread)
                raise RunnerError(STOPPED_STATUS, "turn stopped by user")
            return _session_id(thread, user, workspace), f"mock response: {prompt[:80]}", _billing()
        finally:
            self.active_threads.discard(thread)
            self._active_attempts.pop(thread, None)
            self._active_message_ids.pop(thread, None)
            self._attempt_gates.pop(attempt, None)

    async def stop(self, thread: str, user: str, workspace: str, target_message_id: str, command_id: str) -> bool:
        self.state.call("runner", "stop", thread, user, workspace, target_message_id, command_id)
        if (attempt := self._active_attempts.get(thread)) is not None:
            self._stop_attempts[thread] = attempt
            self.stop_requested.add(thread)
        if not self.hold_stop and attempt is not None:
            for gate in self._attempt_gates.get(attempt, ()):
                if gate is not None:
                    gate.set()
        return True


def _billing() -> dict[str, object]:
    return {
        "model": "mock",
        "usage": {"input_tokens": 128, "output_tokens": 32, "total_tokens": 160},
        "cost": {"input": 0.001, "output": 0.002, "total": 0.003, "currency": "USD"},
    }


def _session_id(thread: str, user: str, workspace: str) -> str:
    return thread if thread and thread != workspace else f"codex-{workspace or user}"
