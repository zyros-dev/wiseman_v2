# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from app.models import DeliveryState, Event, State, TurnWork
from app.runner_status import STOPPED_STATUS, UNKNOWN_STATUS

TRANSPORT_RETRY_POLICY = RetryPolicy(timedelta(seconds=5), 2, timedelta(seconds=30), 2)
DELIVERY_RETRY_POLICY = RetryPolicy(timedelta(seconds=5), 2, timedelta(seconds=30))
HISTORY_COMPACTION_TURNS = 20
PERMANENT_FAILURE_CUTOFF = 503

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from temporalio.client import Client

    from app.engine import Engine
    from app.types import JsonObject, JsonValue


class _ActivityRuntime:
    engine: Engine | None = None


_activity_runtime = _ActivityRuntime()


def configure_engine(engine: Engine) -> None:
    _activity_runtime.engine = engine


def _engine() -> Engine:
    if _activity_runtime.engine is None:
        raise RuntimeError("Temporal Activities are not configured")
    return _activity_runtime.engine


@activity.defn(name="wiseman.workspace")
async def provision_workspace(payload: dict) -> dict:
    event = _event(payload)
    workspace = event.trigger.thread_id or event.trigger.channel_id
    owner = str(_state(payload).get("owner_id") or event.trigger.author_id)
    await _engine().config.runner.acquire(owner, workspace)
    return {"workspace": workspace}


@activity.defn(name="wiseman.codex_start")
async def start_codex(payload: dict) -> dict:
    event, state = _event(payload), _state(payload)
    workspace = event.trigger.thread_id or event.trigger.channel_id
    thread = await _engine().config.runner.start(str(state.get("codex_thread") or ""), str(state.get("owner_id") or event.trigger.author_id), workspace)
    state["codex_thread"] = thread
    return {"state": state, "workspace": workspace, "codex_thread": thread}


@activity.defn(name="wiseman.failure")
async def fail_turn(payload: dict) -> dict:
    event = _event(payload)
    error = str(payload.get("error", "unknown failure"))
    return await _engine().fail(event, error, _state(payload))


@activity.defn(name="wiseman.retire")
async def retire_session(payload: dict) -> dict:
    event, state = _event(payload), _state(payload)
    await _engine().config.runner.release(str(state.get("owner_id") or event.trigger.author_id), event.trigger.thread_id or event.trigger.channel_id)
    return {"state": {**state, "closed": True}}


async def _activity(fn: Callable[[dict], Awaitable[dict]], payload: dict, duration: timedelta) -> dict:
    return await workflow.execute_activity(
        fn,
        payload,
        start_to_close_timeout=duration,
        retry_policy=TRANSPORT_RETRY_POLICY,
    )


@workflow.defn(name="wiseman.turn")
class TurnWorkflow:
    def __init__(self) -> None:
        self.work: TurnWork | None = None
        self.pending_progress: list[str] = []
        self.inferencing = False
        self.stop_requested = False
        self.stop_commands: set[str] = set()
        self.terminal_hold = False
        self.cancellation_hold = False
        self.completion_won = False
        self.recovering = False
        self.outcome_unknown = False
        self.resume_requested = False
        self.resumed = False
        self.retry_exhausted = False
        self.outcome_established = False
        self.cancellation_unknown = False

    @workflow.signal
    def hold_terminal(self) -> None:
        self.terminal_hold = True

    @workflow.signal
    def release_terminal(self) -> None:
        self.terminal_hold = False

    @workflow.signal
    def hold_cancellation(self) -> None:
        self.cancellation_hold = True

    @workflow.signal
    def release_cancellation(self) -> None:
        self.cancellation_hold = False

    @workflow.signal
    def resume_session(self) -> None:
        self.resume_requested = True
        self.resumed = True
        self.recovering = False
        self.inferencing = True

    @workflow.signal
    def exhaust_retries(self) -> None:
        self.retry_exhausted = True

    @workflow.signal
    def establish_outcome(self) -> None:
        self.outcome_established = True
        self.cancellation_unknown = False
        if self.outcome_unknown and self.work is not None:
            self.outcome_unknown = False
            self.work.error = "Execution outcome established without a result"

    @workflow.signal
    def mark_cancellation_unknown(self) -> None:
        self.cancellation_unknown = True
        self.outcome_unknown = True
        self.stop_requested = False
        if self.work is not None:
            self.work.stopped = False
            self.work.error = ""

    @workflow.signal
    def mark_completion_race(self) -> None:
        self.completion_won = True
        self.stop_requested = False
        if self.work is not None:
            self.work.stopped = False
            self.work.error = ""

    @workflow.update
    async def steer(self, event: dict) -> bool:
        incoming = Event.model_validate(event)
        if self.work is None or not self.inferencing or incoming.trigger.reply_to != self.work.state.delivery_id or (incoming.anchor_id and incoming.anchor_id != self.work.event.trigger.id):  # noqa: E501 # fmt: skip
            return False
        if incoming.trigger.id in self.work.state.processed:
            return True
        if await self._control("steer", event):
            self._accept_control(incoming, self.work.state.steering_ids)
            return True
        return False

    @workflow.update
    async def stop(self, event: dict) -> bool:
        incoming = Event.model_validate(event)
        if self.work is None or incoming.trigger.id in self.stop_commands:
            self.stop_requested = True
            self.stop_commands.add(incoming.trigger.id)
            return True
        if incoming.anchor_id and incoming.anchor_id != self.work.event.trigger.id:
            return False
        if not self.inferencing:
            self._accept_stop(incoming)
            return True
        if await self._control("stop", event):
            self._accept_stop(incoming)
            return True
        return False

    def _accept_control(self, incoming: Event, ids: list[str]) -> None:
        assert self.work is not None
        _record_state_message(self.work.state, incoming)
        if incoming.trigger.id not in ids:
            ids.append(incoming.trigger.id)
        self.work.state.processed.add(incoming.trigger.id)

    def _accept_stop(self, incoming: Event) -> None:
        self.stop_requested = True
        self.stop_commands.add(incoming.trigger.id)
        assert self.work is not None
        self._accept_control(incoming, self.work.state.stop_command_ids)

    async def _control(self, name: str, event: dict) -> bool:
        assert self.work is not None
        result = await workflow.execute_activity(
            f"wiseman.{name}",
            {"work": self.work.model_dump(mode="json"), "event": event},
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=1) if name == "stop" else TRANSPORT_RETRY_POLICY,
        )
        return bool(result["accepted"])

    @workflow.signal
    def progress(self, message: str) -> None:
        self.inferencing = True
        self.recovering = False
        key = _progress_key(message)
        self.pending_progress = [item for item in self.pending_progress if _progress_key(item) != key][-31:]
        self.pending_progress.append(message)

    @workflow.query
    def snapshot(self) -> dict:
        return {
            "work": self.work.model_dump(mode="json") if self.work is not None else {},
            "inferencing": self.inferencing,
            "stop_requested": self.stop_requested,
            "resume_requested": self.resume_requested,
            "resumed": self.resumed,
            "pending_progress": list(self.pending_progress),
            "recovering": self.recovering,
            "outcome_unknown": self.outcome_unknown,
        }

    async def _node(self, name: str, *, durable: bool = False) -> None:
        assert self.work is not None
        self.work.state.progress = _merge_progress(self.work.state.progress, self.pending_progress)
        self.pending_progress.clear()
        result = TurnWork.model_validate(
            await workflow.execute_activity(
                f"wiseman.{name}",
                self.work.model_dump(mode="json"),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=DELIVERY_RETRY_POLICY if durable else TRANSPORT_RETRY_POLICY,
            )
        )
        _merge_work_state(result.state, self.work.state)
        self.work = result

    @workflow.run
    async def run(self, payload: dict) -> dict:
        self.work = TurnWork.model_validate(payload)
        self.work.state.delivery = DeliveryState()
        self.work.state.progress = ["🛠️ Workspace provisioning..."] if not self.work.state.codex_thread else []
        try:
            await self._node("render", durable=True)
            await self._node("react")
        except Exception:
            workflow.logger.exception("Processing reaction unavailable; continuing the turn")
        try:
            await self._node("context")
            await self._node("prompt")
            await workflow.wait_condition(lambda: not self.cancellation_hold)
            completion_won = self.completion_won
            self.completion_won = False
            if (not self.stop_requested or completion_won) and not self.work.state.codex_thread:
                await _activity(provision_workspace, self.work.model_dump(mode="json"), timedelta(minutes=5))
                self.progress("🤖 Codex starting...")
                await self._node("render")
                started = await _activity(start_codex, self.work.model_dump(mode="json"), timedelta(minutes=2))
                self.work.state.codex_thread = str(started["codex_thread"])
            if self.stop_requested and not completion_won:
                self.work.stopped = True
                self.work.error = "Turn stopped by user"
            else:
                await self._infer()
            if not self.work.output.strip() and not self.work.stopped and not self.work.error:
                self.work.error = "Codex returned no answer"
        except Exception as exc:
            if self.stop_requested:
                self.work.stopped = True
                self.work.error = "Turn stopped by user"
            else:
                self.work.error = _error_message(exc)
        return await self._finish_turn()

    async def _finish_turn(self) -> dict:
        assert self.work is not None
        self.inferencing = False
        await workflow.wait_condition(workflow.all_handlers_finished)
        await workflow.wait_condition(lambda: not self.cancellation_hold and not self.cancellation_unknown)
        await self._reconcile_completion_race()
        await self._node("deliver", durable=True)
        await self._node("react", durable=True)
        try:
            await self._node("observe")
        except Exception:
            workflow.logger.exception("Terminal telemetry exhausted retries")
        await self._reconcile_established_outcome()
        await workflow.wait_condition(lambda: not self.terminal_hold)
        self.work.state.processed.add(self.work.event.trigger.id)
        self.work.state.turn += 1
        self.work.state.delivery_id = None
        self.work.state.progress = []
        self.recovering = False
        self.outcome_unknown = False
        self.resumed = False
        return {"state": self.work.state.model_dump(mode="json"), "output": self.work.output, "error": self.work.error}

    async def _reconcile_completion_race(self) -> None:
        assert self.work is not None
        if not self.completion_won or self.work.output.strip() or self.work.error:
            return
        self.completion_won = False
        self.work.stopped = False
        self.work.error = ""
        await self._infer()
        self.inferencing = False
        await workflow.wait_condition(workflow.all_handlers_finished)
        if not self.work.output.strip() and not self.work.stopped and not self.work.error:
            self.work.error = "Codex returned no answer"

    async def _reconcile_established_outcome(self) -> None:
        assert self.work is not None
        await workflow.wait_condition(lambda: not self.terminal_hold or self.outcome_established)
        if not self.outcome_established or not self.work.error or self.work.state.delivery.reaction_phase != "success":
            return
        self.work.terminal_emoji = ""
        await self._node("deliver", durable=True)
        await self._node("react", durable=True)
        try:
            await self._node("observe")
        except Exception:
            workflow.logger.exception("Reconciled outcome telemetry exhausted retries")

    async def _infer(self) -> None:
        assert self.work is not None
        while True:
            self.inferencing = True
            self.recovering = False
            pending = workflow.start_activity(
                "wiseman.infer",
                self.work.model_dump(mode="json"),
                start_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=timedelta(seconds=45),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            try:
                while not pending.done():
                    await workflow.wait_condition(lambda pending=pending: pending.done() or bool(self.pending_progress))
                    if self.pending_progress:
                        try:
                            await self._node("render")
                        except Exception:
                            workflow.logger.exception("Progress delivery exhausted retries; inference remains active")
                result = TurnWork.model_validate(await pending)
            except ActivityError as exc:
                status = _activity_status(exc)
                self.inferencing = False
                self.resumed = False
                if await self._handle_infer_failure(status):
                    return
                continue
            self.work.state.codex_thread = result.state.codex_thread
            self.work.output, self.work.billing = result.output, result.billing
            self.resumed = False
            return

    async def _handle_infer_failure(self, status: int | None) -> bool:
        assert self.work is not None
        finished = False
        status = _outcome_failure_status(established=self.outcome_established, status=status)
        if status == STOPPED_STATUS or self.stop_requested:
            self.recovering = False
            self.work.stopped = True
            self.work.error = "Turn stopped by user"
            finished = True
        elif status == UNKNOWN_STATUS:
            self.outcome_unknown = True
            await workflow.wait_condition(lambda: self.outcome_established or self.stop_requested)
            if self.stop_requested and not self.outcome_established:
                self.outcome_unknown = False
                self.work.stopped = True
                self.work.error = "Turn stopped by user"
            else:
                self.outcome_unknown = False
                self.work.error = "Execution outcome established without a result"
            finished = True
        elif status is not None and status < PERMANENT_FAILURE_CUTOFF:
            self.recovering = False
            self.work.error = f"Codex failed with HTTP {status}"
            finished = True
        else:
            self.recovering = True
            await workflow.wait_condition(lambda: self.resume_requested or self.retry_exhausted or self.stop_requested or self.outcome_established)
            if self.outcome_established:
                self.recovering = False
                self.work.stopped = False
                self.work.error = "Execution outcome established without a result"
                finished = True
            elif self.stop_requested:
                self.recovering = False
                self.work.stopped = True
                self.work.error = "Turn stopped by user"
                finished = True
            elif self.retry_exhausted:
                self.recovering = False
                self.work.error = f"Codex failed with HTTP {status}" if status else "Codex failed after retries"
                finished = True
            else:
                self.resume_requested = False
                self.retry_exhausted = False
                self.recovering = False
                self.inferencing = True
        return finished


@workflow.defn(name="wiseman.thread")
class ThreadWorkflow:
    def __init__(self) -> None:
        self.pending: list[dict] = []
        self.state: JsonObject = {}
        self.active_message = self.active_timestamp = ""

    @workflow.signal
    async def submit(self, event: dict) -> None:
        message_id = _object_map(event.get("trigger")).get("id")
        known = [self.active_message, *_sequence(self.state.get("processed")), *_sequence(self.state.get("message_ids"))]
        known.extend(_object_map(item.get("trigger")).get("id") for item in self.pending)
        if message_id and message_id in known:
            return
        if message_id:
            self._record_message(event)
        self.pending.append(dict(event))

    @workflow.signal
    async def touch(self, event: dict) -> None:
        message_id = str(_object_map(event.get("trigger")).get("id", ""))
        if message_id and message_id not in _sequence(self.state.get("message_ids")):
            self._record_message(event)
            self._append_state_id("background_context_ids", message_id)
        self.pending.append({**event, "background": True})

    @workflow.signal
    async def record_control(self, event: dict) -> None:
        message_id = str(_object_map(event.get("trigger")).get("id", ""))
        target = "stop_command_ids" if event.get("kind") == "stop" else "steering_ids"
        if message_id:
            self._record_message(event)
            self._append_state_id(target, message_id)

    def _record_message(self, event: dict) -> None:
        trigger = _object_map(event.get("trigger"))
        message_id = str(trigger.get("id", ""))
        if not message_id:
            return
        self._append_state_id("message_ids", message_id)
        timestamps = self.state.setdefault("message_timestamps", {})
        if isinstance(timestamps, dict):
            timestamps[message_id] = str(trigger.get("timestamp", ""))

    def _append_state_id(self, name: str, message_id: str) -> None:
        values = self.state.setdefault(name, [])
        if isinstance(values, list) and message_id not in values:
            values.append(message_id)

    @workflow.query
    def session(self) -> dict:
        pending = tuple(str(_object_map(item.get("trigger")).get("id", "")) for item in self.pending if not _object_map(item).get("background"))
        return {
            **self.state,
            "active_message": self.active_message,
            "active_timestamp": self.active_timestamp,
            "pending_message_ids": pending,
        }

    @workflow.run
    async def run(self, first: dict) -> dict:
        workflow.patched("split-startup-activities")
        self.state = _object_map(first.get("state"))
        self.pending.extend(_object_map(item) for item in _sequence(first.get("pending")))
        if event := _object_map(first.get("event")):
            message_id = str(_object_map(event.get("trigger")).get("id", ""))
            if message_id:
                self._record_message(event)
            self.pending.insert(0, event)
        self.result, handled = {"state": self.state}, 0
        while True:
            try:
                await workflow.wait_condition(
                    lambda: bool(self.pending),
                    timeout=timedelta(days=3),
                )
            except TimeoutError:
                await self._retire(event)
                return self.result
            turn = self.state.get("turn")
            if handled and isinstance(turn, int) and turn % HISTORY_COMPACTION_TURNS == 0:
                workflow.continue_as_new({"state": self.state, "pending": self.pending})
            event = self.pending.pop(0)
            if _object_map(event).get("background"):
                continue
            self.state["consumed_context_ids"] = [
                *_string_sequence(self.state.get("consumed_context_ids")),
                *_string_sequence(self.state.get("background_context_ids")),
            ]
            self.state["background_context_ids"] = []
            message_id = str(_object_map(event.get("trigger")).get("id", ""))
            if message_id and message_id in _sequence(self.state.get("processed", [])):
                continue
            self.active_message, self.active_timestamp = message_id, str(_object_map(event.get("trigger")).get("timestamp", ""))
            self.state.setdefault("owner_id", _object_map(event.get("trigger")).get("author_id", ""))
            try:
                self.result = await self._turn(event)
            except Exception as exc:
                self.result = await _activity(
                    fail_turn,
                    {"event": event, "state": self.state, "error": str(exc)},
                    timedelta(seconds=30),
                )
            self.state = _merge_thread_state(self.state, _object_map(self.result.get("state", self.state)))
            self.result["state"] = self.state
            self.active_message = self.active_timestamp = ""
            handled += 1

    async def _turn(self, event: dict) -> dict:
        return await workflow.execute_child_workflow(
            TurnWorkflow.run,
            {"event": event, "state": self.state},
            id=f"wiseman-turn-{self.active_message}",
        )

    async def _retire(self, event: dict) -> None:
        await _activity(retire_session, {"event": event, "state": self.state}, timedelta(minutes=5))
        self.state["closed"] = True
        self.result["state"] = self.state
        if self.pending:
            workflow.continue_as_new({"state": {"owner_id": self.state.get("owner_id", "")}, "pending": self.pending})


class TemporalRuntime:
    def __init__(self, address: str | Client, queue: str) -> None:
        self.address = address if isinstance(address, str) else ""
        self.queue = queue
        self.client: Client | None = None if isinstance(address, str) else address
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        from temporalio.client import Client
        from temporalio.contrib.opentelemetry import TracingInterceptor

        if self.client is None:
            self.client = await Client.connect(self.address, interceptors=[TracingInterceptor(always_create_workflow_spans=True)])
        self.worker_task = asyncio.create_task(self._serve())

    async def _serve(self) -> None:
        from temporalio.worker import Worker

        from app.nodes import TurnActivities

        async with Worker(
            cast("Client", self.client),
            task_queue=self.queue,
            workflows=[ThreadWorkflow, TurnWorkflow],
            activities=[
                provision_workspace,
                start_codex,
                fail_turn,
                retire_session,
                *TurnActivities(_engine(), cast("Client", self.client)).registered(),
            ],
        ):
            await asyncio.Event().wait()

    async def submit(self, event: dict) -> dict[str, object] | None:
        if self.client is None:
            raise RuntimeError("Temporal is not connected")
        trigger = _object_map(event["trigger"])
        thread_id = trigger.get("thread_id") or trigger["channel_id"]
        workflow_id = f"wiseman-{thread_id}"
        client = cast("Client", self.client)
        try:
            await client.start_workflow(
                ThreadWorkflow.run,
                {"event": dict(event), "state": {}},
                id=workflow_id,
                task_queue=self.queue,
            )
        except WorkflowAlreadyStartedError:
            handle = client.get_workflow_handle(workflow_id)
            message_id = str(trigger.get("id", ""))
            session = cast("JsonObject", await handle.query(ThreadWorkflow.session))
            if _known_message(session, message_id):
                return {"status": "duplicate", "message_id": message_id}
            await handle.signal(ThreadWorkflow.submit, event)
        return None

    async def touch(self, event: Event) -> None:
        if self.client is None:
            raise RuntimeError("Temporal is not connected")
        if not (thread_id := event.trigger.thread_id):
            return
        try:
            await cast("Client", self.client).get_workflow_handle(f"wiseman-{thread_id}").signal(ThreadWorkflow.touch, event.model_dump(mode="json"))
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise

    async def _update(self, name: str, event: Event) -> bool:
        if self.client is None:
            raise RuntimeError("Temporal is not connected")
        client = cast("Client", self.client)
        thread_id = event.trigger.thread_id or event.trigger.channel_id
        try:
            state = await client.get_workflow_handle(f"wiseman-{thread_id}").query(ThreadWorkflow.session)
            stale = bool(event.trigger.timestamp and event.trigger.timestamp < str(state.get("active_timestamp", "")))
            if not (message_id := state.get("active_message")) or stale:
                return False
            accepted = bool(
                await client.get_workflow_handle(f"wiseman-turn-{message_id}").execute_update(
                    name,
                    event.model_dump(mode="json") | {"anchor_id": str(message_id)},
                    id=event.trigger.id,
                )
            )
            if accepted:
                await client.get_workflow_handle(f"wiseman-{thread_id}").signal(ThreadWorkflow.record_control, event.model_dump(mode="json"))
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return False
            raise
        else:
            return accepted

    async def steer(self, event: Event) -> bool: return await self._update("steer", event)  # fmt: skip

    async def stop(self, event: Event) -> bool: return await self._update("stop", event)  # fmt: skip

    async def snapshot(self, thread_id: str) -> JsonObject:
        if self.client is None:
            raise RuntimeError("Temporal is not connected")
        client = cast("Client", self.client)
        session = await client.get_workflow_handle(f"wiseman-{thread_id}").query(ThreadWorkflow.session)
        snapshot: JsonObject = cast("JsonObject", session)
        if message_id := snapshot.get("active_message"):
            child = await client.get_workflow_handle(f"wiseman-turn-{message_id}").query(TurnWorkflow.snapshot)
            snapshot["active_turn_snapshot"] = cast("JsonObject", child)
        return snapshot

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)


def _object_map(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _event(payload: dict) -> Event:
    return Event.model_validate(payload["event"])


def _state(payload: dict) -> JsonObject:
    return _object_map(payload.get("state"))


def _record_state_message(state: State, event: Event) -> None:
    message_id, timestamp = event.trigger.id, event.trigger.timestamp
    if message_id not in state.message_ids:
        state.message_ids.append(message_id)
    if timestamp:
        state.message_timestamps[message_id] = timestamp


def _progress_key(message: str) -> str:
    return message.split("...", 1)[0].split('"', 1)[0].strip()


def _merge_progress(existing: list[str], updates: list[str]) -> list[str]:
    merged = list(existing)
    for message in updates:
        key = _progress_key(message)
        match = next((index for index, item in enumerate(merged) if _progress_key(item) == key), None)
        if match is None:
            merged.append(message)
        else:
            merged[match] = message
    return merged[-32:]


def _merge_ids(existing: list[str], updates: list[str]) -> list[str]:
    merged = list(existing)
    for value in updates:
        if value not in merged:
            merged.append(value)
    return merged


def _merge_work_state(result: State, previous: State) -> None:
    result.processed.update(previous.processed)
    for result_values, previous_values in (
        (result.message_ids, previous.message_ids),
        (result.background_context_ids, previous.background_context_ids),
        (result.consumed_context_ids, previous.consumed_context_ids),
        (result.steering_ids, previous.steering_ids),
        (result.stop_command_ids, previous.stop_command_ids),
    ):
        result_values[:] = _merge_ids(result_values, previous_values)
    result.message_timestamps.update(previous.message_timestamps)


def _merge_thread_state(previous: JsonObject, result: JsonObject) -> JsonObject:
    merged = dict(result)
    for field_name in ("message_ids", "background_context_ids", "consumed_context_ids", "steering_ids", "stop_command_ids", "processed"):
        merged[field_name] = cast("JsonValue", _merge_ids(_string_sequence(merged.get(field_name)), _string_sequence(previous.get(field_name))))
    timestamps = _object_map(previous.get("message_timestamps"))
    timestamps.update(_object_map(merged.get("message_timestamps")))
    merged["message_timestamps"] = cast("JsonValue", timestamps)
    return merged


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _string_sequence(value: object) -> list[str]:
    return [item for item in _sequence(value) if isinstance(item, str)]


def _known_message(session: JsonObject, message_id: str) -> bool:
    if not message_id:
        return False
    known = {
        str(session.get("active_message", "")),
        *_string_sequence(session.get("processed")),
        *_string_sequence(session.get("message_ids")),
        *_string_sequence(session.get("pending_message_ids")),
    }
    return message_id in known


def _error_message(error: BaseException) -> str:
    cause = getattr(error, "cause", None) or error.__cause__
    return str(cause or error).strip() or type(error).__name__


def _activity_status(error: ActivityError) -> int | None:
    cause = error.cause
    if isinstance(cause, ApplicationError):
        error_type = cause.type or ""
        if error_type.startswith("runner:"):
            try:
                return int(error_type.removeprefix("runner:"))
            except ValueError:
                return None
    message = str(cause or error)
    marker = "HTTP "
    if marker in message:
        try:
            return int(message.split(marker, 1)[1].split(None, 1)[0])
        except (IndexError, ValueError):
            return None
    return None


def _outcome_failure_status(*, established: bool, status: int | None) -> int | None:
    return UNKNOWN_STATUS if established else status
