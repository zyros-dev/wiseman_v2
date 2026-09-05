# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from app.models import Event, TurnWork

TRANSPORT_RETRY_POLICY = RetryPolicy(timedelta(seconds=5), 2, timedelta(seconds=30), 2)
DELIVERY_RETRY_POLICY = RetryPolicy(timedelta(seconds=5), 2, timedelta(seconds=30))
HISTORY_COMPACTION_TURNS = 20

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from temporalio.client import Client

    from app.engine import Engine
    from app.types import JsonObject


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

    @workflow.update
    async def steer(self, event: dict) -> bool:
        incoming = Event.model_validate(event)
        if self.work is None or not self.inferencing or incoming.trigger.reply_to != self.work.state.delivery_id:
            return False
        if incoming.trigger.id in self.work.state.processed:
            return True
        if await self._control("steer", event):
            self.work.state.processed.add(incoming.trigger.id)
            return True
        return False

    @workflow.update
    async def stop(self, event: dict) -> bool:
        incoming = Event.model_validate(event)
        if self.work is None or incoming.trigger.id in self.stop_commands:
            return True
        if not self.inferencing:
            self.stop_requested = True
            self.stop_commands.add(incoming.trigger.id)
            return True
        if await self._control("stop", event):
            self.stop_requested = True
            self.stop_commands.add(incoming.trigger.id)
            return True
        return False

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
        key = _progress_key(message)
        self.pending_progress = [item for item in self.pending_progress if _progress_key(item) != key][-31:]
        self.pending_progress.append(message)

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
        result.state.processed.update(self.work.state.processed)
        self.work = result

    @workflow.run
    async def run(self, payload: dict) -> dict:
        self.work = TurnWork.model_validate(payload)
        self.work.state.progress = ["🛠️ Workspace provisioning..."] if not self.work.state.codex_thread else []
        try:
            await self._node("render", durable=True)
            await self._node("react")
        except Exception:
            workflow.logger.exception("Processing reaction unavailable; continuing the turn")
        try:
            await self._node("context")
            await self._node("prompt")
            if not self.stop_requested and not self.work.state.codex_thread:
                await _activity(provision_workspace, self.work.model_dump(mode="json"), timedelta(minutes=5))
                self.progress("🤖 Codex starting...")
                await self._node("render")
                started = await _activity(start_codex, self.work.model_dump(mode="json"), timedelta(minutes=2))
                self.work.state.codex_thread = str(started["codex_thread"])
            if self.stop_requested:
                self.work.stopped = True
                self.work.error = "Turn stopped by user"
            else:
                await self._infer()
            if not self.work.output.strip() and not self.work.stopped:
                self.work.error = "Codex returned no answer"
        except Exception as exc:
            if self.stop_requested:
                self.work.stopped = True
                self.work.error = "Turn stopped by user"
            else:
                self.work.error = _error_message(exc)
        self.inferencing = False
        await workflow.wait_condition(workflow.all_handlers_finished)
        await self._node("deliver", durable=True)
        await self._node("react", durable=True)
        try:
            await self._node("observe")
        except Exception:
            workflow.logger.exception("Terminal telemetry exhausted retries")
        self.work.state.processed.add(self.work.event.trigger.id)
        self.work.state.turn += 1
        self.work.state.delivery_id = None
        self.work.state.progress = []
        return {"state": self.work.state.model_dump(mode="json"), "output": self.work.output, "error": self.work.error}

    async def _infer(self) -> None:
        assert self.work is not None
        self.inferencing = True
        pending = workflow.start_activity(
            "wiseman.infer",
            self.work.model_dump(mode="json"),
            start_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(seconds=45),
            retry_policy=TRANSPORT_RETRY_POLICY,
        )
        while not pending.done():
            await workflow.wait_condition(lambda: pending.done() or bool(self.pending_progress))
            if self.pending_progress:
                try:
                    await self._node("render")
                except Exception:
                    workflow.logger.exception("Progress delivery exhausted retries; inference remains active")
        result = TurnWork.model_validate(await pending)
        self.work.state.codex_thread = result.state.codex_thread
        self.work.output, self.work.billing = result.output, result.billing


@workflow.defn(name="wiseman.thread")
class ThreadWorkflow:
    def __init__(self) -> None:
        self.pending: list[dict] = []
        self.state: JsonObject = {}
        self.active_message = ""

    @workflow.signal
    async def submit(self, event: dict) -> None:
        message_id = _object_map(event.get("trigger")).get("id")
        known = [self.active_message, *_sequence(self.state.get("processed"))]
        known.extend(_object_map(item.get("trigger")).get("id") for item in self.pending)
        if message_id and message_id in known:
            return
        self.pending.append(dict(event))

    @workflow.signal
    async def touch(self, event: dict) -> None:
        self.pending.append({"background": True, "event": event})

    @workflow.query
    def session(self) -> dict:
        return {**self.state, "active_message": self.active_message}

    @workflow.run
    async def run(self, first: dict) -> dict:
        workflow.patched("split-startup-activities")
        self.state = _object_map(first.get("state"))
        self.pending.extend(_object_map(item) for item in _sequence(first.get("pending")))
        if event := _object_map(first.get("event")):
            self.pending.insert(0, event)
        self.result = {"state": self.state}
        handled = 0
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
            message_id = str(_object_map(event.get("trigger")).get("id", ""))
            if message_id and message_id in _sequence(self.state.get("processed", [])):
                continue
            self.active_message = message_id
            self.state.setdefault("owner_id", _object_map(event.get("trigger")).get("author_id", ""))
            try:
                self.result = await self._turn(event)
            except Exception as exc:
                self.result = await _activity(
                    fail_turn,
                    {"event": event, "state": self.state, "error": str(exc)},
                    timedelta(seconds=30),
                )
            self.state = _object_map(self.result.get("state", self.state))
            self.active_message = ""
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
    def __init__(self, address: str, queue: str) -> None:
        self.address, self.queue = address, queue
        self.client: object | None = None
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        from temporalio.client import Client
        from temporalio.contrib.opentelemetry import TracingInterceptor

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

    async def submit(self, event: dict) -> None:
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
            await client.get_workflow_handle(workflow_id).signal(ThreadWorkflow.submit, event)

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
            if not (message_id := state.get("active_message")):
                return False
            return bool(
                await client.get_workflow_handle(f"wiseman-turn-{message_id}").execute_update(
                    name,
                    event.model_dump(mode="json"),
                    id=event.trigger.id,
                )
            )
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return False
            raise

    async def steer(self, event: Event) -> bool:
        return await self._update("steer", event)

    async def stop(self, event: Event) -> bool:
        return await self._update("stop", event)

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


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _error_message(error: BaseException) -> str:
    cause = getattr(error, "cause", None) or error.__cause__
    return str(cause or error).strip() or type(error).__name__
