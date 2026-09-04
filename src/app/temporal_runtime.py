# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from app.models import THREAD_AUTO_ARCHIVE_MINUTES, Event
from app.runner import LifecycleRunner
from app.types import JsonObject  # noqa: TC001 - Temporal resolves wire annotations

TRANSPORT_RETRY_POLICY = RetryPolicy(timedelta(seconds=5), 2, timedelta(seconds=30), 2)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from temporalio.client import Client

    from app.engine import Engine


class _ActivityRuntime:
    engine: Engine | None = None


_activity_runtime = _ActivityRuntime()


def configure_engine(engine: Engine) -> None:
    _activity_runtime.engine = engine


def _engine() -> Engine:
    if _activity_runtime.engine is None:
        raise RuntimeError("Temporal Activities are not configured")  # noqa: TRY003
    return _activity_runtime.engine


@activity.defn(name="wiseman.turn")
async def run_turn(payload: dict) -> dict:
    event = Event.model_validate(payload["event"])
    if _activity_attempt() > 1:
        event.trigger.content = f"{event.trigger.content}\n\n{_retry_prompt()}"
    state = _object_map(payload.get("state"))
    event.seen_ids = [str(item) for item in _sequence(state.get("seen", event.seen_ids))]
    retry_transport = _activity_attempt() < (TRANSPORT_RETRY_POLICY.maximum_attempts or 1)
    result = cast("JsonObject", await _engine().handle(event, state_data=state, retry_transport=retry_transport))  # fmt: skip  # noqa: E501
    error = result.get("error")
    if retry_transport and isinstance(error, str) and any(marker in error.lower() for marker in ("runner returned http 5", "disconnected", "transport error")):  # fmt: skip  # noqa: E501
        raise ApplicationError(str(result["error"]), type="runner_transport")
    return result


@activity.defn(name="wiseman.workspace")
async def provision_workspace(payload: dict) -> dict:
    event = Event.model_validate(payload["event"])
    workspace = event.trigger.thread_id or event.trigger.channel_id
    await cast(LifecycleRunner, _engine().config.runner).acquire(  # noqa: TC006
        event.trigger.author_id, workspace
    )
    return {"workspace": workspace}


@activity.defn(name="wiseman.codex_start")
async def start_codex(payload: dict) -> dict:
    event = Event.model_validate(payload["event"])
    state = _object_map(payload.get("state"))
    workspace = event.trigger.thread_id or event.trigger.channel_id
    thread = await cast(LifecycleRunner, _engine().config.runner).start(  # noqa: TC006
        str(state.get("codex_thread") or ""), event.trigger.author_id, workspace
    )
    state["codex_thread"] = thread
    return {"state": state, "workspace": workspace, "codex_thread": thread}


@activity.defn(name="wiseman.progress")
async def publish_progress(payload: dict) -> dict:
    event = Event.model_validate(payload["event"])
    phase = str(payload.get("phase", "⏳ Working..."))
    await _engine().preflight(event, phase, _object_map(payload.get("state")))
    return {"phase": phase}


@activity.defn(name="wiseman.failure")
async def fail_turn(payload: dict) -> dict:
    event = Event.model_validate(payload["event"])
    return cast("dict", await _engine().fail(event, str(payload.get("error", "unknown failure")), _object_map(payload.get("state"))))  # fmt: skip  # noqa: E501


async def _activity(fn: Callable[[dict], Awaitable[dict]], payload: dict, duration: timedelta) -> dict:  # fmt: skip  # noqa: E501
    return await workflow.execute_activity(fn, payload, start_to_close_timeout=duration, retry_policy=TRANSPORT_RETRY_POLICY)  # fmt: skip  # noqa: E501


@workflow.defn(name="wiseman.turn")
class TurnWorkflow:
    @workflow.run
    async def run(self, payload: dict) -> dict:
        return await _activity(run_turn, payload, timedelta(minutes=10))


@workflow.defn(name="wiseman.thread")
class ThreadWorkflow:
    def __init__(self) -> None:
        self.pending: list[dict] = []
        self.state: JsonObject = {}

    @workflow.signal
    async def submit(self, event: dict) -> None:
        self.pending.append(dict(event))

    @workflow.run
    async def run(self, first: dict) -> dict:
        self.pending.append(_object_map(first["event"]))
        workflow.patched("split-startup-activities")
        child_workflow = workflow.patched("child-turn-workflow")
        while True:
            event = self.pending.pop(0)
            message_id = str(_object_map(event.get("trigger")).get("id", ""))
            if message_id and message_id in _sequence(self.state.get("processed", [])):
                continue
            try:
                if not self.state.get("codex_thread"):
                    await _activity(publish_progress, {"event": event, "state": self.state, "phase": "🛠️ Workspace provisioning..."}, timedelta(seconds=30))  # fmt: skip  # noqa: E501
                    await _activity(provision_workspace, {"event": event, "state": self.state}, timedelta(seconds=30))  # fmt: skip  # noqa: E501
                    await _activity(publish_progress, {"event": event, "state": self.state, "phase": "🤖 Codex starting..."}, timedelta(seconds=30))  # fmt: skip  # noqa: E501
                    started = await _activity(start_codex, {"event": event, "state": self.state}, timedelta(seconds=90))  # fmt: skip  # noqa: E501
                    self.state = _object_map(started.get("state", self.state))
                await _activity(publish_progress, {"event": event, "state": self.state, "phase": "🤖 Codex turn started..."}, timedelta(seconds=30))  # fmt: skip  # noqa: E501
                if child_workflow:
                    self.result = await workflow.execute_child_workflow(
                        TurnWorkflow.run,
                        {"event": event, "state": self.state},
                        id=f"wiseman-turn-{message_id or len(self.pending)}",
                    )
                else:
                    self.result = await _activity(
                        run_turn, {"event": event, "state": self.state}, timedelta(minutes=10)
                    )
            except Exception as exc:  # noqa: BLE001 - activity failures need Discord delivery
                self.result = await _activity(fail_turn, {"event": event, "state": self.state, "error": str(exc)}, timedelta(seconds=30))  # fmt: skip  # noqa: E501
            self.state = _object_map(self.result.get("state", self.state))
            try:
                await workflow.wait_condition(
                    lambda: bool(self.pending),
                    timeout=timedelta(minutes=THREAD_AUTO_ARCHIVE_MINUTES),
                )
            except TimeoutError:
                return self.result


class TemporalRuntime:
    def __init__(self, address: str, queue: str) -> None:
        self.address, self.queue = address, queue
        self.client: object | None = None
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        from temporalio.client import Client  # noqa: PLC0415 - optional runtime dependency

        self.client = await Client.connect(self.address)
        self.worker_task = asyncio.create_task(self._serve())

    async def _serve(self) -> None:
        from temporalio.worker import Worker  # noqa: PLC0415 - optional runtime dependency

        async with Worker(
            cast("Client", self.client),
            task_queue=self.queue,
            workflows=[ThreadWorkflow, TurnWorkflow],
            activities=[provision_workspace, start_codex, publish_progress, fail_turn, run_turn],
        ):
            await asyncio.Event().wait()

    async def submit(self, event: dict) -> None:
        if self.client is None:
            raise RuntimeError("Temporal is not connected")  # noqa: TRY003
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

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)


def _activity_attempt() -> int:
    try:
        return activity.info().attempt
    except RuntimeError:
        return 1


def _retry_prompt() -> str:
    try:
        return (Path(__file__).parents[2] / "contracts" / "codex-disconnect-retry.j2").read_text()
    except OSError:
        return ""


def _object_map(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
