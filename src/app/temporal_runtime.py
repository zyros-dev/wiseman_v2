# Copyright (c) 2026 Nick van der Merwe
"""Temporal boundary for one durable workflow per Discord thread."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from app.engine import EngineResult  # noqa: TC001 - Temporal resolves the TypedDict annotation
from app.models import Event

TRANSPORT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=2,
)

if TYPE_CHECKING:
    from temporalio.client import Client

    from app.runner import LifecycleRunner


if TYPE_CHECKING:
    from app.types import JsonObject


def _retryable_turn_result(result: Mapping[str, object]) -> bool:
    error = result.get("error")
    return isinstance(error, str) and (
        error.startswith("runner returned HTTP 5")
        or "disconnected" in error.lower()
        or "transport error" in error.lower()
    )


@activity.defn(name="wiseman.turn")
async def run_turn(payload: Mapping[str, object]) -> EngineResult:
    from app.main import engine  # noqa: PLC0415 - entrypoint dependency

    event = Event.model_validate(payload["event"])
    if _activity_attempt() > 1:
        event.trigger.content = f"{event.trigger.content}\n\n{_retry_prompt()}"
    state = _json_object(payload.get("state"))
    event.seen_ids = [str(item) for item in _sequence(state.get("seen", event.seen_ids))]
    result = await engine.handle(event, state_data=state)
    if _retryable_turn_result(result):
        raise ApplicationError(str(result["error"]), type="runner_transport")
    return result


@activity.defn(name="wiseman.workspace")
async def provision_workspace(payload: Mapping[str, object]) -> dict[str, str]:
    """Materialize the warm runner workspace as its own observable Activity."""
    from app.main import engine  # noqa: PLC0415 - entrypoint dependency

    event = Event.model_validate(payload["event"])
    workspace = event.trigger.thread_id or event.trigger.channel_id
    await cast("LifecycleRunner", engine.runner).acquire(event.trigger.author_id, workspace)
    return {"workspace": workspace}


@activity.defn(name="wiseman.codex_start")
async def start_codex(payload: Mapping[str, object]) -> dict[str, object]:
    """Create or resume the Codex SDK thread before the model turn Activity."""
    from app.main import engine  # noqa: PLC0415 - entrypoint dependency

    event = Event.model_validate(payload["event"])
    state = _json_object(payload.get("state"))
    workspace = event.trigger.thread_id or event.trigger.channel_id
    thread = await cast("LifecycleRunner", engine.runner).start(
        str(state.get("codex_thread") or ""), event.trigger.author_id, workspace
    )
    state["codex_thread"] = thread
    return {"state": state, "workspace": workspace, "codex_thread": thread}


@workflow.defn(name="wiseman.thread")
class ThreadWorkflow:
    """Serialize turns and retain the Codex thread/context cursor durably."""

    def __init__(self) -> None:
        self.pending: list[dict[str, object]] = []
        self.state: dict[str, object] = {}
        self.result: EngineResult = {}
        self.started = False

    @workflow.signal
    async def submit(self, event: Mapping[str, object]) -> None:
        self.pending.append(dict(event))

    @workflow.run
    async def run(self, first: Mapping[str, object]) -> EngineResult:
        self.pending.append(_object_map(first["event"]))
        while True:
            event = self.pending.pop(0)
            if not self.started:
                await workflow.execute_activity(
                    provision_workspace,
                    {"event": event, "state": self.state},
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=TRANSPORT_RETRY_POLICY,
                )
                started = await workflow.execute_activity(
                    start_codex,
                    {"event": event, "state": self.state},
                    start_to_close_timeout=timedelta(seconds=90),
                    retry_policy=TRANSPORT_RETRY_POLICY,
                )
                self.state = _object_map(started.get("state", self.state))
                self.started = True
            self.result = await workflow.execute_activity(
                run_turn,
                {"event": event, "state": self.state},
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=TRANSPORT_RETRY_POLICY,
            )
            self.state = _object_map(self.result.get("state", self.state))
            try:
                await workflow.wait_condition(
                    lambda: bool(self.pending), timeout=timedelta(hours=2)
                )
            except TimeoutError:
                return self.result


class TemporalError(RuntimeError):
    """Raised when a workflow is submitted before the client is connected."""

    def __init__(self) -> None:
        super().__init__("Temporal is not connected")


class TemporalRuntime:
    """Start the worker and route each event to its thread workflow."""

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
            workflows=[ThreadWorkflow],
            activities=[provision_workspace, start_codex, run_turn],
        ):
            await asyncio.Event().wait()

    async def submit(self, event: Mapping[str, object]) -> None:
        if self.client is None:
            raise TemporalError
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
    path = Path(__file__).parents[2] / "contracts" / "codex-disconnect-retry.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _json_object(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _object_map(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
