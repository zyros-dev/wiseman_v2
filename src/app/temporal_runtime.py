# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from app.engine import Engine, EngineResult  # noqa: TC001 - Temporal resolves the annotation
from app.models import Event

TRANSPORT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=2,
)
RETRY_MARKERS = ("runner returned http 5", "disconnected", "transport error")

if TYPE_CHECKING:
    from temporalio.client import Client

    from app.runner import LifecycleRunner
    from app.types import JsonObject


class _ActivityRuntime:
    engine: Engine | None = None


_activity_runtime = _ActivityRuntime()


def configure_engine(engine: Engine) -> None:
    _activity_runtime.engine = engine


def _engine() -> Engine:
    if _activity_runtime.engine is None:
        raise RuntimeError("Temporal Activities are not configured")  # noqa: TRY003
    return _activity_runtime.engine


def _retryable_turn_result(result: Mapping[str, object]) -> bool:
    error = result.get("error")
    lowered = error.lower() if isinstance(error, str) else ""
    return any(marker in lowered for marker in RETRY_MARKERS)


@activity.defn(name="wiseman.turn")
async def run_turn(payload: Mapping[str, object]) -> EngineResult:
    event = Event.model_validate(payload["event"])
    if _activity_attempt() > 1:
        event.trigger.content = f"{event.trigger.content}\n\n{_retry_prompt()}"
    state = _json_object(payload.get("state"))
    event.seen_ids = [str(item) for item in _sequence(state.get("seen", event.seen_ids))]
    retry_transport = _activity_attempt() < (TRANSPORT_RETRY_POLICY.maximum_attempts or 1)
    result = await _engine().handle(event, state_data=state, retry_transport=retry_transport)
    if _retryable_turn_result(result) and retry_transport:
        raise ApplicationError(str(result["error"]), type="runner_transport")
    return result


@activity.defn(name="wiseman.workspace")
async def provision_workspace(payload: Mapping[str, object]) -> None:
    event = Event.model_validate(payload["event"])
    workspace = event.trigger.thread_id or event.trigger.channel_id
    await cast("LifecycleRunner", _engine().runner).acquire(event.trigger.author_id, workspace)


@activity.defn(name="wiseman.codex_start")
async def start_codex(payload: Mapping[str, object]) -> dict[str, object]:
    event = Event.model_validate(payload["event"])
    state = _json_object(payload.get("state"))
    workspace = event.trigger.thread_id or event.trigger.channel_id
    thread = await cast("LifecycleRunner", _engine().runner).start(
        str(state.get("codex_thread") or ""), event.trigger.author_id, workspace
    )
    state["codex_thread"] = thread
    return {"state": state, "workspace": workspace, "codex_thread": thread}


@workflow.defn(name="wiseman.turn")
class TurnWorkflow:
    @workflow.run
    async def run(self, payload: Mapping[str, object]) -> EngineResult:
        return await workflow.execute_activity(
            run_turn,
            payload,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=TRANSPORT_RETRY_POLICY,
        )


@workflow.defn(name="wiseman.thread")
class ThreadWorkflow:
    def __init__(self) -> None:
        self.pending: list[dict[str, object]] = []
        self.state: dict[str, object] = {}
        self.result: EngineResult = {}

    @workflow.signal
    async def submit(self, event: Mapping[str, object]) -> None:
        self.pending.append(dict(event))

    @workflow.run
    async def run(self, first: Mapping[str, object]) -> EngineResult:
        self.pending.append(_object_map(first["event"]))
        while True:
            event = self.pending.pop(0)
            message_id = str(_object_map(event.get("trigger")).get("id", ""))
            processed = self.state.get("processed", [])
            if message_id and message_id in _sequence(processed):
                continue
            if not self.state.get("codex_thread"):
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
            self.result = await workflow.execute_child_workflow(
                TurnWorkflow.run,
                {"event": event, "state": self.state},
                id=f"wiseman-turn-{message_id or len(self.pending)}",
            )
            self.state = _object_map(self.result.get("state", self.state))
            try:
                await workflow.wait_condition(
                    lambda: bool(self.pending), timeout=timedelta(hours=2)
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
            activities=[provision_workspace, start_codex, run_turn],
        ):
            await asyncio.Event().wait()

    async def submit(self, event: Mapping[str, object]) -> None:
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
    path = Path(__file__).parents[2] / "contracts" / "codex-disconnect-retry.j2"
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
