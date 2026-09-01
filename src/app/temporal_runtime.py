# Copyright (c) 2026 Nick van der Merwe
"""Temporal boundary for one durable workflow per Discord thread."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.exceptions import WorkflowAlreadyStartedError


@activity.defn(name="wiseman.turn")
async def run_turn(payload: dict[str, Any]) -> dict[str, Any]:
    from app.main import Event, engine  # noqa: PLC0415 - avoid workflow import cycle

    event = Event.model_validate(payload["event"])
    state = payload.get("state", {})
    event.seen_ids = list(state.get("seen", event.seen_ids))
    return await engine.handle(event, state_data=state)


@workflow.defn(name="wiseman.thread")
class ThreadWorkflow:
    """Serialize turns and retain the Codex thread/context cursor durably."""

    def __init__(self) -> None:
        self.pending: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        self.result: dict[str, Any] = {}

    @workflow.signal
    async def submit(self, event: dict[str, Any]) -> None:
        self.pending.append(event)

    @workflow.run
    async def run(self, first: dict[str, Any]) -> dict[str, Any]:
        self.pending.append(first["event"])
        while True:
            event = self.pending.pop(0)
            self.result = await workflow.execute_activity(
                run_turn,
                {"event": event, "state": self.state},
                start_to_close_timeout=timedelta(minutes=5),
            )
            self.state = self.result.get("state", self.state)
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
        self.client: Any = None
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        from temporalio.client import Client  # noqa: PLC0415 - optional runtime dependency

        self.client = await Client.connect(self.address)
        self.worker_task = asyncio.create_task(self._serve())

    async def _serve(self) -> None:
        from temporalio.worker import Worker  # noqa: PLC0415 - optional runtime dependency

        async with Worker(
            self.client,
            task_queue=self.queue,
            workflows=[ThreadWorkflow],
            activities=[run_turn],
        ):
            await asyncio.Event().wait()

    async def submit(self, event: dict[str, Any]) -> None:
        if self.client is None:
            raise TemporalError
        thread_id = event["trigger"].get("thread_id") or event["trigger"]["channel_id"]
        workflow_id = f"wiseman-{thread_id}"
        try:
            await self.client.start_workflow(
                ThreadWorkflow.run,
                {"event": event, "state": {}},
                id=workflow_id,
                task_queue=self.queue,
            )
        except WorkflowAlreadyStartedError:
            await self.client.get_workflow_handle(workflow_id).signal(ThreadWorkflow.submit, event)

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
