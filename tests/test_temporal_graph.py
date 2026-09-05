# Copyright (c) 2026 Nick van der Merwe
import asyncio
from datetime import timedelta

from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.temporal_runtime import ThreadWorkflow, TurnWorkflow


def _activity(name: str):
    @activity.defn(name=f"wiseman.{name}")
    async def run(payload: dict) -> dict:
        result = dict(payload)
        state = dict(result.get("state", {}))
        if name == "codex_start":
            state["codex_thread"] = state.get("codex_thread") or "sdk-session"
            result["codex_thread"] = state["codex_thread"]
        elif name == "infer" and result["event"]["trigger"]["id"] == "q2":
            raise RuntimeError("inference failed")
        elif name == "infer":
            result["output"] = f"answer:{result['event']['trigger']['id']}"
        elif name == "retire":
            state["closed"] = True
        result["state"] = state
        return result

    return run


async def _wait_turn(handle, expected: int) -> None:
    async with asyncio.timeout(15):
        while (await handle.query(ThreadWorkflow.session)).get("turn") != expected:
            await asyncio.sleep(0.05)


async def test_temporal_graph_keeps_session_order_and_retires() -> None:
    names = "workspace codex_start progress context prompt render infer deliver react observe retire".split()
    activities = [_activity(name) for name in names]
    async with (
        await WorkflowEnvironment.start_time_skipping() as env,
        Worker(env.client, task_queue="graph", workflows=[ThreadWorkflow, TurnWorkflow], activities=activities),
    ):
        first = {"id": "q1", "author_id": "alice", "channel_id": "t", "thread_id": "t"}
        handle = await env.client.start_workflow(ThreadWorkflow.run, {"event": {"trigger": first}}, id="wiseman-t", task_queue="graph")
        await _wait_turn(handle, 1)
        await handle.signal(ThreadWorkflow.submit, {"trigger": {**first, "id": "q1"}})
        await handle.signal(ThreadWorkflow.submit, {"trigger": {**first, "id": "q2", "author_id": "bob"}})
        await _wait_turn(handle, 2)
        state = await handle.query(ThreadWorkflow.session)
        assert (state["owner_id"], state["turn"]) == ("alice", 2)
        assert state["codex_thread"] == "sdk-session"
        assert set(state["processed"]) == {"q1", "q2"}
        await env.sleep(timedelta(days=3))
        assert (await handle.result())["state"]["closed"] is True
