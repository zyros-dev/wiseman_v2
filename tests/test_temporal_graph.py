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
        message_id = result["event"]["trigger"]["id"]
        if name == "codex_start":
            state["codex_thread"] = result["codex_thread"] = state.get("codex_thread") or "sdk-session"
        elif name == "infer" and message_id == "qstop":
            await asyncio.sleep(1)
        elif name == "stop":
            result["accepted"] = True
        elif name == "infer":
            result["output"] = f"answer:{message_id}"
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
    names = "workspace codex_start progress context prompt render infer deliver react observe stop retire".split()
    async with (
        await WorkflowEnvironment.start_time_skipping() as env,
        Worker(env.client, task_queue="graph", workflows=[ThreadWorkflow, TurnWorkflow], activities=[_activity(name) for name in names]),
    ):
        first = {"id": "qstop", "author_id": "alice", "channel_id": "t", "thread_id": "t"}
        handle = await env.client.start_workflow(ThreadWorkflow.run, {"event": {"trigger": first}}, id="wiseman-t", task_queue="graph")
        async with asyncio.timeout(15):
            while not (await handle.query(ThreadWorkflow.session)).get("active_message"):
                await asyncio.sleep(0.05)
        child = env.client.get_workflow_handle("wiseman-turn-qstop")
        assert await child.execute_update("stop", {"trigger": {**first, "id": "stop"}}, id="stop")
        await handle.signal(ThreadWorkflow.submit, {"trigger": {**first, "id": "q2", "author_id": "bob"}})
        await _wait_turn(handle, 2)
        state = await handle.query(ThreadWorkflow.session)
        assert (state["owner_id"], state["turn"]) == ("alice", 2)
        assert set(state["processed"]) == {"qstop", "q2"}
        await env.sleep(timedelta(days=3))
        assert (await handle.result())["state"]["closed"] is True
