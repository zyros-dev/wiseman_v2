# Copyright (c) 2026 Nick van der Merwe
import asyncio
from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from app.temporal_runtime import ThreadWorkflow, TurnWorkflow


def fake_node(name):
    @activity.defn(name=name)
    async def node(payload: dict) -> dict:
        state = dict(payload.get("state", {}))
        if name == "wiseman.codex_start":
            state["codex_thread"] = "persisted-sdk-id"
        if name == "wiseman.turn":
            state["turn"] = state.get("turn", 0) + 1
            state["processed"] = [*state.get("processed", []), payload["event"]["trigger"]["id"]]
        if name == "wiseman.retire":
            state["closed"] = True
        result = {**payload, "state": state}
        if name == "wiseman.codex_start":
            result["codex_thread"] = state["codex_thread"]
        if name == "wiseman.infer":
            result["output"] = "answer"
        return result

    return node


async def wait_turn(handle, number):
    async with asyncio.timeout(15):
        for _ in range(300):
            if (await handle.query(ThreadWorkflow.session)).get("turn", 0) == number:
                return
            await asyncio.sleep(0.05)
    pytest.fail("workflow did not finish the expected turn")


@pytest.mark.asyncio
async def test_temporal_retains_session_across_archive_duplicates_and_worker_restart():
    activities = [
        fake_node(name)
        for name in (
            "wiseman.workspace",
            "wiseman.codex_start",
            "wiseman.progress",
            "wiseman.turn",
            "wiseman.failure",
            "wiseman.retire",
            "wiseman.context",
            "wiseman.prompt",
            "wiseman.render",
            "wiseman.infer",
            "wiseman.deliver",
            "wiseman.react",
            "wiseman.observe",
        )
    ]
    async with asyncio.timeout(60), await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="session-test",
            workflows=[ThreadWorkflow, TurnWorkflow],
            activities=activities,
            max_cached_workflows=0,
        ):
            event = {"trigger": {"id": "first", "author_id": "alice", "channel_id": "discord-thread"}}
            handle = await env.client.start_workflow(
                ThreadWorkflow.run, {"event": event}, id="wiseman-discord-thread", task_queue="session-test"
            )
            await wait_turn(handle, 1)
            await env.sleep(timedelta(hours=2))
            assert (await handle.describe()).status == WorkflowExecutionStatus.RUNNING
            await handle.signal(ThreadWorkflow.submit, event)
            await env.sleep(timedelta(minutes=1))
            assert (await handle.query(ThreadWorkflow.session))["turn"] == 1
        async with Worker(
            env.client,
            task_queue="session-test",
            workflows=[ThreadWorkflow, TurnWorkflow],
            activities=activities,
            max_cached_workflows=0,
        ):
            assert (await handle.query(ThreadWorkflow.session))["codex_thread"] == "persisted-sdk-id"
            await handle.signal(
                ThreadWorkflow.submit, {"trigger": {"id": "second", "author_id": "bob", "channel_id": "discord-thread"}}
            )
            await wait_turn(handle, 2)
            assert (await handle.query(ThreadWorkflow.session))["owner_id"] == "alice"
            await env.sleep(timedelta(days=3))
            assert (await handle.result())["state"]["closed"] is True
        await Replayer(workflows=[ThreadWorkflow, TurnWorkflow]).replay_workflow(await handle.fetch_history())
