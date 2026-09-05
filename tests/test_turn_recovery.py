# Copyright (c) 2026 Nick van der Merwe
import asyncio
from collections import Counter

from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from app.temporal_runtime import TurnWorkflow
from tests.test_temporal_server import fake_node


async def test_delivery_retry_and_cold_worker_do_not_repeat_inference():
    calls = Counter()
    first_delivery = asyncio.Event()
    async with asyncio.timeout(60), await WorkflowEnvironment.start_time_skipping() as env:

        @activity.defn(name="wiseman.infer")
        async def infer(payload: dict) -> dict:
            calls["infer"] += 1
            workflow_id = activity.info().workflow_id
            assert workflow_id is not None
            handle = env.client.get_workflow_handle(workflow_id)
            await handle.signal(TurnWorkflow.progress, "Gurt 1: writing file")
            await handle.signal(TurnWorkflow.progress, "Gurt 2: testing file")
            return {**payload, "output": "verified output"}

        @activity.defn(name="wiseman.deliver")
        async def deliver(payload: dict) -> dict:
            calls["deliver"] += 1
            assert payload["output"] == "verified output"
            assert payload["state"]["delivery_id"] == "recorded-answer"
            if calls["deliver"] <= 3:
                first_delivery.set()
                raise RuntimeError("Discord disconnected after inference completed")
            return payload

        activities = [
            infer,
            deliver,
            *(
                fake_node(f"wiseman.{name}")
                for name in (
                    "context",
                    "prompt",
                    "render",
                    "react",
                    "observe",
                )
            ),
        ]
        options = {
            "task_queue": "turn-recovery",
            "workflows": [TurnWorkflow],
            "activities": activities,
            "max_cached_workflows": 0,
        }
        async with Worker(env.client, **options):
            handle = await env.client.start_workflow(
                TurnWorkflow.run,
                {
                    "event": {"trigger": {"id": "m", "author_id": "bob", "channel_id": "t"}},
                    "state": {"owner_id": "alice", "codex_thread": "sdk", "delivery_id": "recorded-answer"},
                },
                id="durable-turn",
                task_queue="turn-recovery",
            )
            await asyncio.wait_for(first_delivery.wait(), 15)
        async with Worker(env.client, **options):
            result = await handle.result()
            assert result["state"]["owner_id"] == "alice"
            assert result["state"]["codex_thread"] == "sdk"
            assert result["state"]["turn"] == 1
            assert result["output"] == "verified output"
        assert calls == {"infer": 1, "deliver": 4}
        history = await handle.fetch_history()
        names = [
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
        ]
        assert names.index("wiseman.context") < names.index("wiseman.prompt") < names.index("wiseman.infer")
        assert names.index("wiseman.infer") < names.index("wiseman.deliver") < names.index("wiseman.observe")
        await Replayer(workflows=[TurnWorkflow]).replay_workflow(history)


async def test_steering_is_a_durable_update_with_original_owner_and_no_second_inference():
    running, finish = asyncio.Event(), asyncio.Event()
    rendering, release_render = asyncio.Event(), asyncio.Event()
    steers = []
    async with asyncio.timeout(60), await WorkflowEnvironment.start_time_skipping() as env:

        @activity.defn(name="wiseman.infer")
        async def infer(payload: dict) -> dict:
            running.set()
            workflow_id = activity.info().workflow_id
            assert workflow_id is not None
            await env.client.get_workflow_handle(workflow_id).signal(TurnWorkflow.progress, "tool progress")
            await finish.wait()
            return {**payload, "output": "steered answer"}

        @activity.defn(name="wiseman.render")
        async def render(payload: dict) -> dict:
            if running.is_set():
                rendering.set()
                await release_render.wait()
            return payload

        @activity.defn(name="wiseman.steer")
        async def steer(payload: dict) -> dict:
            steers.append(payload)
            return {"accepted": True}

        activities = [
            infer,
            steer,
            render,
            *(
                fake_node(f"wiseman.{name}")
                for name in (
                    "context",
                    "prompt",
                    "react",
                    "observe",
                    "deliver",
                )
            ),
        ]
        async with Worker(
            env.client, task_queue="steering", workflows=[TurnWorkflow], activities=activities, max_cached_workflows=0
        ):
            handle = await env.client.start_workflow(
                TurnWorkflow.run,
                {
                    "event": {"trigger": {"id": "initial", "author_id": "alice", "channel_id": "t"}},
                    "state": {"owner_id": "alice", "codex_thread": "sdk", "delivery_id": "answer"},
                },
                id="steering-turn",
                task_queue="steering",
            )
            await asyncio.wait_for(running.wait(), 15)
            await asyncio.wait_for(rendering.wait(), 15)
            event = {
                "trigger": {
                    "id": "reply",
                    "author_id": "bob",
                    "channel_id": "t",
                    "reply_to": "answer",
                    "content": "change direction",
                }
            }
            assert await handle.execute_update("steer", event, id="reply") is True
            assert await handle.execute_update("steer", event, id="reply") is True
            wrong = {"trigger": {**event["trigger"], "id": "unrelated", "reply_to": "old-answer"}}
            assert await handle.execute_update("steer", wrong, id="unrelated") is False
            assert len(steers) == 1
            assert steers[0]["work"]["state"]["owner_id"] == "alice"
            assert steers[0]["event"]["trigger"]["author_id"] == "bob"
            release_render.set()
            finish.set()
            result = await handle.result()
            assert set(result["state"]["processed"]) == {"initial", "reply"}
            assert result["state"]["turn"] == 1
        await Replayer(workflows=[TurnWorkflow]).replay_workflow(await handle.fetch_history())
