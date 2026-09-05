# Copyright (c) 2026 Nick van der Merwe
import asyncio
import json
import os
from pathlib import Path
from typing import cast

import httpx
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.clients.mock_clients import MockRunner, mock_container
from app.engine import Engine, EngineConfig
from app.http_api import create_app
from app.nodes import TurnActivities
from app.temporal_runtime import ThreadWorkflow, TurnWorkflow, fail_turn, provision_workspace, retire_session, start_codex
from tests.graph_model import EDGES, ModelState, Vertex

ACTION_EDGES = set("background-chatter duplicate-question idle-stop steer-active-turn repeat-steer stop-active-turn".split()) | set(
    "stop-active-recovery stop-active-delivery duplicate-stop admit-question queue-question".split()
)


def _message(
    message_id: str,
    content: str,
    *,
    thread: str | None = None,
    reply_to: str | None = None,
    mention: bool = True,
) -> dict[str, object]:
    return {
        "id": message_id,
        "author": {"id": "human", "username": "human"},
        "content": content,
        "channel_id": "home",
        "thread_id": thread,
        "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z",
        "mentions": [{"id": "bot"}] if mention else [],
        "message_reference": {"message_id": reply_to} if reply_to else {},
    }


async def _action(client: httpx.AsyncClient, edge: str, message_id: str) -> httpx.Response:
    if edge in {"background-chatter", "duplicate-question", "idle-stop"}:
        return await client.post("/v1/discord/events", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "background", mention=False)})
    if edge in {"steer-active-turn", "repeat-steer"}:
        return await client.post(
            "/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "steer", thread="thread", reply_to="answer"), "kind": "steer"}
        )
    if edge in {"stop-active-turn", "stop-active-recovery", "stop-active-delivery", "duplicate-stop"}:
        return await client.post("/v1/discord/events", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread="thread"), "kind": "stop"})
    thread = None if edge == "admit-question" else "thread"
    kind = "startup" if edge == "admit-question" else "followup"
    return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "question", thread=thread), "kind": kind})


class _TemporalBoundary:
    def __init__(self, client) -> None:
        self.client = client

    async def submit(self, event: dict[str, object]) -> dict[str, object]:
        trigger = cast("dict[str, object]", event["trigger"])
        workflow_id = f"wiseman-{trigger.get('thread_id') or trigger['channel_id']}"
        try:
            await self.client.start_workflow(ThreadWorkflow.run, {"event": event, "state": {}}, id=workflow_id, task_queue="graph")
        except WorkflowAlreadyStartedError:
            await self.client.get_workflow_handle(workflow_id).signal(ThreadWorkflow.submit, event)
        return {"status": "queued", "message_id": str(trigger["id"])}

    async def stop(self, event) -> bool:
        thread_id = event.trigger.thread_id or event.trigger.channel_id
        state = await self.client.get_workflow_handle(f"wiseman-{thread_id}").query(ThreadWorkflow.session)
        message_id = str(state.get("active_message", ""))
        if not message_id:
            return False
        return bool(
            await self.client.get_workflow_handle(f"wiseman-turn-{message_id}").execute_update("stop", event.model_dump(mode="json"), id=event.trigger.id)
        )

    async def touch(self, event) -> None:
        thread_id = event.trigger.thread_id
        if thread_id:
            await self.client.get_workflow_handle(f"wiseman-{thread_id}").signal(ThreadWorkflow.touch, event.model_dump(mode="json"))


async def _child_result(client, message_id: str) -> dict[str, object]:
    for _ in range(100):
        try:
            return await client.get_workflow_handle(f"wiseman-turn-{message_id}").result()
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
            await asyncio.sleep(0.05)
    raise AssertionError("turn did not start")


async def test_graph_boundary_runs_production_temporal(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    clients = mock_container()
    runner = cast("MockRunner", clients.runner)
    async with await WorkflowEnvironment.start_time_skipping() as env:
        setattr(clients, "temporal", _TemporalBoundary(env.client))
        activities = TurnActivities(Engine(EngineConfig(clients.phoenix, clients.runner, clients.prompts, discord=clients.discord)), env.client)
        async with Worker(
            env.client,
            task_queue="graph",
            workflows=[ThreadWorkflow, TurnWorkflow],
            activities=[provision_workspace, start_codex, fail_turn, retire_session, *activities.registered()],
        ):
            app = create_app(clients=clients)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
                for message_id, content, thread, messages in (
                    ("q1", "one", "thread", []),
                    ("q2", "two", "thread", [_message("chat", "background", mention=False)]),
                ):
                    if message_id == "q2":
                        background = await client.post(
                            "/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("chat", "background", thread="thread", mention=False)}
                        )
                        assert background.json()["status"] == "ignored"
                    event = _message(message_id, content, thread=thread)
                    payload = {"t": "MESSAGE_CREATE", "d": event, "thread_messages": messages}
                    assert (await client.post("/v1/replay/discord", json=payload)).status_code == 200
                    await _child_result(env.client, message_id)
                runner.run_gate, runner.stop_requested = asyncio.Event(), False
                runner.run_started.clear()
                await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("stop-q", "work", thread="thread")})
                await asyncio.wait_for(runner.run_started.wait(), 2)
                assert (
                    await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("stop-c", "/stop", thread="thread"), "kind": "stop"})
                ).json()["status"] == "stopped"
                assert (await _child_result(env.client, "stop-q"))["error"] == "Turn stopped by user"
            state = await env.client.get_workflow_handle("wiseman-thread").query(ThreadWorkflow.session)
            assert state["turn"] == 3


async def test_graph_boundary_covers_tools_and_provider(monkeypatch) -> None:
    for name in ("WISEMAN_DISCORD_BOT_ID", "WISEMAN_PROVIDER_TOKEN", "WISEMAN_MCP_TOKEN", "WISEMAN_ALLOW_PROFILE_EDITS"):
        monkeypatch.setenv(name, "bot" if name.endswith("BOT_ID") else "secret" if not name.endswith("EDITS") else "1")
    clients = mock_container()
    app = create_app(clients=clients)
    headers = {"authorization": "Bearer secret"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        await client.post("/v1/tools/describe-image", headers=headers, json={"url": "https://cdn.test/a.png"})
        await client.post("/v1/tools/send-file", headers=headers, json={"thread_id": "thread", "filename": "a.txt", "data_base64": "b2s="})
        await client.post("/v1/tools/set-profile", headers=headers, json={"username": "Wiseman"})
        provider = await client.post("/v1/responses", headers=headers, json={"model": "mock", "input": "hi"})
    assert provider.status_code == 200


async def test_graphwalker_edges_use_the_discord_admission_boundary(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    path = os.getenv("GRAPHWALKER_PATH")
    if not path:
        return
    clients = mock_container()
    app = create_app(clients=clients)
    source = await asyncio.to_thread(Path(path).read_text)
    elements = [str(json.loads(line)["currentElementName"]) for line in source.splitlines() if line.strip()]
    state = ModelState()
    executed: set[tuple[str, str, str]] = set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        for index in range(0, len(elements) - 2, 2):
            source, edge, target = Vertex(elements[index]), elements[index + 1], Vertex(elements[index + 2])
            message_id = f"graph-{index}"
            executed.add((edge, source, target))
            if edge not in ACTION_EDGES:
                state.advance(edge, source, target)
                continue
            response = await _action(client, edge, message_id)
            assert response.status_code == 200
            state.advance(edge, source, target, message_id)
    if os.getenv("GRAPHWALKER_COVERAGE"):
        assert {(edge, source, target) for edge, source, target in EDGES} <= executed
