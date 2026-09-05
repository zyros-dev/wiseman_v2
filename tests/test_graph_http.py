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

ACTION_EDGES = {"background-chatter", "duplicate-question", "idle-stop", "steer-active-turn", "repeat-steer", "stop-active-turn", "stop-active-recovery", "stop-active-delivery", "duplicate-stop", "admit-question", "queue-question"}  # noqa: E501 # fmt: skip


def _message(message_id: str, content: str, *, thread: str | None = None, reply_to: str | None = None, mention: bool = True) -> dict[str, object]:
    return {"id": message_id, "author": {"id": "human", "username": "human"}, "content": content, "channel_id": "home", "thread_id": thread, "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z", "mentions": [{"id": "bot"}] if mention else [], "message_reference": {"message_id": reply_to} if reply_to else {}}  # noqa: E501 # fmt: skip


async def _action(client: httpx.AsyncClient, edge: str, message_id: str) -> httpx.Response:
    if edge == "duplicate-question":
        payload = {"t": "MESSAGE_CREATE", "d": _message(message_id, "question")}
        await client.post("/v1/replay/discord", json=payload)
        return await client.post("/v1/replay/discord", json=payload)
    if edge == "idle-stop":
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread="thread"), "kind": "stop"})
    if edge == "background-chatter":
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "background", mention=False)})
    if edge in {"steer-active-turn", "repeat-steer"}:
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "steer", thread="thread", reply_to="answer"), "kind": "steer"})  # noqa: E501 # fmt: skip
    if edge in {"stop-active-turn", "stop-active-recovery", "stop-active-delivery", "duplicate-stop"}:
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread="thread"), "kind": "stop"})
    return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "question", thread=None if edge == "admit-question" else "thread"), "kind": "startup" if edge == "admit-question" else "followup"})  # noqa: E501 # fmt: skip


class _TemporalBoundary:
    def __init__(self, client) -> None:
        self.client = client

    async def submit(self, event: dict[str, object]) -> None:
        trigger = cast("dict[str, object]", event["trigger"])
        workflow_id = f"wiseman-{trigger.get('thread_id') or trigger['channel_id']}"
        try:
            await self.client.start_workflow(ThreadWorkflow.run, {"event": event, "state": {}}, id=workflow_id, task_queue="graph")
        except WorkflowAlreadyStartedError:
            await self.client.get_workflow_handle(workflow_id).signal(ThreadWorkflow.submit, event)

    async def stop(self, event) -> bool:
        state = await self.client.get_workflow_handle(f"wiseman-{event.trigger.thread_id or event.trigger.channel_id}").query(ThreadWorkflow.session)
        message_id = str(state.get("active_message", ""))
        return bool(
            message_id
            and await self.client.get_workflow_handle(f"wiseman-turn-{message_id}").execute_update("stop", event.model_dump(mode="json"), id=event.trigger.id)
        )

    async def touch(self, event) -> None:
        return None


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
                for message_id, content, messages in (("q1", "one", []), ("q2", "two", [_message("chat", "background", mention=False)])):
                    if message_id == "q2":
                        assert (
                            await client.post(
                                "/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("chat", "background", thread="thread", mention=False)}
                            )
                        ).json()["status"] == "ignored"
                    assert (
                        await client.post(
                            "/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, content, thread="thread"), "thread_messages": messages}
                        )
                    ).status_code == 200
                    await _child_result(env.client, message_id)
                runner.run_gate, runner.run_started = asyncio.Event(), asyncio.Event()
                await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("stop-q", "work", thread="thread")})
                await asyncio.wait_for(runner.run_started.wait(), 2)
                assert (
                    await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("stop-c", "/stop", thread="thread"), "kind": "stop"})
                ).json()["status"] == "stopped"
                assert (await _child_result(env.client, "stop-q"))["error"] == "Turn stopped by user"
            assert (await env.client.get_workflow_handle("wiseman-thread").query(ThreadWorkflow.session))["turn"] == 3


async def test_boundary_tools_and_provider(monkeypatch) -> None:
    for name in ("WISEMAN_PROVIDER_TOKEN", "WISEMAN_MCP_TOKEN"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("WISEMAN_ALLOW_PROFILE_EDITS", "1")
    app = create_app(clients=mock_container())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        headers = {"authorization": "Bearer secret"}
        requests = (("/v1/tools/describe-image", {"url": "https://cdn.test/a.png"}), ("/v1/tools/send-file", {"thread_id": "thread", "filename": "a.txt", "data_base64": "b2s="}), ("/v1/tools/set-profile", {"username": "Wiseman"}), ("/v1/responses", {"model": "mock", "input": "hi"}))  # noqa: E501 # fmt: skip
        responses = [await client.post(path, headers=headers, json=payload) for path, payload in requests]
        assert all(response.status_code == 200 for response in responses)


async def test_graphwalker_edges_use_admission_boundary(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    path = os.getenv("GRAPHWALKER_PATH")
    if not path: return  # noqa: E701 # fmt: skip
    app, state, executed = create_app(clients=mock_container()), ModelState(), set()
    elements = [str(json.loads(line)["currentElementName"]) for line in (await asyncio.to_thread(Path(path).read_text)).splitlines() if line.strip()]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        for index in range(0, len(elements) - 2, 2):
            source, edge, target = Vertex(elements[index]), elements[index + 1], Vertex(elements[index + 2])
            executed.add((edge, source, target))
            if edge in ACTION_EDGES:
                response = await _action(client, edge, f"graph-{index}")
                assert response.status_code == 200
                if edge == "duplicate-question":
                    assert response.json()["status"] == "duplicate"
                elif edge in {"background-chatter", "idle-stop"}:
                    assert response.json()["status"] == "ignored"
                elif edge in {"admit-question", "queue-question"}:
                    assert response.json()["status"] == "queued"
            state.advance(edge, source, target, f"graph-{index}")
    if os.getenv("GRAPHWALKER_COVERAGE"):
        assert {(edge, source, target) for edge, source, target in EDGES} <= executed
