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

from app.clients.client_interfaces import ClientMode, ClientSettings
from app.clients.mock_clients import MockRunner, mock_container
from app.clients.real_clients import build_clients
from app.engine import Engine, EngineConfig
from app.http_api import create_app
from app.nodes import TurnActivities
from app.temporal_runtime import ThreadWorkflow, TurnWorkflow, fail_turn, provision_workspace, retire_session, start_codex
from tests.graphwalker.edges import EDGE_FUNCTIONS
from tests.graphwalker.graph_utils import GraphContext, GraphElement, GraphHarness
from tests.graphwalker.model import (
    EDGES,
    EDGES_BY_NAME,
    EdgeName,
    ObservedChatState,
    ObservedWisemanState,
    RuntimeObservation,
    Vertex,
)
from tests.graphwalker.vertices import STATE_FUNCTIONS

ACTION_EDGES = {
    EdgeName.BACKGROUND_CHATTER,
    EdgeName.RUNNING_BACKGROUND_CHATTER,
    EdgeName.DUPLICATE_QUESTION,
    EdgeName.IDLE_STOP,
    EdgeName.STEER_ACTIVE_TURN,
    EdgeName.REPEAT_STEER,
    EdgeName.STOP_PREPARING,
    EdgeName.STOP_RUNNING,
    EdgeName.STOP_RECOVERING,
    EdgeName.STOP_DELIVERING,
    EdgeName.DUPLICATE_STOP,
    EdgeName.ADMIT_QUESTION,
    EdgeName.QUEUE_QUESTION,
}


def _message(message_id: str, content: str, *, thread: str | None = None, reply_to: str | None = None, mention: bool = True) -> dict[str, object]:
    return {"id": message_id, "author": {"id": "human", "username": "human"}, "content": content, "channel_id": "home", "thread_id": thread, "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z", "mentions": [{"id": "bot"}] if mention else [], "message_reference": {"message_id": reply_to} if reply_to else {}}  # noqa: E501 # fmt: skip


async def _action(client: httpx.AsyncClient, edge: EdgeName | str, message_id: str) -> httpx.Response:
    edge = EdgeName(edge)
    if edge == EdgeName.DUPLICATE_QUESTION:
        payload = {"t": "MESSAGE_CREATE", "d": _message(message_id, "question")}
        await client.post("/v1/replay/discord", json=payload)
        return await client.post("/v1/replay/discord", json=payload)
    if edge == EdgeName.IDLE_STOP:
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread="thread"), "kind": "stop"})
    if edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER}:
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "background", mention=False)})
    if edge in {EdgeName.STEER_ACTIVE_TURN, EdgeName.REPEAT_STEER}:
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "steer", thread="thread", reply_to="answer"), "kind": "steer"})  # noqa: E501 # fmt: skip
    if edge in {
        EdgeName.STOP_PREPARING,
        EdgeName.STOP_RUNNING,
        EdgeName.STOP_RECOVERING,
        EdgeName.STOP_DELIVERING,
        EdgeName.DUPLICATE_STOP,
    }:
        return await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "/stop", thread="thread"), "kind": "stop"})
    return await client.post(
        "/v1/replay/discord",
        json={
            "t": "MESSAGE_CREATE",
            "d": _message(message_id, "question", thread=None if edge == EdgeName.ADMIT_QUESTION else "thread"),
            "kind": "startup" if edge == EdgeName.ADMIT_QUESTION else "followup",
        },
    )


class _ReplayHarness(GraphHarness):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.response: httpx.Response | None = None

    async def wait_for_state(self, vertex: Vertex, context: GraphContext, *, deadline_seconds: int) -> RuntimeObservation:
        assert context.state.vertex is vertex
        assert deadline_seconds > 0
        state = context.state
        active_question = state.wiseman.active_question
        terminal = vertex in {Vertex.DELIVERING, Vertex.ERROR}
        return RuntimeObservation(
            chat=ObservedChatState(
                message_ids=tuple(message.id for message in state.chat.messages),
                background_context_ids=tuple(state.chat.background_context),
                consumed_context_ids=tuple(state.chat.consumed_context),
                steering_ids=tuple(state.chat.steering_messages),
                stop_command_ids=tuple(state.chat.stop_commands),
                answer_message_ids=(f"answer-{active_question}",) if vertex is Vertex.DELIVERING and active_question else (),
                progress_message_ids=(f"progress-{active_question}",) if vertex in {Vertex.PREPARING, Vertex.RUNNING} and active_question else (),
                progress_edit_count=1 if vertex in {Vertex.PREPARING, Vertex.RUNNING} and active_question else 0,
                answer_edit_count=1 if vertex is Vertex.DELIVERING and active_question else 0,
                typing=vertex is Vertex.RUNNING,
                archived=vertex is Vertex.RETIRED,
            ),
            wiseman=ObservedWisemanState(
                phase=vertex,
                active_question=active_question,
                pending_question_ids=tuple(state.wiseman.pending_questions),
                session_id=state.wiseman.session_id,
                turn=state.wiseman.turns,
                active_turn=state.wiseman.turns + 1 if active_question else None,
                result_known=terminal,
                error=vertex is Vertex.ERROR,
                stop_target_question_id=active_question if vertex is Vertex.CANCELLING else None,
            ),
        )

    async def execute_edge(self, edge: GraphElement, context: GraphContext, *, deadline_seconds: int) -> None:
        assert deadline_seconds > 0
        if edge.name in ACTION_EDGES:
            self.response = await _action(self.client, edge.name, context.message_id)


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
            result = await client.get_workflow_handle(f"wiseman-turn-{message_id}").result()
            for _ in range(100):
                state = await client.get_workflow_handle("wiseman-thread").query(ThreadWorkflow.session)
                if message_id in state.get("processed", []):
                    break
                await asyncio.sleep(0.05)
            return result  # noqa: TRY300
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
                        assert (await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("chat", "background", thread="thread", mention=False)})).json()["status"] == "ignored"  # noqa: E501 # fmt: skip
                    assert (await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, content, thread="thread"), "thread_messages": messages})).status_code == 200  # noqa: E501 # fmt: skip
                    await _child_result(env.client, message_id)
                runner.run_gate, runner.run_started = asyncio.Event(), asyncio.Event()
                await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("stop-q", "work", thread="thread")})
                await asyncio.wait_for(runner.run_started.wait(), 2)
                assert all(response.json()["status"] == "queued" for response in await asyncio.gather(*(client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message(message_id, "queued", thread="thread")}) for message_id in ("queued-1", "queued-2"))))  # noqa: E501 # fmt: skip
                assert (
                    await client.post("/v1/replay/discord", json={"t": "MESSAGE_CREATE", "d": _message("stop-c", "/stop", thread="thread"), "kind": "stop"})
                ).json()["status"] == "stopped"
                assert (await _child_result(env.client, "stop-q"))["error"] == "Turn stopped by user"
                await asyncio.gather(*(_child_result(env.client, message_id) for message_id in ("queued-1", "queued-2")))  # fmt: skip
            assert (await env.client.get_workflow_handle("wiseman-thread").query(ThreadWorkflow.session))["turn"] == 5


async def test_boundary_tools_and_provider(monkeypatch) -> None:
    for name in ("WISEMAN_PROVIDER_TOKEN", "WISEMAN_MCP_TOKEN"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("WISEMAN_ALLOW_PROFILE_EDITS", "1")
    app = create_app(clients=build_clients(ClientMode.MOCK, ClientSettings()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        headers = {"authorization": "Bearer secret"}
        requests = (("/v1/tools/describe-image", {"url": "https://cdn.test/a.png"}), ("/v1/tools/send-file", {"thread_id": "thread", "filename": "a.txt", "data_base64": "b2s="}), ("/v1/tools/set-profile", {"username": "Wiseman"}), ("/v1/responses", {"model": "mock", "input": "hi"}))  # noqa: E501 # fmt: skip
        responses = [await client.post(path, headers=headers, json=payload) for path, payload in requests]
        assert all(response.status_code == 200 for response in responses)
        assert responses[-1].content == b':keep\r\nid: provider-1\r\nretry: 1000\r\ndata: {"type":"response.completed","response":{"model":"mock"}}\r\n\r\n'


async def test_graphwalker_edges_use_admission_boundary(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_DISCORD_BOT_ID", "bot")
    path = os.getenv("GRAPHWALKER_PATH")
    if not path: return  # noqa: E701 # fmt: skip
    app, executed = create_app(clients=mock_container()), set()
    elements = [str(json.loads(line)["currentElementName"]) for line in (await asyncio.to_thread(Path(path).read_text)).splitlines() if line.strip()]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        harness = _ReplayHarness(client)
        context = GraphContext(harness)
        for index in range(0, len(elements) - 2, 2):
            source, edge, target = _vertex(elements[index]), _edge_name(elements[index + 1]), _vertex(elements[index + 2])
            executed.add((edge, source, target))
            context.message_id = f"graph-{index}"
            await STATE_FUNCTIONS[source](context)
            await EDGE_FUNCTIONS[edge](context)
            if edge in ACTION_EDGES:
                response = harness.response
                assert response is not None
                assert response.status_code == 200
                if edge == EdgeName.DUPLICATE_QUESTION:
                    assert response.json()["status"] == "duplicate"
                elif edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER, EdgeName.IDLE_STOP}:
                    assert response.json()["status"] == "ignored"
                elif edge in {EdgeName.ADMIT_QUESTION, EdgeName.QUEUE_QUESTION}:
                    assert response.json()["status"] == "queued"
    if os.getenv("GRAPHWALKER_COVERAGE"):
        assert {(edge, source, target) for edge, source, target in EDGES} <= executed


def _vertex(value: str) -> Vertex:
    return Vertex(value.removeprefix("v-"))


def _edge_name(value: str) -> EdgeName:
    name = value.removeprefix("e-")
    if name in EDGES_BY_NAME:
        return EDGES_BY_NAME[name].name
    return EdgeName(name)
