# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.clients.client_interfaces import ClientMode, ClientSettings
from app.clients.mock_clients import MockHarnessRunner, mock_container
from app.clients.real_clients import build_clients
from app.engine import Engine, EngineConfig
from app.http_api import create_app
from app.nodes import TurnActivities
from app.runner import RunnerError
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
from tests.graphwalker.runner import GraphRunner
from tests.graphwalker.runtime_observer import RuntimeObserver
from tests.graphwalker.vertices import STATE_FUNCTIONS
from tests.graphwalker.wiseman_client import WisemanClient

if TYPE_CHECKING:
    from app.models import Event
    from app.types import JsonObject


def _ordered_history(context: GraphContext, observation: RuntimeObservation) -> bool:
    expected = [message.id for message in context.state.chat.messages]
    observed = list(observation.chat.message_ids)
    cursor = 0
    for message_id in observed:
        if cursor < len(expected) and message_id == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _session_ids(session: JsonObject, name: str) -> tuple[str, ...]:
    value = session.get(name)
    return tuple(str(item) for item in value if item) if isinstance(value, list) else ()


def _edge_observed(context: GraphContext, vertex: Vertex, observation: RuntimeObservation) -> bool:
    message_id = context.last_message_id
    edge = context.last_edge
    if vertex in {Vertex.DELIVERING, Vertex.ERROR} and observation.wiseman.reaction_phase not in {"success", "failure"}:
        return False
    edge_result = False
    if context.last_edge == "background-chatter" and context.previous_state and context.previous_state.vertex is Vertex.IDLE:
        edge_result = True
    elif edge == "idle-stop":
        edge_result = vertex is Vertex.IDLE
    elif edge == "duplicate-question":
        edge_result = message_id not in observation.wiseman.pending_question_ids
    elif edge == "duplicate-stop":
        edge_result = message_id in observation.chat.stop_command_ids
    else:
        collections = {
            "admit-question": observation.wiseman.pending_question_ids,
            "queue-question": observation.wiseman.pending_question_ids,
            "background-chatter": observation.chat.background_context_ids,
            "running-background-chatter": observation.chat.background_context_ids,
            "steer-active-turn": observation.chat.steering_ids,
            "repeat-steer": observation.chat.steering_ids,
            "stop-preparing": observation.chat.stop_command_ids,
            "stop-running": observation.chat.stop_command_ids,
            "stop-recovering": observation.chat.stop_command_ids,
            "stop-delivering": observation.chat.stop_command_ids,
        }
        if edge is not None and (ids := collections.get(edge)) is not None:
            edge_result = message_id in ids
        elif edge == "context-ready":
            edge_result = bool(context.previous_state and observation.chat.consumed_context_ids == tuple(context.state.chat.consumed_context))
        elif edge == "resume-session":
            edge_result = bool(
                context.previous_state
                and observation.chat.consumed_context_ids == tuple(context.state.chat.consumed_context)
                and observation.chat.background_context_ids == tuple(context.state.chat.background_context)
            )
        else:
            required_evidence = {Vertex.DELIVERING: {"completed", "reaction"}, Vertex.ERROR: {"failure", "reaction"}}.get(vertex, set())
            edge_result = required_evidence.issubset(observation.phoenix_nodes)
    return edge_result


def _queued_idle_handoff(context: GraphContext, observation: RuntimeObservation) -> RuntimeObservation | None:
    if context.last_edge not in {"answer-finalized", "error-finalized", "stop-confirmed"}:
        return None
    pending = context.state.wiseman.pending_questions
    active = context.state.wiseman.active_question
    next_question = pending[1] if active and len(pending) > 1 and pending[:1] == [active] else pending[0] if pending else None
    if (
        not next_question
        or observation.wiseman.active_question not in pending
        or observation.wiseman.active_question == active
    ):
        return None
    return replace(
        observation,
        chat=replace(observation.chat, answer_message_ids=(), progress_message_ids=(), typing=False),
        wiseman=replace(
            observation.wiseman,
            phase=Vertex.IDLE,
            active_question=None,
            active_turn=None,
            result_known=False,
            error=False,
            stop_target_question_id=None,
            inferencing=False,
            stop_requested=False,
            recovering=False,
            outcome_unknown=False,
            cancellation_unknown=False,
            delivery_phase="idle",
            reaction_phase="none",
        ),
    )


def _message(message_id: str, content: str, *, thread: str | None = None, reply_to: str | None = None, mention: bool = True) -> dict[str, object]:
    try:
        sequence = int(message_id.rsplit("-", 1)[-1])
    except ValueError:
        sequence = 0
    timestamp = f"2026-09-05T00:{sequence // 60:02d}:{sequence % 60:02d}Z"
    return {"id": message_id, "author": {"id": "human", "username": "human"}, "content": content, "channel_id": "home", "thread_id": thread, "timestamp": timestamp, "mentions": [{"id": "bot"}] if mention else [], "message_reference": {"message_id": reply_to} if reply_to else {}}  # noqa: E501 # fmt: skip


class _ReplayHarness(GraphHarness):
    def __init__(self, observer: RuntimeObserver, runner: GraphRunner, engine: Engine, client, environment: WorkflowEnvironment) -> None:
        self.observer, self.runner, self.engine, self.client, self.environment = observer, runner, engine, client, environment

    async def wait_for_state(self, vertex: Vertex, context: GraphContext, *, deadline_seconds: int) -> RuntimeObservation:
        assert context.state.vertex is vertex
        try:
            async with asyncio.timeout(deadline_seconds):
                while True:
                    try:
                        observation = await self.observer.observe(context.state.chat.thread_id)
                    except RPCError as exc:
                        if exc.status != RPCStatusCode.NOT_FOUND or vertex not in {Vertex.IDLE, Vertex.RETIRED}:
                            raise
                        return RuntimeObservation(ObservedChatState(), ObservedWisemanState(vertex))
                    observation = _queued_idle_handoff(context, observation) or observation
                    if observation.wiseman.phase is vertex and self._edge_observed(context, vertex, observation):
                        return observation
                    await asyncio.sleep(0.02)
        except TimeoutError as exc:
            snapshot = await self.observer.clients.temporal.snapshot(context.state.chat.thread_id)
            message = f"timed out waiting for {vertex} after {context.last_edge}: {snapshot}; runner={self.runner.history[-16:]}"
            raise AssertionError(message) from exc

    @staticmethod
    def _edge_observed(context: GraphContext, vertex: Vertex, observation: RuntimeObservation) -> bool:
        return _ordered_history(context, observation) and _edge_observed(context, vertex, observation)

    async def prepare_edge(self, edge: GraphElement, context: GraphContext, *, deadline_seconds: int) -> None:
        assert deadline_seconds > 0
        if edge.name == EdgeName.ADMIT_QUESTION:
            self.runner = GraphRunner(self.runner.state)
            object.__setattr__(self.engine.config, "runner", self.runner)
            self.runner.hold_stop = False
            self.runner.clear_error()
            self.runner.clear_stops()
            self.runner.run_gate = asyncio.Event()
            self.runner.completion_gate = asyncio.Event()
            self.runner.progress_sent.clear()
            self.runner.run_started.clear()
        elif edge.name is EdgeName.DISPATCH_QUEUED:
            self.runner.clear_error()
            self.runner.clear_stops()
            self.runner.release_next_attempt()
        elif edge.name is EdgeName.QUEUE_QUESTION:
            self.runner.hold_next_attempt()
        elif edge.name in {
            EdgeName.PREPARATION_FAILED,
            EdgeName.PERMANENT_FAILURE,
            EdgeName.TRANSIENT_FAILURE,
            EdgeName.EXECUTION_UNCERTAIN,
            EdgeName.RETRY_EXHAUSTED,
            EdgeName.OUTCOME_ESTABLISHED,
        }:
            await self._prepare_failure(edge, context)
        elif edge.name in {EdgeName.STOP_PREPARING, EdgeName.STOP_RUNNING, EdgeName.STOP_RECOVERING}:
            self.runner.hold_stop = True
            await self._signal_active(context, "hold_cancellation")
        elif edge.name is EdgeName.STOP_DELIVERING:
            await self._signal_active(context, "hold_terminal")

    async def _prepare_failure(self, edge: GraphElement, context: GraphContext) -> None:
        active_message_id = context.state.wiseman.active_question or context.last_message_id
        if edge.name in {EdgeName.PREPARATION_FAILED, EdgeName.PERMANENT_FAILURE, EdgeName.RETRY_EXHAUSTED, EdgeName.OUTCOME_ESTABLISHED}:
            await self._signal_active(context, "hold_terminal")
        errors: dict[str, tuple[str, RunnerError]] = {
            EdgeName.TRANSIENT_FAILURE: ("next_error", RunnerError(503, "graph transient failure")),
            EdgeName.PERMANENT_FAILURE: ("next_error", RunnerError(500, "graph permanent failure")),
            EdgeName.EXECUTION_UNCERTAIN: ("next_error", RunnerError(520, "graph outcome is unknown")),
        }
        if edge.name is EdgeName.PREPARATION_FAILED:
            self.runner.inject_error(RunnerError(400, "graph preparation failure"), active_message_id)
        if setting := errors.get(edge.name):
            self.runner.inject_error(setting[1], active_message_id)

    async def _signal_active(self, context: GraphContext, signal: str, *args: object) -> None:
        if active := context.state.wiseman.active_question:
            await self.client.get_workflow_handle(f"wiseman-turn-{active}").signal(signal, *args)

    async def execute_edge(self, edge: GraphElement, context: GraphContext, *, deadline_seconds: int) -> None:
        assert deadline_seconds > 0
        if edge.name == EdgeName.PROGRESS_PREVIEW:
            await self._execute_progress(context, deadline_seconds)
        elif edge.name == EdgeName.INFERENCE_COMPLETE:
            await self._execute_inference_complete(context)
        elif edge.name is EdgeName.TRANSIENT_FAILURE:
            self._release_active_runner(context)
            await self.runner.wait_until_idle(context.state.wiseman.session_id or f"codex-{context.state.chat.thread_id}")
            self.runner.clear_error()
            self.runner.completion_gate = asyncio.Event()
            self.runner.progress_sent.clear()
        elif edge.name is EdgeName.EXECUTION_UNCERTAIN:
            self._release_active_runner(context)
        elif edge.name in {
            EdgeName.RESUME_SESSION,
            EdgeName.RETRY_EXHAUSTED,
            EdgeName.OUTCOME_ESTABLISHED,
            EdgeName.CANCELLATION_UNKNOWN,
            EdgeName.COMPLETION_RACE,
        }:
            await ({EdgeName.RESUME_SESSION: self._prepare_resume}.get(edge.name, self._noop))(context)
            signals = {
                EdgeName.RESUME_SESSION: "resume_session",
                EdgeName.RETRY_EXHAUSTED: "exhaust_retries",
                EdgeName.OUTCOME_ESTABLISHED: "establish_outcome",
                EdgeName.CANCELLATION_UNKNOWN: "mark_cancellation_unknown",
                EdgeName.COMPLETION_RACE: "mark_completion_race",
            }
            if edge.name is EdgeName.COMPLETION_RACE:
                self.runner.clear_stops()
                self.runner.hold_stop = False
            await self._signal_active(context, signals[edge.name])
            await ({EdgeName.RESUME_SESSION: self._wait_for_resume}.get(edge.name, self._noop))(context, deadline_seconds)
            follow_up = {
                EdgeName.COMPLETION_RACE: self._completion_wins,
                EdgeName.OUTCOME_ESTABLISHED: self._release_active_cancellation,
            }.get(edge.name, self._noop)
            await follow_up(context)
        elif edge.name in {EdgeName.ANSWER_FINALIZED, EdgeName.ERROR_FINALIZED, EdgeName.STOP_CONFIRMED}:
            await self._execute_terminal(edge, context)
        elif edge.name in {EdgeName.PREPARATION_FAILED, EdgeName.PERMANENT_FAILURE}:
            self._release_active_runner(context)
        elif edge.name == EdgeName.IDLE_RETIREMENT:
            await self.environment.sleep(timedelta(days=3))
        elif edge.name is EdgeName.FIXTURE_RESET:
            self.runner = GraphRunner(self.runner.state)
            object.__setattr__(self.engine.config, "runner", self.runner)
            context.state.chat.thread_id = f"thread-reset-{context.state.step}"
            fixture_event = {
                "background": True,
                "trigger": {
                    "id": f"fixture-{context.state.chat.thread_id}",
                    "author_id": "fixture",
                    "channel_id": context.state.chat.thread_id,
                    "thread_id": context.state.chat.thread_id,
                },
            }
            await self.client.start_workflow(
                ThreadWorkflow.run,
                {"state": {}, "pending": [fixture_event]},
                id=f"wiseman-{context.state.chat.thread_id}",
                task_queue="graph",
            )
        await self._wait_for_phase(edge.target, deadline_seconds, context.state.chat.thread_id, context)

    async def _execute_progress(self, context: GraphContext, deadline_seconds: int) -> None:
        if context.state.wiseman.active_question:
            await self._signal_active(context, "progress", f"✍️ Graph preview {context.state.step}")
        if not self.runner.release_attempt_start(context.state.wiseman.active_question or "") and self.runner.run_gate is not None:
            self.runner.run_gate.set()
        await asyncio.wait_for(self.runner.progress_sent.wait(), deadline_seconds)

    async def _completion_wins(self, context: GraphContext) -> None:
        self.runner.clear_stops()
        await self._signal_active(context, "hold_terminal")
        self.runner.hold_stop = False
        self.runner.clear_error()
        await self._signal_active(context, "release_cancellation")
        self._release_active_runner(context)

    async def _release_active_cancellation(self, context: GraphContext) -> None:
        await self._signal_active(context, "release_cancellation")

    async def _prepare_resume(self, _context: GraphContext) -> None:
        self.runner.clear_error()
        self.runner.clear_stops()
        if self.runner.run_gate is not None:
            self.runner.run_gate.set()

    async def _wait_for_resume(self, _context: GraphContext, deadline_seconds: int) -> None:
        await asyncio.wait_for(self.runner.progress_sent.wait(), deadline_seconds)

    async def _noop(self, _context: GraphContext, _deadline_seconds: int = 0) -> None:
        return

    async def _execute_inference_complete(self, context: GraphContext) -> None:
        await self._signal_active(context, "hold_terminal")
        self._release_active_runner(context)

    async def _execute_terminal(self, edge: GraphElement, context: GraphContext) -> None:
        if active := context.state.wiseman.active_question:
            handle = self.client.get_workflow_handle(f"wiseman-turn-{active}")
            if edge.name is EdgeName.STOP_CONFIRMED:
                await handle.signal("release_cancellation")
                await handle.signal("release_terminal")
            else:
                await handle.signal("release_terminal")
            if edge.name is EdgeName.STOP_CONFIRMED:
                self.runner.hold_stop = False
                self.runner.release_attempt(active)

    def _release_runner(self) -> None:
        for gate in (self.runner.run_gate, self.runner.completion_gate):
            if gate is not None:
                gate.set()

    def _release_active_runner(self, context: GraphContext) -> None:
        if not self.runner.release_attempt(context.state.wiseman.active_question or ""):
            self._release_runner()

    async def _wait_for_phase(self, vertex: Vertex, deadline_seconds: int, thread_id: str, context: GraphContext) -> None:
        try:
            async with asyncio.timeout(deadline_seconds):
                while True:
                    try:
                        observation = await self.observer.observe(thread_id)
                    except RPCError as exc:
                        if vertex in {Vertex.IDLE, Vertex.RETIRED} and exc.status == RPCStatusCode.NOT_FOUND:
                            return
                        raise
                    observation = _queued_idle_handoff(context, observation) or observation
                    if observation.wiseman.phase is vertex:
                        return
                    await asyncio.sleep(0.02)
        except TimeoutError as exc:
            snapshot = await self.observer.clients.temporal.snapshot(thread_id)
            message = f"timed out waiting for boundary phase {vertex}: {snapshot}; runner={self.runner.history[-16:]}"
            raise AssertionError(message) from exc


class _TemporalBoundary:
    def __init__(self, client) -> None:
        self.client = client

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def submit(self, event: JsonObject) -> dict[str, object] | None:
        trigger = cast("dict[str, object]", event["trigger"])
        workflow_id = f"wiseman-{trigger.get('thread_id') or trigger['channel_id']}"
        try:
            await self.client.start_workflow(ThreadWorkflow.run, {"event": event, "state": {}}, id=workflow_id, task_queue="graph")
        except WorkflowAlreadyStartedError:
            handle = self.client.get_workflow_handle(workflow_id)
            message_id = str(trigger.get("id", ""))
            session = cast("JsonObject", await handle.query(ThreadWorkflow.session))
            known = {
                str(session.get("active_message", "")),
                *_session_ids(session, "processed"),
                *_session_ids(session, "message_ids"),
                *_session_ids(session, "pending_message_ids"),
            }
            if message_id in known:
                return {"status": "duplicate", "message_id": message_id}
            await handle.signal(ThreadWorkflow.submit, event)
        return None

    async def stop(self, event: Event) -> bool:
        try:
            state = await self.client.get_workflow_handle(f"wiseman-{event.trigger.thread_id or event.trigger.channel_id}").query(ThreadWorkflow.session)
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return False
            raise
        message_id = str(state.get("active_message", ""))
        return bool(
            message_id
            and await self.client.get_workflow_handle(f"wiseman-turn-{message_id}").execute_update("stop", event.model_dump(mode="json"), id=event.trigger.id)
        )

    async def steer(self, event: Event) -> bool:
        state = await self.client.get_workflow_handle(f"wiseman-{event.trigger.thread_id or event.trigger.channel_id}").query(ThreadWorkflow.session)
        message_id = str(state.get("active_message", ""))
        return bool(
            message_id
            and await self.client.get_workflow_handle(f"wiseman-turn-{message_id}").execute_update("steer", event.model_dump(mode="json"), id=event.trigger.id)
        )

    async def touch(self, event: Event) -> None:
        if event.trigger.thread_id:
            try:
                await self.client.get_workflow_handle(f"wiseman-{event.trigger.thread_id}").signal(ThreadWorkflow.touch, event.model_dump(mode="json"))
            except RPCError as exc:
                if exc.status != RPCStatusCode.NOT_FOUND:
                    raise

    async def snapshot(self, thread_id: str) -> JsonObject:
        handle = self.client.get_workflow_handle(f"wiseman-{thread_id}")
        try:
            session = cast("JsonObject", await handle.query(ThreadWorkflow.session))
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
            result = cast("JsonObject", await handle.result())
            state = result.get("state")
            if not isinstance(state, dict):
                raise
            return {**state, "active_message": "", "active_timestamp": "", "closed": True}
        if message_id := session.get("active_message"):
            try:
                session["active_turn_snapshot"] = cast(
                    "JsonObject", await self.client.get_workflow_handle(f"wiseman-turn-{message_id}").query(TurnWorkflow.snapshot)
                )
            except RPCError as exc:
                if exc.status != RPCStatusCode.NOT_FOUND:
                    raise
        return session


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
    async with await WorkflowEnvironment.start_time_skipping() as env:
        clients = mock_container(temporal=_TemporalBoundary(env.client))
        runner = GraphRunner(cast("MockHarnessRunner", clients.runner).state)
        clients.runner = runner
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
    executed: set[tuple[EdgeName, Vertex, Vertex]] = set()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        clients = mock_container(temporal=_TemporalBoundary(env.client))
        runner = GraphRunner(cast("MockHarnessRunner", clients.runner).state)
        clients.runner = runner
        activities = TurnActivities(Engine(EngineConfig(clients.phoenix, clients.runner, clients.prompts, discord=clients.discord)), env.client)
        async with Worker(
            env.client,
            task_queue="graph",
            workflows=[ThreadWorkflow, TurnWorkflow],
            activities=[provision_workspace, start_codex, fail_turn, retire_session, *activities.registered()],
        ):
            app = create_app(clients=clients)
            elements = [str(json.loads(line)["currentElementName"]) for line in (await asyncio.to_thread(Path(path).read_text)).splitlines() if line.strip()]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
                wiseman = WisemanClient(client)
                harness = _ReplayHarness(RuntimeObserver(clients), runner, activities.engine, env.client, env)
                context = GraphContext(harness, clients, wiseman)
                for index in range(0, len(elements) - 2, 2):
                    source, edge, target = _vertex(elements[index]), _edge_name(elements[index + 1]), _vertex(elements[index + 2])
                    executed.add((edge, source, target))
                    context.message_id = f"graph-{index}"
                    await STATE_FUNCTIONS[source](context)
                    await EDGE_FUNCTIONS[edge](context)
                    if edge in {
                        EdgeName.ADMIT_QUESTION,
                        EdgeName.QUEUE_QUESTION,
                        EdgeName.DUPLICATE_QUESTION,
                        EdgeName.BACKGROUND_CHATTER,
                        EdgeName.RUNNING_BACKGROUND_CHATTER,
                        EdgeName.IDLE_STOP,
                        EdgeName.STEER_ACTIVE_TURN,
                        EdgeName.REPEAT_STEER,
                        EdgeName.STOP_PREPARING,
                        EdgeName.STOP_RUNNING,
                        EdgeName.STOP_RECOVERING,
                        EdgeName.STOP_DELIVERING,
                        EdgeName.DUPLICATE_STOP,
                    }:
                        response = context.last_response
                        assert response is not None
                        assert response.status_code == 200
                        if edge == EdgeName.DUPLICATE_QUESTION:
                            expected = "duplicate" if context.previous_state and context.previous_state.wiseman.seen_questions else "ignored"
                            assert response.body["status"] == expected
                        elif edge in {EdgeName.BACKGROUND_CHATTER, EdgeName.RUNNING_BACKGROUND_CHATTER, EdgeName.IDLE_STOP}:
                            assert response.body["status"] == "ignored"
                        elif edge in {EdgeName.ADMIT_QUESTION, EdgeName.QUEUE_QUESTION}:
                            assert response.body["status"] == "queued"
    if os.getenv("GRAPHWALKER_COVERAGE"):
        assert {(edge, source, target) for edge, source, target in EDGES} <= executed


def _vertex(value: str) -> Vertex:
    return Vertex(value.removeprefix("v-"))


def _edge_name(value: str) -> EdgeName:
    name = value.removeprefix("e-")
    if name in EDGES_BY_NAME:
        return EDGES_BY_NAME[name].name
    return EdgeName(name)
