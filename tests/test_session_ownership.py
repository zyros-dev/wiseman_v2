# Copyright (c) 2026 Nick van der Merwe
import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import httpx
import pytest
from openai_codex import ApprovalMode, AsyncThread, AsyncTurnHandle, Sandbox

from app.clients.mock_clients import mock_container
from app.engine import Engine, EngineConfig
from app.models import Event, Message
from app.phoenix import Phoenix, PromptHub
from app.runner import FakeRunner
from app.temporal_runtime import ThreadWorkflow
from runner.api import CodexRunner, Turn, Workspace, create_app

if TYPE_CHECKING:
    from openai_codex import AsyncCodex


@pytest.mark.asyncio
async def test_followup_keeps_original_workspace_owner():
    owners = []

    class Runner(FakeRunner):
        async def run(self, thread, prompt, user, workspace="", progress=None):
            owners.append(user)
            return await super().run(thread, prompt, user, workspace, progress)

    engine = Engine(EngineConfig(Phoenix(), Runner(), PromptHub(), discord=mock_container().discord))
    first = await engine.handle(Event(trigger=Message(id="1", author_id="alice", channel_id="c", thread_id="t")))
    second = await engine.handle(
        Event(trigger=Message(id="2", author_id="bob", channel_id="c", thread_id="t")), state_data=first["state"]
    )
    assert owners == ["alice", "alice"]
    assert second["state"]["owner_id"] == "alice"
    assert second["state"]["codex_thread"] == first["state"]["codex_thread"]


@pytest.mark.asyncio
async def test_session_query_survives_discord_archive_idle_period(monkeypatch):
    instance = ThreadWorkflow()
    waits = []

    async def execute(*args, **kwargs):
        return {"state": args[1].get("state", {})}

    async def child(*args, **kwargs):
        return {"state": {"owner_id": "alice", "codex_thread": "sdk-thread", "turn": 1}}

    async def wait(predicate, **options):
        if predicate():
            return
        waits.append(options["timeout"])
        raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.execute_child_workflow", child)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", wait)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    await instance.run(
        {
            "event": {"trigger": {"id": "m", "author_id": "alice", "channel_id": "t"}},
            "state": {"owner_id": "alice", "codex_thread": "sdk-thread"},
        }
    )
    assert waits == [timedelta(days=3)]
    assert instance.session()["codex_thread"] == "sdk-thread"


@pytest.mark.asyncio
async def test_completed_duplicate_in_restored_pending_waits_instead_of_crashing(monkeypatch):
    async def wait(predicate, **_options):
        if predicate():
            return
        raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", wait)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: False)
    result = await ThreadWorkflow().run({"state": {"processed": ["done"]}, "pending": [{"trigger": {"id": "done"}}]})
    assert result["state"]["processed"] == ["done"]


@pytest.mark.asyncio
async def test_sdk_idle_eviction_never_interrupts_active_turn():
    runner = CodexRunner()
    closed = []

    async def close():
        closed.append("closed")

    runner.codex["t"] = cast("AsyncCodex", SimpleNamespace(close=close))
    runner.threads["t"] = Mock(spec=AsyncThread, id="sdk-thread")
    runner.last_used["t"] = 0
    runner.active_turns["t"] = Mock(spec=AsyncTurnHandle)
    await runner.expire(now=1000)
    assert not closed
    runner.active_turns.clear()
    await runner.expire(now=1000)
    assert closed == ["closed"]
    assert "t" not in runner.threads
    assert "t" not in runner.codex


@pytest.mark.asyncio
async def test_sdk_cache_evicts_oldest_idle_client_at_capacity(monkeypatch):
    monkeypatch.setenv("WISEMAN_MAX_IDLE_CLIENTS", "1")
    runner = CodexRunner()
    closed = []

    async def close():
        closed.append("closed")

    for key in ("old", "new"):
        runner.codex[key] = cast("AsyncCodex", SimpleNamespace(close=close))
        runner.threads[key] = Mock(spec=AsyncThread, id=key)
        runner.last_used[key] = 1 if key == "old" else 2
    await runner.expire(now=3)
    assert closed == ["closed"]
    assert set(runner.codex) == {"new"}


def test_workspace_reacquire_preserves_agent_authored_instructions(tmp_path):
    workspace = Workspace(str(tmp_path))
    path = workspace.thread("alice", "t")
    (path / "AGENTS.md").write_text("Keep this instruction.\n")
    workspace.thread("alice", "t")
    assert (path / "AGENTS.md").read_text() == "Keep this instruction.\n"


def test_workspace_release_is_scoped_idempotent_and_keeps_shared(tmp_path):
    workspace = Workspace(str(tmp_path))
    path = workspace.thread("alice", "t")
    other = workspace.thread("bob", "t")
    shared = (path / "shared").resolve()
    workspace.release("alice", "t")
    workspace.release("alice", "t")
    assert not path.exists()
    assert shared.is_dir()
    assert other.is_dir()
    with pytest.raises(ValueError, match="escapes"):
        workspace.release("../alice", "t")


@pytest.mark.asyncio
async def test_message_arriving_during_retirement_is_carried_forward(monkeypatch):
    instance = ThreadWorkflow()
    next_event = {"trigger": {"id": "next", "author_id": "bob", "channel_id": "t"}}
    continued = []

    class ContinuedError(Exception):
        pass

    async def wait(predicate, **_options):
        if not predicate():
            raise TimeoutError

    async def retire(_activity, payload, **_options):
        await instance.submit(next_event)
        return {"state": {**payload["state"], "closed": True}}

    def continue_as_new(payload):
        continued.append(payload)
        raise ContinuedError

    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", wait)
    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", retire)
    monkeypatch.setattr("app.temporal_runtime.workflow.continue_as_new", continue_as_new)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    with pytest.raises(ContinuedError):
        await instance.run(
            {"event": {"trigger": {"id": "done"}}, "state": {"owner_id": "alice", "processed": ["done"]}}
        )
    assert continued[0]["pending"] == [next_event]
    assert continued[0]["state"] == {"owner_id": "alice"}


@pytest.mark.asyncio
async def test_sdk_resume_after_eviction_keeps_session_and_permissions(tmp_path, monkeypatch):
    calls = []

    class Client:
        async def thread_start(self, **_options):
            calls.append("start")
            return SimpleNamespace(id="sdk-session")

        async def thread_resume(self, thread, **options):
            calls.append("resume")
            assert thread == "sdk-session"
            assert options["sandbox"] is Sandbox.full_access
            assert options["approval_mode"] is ApprovalMode.deny_all
            assert options["cwd"] == str(tmp_path)
            return SimpleNamespace(id=thread)

        async def close(self):
            calls.append("close")

    monkeypatch.setattr("runner.api.AsyncCodex", lambda _config: Client())
    runner = CodexRunner()
    turn = Turn(thread_id="discord", user_id="alice", input="")
    first = await runner.start(turn, tmp_path)
    await runner.expire(now=runner.last_used["discord"] + 901)
    second = await runner.start(turn.model_copy(update={"codex_thread_id": first["thread_id"]}), tmp_path)
    assert first == second
    assert calls == ["start", "close", "resume"]


@pytest.mark.asyncio
async def test_runner_http_release_authenticates_and_preserves_shared(tmp_path, monkeypatch):
    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://runner") as client:
        payload = {"thread_id": "thread", "user_id": "alice", "input": ""}
        headers = {"authorization": "Bearer secret"}
        assert (await client.post("/acquire", json=payload, headers=headers)).status_code == 200
        assert (await client.post("/release", json=payload)).status_code == 401
        for _ in range(2):
            assert (await client.post("/release", json=payload, headers=headers)).json() == {"released": True}
    assert not (tmp_path / "users/alice/threads/thread").exists()
    assert (tmp_path / "users/alice/shared/memories.md").exists()


@pytest.mark.asyncio
async def test_runner_rejects_release_with_an_accepted_job(tmp_path, monkeypatch):
    async def run(*_args):
        await asyncio.Event().wait()

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "secret")
    monkeypatch.setattr(CodexRunner, "run", run)
    app = create_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://runner") as client,
    ):
        payload = {"thread_id": "thread", "user_id": "alice", "input": "", "message_id": "message"}
        headers = {"authorization": "Bearer secret"}
        assert (await client.post("/acquire", json=payload, headers=headers)).status_code == 200
        assert (await client.post("/turn", json=payload, headers=headers)).json()["status"] == "running"
        assert (await client.post("/release", json=payload, headers=headers)).status_code == 409
        assert (tmp_path / "users/alice/threads/thread").is_dir()
