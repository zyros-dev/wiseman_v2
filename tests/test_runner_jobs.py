# Copyright (c) 2026 Nick van der Merwe
import asyncio

import httpx
import pytest

from app.runner import MESSAGE_ID, HttpRunner
from runner.api import create_app
from runner.jobs import Jobs


@pytest.mark.asyncio
async def test_job_survives_caller_disconnect_and_replays_result(tmp_path):
    jobs = Jobs(tmp_path)
    started, finish = asyncio.Event(), asyncio.Event()
    calls = 0

    async def work() -> dict[str, object]:
        nonlocal calls
        calls += 1
        started.set()
        await finish.wait()
        return {"output": "built", "thread_id": "codex"}

    assert jobs.submit("message", "owner/thread", work).status == "running"
    await started.wait()
    assert jobs.submit("message", "owner/thread", work).status == "running"
    finish.set()
    await jobs.tasks["message"]
    assert jobs.get("message").result == {"output": "built", "thread_id": "codex"}
    restored = Jobs(tmp_path)
    assert restored.submit("message", "owner/thread", work).result == jobs.get("message").result
    assert calls == 1
    with pytest.raises(ValueError, match="different workspace"):
        restored.submit("message", "other/thread", work)


@pytest.mark.asyncio
async def test_restart_never_reexecutes_uncertain_job(tmp_path):
    jobs = Jobs(tmp_path)
    started = asyncio.Event()

    async def work():
        started.set()
        await asyncio.Event().wait()
        return {}

    jobs.submit("message", "owner/thread", work)
    await started.wait()
    restored = Jobs(tmp_path)
    receipt = restored.submit("message", "owner/thread", work)
    assert receipt.status == "failed"
    assert "restarted" in receipt.error
    assert not restored.tasks
    await jobs.close()
    assert jobs.get("message").status == "failed"


@pytest.mark.asyncio
async def test_distinct_jobs_execute_concurrently_and_failures_are_retained(tmp_path):
    jobs = Jobs(tmp_path)
    waiting = asyncio.Event()

    async def slow() -> dict[str, object]:
        await waiting.wait()
        return {"output": "slow"}

    async def failure():
        raise RuntimeError("provider disconnected")

    jobs.submit("one", "owner/thread", slow)
    jobs.submit("two", "owner/other", failure)
    await jobs.tasks["two"]
    assert jobs.get("two").error == "provider disconnected"
    assert jobs.get("one").status == "running"
    waiting.set()
    await jobs.tasks["one"]


@pytest.mark.asyncio
async def test_http_disconnect_reattaches_without_repeating_execution(tmp_path, monkeypatch):
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def run(self, turn, path, account):
        calls.append(turn.message_id)
        started.set()
        await finish.wait()
        return {"thread_id": "codex", "output": turn.input}

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "secret")
    monkeypatch.setattr("runner.api.CodexRunner.run", run)
    server = httpx.ASGITransport(app=create_app())

    class DropResponse(httpx.AsyncBaseTransport):
        dropped = False

        async def handle_async_request(self, request):
            response = await server.handle_async_request(request)
            if request.url.path == "/turn" and not self.dropped:
                self.dropped = True
                raise httpx.RemoteProtocolError("Server disconnected without sending a response")
            return response

    transport = DropResponse()
    client_type = httpx.AsyncClient
    monkeypatch.setattr("app.runner.httpx.AsyncClient", lambda **kwargs: client_type(transport=transport, **kwargs))
    runner = HttpRunner("http://sandbox", "secret")
    token = MESSAGE_ID.set("initial")
    try:
        with pytest.raises(httpx.RemoteProtocolError, match="disconnected"):
            await runner.run("", "build", "owner", "thread")
        await started.wait()
        finish.set()
        assert (await runner.run("", "build", "owner", "thread"))[1] == "build"
        assert calls == ["initial"]
    finally:
        MESSAGE_ID.reset(token)
    token = MESSAGE_ID.set("followup")
    try:
        assert (await runner.run("codex", "verify", "owner", "thread"))[1] == "verify"
        assert calls == ["initial", "followup"]
    finally:
        MESSAGE_ID.reset(token)
