"""Contract tests for the canonical raw Discord and runner paths."""

# Copyright (c) 2026 Nick van der Merwe

import json
import os
from datetime import UTC, datetime
from typing import Any, Self

import httpx
import pytest
from fastapi.testclient import TestClient
from temporalio.exceptions import WorkflowAlreadyStartedError

from app.main import (
    Engine,
    FakeRunner,
    Gateway,
    HttpRunner,
    Phoenix,
    PromptHub,
    State,
    _banner,
    _history,
    _provider_values,
    create_app,
    normalize_event,
)
from app.temporal_runtime import TemporalError, TemporalRuntime, ThreadWorkflow, run_turn
from runner.api import ApprovalMode, Sandbox, Workspace
from runner.api import create_app as runner_app


def message(
    mid: str, content: str, channel: str = "parent", thread: str | None = None
) -> dict[str, object]:
    return {
        "id": mid,
        "author_id": "u",
        "author_name": "Nick",
        "content": content,
        "channel_id": channel,
        "thread_id": thread,
        "timestamp": mid,
    }


def discord_message(
    mid: str, content: str, channel: str = "parent", thread: str | None = None
) -> dict[str, object]:
    return {
        "id": mid,
        "author": {"id": "u", "username": "Nick"},
        "content": content,
        "channel_id": channel,
        "thread_id": thread,
        "timestamp": mid,
        "mentions": [],
        "attachments": [],
    }


def test_raw_discord_initial_and_followup_use_distinct_contexts() -> None:
    engine = Engine(Phoenix(), FakeRunner())
    app = create_app(engine)
    client = TestClient(app)
    headers = {"x-replay-token": ""}
    assert client.get("/metrics").status_code == 200
    startup = {
        "trigger": discord_message("1", "hello", thread="t"),
        "kind": "startup",
        "parent_messages": [discord_message(str(i), f"old-{i}") for i in range(120)],
    }
    first = client.post("/v1/discord/events", headers=headers, json=startup)
    assert first.status_code == 200
    followup = {
        "trigger": {
            **discord_message("2", "next", thread="t"),
            "message_reference": {"message_id": "1"},
        },
        "kind": "followup",
        "parent_messages": [discord_message("1", "hello"), discord_message("123", "new-parent")],
        "thread_messages": [discord_message("124", "new-thread", thread="t")],
    }
    second = client.post("/v1/discord/events", headers=headers, json=followup)
    assert second.status_code == 200
    assert first.json()["kind"] == "startup"
    assert second.json()["kind"] == "followup"
    assert "1" in second.json()["selected_ids"]
    assert {"123", "124", "2"}.issubset(second.json()["selected_ids"])
    context = [item for item in engine.phoenix.records if item["node"] == "context"][-1]
    assert context["normalized"]["reply_ancestors"][0]["id"] == "1"
    assert "1" not in {item["id"] for item in context["normalized"]["surrounding"]}
    assert engine.reactions["1"] == ["👀", "✅"]
    duplicate = client.post("/v1/discord/events", headers=headers, json=startup)
    assert duplicate.json()["status"] == "duplicate"
    assert duplicate.json()["state"]["turn"] == 2
    grammar = next(item for item in engine.phoenix.records if item["node"] == "grammar")
    assert {"source", "raw", "normalized", "rendered", "version"} <= grammar.keys()
    assert grammar["parsed"]["schema"] == "wiseman.context.grammar.v2"
    codex = [item for item in engine.phoenix.records if item["node"] == "codex"]
    assert codex[0]["model"] == "local-fake"
    assert "old-" in str(codex[0]["input"])
    assert "new-parent" in str(codex[1]["input"])
    assert "old-" not in str(codex[1]["input"])
    first_prompt = json.loads(str(codex[0]["input"]))
    assert json.loads(first_prompt["context"])["messages"]
    assert not engine.phoenix.roots
    assert len(engine.phoenix.records) >= 8


def test_normalize_discord_gateway_message_create_envelope() -> None:
    payload = {"op": 0, "t": "MESSAGE_CREATE", "d": discord_message("gateway", "hello")}
    event = normalize_event(payload)
    assert event.trigger.id == "gateway"
    assert event.raw_payload == payload


def test_reaction_state_is_idempotent() -> None:
    engine = Engine(Phoenix(), FakeRunner())
    assert engine._react("message", "👀")  # noqa: SLF001
    assert not engine._react("message", "👀")  # noqa: SLF001
    assert engine.reactions["message"] == ["👀"]


def test_workspace_link_cannot_escape_owner(tmp_path) -> None:
    workspace = Workspace(str(tmp_path))
    path = workspace.thread("user", "thread")
    assert (path / "shared").resolve() == (tmp_path / "users/user/shared").resolve()
    assert "shared/AGENTS.md" in (path / "AGENTS.md").read_text()
    with pytest.raises(ValueError, match="owner escapes root"):
        workspace.thread("../outside", "thread")
    with pytest.raises(ValueError, match="workspace path escapes owner"):
        workspace.thread("user", "../outside")
    (path / "shared").unlink()
    (path / "shared").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="invalid shared link"):
        workspace.thread("user", "thread")


def test_nested_provider_event_preserves_model_usage_and_cost() -> None:
    usage, cost, model = _provider_values(
        {
            "type": "response.completed",
            "response": {"model": "served-model", "usage": {"input_tokens": 4, "cost": 0.02}},
        },
        None,
        None,
        None,
    )
    assert usage == {"input_tokens": 4, "cost": 0.02}
    assert cost == 0.02
    assert model == "served-model"


def test_banner_omits_unknowns_and_shows_configured_route(monkeypatch) -> None:
    monkeypatch.setenv(
        "WISEMAN_ROUTE_INFO",
        json.dumps({"requested_model": "model", "provider": "provider", "input_price": "$1/M"}),
    )
    banner = _banner()
    assert "`model` via provider" in banner
    assert "Input: `$1/M`" in banner
    assert "Context:" not in banner


@pytest.mark.asyncio
async def test_http_runner_forwards_thread_and_returns_billing(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"thread_id": "next", "output": "answer", "model": "served", "cost": 0.1}

    class Client:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def post(self, url: str, **kwargs: object) -> Response:
            assert url.endswith("/turn")
            body = kwargs["json"]
            assert isinstance(body, dict)
            assert body["thread_id"] == "workspace"
            assert body["codex_thread_id"] == "old"
            return Response()

    monkeypatch.setattr("app.main.httpx.AsyncClient", lambda **kwargs: Client())
    result = await HttpRunner("http://runner", "secret").run("old", "prompt", "user", "workspace")
    assert result == ("next", "answer", {"model": "served", "cost": 0.1})


@pytest.mark.asyncio
async def test_failure_keeps_processing_reaction_and_records_error() -> None:
    class FailingRunner:
        async def run(
            self, thread: str, prompt: str, user: str, workspace: str = ""
        ) -> tuple[str, str, dict[str, object]]:
            del thread, prompt, user, workspace
            raise RuntimeError("runner down")  # noqa: TRY003

    engine = Engine(Phoenix(), FailingRunner())
    result = await engine.handle(normalize_event(discord_message("failure", "hello", thread="t")))
    assert result["error"] == "runner down"
    assert result["reactions"] == ["👀", "❌"]
    assert any(item["node"] == "failure" for item in engine.phoenix.records)


@pytest.mark.asyncio
async def test_closed_thread_rejects_new_work() -> None:
    engine = Engine(Phoenix(), FakeRunner())
    engine.states["t"] = State(closed=True)
    result = await engine.handle(normalize_event(discord_message("closed", "hello", thread="t")))
    assert result["error"] == "thread is closed"


@pytest.mark.asyncio
async def test_temporal_activity_restores_seen_state(monkeypatch) -> None:
    engine = Engine(Phoenix(), FakeRunner())
    monkeypatch.setattr("app.main.engine", engine)
    result = await run_turn(
        {
            "event": {
                "trigger": message("activity", "hello", thread="t"),
                "kind": "followup",
                "parent_messages": [],
                "thread_messages": [],
            },
            "state": {"seen": ["old"], "turn": 1},
        }
    )
    assert result["state"]["turn"] == 2


@pytest.mark.asyncio
async def test_temporal_submit_without_client_is_explicit() -> None:
    with pytest.raises(TemporalError):
        await TemporalRuntime("temporal", "wiseman").submit(
            {"trigger": {"id": "m", "channel_id": "c"}}
        )


@pytest.mark.asyncio
async def test_live_delivery_sends_banner_progress_and_answer(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_ROUTE_INFO", json.dumps({"requested_model": "model"}))

    class Channel:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, content: str) -> None:
            self.sent.append(content)

    class Live:
        def __init__(self) -> None:
            self.channel = Channel()
            self.reactions: list[str] = []

        async def add_reaction(self, emoji: str) -> None:
            self.reactions.append(emoji)

    live: Any = Live()
    result = await Engine(Phoenix(), FakeRunner()).handle(
        normalize_event(discord_message("live", "hello", thread="t")), live
    )
    assert result["reactions"] == ["👀", "✅"]
    assert live.reactions == ["👀", "✅"]
    assert "model" in live.channel.sent[0]
    assert live.channel.sent[1:] == [
        "🛠️ Workspace provisioning...",
        "🤖 Codex starting...",
        "✍️ Finishing the response...",
        live.channel.sent[-1],
    ]


@pytest.mark.asyncio
async def test_history_normalizes_discord_fields() -> None:
    class Author:
        id, name, bot = "u", "User", False

    class Attachment:
        id, filename, url, content_type, size = "a", "a.txt", "https://a", "text/plain", 3

    class Item:
        id, author, content, created_at = (
            "m",
            Author(),
            "hello",
            __import__("datetime").datetime.now(),
        )
        reference = None
        mentions = ()
        attachments = (Attachment(),)

    class Channel:
        id = 42

        async def history(self, **kwargs: object):
            assert kwargs == {"limit": 2}
            yield Item()

    messages = await _history(Channel(), 2)
    assert messages[0].channel_id == "42"
    assert messages[0].attachments[0]["filename"] == "a.txt"


def test_app_health_and_replay_authentication() -> None:
    client = TestClient(create_app(Engine(Phoenix(), FakeRunner()), token="secret"))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.get("/v1/phoenix/events").json() == []
    assert client.post("/v1/replay/discord", json={}).status_code == 401
    assert (
        client.post("/v1/replay/discord", headers={"x-replay-token": "secret"}, json={}).status_code
        == 422
    )


def test_app_starts_discord_with_discord_token_not_replay_token(monkeypatch) -> None:
    received: list[str] = []

    async def fake_run(_bot: Gateway, token: str) -> None:
        received.append(token)

    monkeypatch.setattr(Gateway, "run_forever", fake_run)
    with TestClient(
        create_app(Engine(Phoenix(), FakeRunner()), token="replay", discord_token="discord")
    ):
        pass
    assert received == ["discord"]


@pytest.mark.asyncio
async def test_gateway_retries_a_fatal_session_error(monkeypatch) -> None:
    bot = Gateway(Engine(Phoenix(), FakeRunner()), {1})
    calls = 0

    async def fake_start(_token: str, **kwargs: object) -> None:
        nonlocal calls
        assert kwargs["reconnect"] is True
        calls += 1
        if calls > 1:
            await bot.close()
            return
        raise RuntimeError

    async def fake_sleep(_delay: float) -> None:
        del _delay

    monkeypatch.setattr(bot, "start", fake_start)
    monkeypatch.setattr("app.main.asyncio.sleep", fake_sleep)
    await bot.run_forever("discord")
    assert calls == 2


def test_gateway_requests_only_enabled_discord_intents() -> None:
    bot = Gateway(Engine(Phoenix(), FakeRunner()), {1})
    assert bot.intents.message_content
    assert not bot.intents.members
    assert not bot.intents.presences


def test_phoenix_provider_relay_records_wrapped_billing(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None: ...

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def aiter_lines(self):
            yield 'data: {"response":{"model":"served","usage":{"cost":0.4}}}'
            yield "data: [DONE]"

    class Client:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        def stream(self, method: str, url: str, **kwargs: object) -> Response:
            assert method == "POST"
            assert url.endswith("/api/v1/responses")
            assert kwargs["json"] == {"model": "requested"}
            return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setattr("app.main.httpx.AsyncClient", lambda **kwargs: Client())
    engine = Engine(Phoenix(), FakeRunner())
    response = TestClient(create_app(engine)).post("/v1/responses", json={"model": "requested"})
    assert response.status_code == 200
    provider = engine.phoenix.records[-1]
    assert provider["served_model"] == "served"
    assert provider["cost"] == 0.4
    assert provider["request"] == {"model": "requested"}


@pytest.mark.asyncio
async def test_temporal_workflow_processes_one_turn_then_times_out(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    await workflow.submit({"id": "queued"})

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {"state": {"turn": 1}}

    async def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", timeout)
    result = await workflow.run({"event": {"id": "first"}})
    assert result == {"state": {"turn": 1}}


@pytest.mark.asyncio
async def test_gateway_uses_same_admission_for_parent_and_thread(monkeypatch) -> None:  # noqa: C901
    class Guild:
        def __init__(self, gid: int) -> None:
            self.id = gid

    class User:
        id, name, bot = 7, "Nick", False

    class Channel:
        id, parent_id = 1, None

        def __init__(self) -> None:
            self.guild = Guild(1)
            self.sent: list[str] = []
            self.thread: Thread | None = None

        async def history(self, **kwargs: object):
            del kwargs
            yield raw_message("parent-history", self)

        async def send(self, content: str) -> None:
            self.sent.append(content)

    class Thread:
        parent_id = 1

        def __init__(self, parent: Channel) -> None:
            self.id, self.parent, self.guild = 2, parent, parent.guild
            self.sent: list[str] = []

        async def history(self, **kwargs: object):
            del kwargs
            yield raw_message("thread-history", self)

        async def send(self, content: str) -> None:
            self.sent.append(content)

        async def edit(self, **kwargs: object) -> None:
            del kwargs

    class Message:
        def __init__(self, mid: str, channel: Any, content: str) -> None:
            self.id, self.channel, self.content = mid, channel, content
            self.author, self.mentions = User(), [bot_user]
            self.created_at = datetime.now(UTC)
            self.attachments, self.reference = (), None
            self.reactions: list[str] = []

        async def create_thread(self, name: str) -> Thread:
            assert name == "wiseman"
            self.channel.thread = Thread(self.channel)
            return self.channel.thread

        async def add_reaction(self, emoji: str) -> None:
            self.reactions.append(emoji)

    def raw_message(mid: str, channel: Any) -> Any:
        item = Message(mid, channel, "context")
        item.mentions = []
        return item

    parent = Channel()
    bot_user = User()
    engine = Engine(Phoenix(), FakeRunner())
    monkeypatch.setattr("app.main.discord.Thread", Thread)
    bot = __import__("app.main", fromlist=["Gateway"]).Gateway(engine, {1})
    bot._connection.user = bot_user  # noqa: SLF001
    startup = Message("start", parent, "hello")
    await bot.on_message(startup)
    followup = Message("follow", parent.thread, "next")
    await bot.on_message(followup)
    assert engine.states["2"].turn == 2
    assert startup.reactions == ["👀", "✅"]
    assert followup.reactions == ["👀", "✅"]
    assert parent.sent
    assert parent.thread is not None
    assert parent.thread.sent

    rejected = Channel()
    rejected.guild = Guild(2)
    ignored = Message("ignored", rejected, "hello")
    await bot.on_message(ignored)
    assert "ignored" not in engine.states


def test_replay_accepts_discord_message_json() -> None:
    engine = Engine(Phoenix(), FakeRunner())
    client = TestClient(create_app(engine))
    payload = {
        "id": "discord-1",
        "author": {"id": "u", "username": "nick"},
        "content": "hello",
        "channel_id": "channel",
        "thread_id": "thread",
        "timestamp": "2026-09-02T00:00:00Z",
        "mentions": [{"id": "bot"}],
        "attachments": [{"id": "file", "filename": "x.png", "url": "https://x/x.png"}],
        "kind": "startup",
        "parent_messages": [],
    }
    response = client.post("/v1/discord/events", json=payload)
    assert response.status_code == 200
    context = next(item for item in engine.phoenix.records if item["node"] == "context")
    assert "file" in str(context["raw"])
    assert "username" in str(context["raw"])


def test_runner_requires_bearer_and_materializes_shared_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "secret")
    client = TestClient(runner_app())
    payload = {"thread_id": "t", "user_id": "u", "input": "hi"}
    assert client.post("/acquire", json=payload).status_code == 401
    response = client.post("/acquire", headers={"authorization": "Bearer secret"}, json=payload)
    assert response.status_code == 200
    assert (tmp_path / "users/u/shared/memories.md").exists()


def test_cleanup_removes_old_thread_but_keeps_shared(tmp_path) -> None:
    workspace = Workspace(str(tmp_path))
    path = workspace.thread("user", "old")
    os.utime(path, (0, 0))
    assert workspace.cleanup(1) == 1
    assert (tmp_path / "users/user/shared").exists()


def test_managed_account_name_is_stable_without_touching_host_accounts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("WISEMAN_MANAGE_ACCOUNTS", "1")
    workspace = Workspace(str(tmp_path))
    commands: list[list[str]] = []
    monkeypatch.setattr("runner.api.pwd.getpwnam", lambda name: (_ for _ in ()).throw(KeyError))
    monkeypatch.setattr(workspace, "_admin", commands.append)
    monkeypatch.setattr("runner.api.shutil.chown", lambda *args, **kwargs: None)
    workspace.thread("discord-user", "thread")
    assert commands[0][0].endswith("groupadd")
    assert commands[1][0].endswith("useradd")


@pytest.mark.asyncio
async def test_prompt_hub_reads_phoenix_latest_version(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"data": {"model_name": "model", "template": {"messages": []}}}

    class Client:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def get(self, url: str, **kwargs: object) -> Response:
            assert url.endswith("/v1/prompts/startup/latest")
            assert kwargs["headers"] == {"Authorization": "Bearer key"}
            return Response()

    monkeypatch.setattr("app.main.httpx.AsyncClient", lambda **kwargs: Client())
    source = await PromptHub("http://phoenix", "key").source("startup")
    assert '"model": "model"' in source


def test_runner_starts_and_resumes_codex_thread(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class Result:
        usage = None

        def __init__(self, text: str) -> None:
            self.final_response = text

    class Thread:
        id = "codex-thread"

        async def run(self, prompt: str, **kwargs: object) -> Result:
            return Result(prompt)

    class Codex:
        def __init__(self, config: object) -> None:
            del config
            self.starts = 0
            self.resumes = 0

        async def thread_start(self, **kwargs: object) -> Thread:
            self.starts += 1
            calls.append(kwargs)
            return Thread()

        async def thread_resume(self, thread: str, **kwargs: object) -> Thread:
            self.resumes += 1
            assert thread == "codex-thread"
            calls.append(kwargs)
            return Thread()

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "secret")
    monkeypatch.setenv("WISEMAN_RELAY_URL", "http://relay/v1")
    monkeypatch.setenv("WISEMAN_MODEL", "provider/model")
    monkeypatch.setattr("runner.api.AsyncCodex", Codex)
    client = TestClient(runner_app())
    headers = {"authorization": "Bearer secret"}
    first = client.post(
        "/turn", headers=headers, json={"thread_id": "t", "user_id": "u", "input": "one"}
    )
    second = client.post(
        "/turn",
        headers=headers,
        json={"thread_id": "t", "codex_thread_id": "codex-thread", "user_id": "u", "input": "two"},
    )
    assert first.json()["thread_id"] == second.json()["thread_id"] == "codex-thread"
    assert calls[0]["approval_mode"] is ApprovalMode.deny_all
    assert calls[0]["sandbox"] is Sandbox.full_access
    assert calls[0]["model"] == "provider/model"
    assert calls[0]["model_provider"] == "wiseman-relay"
    assert (
        'base_url = "http://relay/v1"'
        in (tmp_path / "users/u/threads/t/.codex/config.toml").read_text()
    )


@pytest.mark.asyncio
async def test_http_runner_reaches_sandbox_for_start_and_followup(tmp_path, monkeypatch) -> None:
    class Result:
        usage = None

        def __init__(self, text: str) -> None:
            self.final_response = text

    class Thread:
        id = "codex-thread"

        async def run(self, prompt: str, **kwargs: object) -> Result:
            del kwargs
            return Result(prompt)

    class Codex:
        def __init__(self, config: object) -> None:
            del config

        async def thread_start(self, **kwargs: object) -> Thread:
            del kwargs
            return Thread()

        async def thread_resume(self, thread: str, **kwargs: object) -> Thread:
            del kwargs
            assert thread == "codex-thread"
            return Thread()

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "secret")
    monkeypatch.setattr("runner.api.AsyncCodex", Codex)
    transport = httpx.ASGITransport(app=runner_app())
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "app.main.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=transport, base_url="http://sandbox", **kwargs),
    )
    runner = HttpRunner("http://sandbox", "secret")
    first = await runner.run("", "one", "u", "t")
    second = await runner.run(first[0], "two", "u", "t")
    assert first[0] == second[0] == "codex-thread"
    assert "one" in first[1]
    assert "two" in second[1]
    assert (tmp_path / "users/u/shared/AGENTS.md").exists()
    assert (tmp_path / "users/u/threads/t/.codex").is_dir()


@pytest.mark.asyncio
async def test_temporal_routes_followup_to_existing_thread() -> None:
    class Handle:
        def __init__(self) -> None:
            self.event: dict[str, object] | None = None

        async def signal(self, method: object, event: dict[str, object]) -> None:
            del method
            self.event = event

    class Client:
        def __init__(self) -> None:
            self.started = False
            self.handle = Handle()

        async def start_workflow(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            if self.started:
                raise WorkflowAlreadyStartedError("wiseman-t", "wiseman.thread")
            self.started = True

        def get_workflow_handle(self, workflow_id: str) -> Handle:
            assert workflow_id == "wiseman-t"
            return self.handle

    runtime = TemporalRuntime("temporal", "wiseman")
    runtime.client = Client()
    event = {"trigger": {"id": "m", "channel_id": "c", "thread_id": "t"}}
    await runtime.submit(event)
    await runtime.submit({"trigger": {"id": "m2", "channel_id": "c", "thread_id": "t"}})
    assert runtime.client.handle.event is not None
    trigger = runtime.client.handle.event["trigger"]
    assert isinstance(trigger, dict)
    assert trigger["id"] == "m2"
