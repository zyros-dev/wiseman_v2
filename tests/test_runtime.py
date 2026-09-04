# Copyright (c) 2026 Nick van der Merwe

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast

import discord
import httpx
import pytest
from fastapi.testclient import TestClient
from openai_codex import CodexConfig
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    CommandExecutionOutputDeltaNotification,
    ItemCompletedNotification,
    ThreadItem,
    TurnCompletedNotification,
    TurnStartedNotification,
    TurnStatus,
)
from openai_codex.generated.v2_all import (
    Turn as CodexTurn,
)
from openai_codex.models import Notification
from temporalio.converter import JSONPlainPayloadConverter
from temporalio.exceptions import WorkflowAlreadyStartedError

from app.admission import ContextConfig, context, image_tool_instruction, normalize_event
from app.engine import Engine, EngineConfig
from app.gateway import Gateway, _history, mention_ids
from app.http_api import create_app
from app.models import Event
from app.phoenix import Phoenix, PromptHub, provider_values
from app.presentation import (
    banner,
    deliver_content,
    describe_images,
    normalize_image_url,
    render_progress,
    split_discord_content,
    thread_name,
)
from app.runner import FakeRunner, HttpRunner, Runner
from app.temporal_runtime import (
    TRANSPORT_RETRY_POLICY,
    TemporalRuntime,
    ThreadWorkflow,
    TurnWorkflow,
    fail_turn,
    provision_workspace,
    publish_progress,
    run_turn,
    start_codex,
)
from app.types import EngineResult, JsonObject, StateData
from runner.api import (
    CODEX_TEXT_ONLY_OVERRIDES,
    ApprovalMode,
    CodexRunner,
    Sandbox,
    Turn,
    Workspace,
    _progress_message,
)
from runner.api import create_app as runner_app


class EchoHandle:
    def __init__(self, text: str) -> None:
        self.text = text

    async def stream(self):
        yield Notification(
            "item/completed",
            ItemCompletedNotification(
                completed_at_ms=1,
                thread_id="codex-thread",
                turn_id="turn",
                item=ThreadItem(root=AgentMessageThreadItem(id="answer", text=self.text, type="agentMessage")),
            ),
        )
        yield Notification(
            "turn/completed",
            TurnCompletedNotification(
                thread_id="codex-thread",
                turn=CodexTurn(id="turn", items=[], status=TurnStatus.completed),
            ),
        )


def message(mid: str, content: str, channel: str = "parent", thread: str | None = None) -> JsonObject:
    return {
        "id": mid,
        "author_id": "u",
        "author_name": "Nick",
        "content": content,
        "channel_id": channel,
        "thread_id": thread,
        "timestamp": mid,
    }


def discord_message(mid: str, content: str, channel: str = "parent", thread: str | None = None) -> dict[str, object]:
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


def configured_engine(phoenix: Phoenix | None = None, runner: Runner | None = None) -> Engine:
    return Engine(EngineConfig(phoenix or Phoenix(), runner or FakeRunner(), PromptHub()))


def test_temporal_payload_round_trip_preserves_nested_event() -> None:
    payload = {
        "event": {"trigger": {"id": "message", "channel_id": "channel"}},
        "state": {},
    }
    converter = JSONPlainPayloadConverter()
    encoded = converter.to_payload(payload)
    assert encoded is not None
    decoded = converter.from_payload(encoded, dict)
    assert decoded == payload


def test_raw_discord_initial_and_followup_use_distinct_contexts() -> None:
    engine = configured_engine()
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
    context = cast(
        "dict[str, object]",
        [item for item in engine.config.phoenix.records if item["node"] == "context"][-1],
    )
    normalized = cast("dict[str, object]", context["normalized"])
    ancestors = cast("list[dict[str, object]]", normalized["reply_ancestors"])
    surrounding = cast("list[dict[str, object]]", normalized["surrounding"])
    assert ancestors[0]["id"] == "1"
    assert "1" not in {item["id"] for item in surrounding}
    assert engine.reactions["1"] == ["✅"]
    duplicate = client.post("/v1/discord/events", headers=headers, json=startup)
    assert duplicate.json()["status"] == "duplicate"
    assert duplicate.json()["state"]["turn"] == 2
    grammar = cast(
        "dict[str, object]",
        next(item for item in engine.config.phoenix.records if item["node"] == "grammar"),
    )
    assert {"source", "raw", "normalized", "rendered", "version"} <= grammar.keys()
    assert cast("dict[str, object]", grammar["parsed"])["schema"] == "wiseman.context.grammar.v2"
    codex = [item for item in engine.config.phoenix.records if item["node"] == "codex"]
    assert codex[0]["model"] == "local-fake"
    assert "old-" in str(codex[0]["input"])
    assert "new-parent" in str(codex[1]["input"])
    assert "old-" not in str(codex[1]["input"])
    first_prompt = json.loads(str(codex[0]["input"]))
    assert json.loads(first_prompt["context"])["messages"]
    assert not cast("Phoenix", engine.config.phoenix).roots
    assert len(engine.config.phoenix.records) >= 8


@pytest.mark.asyncio
async def test_image_turn_makes_agent_tool_call_explicit() -> None:
    engine = configured_engine()
    event = normalize_event(
        {
            "trigger": {
                **discord_message("image-turn", "What is in this?", thread="t"),
                "attachments": [
                    {
                        "id": "image-1",
                        "filename": "photo.png",
                        "url": "https://cdn.example/photo.png",
                        "content_type": "image/png",
                    }
                ],
            },
            "parent_messages": [
                {
                    **discord_message("historical-image", "Old image", thread="t"),
                    "attachments": [
                        {
                            "id": "old-image",
                            "filename": "old.png",
                            "url": "https://cdn.example/old.png",
                            "content_type": "image/png",
                        }
                    ],
                }
            ],
            "kind": "startup",
        }
    )
    await engine.handle(event)
    codex = next(item for item in engine.config.phoenix.records if item["node"] == "codex")
    prompt = json.loads(str(codex["input"]))
    assert "/usr/local/bin/wiseman-discord describe-image" in str(codex["input"])
    assert "https://cdn.example/photo.png" in str(codex["input"])
    assert "https://cdn.example/old.png" not in prompt["context"]
    assert "https://cdn.example/old.png" not in prompt["user"]


def test_image_tool_instruction_uses_nearest_referenced_image_only() -> None:
    instruction = image_tool_instruction(
        {
            "content": "what is in this image?",
            "attachments": [],
        },
        [
            {
                "content": "old photo",
                "attachments": [{"content_type": "image/png", "url": "https://cdn.example/old.png"}],
            },
            {
                "content": "latest photo",
                "attachments": [{"content_type": "image/png", "url": "https://cdn.example/latest.png"}],
            },
        ],
    )
    assert "https://cdn.example/latest.png" in instruction
    assert "https://cdn.example/old.png" not in instruction


def test_image_tool_instruction_ignores_historical_images_for_ordinary_text() -> None:
    instruction = image_tool_instruction(
        {"content": "continue the task", "attachments": []},
        [{"attachments": [{"content_type": "image/png", "url": "https://cdn.example/old.png"}]}],
    )
    assert instruction == ""


def test_context_uses_injected_ancestor_bound() -> None:
    messages = [message(str(index), f"m-{index}") for index in range(4)]
    for index in range(1, 4):
        messages[index]["reply_to"] = str(index - 1)
    event = Event(
        trigger=normalize_event({**message("trigger", "question"), "reply_to": "3"}).trigger,
        kind="followup",
        parent_messages=[normalize_event(item).trigger for item in messages],
    )
    current = context(event, ContextConfig(max_ancestors=2))
    ancestors = current["reply_ancestors"]
    assert isinstance(ancestors, list)
    assert [item["id"] for item in ancestors if isinstance(item, dict)] == ["2", "3"]


def test_normalize_discord_gateway_message_create_envelope() -> None:
    payload = {"op": 0, "t": "MESSAGE_CREATE", "d": discord_message("gateway", "hello")}
    event = normalize_event(payload)
    assert event.trigger.id == "gateway"
    assert event.raw_payload == payload


def test_gateway_mention_fallback_reads_raw_discord_content() -> None:
    message = type("Message", (), {"raw_mentions": [], "mentions": [], "content": "<@!42> hi"})()
    assert mention_ids(message) == ["42"]


def test_gateway_mention_fallback_merges_parsed_and_raw_content() -> None:
    other = type("Mention", (), {"id": 7})()
    message = type("Message", (), {"raw_mentions": [other], "mentions": [], "content": "<@42> hi"})()
    assert mention_ids(message) == ["7", "42"]


@pytest.mark.asyncio
async def test_vision_assist_sends_discord_image_to_glm(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "model": "z-ai/glm-5.3-flash",
                "choices": [{"message": {"content": "Three black office chairs."}}],
                "usage": {"cost": 0.01},
            }

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def post(self, url: str, **kwargs: object) -> Response:
            requests.append({"url": url, **kwargs})
            return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setattr("app.http_api.httpx.AsyncClient", Client)
    result = await describe_images(
        [{"attachments": [{"id": "image-1", "filename": "image.jpg", "url": "https://cdn/image.jpg"}]}],
        "How many chairs are visible?",
    )
    assert result["text"] == "Three black office chairs."
    assert result["attachments"] == ["image-1"]
    assert result["question"] == "How many chairs are visible?"
    request = cast("dict[str, object]", requests[0]["json"])
    messages = cast("list[dict[str, object]]", request["messages"])
    content = cast("list[dict[str, object]]", messages[0]["content"])
    prompt = cast("str", content[0]["text"])
    assert request["model"] == "z-ai/glm-5.3-flash"
    assert "How many chairs" in prompt
    assert cast("dict[str, object]", content[1]["image_url"])["url"] == "https://cdn/image.jpg"
    generic = await describe_images(
        [{"attachments": [{"id": "image-1", "filename": "image.jpg", "url": "https://cdn/image.jpg"}]}]
    )
    assert generic["text"] == result["text"]
    assert generic["question"] is None
    generic_request = cast("dict[str, object]", requests[1]["json"])
    generic_messages = cast("list[dict[str, object]]", generic_request["messages"])
    generic_content = cast("list[dict[str, object]]", generic_messages[0]["content"])
    generic_prompt = cast("str", generic_content[0]["text"])
    assert "Please describe this image generally" in generic_prompt
    phoenix = Phoenix()
    await phoenix.record("vision-tool", "vision_tool", **generic)
    assert not phoenix.roots


def test_reaction_state_is_idempotent() -> None:
    engine = configured_engine()
    assert engine._react("message", "👀")
    assert not engine._react("message", "👀")
    assert engine.reactions["message"] == ["👀"]


def test_discord_content_splits_long_answers_at_readable_boundaries() -> None:
    content = "first paragraph\n\n" + ("word " * 600)
    chunks = split_discord_content(content)
    assert len(chunks) > 1
    assert all(len(chunk) <= 2_000 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == content.replace(" ", "")


def test_image_url_normalization_accepts_model_wrappers() -> None:
    assert normalize_image_url(" <https://cdn.example/image.png> ") == ("https://cdn.example/image.png")
    assert normalize_image_url("attachment://image.png") == ""


@pytest.mark.asyncio
async def test_long_delivery_edits_first_chunk_and_sends_overflow() -> None:
    class Channel:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, content: str = "") -> object:
            index = len(self.sent)
            self.sent.append(content)
            channel = self

            class Delivery:
                async def edit(self, *, content: str) -> None:
                    channel.sent[index] = content

            return Delivery()

    channel = Channel()
    progress = await channel.send("working")
    content = "paragraph\n\n" + ("word " * 600)
    await deliver_content(progress, channel, content)
    assert len(channel.sent) >= 2
    assert all(len(chunk) <= 2_000 for chunk in channel.sent)
    assert "".join(channel.sent).replace(" ", "") == content.replace(" ", "")


@pytest.mark.asyncio
async def test_describe_image_route_accepts_wrapped_http_url(monkeypatch) -> None:
    seen: list[str] = []

    async def describe(messages: list[dict[str, object]], question: str = "") -> dict[str, object]:
        attachments = cast("list[dict[str, object]]", messages[0]["attachments"])
        seen.append(str(attachments[0]["url"]))
        return {"text": "description", "attachments": ["image"], "question": question or None}

    monkeypatch.setattr("app.http_api.describe_images", describe)
    response = TestClient(create_app(configured_engine())).post(
        "/v1/tools/describe-image", json={"url": " <https://cdn.example/image.png> "}
    )
    assert response.status_code == 200
    assert seen == ["https://cdn.example/image.png"]


def test_reaction_configuration_changes_future_turns() -> None:
    engine = configured_engine()
    assert engine.set_reaction_emojis({"processing": "🔵", "success": "🟩", "failure": "🟥"}) == {
        "processing": "🔵",
        "success": "🟩",
        "failure": "🟥",
    }
    assert engine.reaction_emojis["processing"] == "🔵"


def test_discord_tools_update_reactions_profile_and_send_file(monkeypatch) -> None:
    class User:
        name = "Wiseman"

        def __init__(self) -> None:
            self.edits: list[dict[str, object]] = []

        async def edit(self, **kwargs: object) -> None:
            self.edits.append(kwargs)

    class Sent:
        id = 42
        jump_url = "https://discord.test/messages/42"

    class Thread:
        async def send(self, **kwargs: object) -> Sent:
            self.payload = kwargs
            return Sent()

    monkeypatch.setenv("WISEMAN_PROVIDER_TOKEN", "secret")
    monkeypatch.setenv("WISEMAN_ALLOW_PROFILE_EDITS", "1")
    monkeypatch.setattr("app.gateway.discord.Thread", Thread)
    engine = configured_engine()
    app = create_app(engine, token="secret")
    app.state.gateway._connection.user = User()

    with TestClient(app) as client:
        headers = {"authorization": "Bearer secret"}
        reactions = client.post(
            "/v1/tools/set-reactions",
            headers=headers,
            json={"processing": "🔵", "success": "🟩", "failure": "🟥"},
        )
        assert reactions.json()["reaction_emojis"] == {
            "processing": "🔵",
            "success": "🟩",
            "failure": "🟥",
        }
        replay = client.post(
            "/v1/replay/discord",
            headers={"x-replay-token": "secret"},
            json={
                "trigger": discord_message("configured", "hello", thread="t"),
                "kind": "startup",
            },
        )
        assert replay.json()["reactions"] == ["🟩"]
        profile = client.post(
            "/v1/tools/set-profile",
            headers=headers,
            json={"username": "New Wiseman", "avatar_base64": "aGVsbG8="},
        )
        assert profile.status_code == 200
        assert app.state.gateway.user.edits == [{"username": "New Wiseman", "avatar": b"hello"}]

        monkeypatch.setattr(
            app.state.gateway,
            "fetch_channel",
            lambda _channel_id: __import__("asyncio").sleep(0, result=Thread()),
        )
        uploaded = client.post(
            "/v1/tools/send-file",
            headers=headers,
            json={
                "thread_id": "123",
                "filename": "report.png",
                "caption": "Here it is",
                "data_base64": "aGVsbG8=",
            },
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["message_id"] == "42"

    assert app.state.gateway.profile_path is None


def test_discord_tools_require_authentication() -> None:
    client = TestClient(create_app(configured_engine(), token="secret"))
    assert client.post("/v1/tools/set-reactions", json={"success": "🎉"}).status_code == 401
    assert client.post("/v1/tools/send-file", json={}).status_code == 401


def test_thread_name_uses_persisted_sequence() -> None:
    assert thread_name(1) == "Gurt 1"
    assert thread_name(42) == "Gurt 42"
    with pytest.raises(ValueError, match="positive"):
        thread_name(0)


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


def test_workspace_ownership_repairs_nested_codex_state_without_following_links(tmp_path, monkeypatch) -> None:
    thread = tmp_path / "thread"
    (thread / ".codex" / ".tmp" / "plugin").mkdir(parents=True)
    volatile = thread / ".codex" / ".tmp" / "plugin" / "removed.md"
    volatile.write_text("plugin", encoding="utf-8")
    (thread / "AGENTS.md").write_text("instructions", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (thread / "outside-link").symlink_to(outside)
    calls: list[Path] = []

    def chown(path: str | os.PathLike[str], **kwargs: object) -> None:
        del kwargs
        calls.append(Path(path))

    monkeypatch.setattr("runner.api.shutil.chown", chown)
    Workspace._own_tree(thread, "wsm_user")

    assert thread in calls
    assert thread / ".codex" in calls
    assert volatile in calls
    assert outside not in calls


def test_nested_provider_event_preserves_model_usage_and_cost() -> None:
    usage, cost, model = provider_values(
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
    rendered = banner()
    assert "`model` via provider" in rendered
    assert "Input: `$1/M`" in rendered
    assert "Context:" not in rendered


@pytest.mark.asyncio
async def test_http_runner_forwards_thread_and_returns_billing(monkeypatch) -> None:
    class Response:
        is_error = False

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

    monkeypatch.setattr("app.http_api.httpx.AsyncClient", lambda **kwargs: Client())
    result = await HttpRunner("http://runner", "secret").run("old", "prompt", "user", "workspace")
    assert result == ("next", "answer", {"model": "served", "cost": 0.1})


@pytest.mark.asyncio
async def test_http_runner_steers_active_turn(monkeypatch) -> None:
    class Response:
        is_error = False

        def json(self) -> dict[str, object]:
            return {"steered": True}

    class Client:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def post(self, url: str, **kwargs: object) -> Response:
            assert url.endswith("/steer")
            body = kwargs["json"]
            assert isinstance(body, dict)
            assert body == {
                "thread_id": "workspace",
                "codex_thread_id": "codex",
                "user_id": "user",
                "input": "use the other file",
            }
            return Response()

    monkeypatch.setattr("app.http_api.httpx.AsyncClient", lambda **kwargs: Client())
    assert await HttpRunner("http://runner", "secret").steer("codex", "use the other file", "user", "workspace")


def test_progress_renderer_keeps_turn_count_and_bounded_recent_steps() -> None:
    rendered = render_progress(["🤖 Codex turn started...", "⚙️ Running command..."], 1)
    assert rendered.startswith("⏳ Working · Gurt 1\n")
    assert rendered.count("Gurt 1") == 1

    bounded = render_progress(["🤖 Codex turn started..."] + ["⚙️ Running command..." for _ in range(10)], 1)
    assert bounded.startswith("⏳ Working · Gurt 1\n")
    assert bounded.count("\n") == 8


def test_progress_message_names_only_actual_turn_starts() -> None:
    codex_turn = CodexTurn(id="turn", items=[], status=TurnStatus.completed)
    started = _progress_message(
        Notification("turn/started", TurnStartedNotification(thread_id="t", turn=codex_turn)),
        3,
    )
    command = _progress_message(
        Notification(
            "item/commandExecution/outputDelta",
            CommandExecutionOutputDeltaNotification(delta="output", item_id="command", thread_id="t", turn_id="turn"),
        ),
        3,
    )
    assert started == "🤖 Gurt 3: Codex turn started..."
    assert command == '⚙️ Running command... "output"'


@pytest.mark.asyncio
async def test_reply_to_active_delivery_is_steering_not_a_second_turn() -> None:
    gate = asyncio.Event()
    steers: list[str] = []

    class Channel:
        async def send(self, content: str = "", **kwargs: object) -> object:
            del kwargs

            class Delivery:
                id = "delivery"

                async def edit(self, *, content: str) -> None:
                    del content

            return Delivery()

    class Runner:
        async def run(
            self,
            thread: str,
            prompt: str,
            user: str,
            workspace: str = "",
            progress: Callable[[str], Awaitable[None]] | None = None,
        ) -> tuple[str, str, dict[str, object]]:
            del thread, prompt, user, workspace, progress
            await gate.wait()
            return "codex", "answer", {}

        async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
            del thread, user, workspace
            steers.append(prompt)
            return True

    engine = configured_engine(runner=Runner())
    task = asyncio.create_task(
        engine.handle(
            normalize_event(discord_message("trigger", "hello", thread="t")),
            delivery_channel=Channel(),
        )
    )
    for _ in range(10):
        if "t" in engine.active_turns:
            break
        await asyncio.sleep(0)
    assert await engine.steer_if_active("t", "delivery", "change direction", "user")
    gate.set()
    await task
    assert steers == ["change direction"]


@pytest.mark.asyncio
async def test_http_runner_forwards_live_progress(monkeypatch) -> None:
    class Response:
        is_error = False
        status_code = 200

        def __init__(self, body: dict[str, object]) -> None:
            self.body = body

        def json(self) -> dict[str, object]:
            return self.body

        def raise_for_status(self) -> None: ...

    class Client:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def post(self, url: str, **kwargs: object) -> Response:
            del url, kwargs
            return Response({"status": "running"})

        async def get(self, url: str, **kwargs: object) -> Response:
            assert "/jobs/" in url
            del kwargs
            return Response(
                {
                    "status": "completed",
                    "steps": ["🤖 Codex starting..."],
                    "result": {"thread_id": "next", "output": "answer", "model": "served"},
                }
            )

    monkeypatch.setattr("app.http_api.httpx.AsyncClient", lambda **kwargs: Client())
    updates: list[str] = []

    async def receive(message: str) -> None:
        updates.append(message)

    result = await HttpRunner("http://runner", "secret").run("old", "prompt", "user", "workspace", progress=receive)
    assert result == ("next", "answer", {"model": "served"})
    assert updates == ["🤖 Codex starting..."]


@pytest.mark.asyncio
async def test_failure_keeps_processing_reaction_and_records_error() -> None:
    class FailingRunner:
        async def run(
            self,
            thread: str,
            prompt: str,
            user: str,
            workspace: str = "",
            progress: Callable[[str], Awaitable[None]] | None = None,
        ) -> tuple[str, str, dict[str, object]]:
            del thread, prompt, user, workspace, progress
            raise RuntimeError("runner down")

    engine = configured_engine(runner=FailingRunner())
    result = await engine.handle(normalize_event(discord_message("failure", "hello", thread="t")))
    assert result["error"] == "runner down"
    assert result["reactions"] == ["❌"]
    assert any(item["node"] == "failure" for item in engine.config.phoenix.records)


@pytest.mark.asyncio
async def test_discord_reaction_failures_do_not_abort_terminal_delivery() -> None:
    class Delivery:
        async def edit(self, *, content: str) -> None:
            del content

    class Channel:
        async def send(self, content: str = "", **kwargs: object) -> Delivery:
            del content, kwargs
            return Delivery()

    class Live:
        channel = Channel()

        async def add_reaction(self, emoji: str) -> None:
            del emoji
            raise discord.DiscordException

        async def remove_reaction(self, emoji: str, member: object) -> None:
            del emoji, member
            raise discord.DiscordException

    engine = configured_engine()
    engine.reaction_user = object()
    result = await engine.handle(
        normalize_event(discord_message("reaction-failure", "hello", thread="t")),
        cast("discord.Message", Live()),
    )
    assert result["output"].startswith("Codex received: ")
    assert result["reactions"] == ["✅"]


@pytest.mark.asyncio
async def test_discord_delivery_failure_does_not_fail_turn() -> None:
    class Channel:
        async def send(self, content: str = "", **kwargs: object) -> object:
            del content, kwargs
            raise discord.DiscordException

    engine = configured_engine()
    result = await engine.handle(
        normalize_event(discord_message("delivery-failure", "hello", thread="t")),
        delivery_channel=Channel(),
    )
    assert result["output"].startswith("Codex received: ")
    assert result["reactions"] == ["✅"]
    assert any(item["node"] == "delivery" for item in engine.config.phoenix.records)


@pytest.mark.asyncio
async def test_temporal_transport_retry_reuses_startup_delivery() -> None:
    class Runner:
        attempts = 0

        class DisconnectionError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("Server disconnected without sending a response")

        async def run(
            self,
            thread: str,
            prompt: str,
            user: str,
            workspace: str = "",
            progress: Callable[[str], Awaitable[None]] | None = None,
        ) -> tuple[str, str, dict[str, object]]:
            del thread, prompt, user, workspace, progress
            self.attempts += 1
            if self.attempts == 1:
                raise self.DisconnectionError
            return "codex", "answer", {}

    class Delivery:
        id = "progress"

        async def edit(self, *, content: str) -> None:
            del content

    class Channel:
        def __init__(self) -> None:
            self.embeds = 0
            self.progress = 0

        async def send(self, content: str = "", **kwargs: object) -> Delivery:
            del content
            if kwargs.get("embed") is not None:
                self.embeds += 1
            else:
                self.progress += 1
            return Delivery()

    channel = Channel()
    engine = configured_engine(runner=Runner())
    event = normalize_event(discord_message("retry-delivery", "hello", thread="t"))
    with pytest.raises(RuntimeError, match="disconnected"):
        await engine.handle(event, delivery_channel=channel, state_data={}, retry_transport=True)
    result = await engine.handle(event, delivery_channel=channel, state_data={})
    assert result["output"] == "answer"
    assert channel.embeds == 1
    assert channel.progress == 1


@pytest.mark.asyncio
async def test_same_thread_turns_are_serialized() -> None:
    active = maximum = 0

    class Runner:
        async def run(
            self,
            thread: str,
            prompt: str,
            user: str,
            workspace: str = "",
            progress: Callable[[str], Awaitable[None]] | None = None,
        ) -> tuple[str, str, dict[str, object]]:
            nonlocal active, maximum
            del prompt, user, workspace, progress
            active += 1
            maximum = max(maximum, active)
            await __import__("asyncio").sleep(0)
            active -= 1
            return thread or "codex", "answer", {}

    engine = configured_engine(runner=Runner())
    events = [normalize_event(discord_message(str(index), "hello", thread="same")) for index in (1, 2)]
    state: JsonObject = {}
    results = []
    for event in events:
        result = await engine.handle(event, state_data=state)
        results.append(result)
        state = cast("JsonObject", result["state"])
    assert maximum == 1
    assert [result["state"]["turn"] for result in results] == [1, 2]


@pytest.mark.asyncio
async def test_new_turn_recovers_after_previous_failure() -> None:
    attempts = 0

    class Runner:
        async def run(
            self,
            thread: str,
            prompt: str,
            user: str,
            workspace: str = "",
            progress: Callable[[str], Awaitable[None]] | None = None,
        ) -> tuple[str, str, dict[str, object]]:
            nonlocal attempts
            del prompt, user, workspace, progress
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary runner failure")
            return thread or "codex", "recovered", {}

    engine = configured_engine(runner=Runner())
    first = await engine.handle(normalize_event(discord_message("failed", "hello", thread="recover")), state_data={})
    second = await engine.handle(
        normalize_event(discord_message("recovered", "retry", thread="recover")),
        state_data=cast("JsonObject", first["state"]),
    )
    assert first["reactions"] == ["❌"]
    assert second["reactions"] == ["✅"]
    assert second["output"] == "recovered"


@pytest.mark.asyncio
async def test_closed_thread_rejects_new_work() -> None:
    engine = configured_engine()
    result = await engine.handle(
        normalize_event(discord_message("closed", "hello", thread="t")),
        state_data={"closed": True},
    )
    assert result["error"] == "thread is closed"


@pytest.mark.asyncio
async def test_temporal_activity_restores_seen_state(monkeypatch) -> None:
    engine = configured_engine()
    monkeypatch.setattr("app.temporal_runtime._activity_runtime.engine", engine)
    result = cast(
        "EngineResult",
        await run_turn(
            {
                "event": {
                    "trigger": message("activity", "hello", thread="t"),
                    "kind": "followup",
                    "parent_messages": [],
                    "thread_messages": [],
                },
                "state": {"seen": ["old"], "turn": 1},
            }
        ),
    )
    assert result["state"]["turn"] == 2


@pytest.mark.asyncio
async def test_temporal_retry_preserves_original_request(monkeypatch) -> None:
    seen: list[str] = []

    class Engine:
        async def handle(
            self,
            event: Event,
            state_data: dict[str, object] | None = None,
            *,
            retry_transport: bool = False,
        ) -> dict[str, object]:
            del state_data, retry_transport
            seen.append(event.trigger.content)
            return {"state": {"turn": 1}}

    class ActivityInfo:
        attempt = 2

    monkeypatch.setattr("app.temporal_runtime._activity_runtime.engine", Engine())

    def activity_info() -> ActivityInfo:
        return ActivityInfo()

    monkeypatch.setattr("app.temporal_runtime.activity.info", activity_info)
    await run_turn(
        {
            "event": {
                "trigger": message("retry", "continue", thread="t"),
                "kind": "startup",
            },
            "state": {},
        }
    )
    assert seen == ["continue"]


def test_temporal_retry_policy_is_bounded() -> None:
    assert TRANSPORT_RETRY_POLICY.maximum_attempts == 2
    assert TRANSPORT_RETRY_POLICY.maximum_interval == timedelta(seconds=30)


@pytest.mark.asyncio
async def test_temporal_final_transport_attempt_returns_failure(monkeypatch) -> None:
    class Engine:
        async def handle(
            self,
            event: Event,
            state_data: dict[str, object] | None = None,
            *,
            retry_transport: bool = False,
        ) -> dict[str, object]:
            del event, state_data, retry_transport
            return {"error": "Server disconnected without sending a response."}

    class ActivityInfo:
        attempt = 2

    def activity_info() -> ActivityInfo:
        return ActivityInfo()

    monkeypatch.setattr("app.temporal_runtime._activity_runtime.engine", Engine())
    monkeypatch.setattr("app.temporal_runtime.activity.info", activity_info)
    result = cast(
        "EngineResult",
        await run_turn({"event": {"trigger": message("final", "hello", thread="t")}, "state": {}}),
    )
    assert result["error"].startswith("Server disconnected")


@pytest.mark.asyncio
async def test_temporal_submit_without_client_is_explicit() -> None:
    with pytest.raises(RuntimeError):
        await TemporalRuntime("temporal", "wiseman").submit({"trigger": {"id": "m", "channel_id": "c"}})


@pytest.mark.asyncio
async def test_live_delivery_sendsbanner_progress_and_answer(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_ROUTE_INFO", json.dumps({"requested_model": "model"}))

    class Channel:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.embeds: list[object] = []

        async def send(self, content: str = "", **kwargs: object) -> object:
            index = len(self.sent)
            self.sent.append(content)
            self.embeds.append(kwargs.get("embed"))

            class Delivery:
                async def edit(self, *, content: str) -> None:
                    channel.sent[index] = content

            channel = self
            return Delivery()

    class Live:
        def __init__(self) -> None:
            self.channel = Channel()
            self.reactions: list[str] = []

        async def add_reaction(self, emoji: str) -> None:
            self.reactions.append(emoji)

        async def remove_reaction(self, emoji: str, member: object) -> None:
            del member
            self.reactions.remove(emoji)

    live = Live()
    engine = configured_engine()
    engine.reaction_user = object()
    result = await engine.handle(
        normalize_event(discord_message("live", "hello", thread="t")),
        cast("discord.Message", live),
    )
    assert result["reactions"] == ["✅"]
    assert live.reactions == ["✅"]
    assert live.channel.sent[0] == ""
    assert live.channel.sent[1].startswith("Codex received: ")
    assert live.channel.embeds[0] is not None
    assert cast("discord.Embed", live.channel.embeds[0]).title == "⚡ Wiseman thread startup"


@pytest.mark.asyncio
async def test_live_delivery_edits_progress_for_http_runner(monkeypatch) -> None:
    class Channel:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, content: str = "", **kwargs: object) -> object:
            del kwargs
            index = len(self.sent)
            self.sent.append(content)
            channel = self

            class Delivery:
                async def edit(self, *, content: str) -> None:
                    channel.sent[index] = content

            return Delivery()

    class Live:
        def __init__(self) -> None:
            self.channel = Channel()
            self.reactions: list[str] = []

        async def add_reaction(self, emoji: str) -> None:
            self.reactions.append(emoji)

        async def remove_reaction(self, emoji: str, member: object) -> None:
            del member
            self.reactions.remove(emoji)

    class ProgressRunner(HttpRunner):
        async def run(
            self,
            thread: str,
            prompt: str,
            user: str,
            workspace: str = "",
            progress=None,
        ) -> tuple[str, str, dict[str, object]]:
            del prompt, user, workspace
            assert progress is not None
            await progress("⚙️ Running command...")
            await progress("⚙️ Running command...")
            await progress("✍️ Writing response...")
            return thread or "codex", "answer", {}

    live = Live()
    engine = configured_engine(runner=ProgressRunner("http://runner"))
    engine.reaction_user = object()
    result = await engine.handle(
        normalize_event(discord_message("progress", "hello", thread="t")),
        cast("discord.Message", live),
    )
    assert result["output"] == "answer"
    assert live.channel.sent[-1] == "answer"
    assert "⚙️ Running command..." in result["progress"]
    assert "✍️ Writing response..." in result["progress"]


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
    client = TestClient(create_app(configured_engine(), token="secret"))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.get("/v1/phoenix/events").json() == []
    assert client.post("/v1/replay/discord", json={}).status_code == 401
    assert client.post("/v1/replay/discord", headers={"x-replay-token": "secret"}, json={}).status_code == 422


def test_app_starts_discord_with_discord_token_not_replay_token(monkeypatch) -> None:
    received: list[str] = []

    async def fake_run(_bot: Gateway, token: str) -> None:
        received.append(token)

    monkeypatch.setattr(Gateway, "run_forever", fake_run)
    with TestClient(create_app(configured_engine(), token="replay", discord_token="discord")):
        pass
    assert received == ["discord"]


@pytest.mark.asyncio
async def test_gateway_retries_a_fatal_session_error(monkeypatch) -> None:
    bot = Gateway(configured_engine(), {1})
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
    monkeypatch.setattr("app.gateway.asyncio.sleep", fake_sleep)
    await bot.run_forever("discord")
    assert calls == 2


def test_gateway_requests_only_enabled_discord_intents() -> None:
    bot = Gateway(configured_engine(), {1})
    assert bot.intents.message_content
    assert not bot.intents.members
    assert not bot.intents.presences


@pytest.mark.asyncio
async def test_gateway_persists_managed_threads_and_expires_only_idle_threads(tmp_path, monkeypatch) -> None:
    activity_file = tmp_path / "activity.json"
    bot = Gateway(configured_engine(), {1}, activity_file)
    bot._touch_thread("123", timestamp=0)
    restored = Gateway(configured_engine(), {1}, activity_file)
    assert restored.thread_activity == {"123": 0.0}
    restored.thread_activity["123"] = 7_200.0

    restored.thread_activity["456"] = 1

    class Thread:
        async def edit(self, **kwargs: object) -> None:
            assert kwargs == {"archived": True, "locked": True}

    async def fetch_channel(thread_id: int) -> Thread:
        assert thread_id == 456
        return Thread()

    monkeypatch.setattr("app.gateway.discord.Thread", Thread)
    monkeypatch.setattr(restored, "fetch_channel", fetch_channel)
    await restored._expire_once(now=7_201)
    assert restored.thread_activity == {"123": 7_200.0}


@pytest.mark.asyncio
async def test_gateway_rediscovery_uses_owner_notthread_name(monkeypatch) -> None:
    bot = Gateway(configured_engine(), {1})
    bot._connection.user = cast("discord.ClientUser", SimpleNamespace(id=42))

    class Thread:
        id, owner_id, last_message_id, name, auto_archive_duration = 9, 42, 9, "rust", 1440

        async def edit(self, **kwargs: object) -> None:
            assert kwargs == {"auto_archive_duration": 60}

    class OtherThread(Thread):
        owner_id, name = 99, "wiseman"

    class Guild:
        async def fetch_active_threads(self) -> object:
            return SimpleNamespace(threads=[Thread(), OtherThread()])

    cast("dict[int, object]", bot._connection._guilds)[1] = Guild()
    await bot._discover_managed_threads()
    assert set(bot.thread_activity) == {"9"}


def test_gateway_does_not_track_unmanaged_threads() -> None:
    bot = Gateway(configured_engine(), {1})
    assert bot.thread_activity == {}


def test_phoenix_provider_relay_records_wrapped_billing(monkeypatch) -> None:
    class Response:
        is_error = False
        status_code = 200

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
    monkeypatch.setattr("app.http_api.httpx.AsyncClient", lambda **kwargs: Client())
    engine = configured_engine()
    response = TestClient(create_app(engine)).post("/v1/responses", json={"model": "requested"})
    assert response.status_code == 200
    provider = engine.config.phoenix.records[-1]
    assert provider["served_model"] == "served"
    assert provider["cost"] == 0.4
    assert provider["request"] == {"model": "requested"}


def test_provider_relay_propagates_disconnect_for_temporal_retry(monkeypatch) -> None:
    class Response:
        is_error = False
        status_code = 200

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def aiter_lines(self):
            yield 'data: {"response":{"model":"served"}}'
            yield "data: [DONE]"

    class Client:
        attempts = 0

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        def stream(self, method: str, url: str, **kwargs: object) -> Response:
            del method, url, kwargs
            self.attempts += 1
            if self.attempts == 1:
                raise httpx.RemoteProtocolError("disconnected")
            return Response()

    client = Client()
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setattr("app.http_api.httpx.AsyncClient", lambda **kwargs: client)
    with pytest.raises(httpx.RemoteProtocolError, match="disconnected"):
        TestClient(create_app(configured_engine())).post("/v1/responses", json={"model": "requested"})
    assert client.attempts == 1


@pytest.mark.asyncio
async def test_temporal_workflow_processes_one_turn_then_times_out(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    await workflow.submit({"id": "queued"})
    activities: list[object] = []
    child_states: list[dict[str, object]] = []
    turns = 0

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        activities.append(args[0])
        if args[0] is start_codex:
            return {"state": {"codex_thread": "codex-thread"}}
        return {"state": {"turn": 1}}

    async def child(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal turns
        del kwargs
        turns += 1
        activities.append(args[0])
        child_states.append(cast("dict[str, object]", cast("dict[str, object]", args[1])["state"]))
        return {"state": {"turn": turns}}

    async def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.execute_child_workflow", child)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", timeout)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    result = await workflow.run({"event": {"id": "first"}})
    assert result == {"state": {"turn": 1}}
    assert child_states == [{"turn": 1}]
    assert activities == [
        publish_progress,
        provision_workspace,
        publish_progress,
        start_codex,
        publish_progress,
        TurnWorkflow.run,
    ]


@pytest.mark.asyncio
async def test_temporal_thread_compacts_history_with_pending_signal(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    turns = 0
    continuation: list[dict[str, object]] = []

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        if args[0] is start_codex:
            return {"state": {"codex_thread": "codex"}}
        return {"state": {"codex_thread": "codex"}}

    async def child(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal turns
        del args, kwargs
        turns += 1
        return {"state": {"codex_thread": "codex", "turn": turns}}

    async def wait_for_signal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        if turns > 20:
            raise TimeoutError
        workflow.pending.append({"id": f"message-{turns + 1}"})

    class ContinuedError(Exception):
        pass

    def continue_as_new(value: object) -> None:
        continuation.append(cast("dict[str, object]", value))
        raise ContinuedError

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.execute_child_workflow", child)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", wait_for_signal)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    monkeypatch.setattr("app.temporal_runtime.workflow.continue_as_new", continue_as_new)
    with pytest.raises(ContinuedError):
        await workflow.run({"event": {"id": "message-1"}})
    assert continuation == [
        {
            "state": {"codex_thread": "codex", "turn": 20},
            "pending": [{"id": "message-21"}],
        }
    ]

    async def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", timeout)
    resumed = await ThreadWorkflow().run(continuation[0])
    assert resumed == {"state": {"codex_thread": "codex", "turn": 21}}


@pytest.mark.asyncio
async def test_temporal_followup_skips_workspace_and_codex_start(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    activities: list[object] = []
    turns = 0

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        activities.append(args[0])
        if args[0] is start_codex:
            return {"state": {"codex_thread": "codex-thread"}}
        return {"state": {"turn": len(activities)}}

    async def child(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal turns
        del kwargs
        turns += 1
        activities.append(args[0])
        return {"state": {"codex_thread": "codex-thread", "turn": turns}}

    async def wait_for_signal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        workflow.pending.append({"id": "followup"})
        if turns > 1:
            raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.execute_child_workflow", child)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", wait_for_signal)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    result = await workflow.run({"event": {"id": "first"}})
    assert result == {"state": {"codex_thread": "codex-thread", "turn": 2}}
    assert activities == [
        publish_progress,
        provision_workspace,
        publish_progress,
        start_codex,
        publish_progress,
        TurnWorkflow.run,
        publish_progress,
        TurnWorkflow.run,
    ]


@pytest.mark.asyncio
async def test_temporal_preflight_activities_use_runner_lifecycle(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class Runner:
        async def acquire(self, user: str, workspace: str) -> None:
            calls.append(("acquire", (user, workspace)))

        async def start(self, thread: str, user: str, workspace: str) -> str:
            calls.append(("start", (thread, user, workspace)))
            return "codex-thread"

    class Engine:
        runner = Runner()

    engine = Engine()
    engine.config = engine  # type: ignore[attr-defined]
    monkeypatch.setattr("app.temporal_runtime._activity_runtime.engine", engine)
    payload = {"event": normalize_event(discord_message("m", "hello", thread="t")).model_dump(mode="json")}
    assert await provision_workspace(cast("JsonObject", payload)) == {"workspace": "t"}
    assert await start_codex({**payload, "state": {"turn": 0}}) == {
        "state": {"turn": 0, "codex_thread": "codex-thread"},
        "workspace": "t",
        "codex_thread": "codex-thread",
    }
    assert calls == [("acquire", ("u", "t")), ("start", ("", "u", "t"))]


@pytest.mark.asyncio
async def test_engine_preflight_reuses_progress_delivery() -> None:
    class Delivery:
        def __init__(self) -> None:
            self.id = "delivery"
            self.content = ""

        async def edit(self, *, content: str) -> None:
            self.content = content

    class Channel:
        def __init__(self) -> None:
            self.sends: list[tuple[str, object | None]] = []
            self.delivery = Delivery()

        async def send(self, content: str = "", **kwargs: object) -> Delivery:
            self.sends.append((content, kwargs.get("embed")))
            return self.delivery

    channel = Channel()
    engine = configured_engine()

    async def lookup(_event: Event) -> Channel:
        return channel

    engine.lookup_channel = lookup
    event = Event(
        trigger=message("preflight", "hello", thread="thread"),
        kind="startup",
    )
    persisted = await engine.preflight(event, "🛠️ Workspace provisioning...", {"turn": 2})
    persisted = cast("StateData", persisted)
    persisted = await engine.preflight(event, "🤖 Codex starting...", persisted)
    assert len(channel.sends) == 2
    assert channel.sends[0][1] is not None
    assert channel.delivery.content.startswith("⏳ Working · Gurt 3")
    assert channel.delivery.content.endswith("🤖 Codex starting...")
    assert persisted["delivery_id"] == "delivery"
    assert persisted["banner_sent"] is True

    restored = configured_engine()
    restored.lookup_channel = lookup

    async def lookup_delivery(_event: Event, delivery_id: str) -> Delivery:
        assert delivery_id == "delivery"
        return channel.delivery

    restored.lookup_delivery = lookup_delivery
    resumed = await restored.preflight(event, "resumed", persisted)
    assert resumed["delivery_id"] == "delivery"
    assert len(channel.sends) == 2
    assert channel.delivery.content.endswith("resumed")
    failed = await engine.fail(event, "runner unavailable", {"turn": 2})
    assert failed["error"] == "runner unavailable"
    assert channel.delivery.content == "Codex failed: runner unavailable"


@pytest.mark.asyncio
async def test_temporal_setup_failure_uses_failure_activity(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    activities: list[object] = []

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        activities.append(args[0])
        if args[0] is provision_workspace:
            raise RuntimeError("runner unavailable")
        return {"error": "runner unavailable", "state": {}}

    async def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TimeoutError

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", timeout)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    result = await workflow.run({"event": {"id": "failed"}})
    assert result["error"] == "runner unavailable"
    assert activities == [publish_progress, provision_workspace, fail_turn]


@pytest.mark.asyncio
async def test_temporal_workflow_always_runs_split_startup_activities(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    activities: list[object] = []
    wait_timeouts: list[timedelta] = []

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        activities.append(args[0])
        return {"state": {"turn": 1}}

    async def timeout(*args: object, **kwargs: object) -> None:
        del args
        wait_timeouts.append(cast("timedelta", kwargs["timeout"]))
        raise TimeoutError

    async def child(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {"state": {"turn": 1}}

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.execute_child_workflow", child)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", timeout)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", lambda _: True)
    result = await workflow.run({"event": {"id": "old"}})
    assert result == {"state": {"turn": 1}}
    assert activities == [
        publish_progress,
        provision_workspace,
        publish_progress,
        start_codex,
        publish_progress,
    ]
    assert wait_timeouts == [timedelta(minutes=60)]


@pytest.mark.asyncio
async def test_temporal_workflow_replays_legacy_activity_history(monkeypatch) -> None:
    workflow = ThreadWorkflow()
    activities: list[object] = []

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        activities.append(args[0])
        return {"state": {"turn": 1}}

    async def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TimeoutError

    def patched(change_id: str) -> bool:
        return change_id == "split-startup-activities"

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    monkeypatch.setattr("app.temporal_runtime.workflow.wait_condition", timeout)
    monkeypatch.setattr("app.temporal_runtime.workflow.patched", patched)
    result = await workflow.run({"event": {"id": "legacy"}})

    assert result == {"state": {"turn": 1}}
    assert activities == [
        publish_progress,
        provision_workspace,
        publish_progress,
        start_codex,
        publish_progress,
        run_turn,
    ]


@pytest.mark.asyncio
async def test_temporal_turn_workflow_owns_activity_retry_policy(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    async def execute(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args[0], kwargs))
        return {"status": "ok"}

    monkeypatch.setattr("app.temporal_runtime.workflow.execute_activity", execute)
    result = await TurnWorkflow().run({"event": {}, "state": {}})

    assert result == {"status": "ok"}
    assert calls[0][0] is run_turn
    assert calls[0][1]["retry_policy"] is TRANSPORT_RETRY_POLICY


@pytest.mark.asyncio
async def test_gateway_uses_same_admission_for_parent_and_thread(monkeypatch) -> None:
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
            self.embeds: list[object] = []
            self.thread: Thread | None = None

        async def history(self, **kwargs: object):
            del kwargs
            yield raw_message("parent-history", self)

        async def send(self, content: str = "", **kwargs: object) -> object:
            self.sent.append(content)
            self.embeds.append(kwargs.get("embed"))
            return object()

    class Thread(Channel):
        parent_id = 1

        def __init__(self, parent: Channel) -> None:
            self.id, self.parent, self.guild = 2, parent, parent.guild
            self.sent: list[str] = []
            self.embeds: list[object] = []

        async def history(self, **kwargs: object):
            del kwargs
            yield raw_message("thread-history", self)

        async def send(self, content: str = "", **kwargs: object) -> object:
            index = len(self.sent)
            self.sent.append(content)
            self.embeds.append(kwargs.get("embed"))

            class Delivery:
                async def edit(self, *, content: str) -> None:
                    thread.sent[index] = content

            thread = self
            return Delivery()

        async def edit(self, **kwargs: object) -> None:
            del kwargs

    class Message:
        def __init__(self, mid: str, channel: Channel, content: str) -> None:
            self.id, self.channel, self.content = mid, channel, content
            self.author, self.mentions = User(), []
            self.raw_mentions = [bot_user.id]
            self.created_at = datetime.now(UTC)
            self.attachments, self.reference = (), None
            self.reactions: list[str] = []

        async def create_thread(self, name: str, auto_archive_duration: int) -> Thread:
            assert name == "Gurt 1"
            assert auto_archive_duration == 60
            self.channel.thread = Thread(self.channel)
            return self.channel.thread

        async def add_reaction(self, emoji: str) -> None:
            self.reactions.append(emoji)

        async def remove_reaction(self, emoji: str, member: object) -> None:
            del member
            self.reactions.remove(emoji)

    def raw_message(mid: str, channel: Channel) -> Message:
        item = Message(mid, channel, "context")
        item.mentions = []
        return item

    parent = Channel()
    bot_user = User()
    engine = configured_engine()
    monkeypatch.setattr("app.gateway.discord.Thread", Thread)
    bot = Gateway(engine, {1})
    bot._connection.user = cast("discord.ClientUser", bot_user)
    startup = Message("start", parent, "hello")
    startup.author.bot = True
    startup.author.id = 8
    await bot.on_socket_raw_receive('{"op":0,"t":"MESSAGE_CREATE","d":{"id":"start"}}')
    await bot.on_message(cast("discord.Message", startup))
    assert parent.thread is not None
    audit = engine.config.phoenix.audit("discord-start")
    assert audit is not None
    assert cast("dict[str, object]", audit["raw_request"])["t"] == "MESSAGE_CREATE"
    followup = Message("follow", parent.thread, "next")
    followup.author.bot = True
    followup.author.id = 8
    await bot.on_message(cast("discord.Message", followup))
    assert bot.fallback_state["2"]["turn"] == 2
    assert startup.reactions == ["✅"]
    assert followup.reactions == ["✅"]
    assert len(parent.thread.sent) == 3
    assert parent.thread.sent[0] == ""
    assert parent.thread.sent[1].startswith("Codex received: ")
    assert parent.thread.sent[2].startswith("Codex received: ")
    assert not parent.sent
    assert parent.thread.embeds[0] is not None
    assert cast("discord.Embed", parent.thread.embeds[0]).title == "⚡ Wiseman thread startup"
    assert parent.thread.sent[-1].startswith("Codex received: ")

    rejected = Channel()
    rejected.guild = Guild(2)
    ignored = Message("ignored", rejected, "hello")
    await bot.on_message(cast("discord.Message", ignored))
    assert "ignored" not in bot.fallback_state


@pytest.mark.asyncio
async def test_gateway_admits_reply_to_bot_without_explicit_mention() -> None:
    class Guild:
        id = 1

    class User:
        id, name, bot = 7, "Nick", False

    class BotMessage:
        class Author:
            id, bot = 7, True

        author = Author()

    class Reference:
        message_id = 99
        resolved = None

    class Channel:
        guild = Guild()

        async def fetch_message(self, message_id: int) -> BotMessage:
            assert message_id == 99
            return BotMessage()

    class Message:
        id, content, channel = 100, "continue", Channel()
        author, raw_mentions, mentions, reference = User(), [], [], Reference()

    gateway = Gateway(configured_engine(), {1})
    gateway._connection.user = cast("discord.ClientUser", User())

    assert await gateway._eligible(cast("discord.Message", Message()))


@pytest.mark.asyncio
async def test_gateway_preserves_raw_message_create_envelope() -> None:
    gateway = Gateway(configured_engine(), {1})
    await gateway.on_socket_raw_receive('{"op":0,"t":"MESSAGE_CREATE","d":{"id":"42"}}')

    assert gateway.raw_gateway_payload == {"op": 0, "t": "MESSAGE_CREATE", "d": {"id": "42"}}


@pytest.mark.asyncio
async def test_gateway_persists_gurt_thread_sequence(tmp_path) -> None:
    sequence = tmp_path / "thread-sequence.json"
    first = Gateway(configured_engine(), {1}, sequence_path=sequence)
    assert first._next_thread_name() == "Gurt 1"
    assert first._next_thread_name() == "Gurt 2"

    restored = Gateway(configured_engine(), {1}, sequence_path=sequence)
    assert restored._next_thread_name() == "Gurt 3"


def test_replay_accepts_discord_message_json() -> None:
    engine = configured_engine()
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
    context = next(item for item in engine.config.phoenix.records if item["node"] == "context")
    assert "file" in str(context["raw"])
    assert "username" in str(context["raw"])


def test_admission_audit_is_exact_and_replayable() -> None:
    engine = configured_engine()
    client = TestClient(create_app(engine, token="replay-secret"))
    payload = {
        "t": "MESSAGE_CREATE",
        "d": {
            **discord_message("audit-1", "write an engine in rust for balatro scoring", thread="audit-t"),
            "author": {"id": "u", "username": "Nick", "bot": False},
        },
        "kind": "startup",
        "parent_messages": [],
    }
    response = client.post("/v1/replay/discord", headers={"x-replay-token": "replay-secret"}, json=payload)
    assert response.status_code == 200
    audit = client.get("/v1/phoenix/audits/discord-audit-1", headers={"x-replay-token": "replay-secret"})
    assert audit.status_code == 200
    artifact = audit.json()
    assert artifact["raw_request"] == payload
    assert artifact["normalized_request"]["trigger"]["content"] == payload["d"]["content"]
    assert artifact["normalizer"] == "normalize_event:v2"

    replay = client.post(
        "/v1/replay/phoenix/discord-audit-1",
        headers={"x-replay-token": "replay-secret"},
    )
    assert replay.status_code == 200
    assert replay.json()["status"] == "duplicate"
    assert client.get("/v1/phoenix/audits/missing", headers={"x-replay-token": "replay-secret"}).status_code == 404
    assert client.get("/v1/phoenix/audits/discord-audit-1", headers={"x-replay-token": "bad"}).status_code == 401


def test_admission_audit_survives_gateway_restart(tmp_path: Path) -> None:
    payload = {
        "t": "MESSAGE_CREATE",
        "d": {
            **discord_message("persisted-audit-1", "replay this", thread="persisted-t"),
            "author": {"id": "u", "username": "Nick", "bot": False},
        },
        "kind": "startup",
        "parent_messages": [],
    }
    first = Phoenix(audit_dir=tmp_path)
    first_client = TestClient(create_app(configured_engine(phoenix=first), token="secret"))
    assert (
        first_client.post("/v1/replay/discord", headers={"x-replay-token": "secret"}, json=payload).status_code == 200
    )

    restarted = Phoenix(audit_dir=tmp_path)
    client = TestClient(create_app(configured_engine(phoenix=restarted), token="secret"))
    audit = client.get(
        "/v1/phoenix/audits/discord-persisted-audit-1",
        headers={"x-replay-token": "secret"},
    )
    assert audit.status_code == 200
    assert audit.json()["raw_request"] == payload
    replay = client.post(
        "/v1/replay/phoenix/discord-persisted-audit-1",
        headers={"x-replay-token": "secret"},
    )
    assert replay.status_code == 200
    assert '"user": "replay this"' in replay.json()["output"]


def test_default_app_persists_admission_audit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_AUDIT_DIR", str(tmp_path))
    payload = {
        "t": "MESSAGE_CREATE",
        "d": {
            **discord_message("default-audit-1", "persist this", thread="default-t"),
            "author": {"id": "u", "username": "Nick", "bot": False},
        },
        "kind": "startup",
    }
    client = TestClient(create_app(token="secret"))
    assert client.post("/v1/replay/discord", headers={"x-replay-token": "secret"}, json=payload).status_code == 200
    assert list(tmp_path.glob("*.json"))


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


def test_managed_account_name_is_stable_without_touching_host_accounts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_MANAGE_ACCOUNTS", "1")
    workspace = Workspace(str(tmp_path))
    commands: list[list[str]] = []
    monkeypatch.setattr("runner.api.pwd.getpwnam", lambda name: (_ for _ in ()).throw(KeyError))
    monkeypatch.setattr(workspace, "_admin", commands.append)
    monkeypatch.setattr("runner.api.shutil.chown", lambda *args, **kwargs: None)
    workspace.thread("discord-user", "thread")
    assert commands[0][0].endswith("groupadd")
    assert commands[1][0].endswith("useradd")
    assert commands[1][commands[1].index("--shell") + 1] == "/bin/bash"
    assert commands[1][commands[1].index("--groups") + 1] == "wsm_sudo"


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

    monkeypatch.setattr("app.http_api.httpx.AsyncClient", lambda **kwargs: Client())
    source = await PromptHub("http://phoenix", "key").source("startup")
    assert '"model": "model"' in source


def test_runner_starts_and_resumes_codex_thread(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    configs: list[CodexConfig] = []

    class Thread:
        id = "codex-thread"

        async def turn(self, prompt: str, **kwargs: object) -> EchoHandle:
            return EchoHandle(prompt)

    class Codex:
        def __init__(self, config: CodexConfig) -> None:
            configs.append(config)
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
    started = client.post("/start", headers=headers, json={"thread_id": "t", "user_id": "u", "input": ""})
    assert started.json()["thread_id"] == "codex-thread"
    retried_start = client.post("/start", headers=headers, json={"thread_id": "t", "user_id": "u", "input": ""})
    assert retried_start.json()["thread_id"] == "codex-thread"
    first = client.post(
        "/turn",
        headers=headers,
        json={
            "thread_id": "t",
            "codex_thread_id": "codex-thread",
            "user_id": "u",
            "input": "one",
        },
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
    assert len(calls) == 1
    assert "Never claim to have searched" in str(calls[0]["developer_instructions"])
    assert "sudo -n apt-get" in str(calls[0]["developer_instructions"])
    assert configs[0].config_overrides == CODEX_TEXT_ONLY_OVERRIDES
    assert 'base_url = "http://relay/v1"' in (tmp_path / "users/u/threads/t/.codex/config.toml").read_text()


@pytest.mark.asyncio
async def test_codex_runner_records_each_sdk_progress_phase(tmp_path, monkeypatch) -> None:
    observed_progress: list[str] = []

    class Handle:
        async def stream(self):
            observed_progress.append(runner.progress["t"])
            completed_turn = CodexTurn(id="turn", items=[], status=TurnStatus.completed)
            yield Notification(
                "turn/started",
                TurnStartedNotification(thread_id="t", turn=completed_turn),
            )
            yield Notification(
                "item/commandExecution/outputDelta",
                CommandExecutionOutputDeltaNotification(
                    delta="output",
                    item_id="command",
                    thread_id="t",
                    turn_id="turn",
                ),
            )
            yield Notification(
                "item/agentMessage/delta",
                AgentMessageDeltaNotification(
                    delta="answer",
                    item_id="item",
                    thread_id="t",
                    turn_id="turn",
                ),
            )
            yield Notification(
                "item/completed",
                ItemCompletedNotification(
                    completed_at_ms=1,
                    item=ThreadItem(root=AgentMessageThreadItem(id="item", text="answer", type="agentMessage")),
                    thread_id="t",
                    turn_id="turn",
                ),
            )
            yield Notification(
                "turn/completed",
                TurnCompletedNotification(
                    thread_id="t",
                    turn=completed_turn,
                ),
            )

    class Thread:
        id = "codex-thread"

        async def turn(self, prompt: str, **kwargs: object) -> Handle:
            del prompt, kwargs
            return Handle()

    class Codex:
        def __init__(self, config: object) -> None:
            del config

        async def thread_start(self, **kwargs: object) -> Thread:
            del kwargs
            return Thread()

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr("runner.api.AsyncCodex", Codex)
    runner = CodexRunner()
    turn = Turn(thread_id="t", user_id="u", input="hello")
    await runner.start(turn, Workspace(str(tmp_path)).thread("u", "t"))
    result = await runner.run(turn, Workspace(str(tmp_path)).thread("u", "t"))
    assert result["output"] == "answer"
    assert observed_progress == ["🤖 Gurt 1: Codex turn started..."]
    assert runner.progress["t"] == '✍️ Writing response... "answer"'


@pytest.mark.asyncio
async def test_codex_runner_propagates_midstream_disconnect_to_temporal(tmp_path, monkeypatch) -> None:
    prompts: list[str] = []
    stopped: list[str] = []

    class Handle:
        def __init__(self, attempt: int) -> None:
            self.failed = attempt == 1

        async def interrupt(self) -> None:
            stopped.append("interrupted")

        async def stream(self):
            if self.failed:
                raise RuntimeError("disconnect")
            yield Notification(
                "turn/completed",
                TurnCompletedNotification(
                    thread_id="t",
                    turn=CodexTurn(id="turn", items=[], status=TurnStatus.completed),
                ),
            )

    class Thread:
        id = "codex-thread"
        calls = 0

        async def turn(self, prompt: str, **kwargs: object) -> Handle:
            del kwargs
            prompts.append(prompt)
            self.calls += 1
            return Handle(self.calls)

    class Codex:
        def __init__(self, config: object) -> None:
            del config

        async def thread_start(self, **kwargs: object) -> Thread:
            del kwargs
            return Thread()

        async def close(self) -> None:
            stopped.append("closed")

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr("runner.api.AsyncCodex", Codex)
    runner = CodexRunner()
    path = Workspace(str(tmp_path)).thread("u", "t")
    with pytest.raises(RuntimeError, match="disconnect"):
        await runner.run(Turn(thread_id="t", user_id="u", input="hello"), path)
    assert prompts == ["hello"]
    assert stopped == ["interrupted", "closed"]
    assert not runner.active_turns
    assert not runner.threads


@pytest.mark.asyncio
async def test_codex_runner_steers_active_sdk_turn(tmp_path, monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    steers: list[str] = []

    class Handle:
        async def steer(self, prompt: str) -> None:
            steers.append(prompt)

        async def stream(self):
            started.set()
            await release.wait()
            yield Notification(
                "turn/completed",
                TurnCompletedNotification(
                    thread_id="t",
                    turn=CodexTurn(id="turn", items=[], status=TurnStatus.completed),
                ),
            )

    class Thread:
        id = "codex-thread"

        async def turn(self, prompt: str, **kwargs: object) -> Handle:
            del prompt, kwargs
            return Handle()

    class Codex:
        def __init__(self, config: object) -> None:
            del config

        async def thread_start(self, **kwargs: object) -> Thread:
            del kwargs
            return Thread()

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr("runner.api.AsyncCodex", Codex)
    runner = CodexRunner()
    path = Workspace(str(tmp_path)).thread("u", "t")
    turn = Turn(thread_id="t", user_id="u", input="hello")
    await runner.start(turn, path)
    task = asyncio.create_task(runner.run(turn, path))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert await runner.steer(Turn(thread_id="t", user_id="u", input="pivot"), path)
    release.set()
    await task
    assert steers == ["pivot"]


@pytest.mark.asyncio
async def test_codex_runner_limits_cross_thread_turns(tmp_path, monkeypatch) -> None:
    active = maximum = 0

    class Handle:
        async def stream(self):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
            yield Notification(
                "turn/completed",
                TurnCompletedNotification(
                    thread_id="t", turn=CodexTurn(id="turn", items=[], status=TurnStatus.completed)
                ),
            )

    class Thread:
        def __init__(self, identifier: str) -> None:
            self.id = identifier

        async def turn(self, prompt: str, **kwargs: object) -> Handle:
            del prompt, kwargs
            return Handle()

    class Codex:
        def __init__(self, config: object) -> None:
            del config

        async def thread_start(self, **kwargs: object) -> Thread:
            return Thread(str(kwargs.get("cwd", "")).rsplit("/", 1)[-1])

    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WISEMAN_MAX_CONCURRENT_TURNS", "1")
    monkeypatch.setattr("runner.api.AsyncCodex", Codex)
    runner = CodexRunner()
    workspace = Workspace(str(tmp_path))
    turns = [Turn(thread_id=name, user_id="u", input=name) for name in ("one", "two")]
    await asyncio.gather(*(runner.run(turn, workspace.thread("u", turn.thread_id)) for turn in turns))
    assert maximum == 1


@pytest.mark.asyncio
async def test_http_runner_reaches_sandbox_for_start_and_followup(tmp_path, monkeypatch) -> None:
    class Thread:
        id = "codex-thread"

        async def turn(self, prompt: str, **kwargs: object) -> EchoHandle:
            del kwargs
            return EchoHandle(prompt)

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
        "app.http_api.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=transport, base_url="http://sandbox", **kwargs),
    )
    runner = HttpRunner("http://sandbox", "secret")
    first = await runner.run("", "one", "u", "t")
    second = await runner.run(first[0], "two", "u", "t")
    assert first[0] == second[0] == "codex-thread"
    assert "one" in first[1]
    assert "two" in second[1]
    async with client_type(transport=transport, base_url="http://sandbox") as probe:
        progress = await probe.get("/progress/t", headers={"authorization": "Bearer secret"})
    assert progress.status_code == 200
    assert isinstance(progress.json()["steps"], list)
    assert (tmp_path / "users/u/shared/AGENTS.md").exists()
    assert (tmp_path / "users/u/threads/t/.codex").is_dir()


def test_runner_exposes_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_WORKSPACE_ROOT", str(tmp_path))
    client = TestClient(runner_app())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_info" in response.text


def test_sandbox_release_budget_supports_codex_workloads() -> None:
    path = Path(__file__).parents[1] / ("deploy/midgard/registry-inputs/wiseman-v2-sandbox.json")
    config = json.loads(path.read_text())
    assert config["mem_mi"] >= 4096


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
    await runtime.submit(cast("JsonObject", event))
    await runtime.submit({"trigger": {"id": "m2", "channel_id": "c", "thread_id": "t"}})
    assert runtime.client.handle.event is not None
    trigger = runtime.client.handle.event["trigger"]
    assert isinstance(trigger, dict)
    assert trigger["id"] == "m2"
