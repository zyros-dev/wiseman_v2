# Copyright (c) 2026 Nick van der Merwe
import base64
import json
from typing import cast

import httpx

from app.admission import ContextConfig, context, image_tool_instruction, normalize_event, render_grammar
from app.clients.mock_clients import MockDiscord, mock_container
from app.engine import Engine, EngineConfig
from app.http_api import create_app
from app.models import State, TurnWork


def _raw(message_id: str, content: str, *, thread: str | None = "t", attachments: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "id": message_id,
        "author": {"id": "alice", "username": "alice"},
        "content": content,
        "channel_id": "home",
        "thread_id": thread,
        "timestamp": f"2026-09-05T00:00:{message_id[-1:]}Z",
        "mentions": [{"id": "bot"}],
        "attachments": attachments or [],
    }


def test_admission_normalizes_discord_and_selects_only_new_images() -> None:
    image: dict[str, object] = {"id": "img", "filename": "chart.png", "url": "https://cdn.test/chart.png"}
    event = normalize_event({"t": "MESSAGE_CREATE", "d": _raw("q1", "describe this", attachments=[image])})
    event.kind = "startup"
    event.parent_messages = [normalize_event({"trigger": _raw("old", "old", attachments=[image])}).trigger]
    event.parent_messages[0].timestamp = "2026-09-04T23:59:00Z"
    selected = context(event, ContextConfig(1))
    assert selected["selected_ids"] == ["old", "q1"]
    instruction = image_tool_instruction(event.trigger.model_dump(), [])
    assert "chart.png" in instruction
    assert "old" not in instruction
    grammar = render_grammar("test", '{"mode":"{{ mode }}","count":{{ messages|length }}}', {}, mode="startup", messages=[])
    assert json.loads(str(grammar["rendered"])) == {"mode": "startup", "count": 0}


async def test_engine_delivers_progress_and_reconciles_terminal_reaction() -> None:
    clients = mock_container()
    engine = Engine(EngineConfig(clients.phoenix, clients.runner, clients.prompts, discord=clients.discord))
    work = TurnWork(event=normalize_event({"trigger": _raw("q1", "answer")}), state=State())
    await engine.prepare_context(work)
    await engine.prepare_prompt(work)
    await engine.render(work)
    await engine.execute(work, lambda message: engine.config.phoenix.record(work.trace, "progress", message=message))
    result = await engine.finish(work)
    discord = cast("MockDiscord", clients.discord)
    assert result["output"]
    assert discord.state.reactions["q1"] == ["✅"]
    assert any(item[1] == "edit" for item in discord.state.calls)


async def test_http_tools_provider_and_audit_replay(monkeypatch) -> None:
    monkeypatch.setenv("WISEMAN_PROVIDER_TOKEN", "secret")
    monkeypatch.setenv("WISEMAN_MCP_TOKEN", "secret")
    monkeypatch.setenv("WISEMAN_ALLOW_PROFILE_EDITS", "1")
    clients = mock_container()
    app = create_app(clients=clients)
    headers = {"authorization": "Bearer secret"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wiseman") as client:
        vision = await client.post("/v1/tools/describe-image", headers=headers, json={"url": "https://cdn.test/a.png", "question": "what?"})
        upload = await client.post(
            "/v1/tools/send-file",
            headers=headers,
            json={"thread_id": "t", "filename": "a.txt", "data_base64": base64.b64encode(b"ok").decode()},
        )
        profile = await client.post("/v1/tools/set-profile", headers=headers, json={"username": "Wiseman"})
        provider = await client.post("/v1/responses", headers=headers, json={"model": "mock", "input": "hi"})
        assert vision.json()["text"] == "mock image description"
        assert upload.json()["status"] == "sent"
        assert profile.json()["username"] == "Wiseman"
        assert provider.status_code == 200
        assert "response.completed" in provider.text
    discord = cast("MockDiscord", clients.discord)
    assert discord.state.uploads
