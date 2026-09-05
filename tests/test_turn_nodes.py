# Copyright (c) 2026 Nick van der Merwe
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.clients.client_interfaces import DiscordClient
from app.engine import Engine, EngineConfig
from app.models import Event, Message, MessageRef, State, TurnWork
from app.nodes import TurnActivities
from app.phoenix import Phoenix, PromptHub
from app.runner import FakeRunner


def work():
    return TurnWork(
        event=Event(trigger=Message(id="trigger", author_id="bob", channel_id="parent", thread_id="thread")),
        state=State(owner_id="alice", codex_thread="sdk", delivery_id="answer"),
    )


def engine():
    return Engine(EngineConfig(Phoenix(), FakeRunner(), PromptHub(), discord=AsyncMock(spec=DiscordClient)))


@pytest.mark.asyncio
async def test_context_and_prompt_nodes_do_not_execute_runner():
    service = engine()
    service.config.runner.run = AsyncMock(side_effect=AssertionError("inference during preparation"))
    prepared = await service.prepare_context(work())
    prepared = await service.prepare_prompt(prepared)
    assert prepared.prompt
    assert prepared.state.owner_id == "alice"
    assert "trigger" in prepared.state.seen
    assert not prepared.state.processed
    service.config.runner.run.assert_not_called()


@pytest.mark.asyncio
async def test_restarted_delivery_uses_recorded_answer_without_inference():
    first = engine()
    prepared = await first.prepare_prompt(await first.prepare_context(work()))
    executed = await first.execute(prepared, AsyncMock())
    restored = TurnWork.model_validate_json(executed.model_dump_json())
    second = engine()
    assert isinstance(second.discord, AsyncMock)
    second.config.runner.run = AsyncMock(side_effect=AssertionError("repeated inference"))
    await second.deliver(restored)
    second.discord.edit.assert_awaited_once_with(MessageRef("thread", "answer"), executed.output, upload=None)


@pytest.mark.asyncio
async def test_reaction_retry_reconciles_on_fresh_worker():
    request = work()
    request.processing_emoji = "eyes"
    request.terminal_emoji = "done"
    for _ in range(2):
        service = engine()
        assert isinstance(service.discord, AsyncMock)
        await service.reconcile(request)
        service.discord.add_reaction.assert_awaited_once_with(MessageRef("parent", "trigger"), "done")
        service.discord.remove_reaction.assert_awaited_once_with(MessageRef("parent", "trigger"), "eyes")
        assert [call[0] for call in service.discord.mock_calls] == ["add_reaction", "remove_reaction"]


async def test_registered_nodes_use_recorded_state_and_emit_live_progress(monkeypatch):
    service, client = engine(), AsyncMock()
    assert isinstance(service.discord, AsyncMock)
    info = SimpleNamespace(workflow_id="turn", workflow_run_id="run")
    monkeypatch.setattr("app.nodes.activity.info", lambda: info)

    async def run(thread, prompt, user, workspace, progress):
        assert (thread, user, workspace) == ("sdk", "alice", "thread")
        await progress("tool completed")
        return "sdk", "answer content", {"model": "test-model"}

    service.config.runner.run = AsyncMock(side_effect=run)
    handle = AsyncMock()
    client.get_workflow_handle = lambda *args, **kwargs: handle
    nodes = TurnActivities(service, client)
    payload = work().model_dump(mode="json")
    for name in ("context", "prompt", "render", "react", "infer", "deliver", "react", "observe"):
        payload = await getattr(nodes, name)(payload)
    handle.signal.assert_awaited_once_with("progress", "tool completed")
    service.discord.edit.assert_any_await(MessageRef("thread", "answer"), "answer content", upload=None)
    assert [call.args[1] for call in service.discord.add_reaction.await_args_list] == ["👀", "✅"]
    service.discord.remove_reaction.assert_awaited_once()
    assert service.config.phoenix.records[-1]["node"] == "completed"
    assert isinstance(service.config.phoenix, Phoenix)
    assert not service.config.phoenix.roots
    service.config.runner.steer = AsyncMock(return_value=True)
    assert await nodes.steer({"work": payload, "event": work().event.model_dump(mode="json")}) == {"accepted": True}
    service.config.runner.steer.assert_awaited_once_with("sdk", "", "alice", "thread")
