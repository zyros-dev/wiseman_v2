# Copyright (c) 2026 Nick van der Merwe
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.engine import Engine, EngineConfig
from app.models import Event, Message, State, TurnWork
from app.nodes import TurnActivities
from app.phoenix import Phoenix, PromptHub
from app.runner import FakeRunner


def work():
    return TurnWork(
        event=Event(trigger=Message(id="trigger", author_id="bob", channel_id="parent", thread_id="thread")),
        state=State(owner_id="alice", codex_thread="sdk", delivery_id="answer"),
    )


def engine():
    return Engine(EngineConfig(Phoenix(), FakeRunner(), PromptHub()))


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
    second.config.runner.run = AsyncMock(side_effect=AssertionError("repeated inference"))
    answer = AsyncMock()
    second.lookup_delivery = AsyncMock(return_value=answer)
    second.lookup_channel = AsyncMock(return_value=AsyncMock())
    await second.deliver(restored)
    answer.edit.assert_awaited_once_with(content=executed.output)
    second.lookup_delivery.assert_awaited_once_with(restored.event, "answer")


@pytest.mark.asyncio
async def test_reaction_retry_reconciles_on_fresh_worker():
    request = work()
    request.processing_emoji = "eyes"
    request.terminal_emoji = "done"
    live = AsyncMock()
    for _ in range(2):
        service = engine()
        service.lookup = AsyncMock(return_value=live)
        service.reaction_user = object()
        await service.reconcile(request)
    assert live.add_reaction.await_count == 2
    assert live.remove_reaction.await_count == 2
    assert live.mock_calls[0].args == ("done",)
    assert live.mock_calls[1].args[0] == "eyes"


async def test_registered_nodes_use_recorded_state_and_emit_live_progress(monkeypatch):
    service, client = engine(), AsyncMock()
    live, answer, channel = AsyncMock(), AsyncMock(), AsyncMock()
    service.lookup = AsyncMock(return_value=live)
    service.lookup_channel = AsyncMock(return_value=channel)
    service.lookup_delivery = AsyncMock(return_value=answer)
    service.reaction_user = object()
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
    answer.edit.assert_any_await(content="answer content")
    assert [call.args[0] for call in live.add_reaction.await_args_list] == ["👀", "✅"]
    live.remove_reaction.assert_awaited_once()
    assert service.config.phoenix.records[-1]["node"] == "completed"
    assert isinstance(service.config.phoenix, Phoenix)
    assert not service.config.phoenix.roots
    service.config.runner.steer = AsyncMock(return_value=True)
    assert await nodes.steer({"work": payload, "event": work().event.model_dump(mode="json")}) == {"accepted": True}
    service.config.runner.steer.assert_awaited_once_with("sdk", "", "alice", "thread")
