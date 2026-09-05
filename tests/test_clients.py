# Copyright (c) 2026 Nick van der Merwe

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from app.clients import ClientContainer, ClientMode, ClientSettings, build_clients
from app.clients.mock_clients import MockDiscord, MockState, mock_container
from app.clients.real_clients import RealDependencies, real_container
from app.engine import Engine, EngineConfig
from app.models import Event, Message

if TYPE_CHECKING:
    from app.types import JsonObject


@pytest.mark.asyncio
async def test_mock_container_records_complete_discord_lifecycle() -> None:
    container = build_clients(ClientMode.MOCK, ClientSettings())
    thread = await container.discord.create_thread("channel", "Gurt 1", 60)
    message = await container.discord.send(thread, "working")
    await container.discord.add_reaction(message, "👀")
    await container.discord.add_reaction(message, "✅")
    await container.discord.remove_reaction(message, "👀")
    await container.discord.remove_reaction(message, "👀")

    state = container.discord.state  # type: ignore[attr-defined]
    assert isinstance(state, MockState)
    assert state.threads[thread] == "Gurt 1"
    assert state.messages[message] == "working"
    assert state.reactions[message] == ["✅"]
    assert [call.operation for call in state.calls] == [
        "create_thread",
        "send",
        "add_reaction",
        "add_reaction",
        "remove_reaction",
        "remove_reaction",
    ]


@pytest.mark.asyncio
async def test_mock_discord_models_history_files_and_native_one_hour_archive() -> None:
    container = mock_container()
    discord = cast("MockDiscord", container.discord)
    thread = await discord.create_thread("channel", "Gurt 1", 60)
    message = await discord.send(thread, "hello")
    assert (await discord.history(thread, 10))[0]["id"] == message
    assert await discord.send_file(thread, "image.png") == "file-3"

    discord.advance(3_600)
    state = discord.state  # type: ignore[attr-defined]
    assert thread in state.archived
    assert thread not in state.locked
    await discord.lock_thread(thread)
    assert thread in state.locked


@pytest.mark.asyncio
async def test_mock_clients_model_progress_profile_and_files() -> None:
    container = mock_container()
    discord = cast("MockDiscord", container.discord)
    thread = await discord.create_thread("channel", "Gurt 1", 60)
    file_message = await discord.send_file(thread, "artifact.bin", "result")
    await discord.set_profile("Wise Man", "avatar.png")
    progress: list[str] = []

    async def record(message: str) -> None:
        progress.append(message)

    await container.runner.run("codex-1", "hello", "user", thread, progress=record)

    state = cast("MockState", discord.state)  # type: ignore[attr-defined]
    assert state.files[file_message] == "artifact.bin"
    assert state.profile == {"username": "Wise Man", "avatar": "avatar.png"}
    assert progress == ["🤖 Codex turn started...", "✍️ Writing response..."]


@pytest.mark.asyncio
async def test_engine_uses_the_injected_client_container() -> None:
    container = mock_container()
    engine = Engine(clients=container)
    event = Event(
        trigger=Message(
            id="message-1",
            author_id="user-1",
            author_name="user",
            content="hello",
            channel_id="channel-1",
            thread_id="thread-1",
            timestamp="2026-09-04T00:00:00+00:00",
        ),
        kind="startup",
    )

    result = await engine.handle(event)

    assert str(result["output"]).startswith('mock response: {"context":')
    state = cast("MockState", container.discord.state)  # type: ignore[attr-defined]
    assert any(call.client == "runner" and call.operation == "run" for call in state.calls)
    assert any(call.client == "phoenix" and call.operation == "record" for call in state.calls)
    assert state.records


@pytest.mark.asyncio
async def test_engine_restarts_from_returned_state_without_local_durable_store() -> None:
    container = mock_container()
    first_engine = Engine(clients=container)
    first_event = Event(
        trigger=Message(
            id="message-1",
            author_id="user-1",
            author_name="user",
            content="hello",
            channel_id="channel-1",
            thread_id="thread-1",
            timestamp="2026-09-04T00:00:00+00:00",
        ),
        kind="startup",
    )

    first = await first_engine.handle(first_event, state_data={})
    restarted = Engine(clients=container)
    followup = first_event.model_copy(
        update={
            "kind": "followup",
            "trigger": first_event.trigger.model_copy(update={"id": "message-2", "content": "next"}),
        }
    )
    second = await restarted.handle(followup, state_data=cast("JsonObject", first["state"]))

    assert not hasattr(first_engine, "states")
    assert second["state"]["turn"] == 2
    assert second["state"]["codex_thread"] == first["state"]["codex_thread"]


@pytest.mark.asyncio
async def test_injected_clients_isolate_concurrent_threads() -> None:
    container = mock_container()
    engine = Engine(clients=container)

    def event(message_id: str, thread_id: str) -> Event:
        return Event(
            trigger=Message(
                id=message_id,
                author_id="user-1",
                author_name="user",
                content=message_id,
                channel_id="channel-1",
                thread_id=thread_id,
                timestamp=message_id,
            ),
            kind="startup",
        )

    results = await asyncio.gather(
        engine.handle(event("message-1", "thread-1")),
        engine.handle(event("message-2", "thread-2")),
    )
    state = cast("MockState", container.discord.state)  # type: ignore[attr-defined]
    calls = [call for call in state.calls if call.client == "runner" and call.operation == "run"]
    assert {call.values[-1] for call in calls} == {"thread-1", "thread-2"}
    assert {result["state"]["turn"] for result in results} == {1}


def test_mock_containers_are_isolated_and_installed_explicitly() -> None:
    first = mock_container()
    second = mock_container()
    assert first is not second
    assert first.discord is not second.discord
    assert first.mode is ClientMode.MOCK

    ClientContainer.install(first)
    assert ClientContainer.current() is first
    ClientContainer.install(second)
    assert ClientContainer.current() is second
    ClientContainer.reset()
    with pytest.raises(RuntimeError, match="not been configured"):
        ClientContainer.current()


def test_real_mode_requires_explicit_adapters_and_never_falls_back() -> None:
    settings = ClientSettings()
    with pytest.raises(RuntimeError, match="adapters must be supplied"):
        real_container(settings)

    mock = mock_container()
    real = real_container(
        settings,
        RealDependencies(
            discord=mock.discord,
            temporal=mock.temporal,
            phoenix=mock.phoenix,
            prompts=mock.prompts,
            runner=mock.runner,
        ),
    )
    assert real.mode is ClientMode.REAL


def test_engine_requires_typed_config_or_client_container() -> None:
    container = mock_container()
    config = EngineConfig(container.phoenix, container.runner, container.prompts)
    engine = Engine(config)
    assert engine.config is config
    with pytest.raises(ValueError, match="engine config is required"):
        Engine()
    with pytest.raises(ValueError, match="choose config or clients"):
        Engine(config, clients=container)
