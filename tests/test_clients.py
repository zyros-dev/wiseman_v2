# Copyright (c) 2026 Nick van der Merwe
"""Contract tests for the explicit real/mock client boundary."""

from __future__ import annotations

from typing import cast

import pytest

from app.clients import ClientContainer, ClientMode, ClientSettings, build_clients
from app.clients.mock_clients import MockDiscord, MockState, mock_container
from app.clients.real_clients import RealDependencies, real_container


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
async def test_mock_discord_models_history_files_and_one_hour_closure() -> None:
    container = mock_container()
    discord = cast("MockDiscord", container.discord)
    thread = await discord.create_thread("channel", "Gurt 1", 60)
    message = await discord.send(thread, "hello")
    assert (await discord.history(thread, 10))[0]["id"] == message
    assert await discord.send_file(thread, "image.png") == "file-3"

    discord.advance(3_600)
    state = discord.state  # type: ignore[attr-defined]
    assert thread in state.archived
    assert thread in state.locked


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
            runner=mock.runner,
            provider=mock.provider,
        ),
    )
    assert real.mode is ClientMode.REAL
