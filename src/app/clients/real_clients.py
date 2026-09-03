# Copyright (c) 2026 Nick van der Merwe
"""Real client container construction at the application boundary."""

from __future__ import annotations

from dataclasses import dataclass

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    DiscordClient,
    PhoenixClient,
    ProviderClient,
    RunnerClient,
    TemporalClient,
)


class RealClientError(RuntimeError):
    """Raised when production adapters were not supplied."""

    def __init__(self) -> None:
        super().__init__("real client adapters must be supplied by the production entrypoint")


@dataclass(frozen=True, slots=True)
class RealDependencies:
    """Already-constructed production adapters supplied by the entrypoint."""

    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    runner: RunnerClient
    provider: ProviderClient


def real_container(
    settings: ClientSettings, dependencies: RealDependencies | None = None
) -> ClientContainer:
    """Build the real graph from explicit adapters; never silently use mocks."""
    del settings
    if dependencies is None:
        raise RealClientError
    return ClientContainer(
        mode=ClientMode.REAL,
        discord=dependencies.discord,
        temporal=dependencies.temporal,
        phoenix=dependencies.phoenix,
        runner=dependencies.runner,
        provider=dependencies.provider,
    )
