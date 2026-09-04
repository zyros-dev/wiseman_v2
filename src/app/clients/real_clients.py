# Copyright (c) 2026 Nick van der Merwe
from dataclasses import dataclass

import httpx

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
    pass


class HttpProvider:
    def __init__(self, url: str, token: str) -> None:
        self.url, self.token = url.rstrip("/"), token

    async def response(self, payload: dict[str, object]) -> dict[str, object]:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{self.url}/api/v1/responses",
                headers={"authorization": f"Bearer {self.token}"},
                json=payload,
            )
            response.raise_for_status()
            value: object = response.json()
        if not isinstance(value, dict):
            raise RealClientError
        return value


class UnavailableTemporal:
    async def submit(self, event: dict[str, object]) -> None:
        del event
        raise RealClientError

    async def signal(self, workflow_id: str, event: dict[str, object]) -> None:
        del workflow_id, event
        raise RealClientError


@dataclass(frozen=True, slots=True)
class RealDependencies:
    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    runner: RunnerClient
    provider: ProviderClient


def real_container(
    settings: ClientSettings, dependencies: RealDependencies | None = None
) -> ClientContainer:
    if dependencies is None:
        raise RealClientError("real client adapters must be supplied")  # noqa: TRY003
    return ClientContainer(
        ClientMode.REAL,
        dependencies.discord,
        dependencies.temporal,
        dependencies.phoenix,
        dependencies.runner,
        dependencies.provider,
        settings,
    )
