# Copyright (c) 2026 Nick van der Merwe
import os
from dataclasses import dataclass

import httpx

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    DiscordClient,
    PhoenixClient,
    PromptClient,
    ProviderClient,
    RunnerClient,
    TemporalClient,
)
from app.phoenix import Phoenix, PromptHub
from app.runner import HttpRunner


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
            raise TypeError
        return value


class UnavailableTemporal:
    async def submit(self, event: dict[str, object]) -> None:
        del event
        raise RuntimeError

    async def signal(self, workflow_id: str, event: dict[str, object]) -> None:
        del workflow_id, event
        raise RuntimeError


@dataclass(frozen=True, slots=True)
class RealDependencies:
    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    prompts: PromptClient
    runner: RunnerClient
    provider: ProviderClient


def real_container(
    settings: ClientSettings, dependencies: RealDependencies | None = None
) -> ClientContainer:
    if dependencies is None:
        raise RuntimeError("real client adapters must be supplied")  # noqa: TRY003
    return ClientContainer(
        ClientMode.REAL,
        dependencies.discord,
        dependencies.temporal,
        dependencies.phoenix,
        dependencies.prompts,
        dependencies.runner,
        dependencies.provider,
        settings,
    )


def real_services(settings: ClientSettings) -> tuple[Phoenix, PromptHub, HttpRunner, HttpProvider]:
    return (
        Phoenix(
            settings.phoenix_endpoint,
            settings.phoenix_key,
            settings.phoenix_project,
            os.getenv("WISEMAN_AUDIT_DIR"),
        ),
        PromptHub(settings.prompt_hub_url, settings.phoenix_key),
        HttpRunner(settings.runner_url, settings.runner_token),
        HttpProvider(settings.provider_url, settings.provider_token),
    )
