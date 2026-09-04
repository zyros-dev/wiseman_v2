# Copyright (c) 2026 Nick van der Merwe
import os
from dataclasses import dataclass

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    DiscordClient,
    PhoenixClient,
    PromptClient,
    RunnerClient,
    TemporalClient,
)
from app.phoenix import Phoenix, PromptHub
from app.runner import HttpRunner


@dataclass(frozen=True, slots=True)
class RealDependencies:
    discord: DiscordClient
    temporal: TemporalClient
    phoenix: PhoenixClient
    prompts: PromptClient
    runner: RunnerClient


def real_container(settings: ClientSettings, dependencies: RealDependencies | None = None) -> ClientContainer:
    if dependencies is None:
        raise RuntimeError("real client adapters must be supplied")
    return ClientContainer(
        ClientMode.REAL,
        dependencies.discord,
        dependencies.temporal,
        dependencies.phoenix,
        dependencies.prompts,
        dependencies.runner,
        settings,
    )


def real_services(settings: ClientSettings) -> tuple[Phoenix, PromptHub, HttpRunner]:
    return (
        Phoenix(
            settings.phoenix_endpoint,
            settings.phoenix_key,
            settings.phoenix_project,
            os.getenv("WISEMAN_AUDIT_DIR"),
        ),
        PromptHub(settings.prompt_hub_url, settings.phoenix_key),
        HttpRunner(settings.runner_url, settings.runner_token),
    )
