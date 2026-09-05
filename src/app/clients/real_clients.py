# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import os

from app.clients.client_interfaces import ClientContainer, ClientMode, ClientSettings
from app.clients.discord_client import RealDiscord
from app.clients.mock_clients import mock_container
from app.clients.provider import OpenRouter
from app.phoenix import Phoenix, PromptHub
from app.runner import HttpRunner
from app.temporal_runtime import TemporalRuntime


def build_clients(mode: ClientMode, settings: ClientSettings) -> ClientContainer:
    if mode is ClientMode.MOCK:
        return ClientContainer.install(mock_container(settings))
    required = (settings.discord_token, settings.temporal_address, settings.runner_url, settings.runner_token, settings.phoenix_endpoint, settings.prompt_hub_url, settings.provider_key)  # noqa: E501 # fmt: skip
    if not all(required):
        raise ValueError("real mode requires Discord, Temporal, runner and Phoenix configuration")
    phoenix = Phoenix(settings.phoenix_endpoint, settings.phoenix_key, settings.phoenix_project, os.getenv("WISEMAN_AUDIT_DIR"))
    prompts, runner = PromptHub(settings.prompt_hub_url, settings.phoenix_key), HttpRunner(settings.runner_url, settings.runner_token)  # fmt: skip
    container = ClientContainer(ClientMode.REAL, RealDiscord(), TemporalRuntime(settings.temporal_address, settings.temporal_queue), phoenix, prompts, runner, OpenRouter(settings, prompts), settings)  # noqa: E501 # fmt: skip
    return ClientContainer.install(container)
