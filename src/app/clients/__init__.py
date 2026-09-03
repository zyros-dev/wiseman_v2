# Copyright (c) 2026 Nick van der Merwe
"""Explicit external-client construction for production and tests."""

from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    current_clients,
)
from app.clients.mock_clients import mock_container
from app.clients.real_clients import real_container


def build_clients(mode: ClientMode, settings: ClientSettings) -> ClientContainer:
    """Build and install either the complete mock or real dependency graph."""
    container = mock_container() if mode is ClientMode.MOCK else real_container(settings)
    return ClientContainer.install(container)


__all__ = ["ClientContainer", "ClientMode", "ClientSettings", "build_clients", "current_clients"]
