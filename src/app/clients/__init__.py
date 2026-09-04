# Copyright (c) 2026 Nick van der Merwe
from app.clients.client_interfaces import (
    ClientContainer,
    ClientMode,
    ClientSettings,
    current_clients,
)
from app.clients.mock_clients import mock_container
from app.clients.real_clients import RealDependencies, real_container


def build_clients(
    mode: ClientMode,
    settings: ClientSettings,
    dependencies: RealDependencies | None = None,
) -> ClientContainer:
    container = (
        mock_container(settings)
        if mode is ClientMode.MOCK
        else real_container(settings, dependencies)
    )
    return ClientContainer.install(container)


__all__ = ["ClientContainer", "ClientMode", "ClientSettings", "build_clients", "current_clients"]
