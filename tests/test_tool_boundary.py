# Copyright (c) 2026 Nick van der Merwe
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from fastapi.testclient import TestClient

from app.clients.discord_client import RealDiscord
from app.clients.mock_clients import MockDiscord, mock_container
from app.http_api import create_app
from app.models import Upload


def test_media_tools_use_injected_client_without_discord_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WISEMAN_ALLOW_PROFILE_EDITS", "1")
    clients = mock_container()
    app = create_app(clients=clients, token="tool-secret")
    monkeypatch.setattr(app.state.gateway, "fetch_channel", AsyncMock(side_effect=AssertionError("real Discord")))
    with TestClient(app) as api:
        headers = {"authorization": "Bearer tool-secret"}
        profile = api.post(
            "/v1/tools/set-profile", headers=headers, json={"username": "Gurt", "avatar_base64": "AAEC/w=="}
        )
        assert profile.status_code == 200
        assert profile.json()["username"] == "Gurt"
        uploaded = api.post(
            "/v1/tools/send-file",
            headers=headers,
            json={"thread_id": "thread", "filename": "../image.png", "data_base64": "AAEC/w==", "caption": "image"},
        )
        assert uploaded.status_code == 200
    assert isinstance(clients.discord, MockDiscord)
    state = clients.discord.state
    assert state.profile == {"username": "Gurt", "avatar": b"\x00\x01\x02\xff"}
    upload = state.uploads[uploaded.json()["message_id"]]
    assert upload.name == "image.png"
    assert upload.data == b"\x00\x01\x02\xff"


async def test_real_media_adapter_preserves_upload_and_combines_profile_update() -> None:
    channel = Mock(spec=discord.Thread)
    channel.send = AsyncMock(return_value=SimpleNamespace(id=20, jump_url="https://discord.test/20"))
    gateway = Mock()
    gateway.get_channel.return_value = channel
    gateway.user.edit = AsyncMock(return_value=SimpleNamespace(name="Gurt"))
    adapter = RealDiscord(gateway)
    receipt = await adapter.send_file("10", Upload("image.png", b"\x00\xff"), "caption")
    assert receipt.message_id == "20"
    assert receipt.url == "https://discord.test/20"
    sent = channel.send.await_args
    assert sent is not None
    assert sent.args == ("caption",)
    assert sent.kwargs["file"].filename == "image.png"
    assert sent.kwargs["file"].fp.read() == b"\x00\xff"
    assert await adapter.set_profile("Gurt", b"avatar") == "Gurt"
    gateway.user.edit.assert_awaited_once_with(username="Gurt", avatar=b"avatar")
    gateway.get_channel.return_value = Mock(spec=discord.TextChannel)
    with pytest.raises(TypeError, match="thread"):
        await adapter.send_file("11", Upload("image.png", b"image"))


@pytest.mark.parametrize(("username", "avatar"), [("Gurt", None), (None, b"avatar")])
async def test_profile_preserves_the_field_not_supplied(username: str | None, avatar: bytes | None) -> None:
    gateway = Mock()
    gateway.user.edit = AsyncMock(return_value=SimpleNamespace(name="Gurt"))
    assert await RealDiscord(gateway).set_profile(username, avatar) == "Gurt"
    gateway.user.edit.assert_awaited_once_with(**({"username": username} if username else {"avatar": avatar}))


@pytest.mark.parametrize(
    ("route", "payload", "permission", "status"),
    [
        ("set-profile", {"username": "Gurt"}, "0", 403),
        ("set-profile", {"username": "x"}, "1", 422),
        ("set-profile", {"avatar_base64": "invalid"}, "1", 422),
        ("set-profile", {}, "1", 422),
        ("send-file", {"thread_id": "t", "filename": "f", "data_base64": "invalid"}, "1", 422),
        ("send-file", {"thread_id": "t", "filename": "..", "data_base64": "YQ=="}, "1", 422),
    ],
)
def test_invalid_tools_do_not_mutate_clients(
    monkeypatch: pytest.MonkeyPatch, route: str, payload: dict[str, str], permission: str, status: int
) -> None:
    monkeypatch.setenv("WISEMAN_ALLOW_PROFILE_EDITS", permission)
    clients = mock_container()
    with TestClient(create_app(clients=clients, token="tool-secret")) as api:
        result = api.post(f"/v1/tools/{route}", json=payload, headers={"authorization": "Bearer tool-secret"})
    assert result.status_code == status
    assert isinstance(clients.discord, MockDiscord)
    assert not clients.discord.state.calls
