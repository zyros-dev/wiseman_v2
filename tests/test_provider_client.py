# Copyright (c) 2026 Nick van der Merwe
import asyncio
import json
import socket
from collections.abc import AsyncIterator
from typing import override
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.clients.client_interfaces import ClientSettings
from app.clients.mock_clients import mock_container
from app.clients.provider import OpenRouter
from app.http_api import _context, _provider_stream, create_app
from app.phoenix import PromptHub


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_upstream_error_status_is_preserved_before_starting_stream(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    clients = mock_container()
    upstream = httpx.Response(status, json={"error": {"message": "provider failure"}}, headers={"retry-after": "7"})
    provider = AsyncMock()
    provider.responses.return_value = upstream
    monkeypatch.setattr(clients, "provider", provider)
    with TestClient(create_app(clients=clients, token="fixture")) as api:
        response = api.post("/v1/responses", headers={"authorization": "Bearer fixture"}, json={"model": "requested"})
    assert response.status_code == status
    assert response.json() == {"error": {"message": "provider failure"}}
    assert response.headers["retry-after"] == "7"
    assert upstream.is_closed
    provider.responses.assert_awaited_once_with({"model": "requested"})


def test_mock_provider_never_uses_real_credentials_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "do-not-use")
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", AsyncMock(side_effect=AssertionError("network"))
    )
    clients = mock_container()
    with TestClient(create_app(clients=clients, token="fixture")) as api:
        headers = {"authorization": "Bearer fixture"}
        stream = api.post("/v1/responses", headers=headers, json={"model": "requested", "stream": True})
        assert stream.status_code == 200
        assert "response.completed" in stream.text
        image = api.post("/v1/tools/describe-image", headers=headers, json={"url": "https://cdn.test/a.png"})
        assert image.status_code == 200
        assert image.json()["text"] == "mock image description"
        assert image.json()["attachments"] == ["https://cdn.test/a.png"]
    assert "do-not-use" not in json.dumps(clients.phoenix.records)


@pytest.mark.parametrize("question", ["How many chairs cost < $40?", ""])
async def test_vision_request_contains_only_the_selected_image(question: str) -> None:
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "served-vision",
                "choices": [{"message": {"content": "Three chairs"}}],
                "usage": {"cost": 0.01},
            },
        )

    provider = OpenRouter(
        ClientSettings(provider_key="fixture", vision_model="selected-vision"),
        PromptHub(),
        httpx.MockTransport(upstream),
    )
    try:
        answer = await provider.describe("https://cdn.test/chair.png", question)
    finally:
        await provider.close()
    assert answer == {
        "text": "Three chairs",
        "model": "served-vision",
        "usage": {"cost": 0.01},
        "cost": 0.01,
        "question": question or None,
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer fixture"
    payload = json.loads(request.content)
    assert payload["model"] == "selected-vision"
    content = payload["messages"][0]["content"]
    assert len(content) == 2
    assert content[1] == {"type": "image_url", "image_url": {"url": "https://cdn.test/chair.png"}}
    assert (question or "Please describe this image generally") in content[0]["text"]


@pytest.mark.parametrize("telemetry_fails", [False, True])
def test_stream_preserves_events_and_records_billing(monkeypatch: pytest.MonkeyPatch, *, telemetry_fails: bool) -> None:
    clients = mock_container()
    wire = 'event: response.completed\ndata: {"response":{"model":"served","usage":{"cost":0.4}}}\n\ndata: [DONE]\n\n'
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=wire, headers={"content-type": "text/event-stream"})

    clients.provider = OpenRouter(
        ClientSettings(provider_key="fixture"), clients.prompts, httpx.MockTransport(upstream)
    )
    if telemetry_fails:
        monkeypatch.setattr(clients.phoenix, "record", AsyncMock(side_effect=RuntimeError("offline")))
    payload = {"model": "requested", "input": [{"role": "user", "content": "hello"}], "stream": True}
    with TestClient(create_app(clients=clients)) as api:
        response = api.post("/v1/responses", json=payload)
    assert response.status_code == 200
    assert response.text == wire
    assert len(requests) == 1
    assert json.loads(requests[0].content) == payload
    if not telemetry_fails:
        record = clients.phoenix.records[-1]
        assert record["served_model"] == "served"
        assert record["cost"] == 0.4
        assert record["request"] == payload
        assert record["transport_complete"] is True


def test_connection_failure_is_503_with_no_local_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = mock_container()
    connect = AsyncMock(side_effect=httpx.RemoteProtocolError("disconnected"))
    monkeypatch.setattr(clients.provider, "responses", connect)
    with TestClient(create_app(clients=clients)) as api:
        result = api.post("/v1/responses", json={"model": "requested"})
    assert result.status_code == 503
    connect.assert_awaited_once()
    assert clients.phoenix.records[-1]["error"] == "RemoteProtocolError"


async def test_midstream_disconnect_is_not_retried_and_closes_upstream() -> None:
    closed: list[bool] = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"type":"response.created"}\n\n'
            raise httpx.RemoteProtocolError("disconnected")

        async def aclose(self) -> None:
            closed.append(True)

    clients = mock_container()
    response = httpx.Response(200, stream=Broken())
    try:
        with pytest.raises(httpx.RemoteProtocolError):
            async for _ in _provider_stream(_context(None, "", "", clients), response, {}, "stream-test"):
                pass
    finally:
        await clients.provider.close()
    assert closed == [True]
    assert clients.phoenix.records[-1]["transport_complete"] is False


@pytest.mark.parametrize("cancel", [False, True])
async def test_real_http_stream_delivers_progress_before_completion_and_closes(*, cancel: bool) -> None:
    finish, closed, started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Server(uvicorn.Server):
        @override
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets)
            started.set()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"type":"response.created"}\n\n'
            await finish.wait()
            yield b'data: {"type":"response.completed"}\n\n'

        async def aclose(self) -> None:
            closed.set()

    clients = mock_container()
    provider = AsyncMock()
    provider.responses.return_value = httpx.Response(
        200, stream=Stream(), headers={"content-type": "text/event-stream"}
    )
    await clients.provider.close()
    clients.provider = provider
    app = create_app(clients=clients, token="fixture")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = Server(uvicorn.Config(app, log_level="error", timeout_graceful_shutdown=2))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(started.wait(), 5)
            async with (
                httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3) as api,
                api.stream(
                    "POST", "/v1/responses", json={"model": "requested"}, headers={"authorization": "Bearer fixture"}
                ) as response,
            ):
                assert response.status_code == 200
                lines = response.aiter_lines()
                assert "response.created" in await anext(lines)
                assert not finish.is_set()
                if not cancel:
                    finish.set()
                    assert "response.completed" in "".join([line async for line in lines])
            await asyncio.wait_for(closed.wait(), 3)
            provider.responses.assert_awaited_once()
        finally:
            finish.set()
            server.should_exit = True
            await asyncio.wait_for(serving, 5)
