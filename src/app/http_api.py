# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import discord
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Security
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from httpx_sse import EventSource
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from app.models import Event

from app.admission import admitted, normalize_event
from app.clients.client_interfaces import ClientContainer, ClientMode, ClientSettings
from app.clients.discord_client import RealDiscord
from app.clients.provider import OpenRouter
from app.engine import Engine, EngineConfig
from app.gateway import Gateway
from app.models import Upload
from app.phoenix import Phoenix, PromptHub, json_text, provider_values
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_UPLOAD_BYTES,
    normalize_image_url,
)
from app.runner import HttpRunner
from app.temporal_runtime import TemporalRuntime, configure_engine
from app.types import JsonObject

BEARER = HTTPBearer(auto_error=False)


@dataclass(slots=True)
class _Context:
    engine: Engine
    bot: Gateway
    clients: ClientContainer
    token: str
    discord_token: str
    discord_task: asyncio.Task[None] | None = None


class ImageToolRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2_048)
    question: str = Field(default="", max_length=2_000)
    attachment_id: str | None = Field(default=None, max_length=200)
    thread_id: str | None = Field(default=None, max_length=100)


class ProfileRequest(BaseModel):
    username: str | None = Field(default=None, min_length=2, max_length=32)
    avatar_base64: str | None = None


class FileRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=255)
    data_base64: str
    caption: str = Field(default="", max_length=2_000)


def create_app(clients: ClientContainer | None = None) -> FastAPI:
    context = _context(clients)
    app = FastAPI(
        title="wiseman-v2",
        docs_url=None,
        redoc_url=None,
        lifespan=partial(_lifespan, context),
    )
    app.state.clients = context.clients
    _register_health(app, context)
    _register_replay(app, context)
    _register_provider(app, context)
    _register_tools(app, context)
    return app


def _context(clients: ClientContainer | None) -> _Context:
    settings = clients.settings if clients is not None else ClientSettings.from_env()
    token, discord_token = settings.runner_token, settings.discord_token
    real = clients is None or clients.mode is ClientMode.REAL
    if real and not all(
        (
            discord_token,
            settings.temporal_address,
            settings.runner_url,
            token,
            settings.phoenix_endpoint,
            settings.prompt_hub_url,
            settings.provider_key,
        )
    ):
        raise ValueError("real mode requires Discord, Temporal, runner and Phoenix configuration")
    services = None
    if clients is None:
        services = (
            Phoenix(
                settings.phoenix_endpoint,
                settings.phoenix_key,
                settings.phoenix_project,
                os.getenv("WISEMAN_AUDIT_DIR"),
            ),
            PromptHub(settings.prompt_hub_url, settings.phoenix_key),
            HttpRunner(settings.runner_url, settings.runner_token),
        )
    if clients is not None:
        engine = Engine(EngineConfig(clients.phoenix, clients.runner, clients.prompts, discord=clients.discord))
    else:
        assert services is not None
        engine = Engine(EngineConfig(services[0], services[2], services[1]))
    configure_engine(engine)
    allowlist = {int(value) for value in os.getenv("WISEMAN_DISCORD_ALLOWLIST", "").split(",") if value}
    activity_file = os.getenv("WISEMAN_ACTIVITY_FILE")
    profile_file = os.getenv("WISEMAN_PROFILE_FILE") or (str(Path(activity_file).with_name("profile.json")) if activity_file else None)
    bot = Gateway(engine, allowlist, activity_file, profile_file)
    if clients is None:
        clients = ClientContainer.install(
            ClientContainer(
                ClientMode.REAL,
                RealDiscord(bot),
                TemporalRuntime(settings.temporal_address, settings.temporal_queue),
                engine.config.phoenix,
                engine.config.prompts,
                engine.config.runner,
                OpenRouter(settings, engine.config.prompts),
                settings,
            )
        )
    engine.config = EngineConfig(clients.phoenix, clients.runner, clients.prompts, engine.config.context, clients.discord)
    bot.temporal = clients.temporal
    if not real:
        discord_token = ""
    return _Context(engine, bot, clients, token, discord_token)


async def _start(context: _Context) -> None:
    if context.clients.mode is ClientMode.REAL:
        await context.clients.temporal.start()
    if context.discord_token:
        context.discord_task = asyncio.create_task(context.bot.run_forever(context.discord_token))


async def _stop(context: _Context) -> None:
    if context.discord_task is not None:
        context.discord_task.cancel()
        await asyncio.gather(context.discord_task, return_exceptions=True)
    await context.bot.close()
    if context.clients.mode is ClientMode.REAL:
        await context.clients.temporal.close()
    await context.clients.provider.close()


@asynccontextmanager
async def _lifespan(context: _Context, _app: FastAPI) -> AsyncGenerator[None, None]:
    await _start(context)
    try:
        yield
    finally:
        await _stop(context)


def _register_health(app: FastAPI, context: _Context) -> None:
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def ready() -> dict[str, str]:
        if context.discord_token and not context.bot.is_ready():
            raise HTTPException(503, "Discord gateway is not ready")
        return {"status": "ready"}

    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.add_api_route("/healthz", health, methods=["GET"])
    app.add_api_route("/readyz", ready, methods=["GET"])
    app.add_api_route("/metrics", metrics, methods=["GET"])


def _auth(context: _Context, supplied: str | None, variable: str, detail: str) -> None:
    expected = os.getenv(variable, context.token)
    token = (supplied or "").removeprefix("Bearer ").strip()
    if expected and not hmac.compare_digest(token, expected):
        raise HTTPException(401, detail)


def _auth_dependency(context: _Context, variable: str, detail: str) -> Callable[[], None]:
    def check(credentials: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER)] = None) -> None:
        _auth(context, credentials.credentials if credentials else None, variable, detail)

    return check


async def _admit(context: _Context, event: Event) -> dict[str, object]:
    if event.kind == "stop":
        accepted = await context.clients.temporal.stop(event)
        return {"status": "stopped" if accepted else "ignored", "message_id": event.trigger.id}
    if event.kind == "steer":
        accepted = await context.clients.temporal.steer(event)
        return {"status": "steered" if accepted else "ignored", "message_id": event.trigger.id}
    bot_id = str(getattr(context.bot.user, "id", "") or os.getenv("WISEMAN_DISCORD_BOT_ID", ""))
    if not bot_id:
        raise HTTPException(503, "Discord identity is unavailable")
    replies = (*event.parent_messages, *event.thread_messages)
    if not admitted(event.trigger, bot_id, reply_to_bot=any(item.id == event.trigger.reply_to and item.bot for item in replies)):
        if event.trigger.thread_id:
            await context.clients.temporal.touch(event)
        return {"status": "ignored", "message_id": event.trigger.id}
    result = await context.clients.temporal.submit(event.model_dump(mode="json"))
    return result or {"status": "queued", "message_id": event.trigger.id}


def _register_replay(app: FastAPI, context: _Context) -> None:
    async def replay(payload: dict[str, object], x_replay_token: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _auth(context, x_replay_token, "WISEMAN_REPLAY_TOKEN", "invalid replay token")
        event = _event(payload, "invalid Discord event", strict=context.clients.mode is ClientMode.REAL)
        return await _admit(context, event)

    async def replay_audit(audit_id: str, x_replay_token: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _auth(context, x_replay_token, "WISEMAN_REPLAY_TOKEN", "invalid replay token")
        artifact = _audit(context, audit_id)
        raw = artifact.get("raw_request")
        payload = artifact.get("normalized_request")
        if not isinstance(raw, dict) or not isinstance(payload, dict):
            raise HTTPException(422, "Phoenix audit has no replayable raw request")
        event = _event(payload, "Phoenix audit contains an invalid Discord event", strict=context.clients.mode is ClientMode.REAL)
        event.raw_payload = cast("JsonObject", raw)
        return {**(await _admit(context, event)), "audit_id": audit_id}

    app.add_api_route("/v1/replay/discord", replay, methods=["POST"])
    app.add_api_route("/v1/replay/phoenix/{audit_id}", replay_audit, methods=["POST"])
    app.add_api_route("/v1/discord/events", replay, methods=["POST"])


def _audit(context: _Context, audit_id: str) -> dict[str, object]:
    value = context.engine.config.phoenix.audit(audit_id)
    if value is None:
        raise HTTPException(404, "Phoenix admission audit was not found")
    return value


def _event(payload: dict[str, object], detail: str, *, strict: bool = False) -> Event:
    try:
        event = normalize_event(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, detail) from exc
    if strict and not all(_snowflake(value) for value in (event.trigger.id, event.trigger.channel_id, event.trigger.thread_id, event.trigger.reply_to)):
        raise HTTPException(422, "Discord IDs must be valid snowflakes")
    return event


def _snowflake(value: str | None) -> bool:
    return value is None or (value.isascii() and value.isdigit() and int(value) <= 2**63 - 1)


def _register_provider(app: FastAPI, context: _Context) -> None:
    async def responses(payload: JsonObject) -> Response:
        trace = f"provider-{hashlib.sha256(json_text(payload).encode()).hexdigest()[:16]}"
        try:
            upstream = await context.clients.provider.responses(payload)
        except (httpx.HTTPError, RuntimeError) as exc:
            await _external_record(context, trace, request=payload, error=type(exc).__name__)
            raise HTTPException(503, "Provider connection unavailable") from exc
        headers = {key: upstream.headers[key] for key in ("content-type", "retry-after", "x-request-id") if key in upstream.headers}
        if upstream.is_error or "text/event-stream" not in upstream.headers.get("content-type", ""):
            try:
                body = await upstream.aread()
                await _external_record(context, trace, request=payload, status=upstream.status_code)
                return Response(body, status_code=upstream.status_code, headers=headers)
            finally:
                await upstream.aclose()
        return StreamingResponse(_provider_stream(context, upstream, payload, trace), status_code=upstream.status_code, headers=headers)

    app.add_api_route(
        "/v1/responses", responses, methods=["POST"], dependencies=[Depends(_auth_dependency(context, "WISEMAN_PROVIDER_TOKEN", "invalid provider token"))]
    )


async def _provider_stream(context: _Context, response: httpx.Response, payload: JsonObject, trace: str) -> AsyncIterator[bytes]:
    usage: object = None
    cost: object = None
    served_model: object = None
    complete = False
    try:
        async for event in EventSource(response).aiter_sse():
            try:
                value: object = event.json()
                if isinstance(value, dict):
                    usage, cost, served_model = provider_values(value, usage, cost, served_model)
            except ValueError:
                pass
            if event.event:
                yield f"event: {event.event}\n".encode()
            for line in event.data.splitlines() or [""]:
                yield f"data: {line}\n".encode()
            yield b"\n"
        complete = True
    finally:
        await asyncio.shield(response.aclose())
        await _external_record(
            context,
            trace,
            request=payload,
            requested_model=payload.get("model"),
            served_model=served_model,
            usage=usage,
            cost=cost,
            transport_complete=complete,
        )


async def _external_record(context: _Context, trace: str, node: str = "provider", **values: object) -> None:
    try:
        await asyncio.wait_for(context.clients.phoenix.record(trace, node, **values), timeout=5)
    except Exception:
        logging.getLogger("wiseman").exception("External telemetry failed trace=%s", trace)


def _register_tools(app: FastAPI, context: _Context) -> None:
    auth = Depends(_auth_dependency(context, "WISEMAN_MCP_TOKEN", "invalid tool token"))

    async def describe_image(payload: ImageToolRequest) -> dict[str, object]:
        url = normalize_image_url(payload.url)
        if not url:
            raise HTTPException(422, "image URL must use HTTP or HTTPS")
        try:
            result = await context.clients.provider.describe(url, payload.question)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise HTTPException(502, "Vision provider unavailable") from exc
        result["attachments"] = [payload.attachment_id or url]
        trace = payload.thread_id or f"vision-tool-{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        await _external_record(context, trace, "vision_tool", **result)
        return result

    async def set_reactions(payload: dict[str, object]) -> dict[str, object]:
        try:
            values = {phase: str(payload[phase]) for phase in DEFAULT_REACTION_EMOJIS if phase in payload}
            configured = context.engine.set_reaction_emojis(values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        context.bot.persist_profile()
        return {"reaction_emojis": configured}

    async def set_profile(payload: ProfileRequest) -> dict[str, object]:
        if os.getenv("WISEMAN_ALLOW_PROFILE_EDITS", "0") != "1":
            raise HTTPException(403, "profile edits are disabled")
        avatar = _decode_upload(payload.avatar_base64, "avatar_base64")
        if payload.username is None and avatar is None:
            raise HTTPException(422, "provide username or avatar")
        try:
            username = await context.clients.discord.set_profile(payload.username, avatar)
        except RuntimeError as exc:
            raise HTTPException(503, "Discord profile is unavailable") from exc
        return {"status": "updated", "username": username}

    async def send_file(payload: FileRequest) -> dict[str, object]:
        filename = Path(payload.filename).name
        if filename in {".", ".."}:
            raise HTTPException(422, "filename is invalid")
        data = _decode_upload(payload.data_base64, "data_base64")
        if data is None:
            raise HTTPException(422, "file must be non-empty and no larger than 8 MiB")
        try:
            receipt = await context.clients.discord.send_file(payload.thread_id, Upload(filename, data), payload.caption)
        except (ValueError, discord.DiscordException) as exc:
            raise HTTPException(404, "Discord thread was not found") from exc
        except TypeError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"status": "sent", "message_id": receipt.message_id, "url": receipt.url}

    app.add_api_route("/v1/tools/describe-image", describe_image, methods=["POST"], dependencies=[auth])
    app.add_api_route("/v1/tools/set-reactions", set_reactions, methods=["POST"], dependencies=[auth])
    app.add_api_route("/v1/tools/set-profile", set_profile, methods=["POST"], dependencies=[auth])
    app.add_api_route("/v1/tools/send-file", send_file, methods=["POST"], dependencies=[auth])


def _decode_upload(encoded: str | None, name: str) -> bytes | None:
    if encoded is None:
        return None
    data: bytes | None = None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"{name} is invalid") from exc
    if not data or len(data) > MAX_DISCORD_UPLOAD_BYTES:
        raise HTTPException(422, f"{name} exceeds the 8 MiB limit")
    return data
