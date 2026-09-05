# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import discord
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from app.models import Event

from app.admission import admitted, normalize_event
from app.clients import ClientContainer, ClientMode, ClientSettings
from app.clients.discord_client import RealDiscord
from app.clients.provider import OpenRouter
from app.engine import Engine, EngineConfig
from app.gateway import Gateway
from app.models import Upload
from app.phoenix import Phoenix, PromptHub, json_text, provider_values
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_UPLOAD_BYTES,
    MAX_DISCORD_USERNAME_LENGTH,
    MIN_DISCORD_USERNAME_LENGTH,
    normalize_image_url,
)
from app.runner import HttpRunner
from app.temporal_runtime import TemporalRuntime, configure_engine
from app.types import JsonObject


@dataclass(slots=True)
class _Context:
    engine: Engine
    bot: Gateway
    clients: ClientContainer
    token: str
    discord_token: str
    discord_task: asyncio.Task[None] | None = None


def create_app(clients: ClientContainer | None = None) -> FastAPI:
    context = _context(clients)
    app = FastAPI(
        title="wiseman-v2",
        docs_url=None,
        redoc_url=None,
        lifespan=partial(_lifespan, context),
    )
    app.state.gateway = context.bot
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


def _replay_auth(context: _Context, supplied: str | None) -> None:
    expected = os.getenv("WISEMAN_REPLAY_TOKEN", context.token)
    if expected and not hmac.compare_digest(supplied or "", expected):
        raise HTTPException(401, "invalid replay token")


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
        return {"status": "ignored", "message_id": event.trigger.id}
    result = await context.clients.temporal.submit(event.model_dump(mode="json"))
    return result or {"status": "queued", "message_id": event.trigger.id}


def _register_replay(app: FastAPI, context: _Context) -> None:
    async def replay(payload: dict[str, object], x_replay_token: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _replay_auth(context, x_replay_token)
        event = _event(payload, "invalid Discord event")
        return await _admit(context, event)

    async def replay_audit(audit_id: str, x_replay_token: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _replay_auth(context, x_replay_token)
        artifact = _audit(context, audit_id)
        payload = artifact.get("raw_request")
        if not isinstance(payload, dict):
            raise HTTPException(422, "Phoenix audit has no replayable raw request")
        event = _event(payload, "Phoenix audit contains an invalid Discord event")
        return {**(await _admit(context, event)), "audit_id": audit_id}

    app.add_api_route("/v1/replay/discord", replay, methods=["POST"])
    app.add_api_route("/v1/replay/phoenix/{audit_id}", replay_audit, methods=["POST"])
    app.add_api_route("/v1/discord/events", replay, methods=["POST"])


def _audit(context: _Context, audit_id: str) -> dict[str, object]:
    value = context.engine.config.phoenix.audit(audit_id)
    if value is None:
        raise HTTPException(404, "Phoenix admission audit was not found")
    return value


def _event(payload: dict[str, object], detail: str) -> Event:
    try:
        return normalize_event(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, detail) from exc


def _register_provider(app: FastAPI, context: _Context) -> None:
    async def responses(payload: JsonObject, authorization: Annotated[str | None, Header()] = None) -> Response:
        expected = os.getenv("WISEMAN_PROVIDER_TOKEN", context.token)
        if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
            raise HTTPException(401, "invalid provider token")
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

    app.add_api_route("/v1/responses", responses, methods=["POST"])


async def _provider_stream(context: _Context, response: httpx.Response, payload: JsonObject, trace: str) -> AsyncIterator[bytes]:
    usage: object = None
    cost: object = None
    served_model: object = None
    complete = False
    try:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                try:
                    value: object = json.loads(line[5:].strip())
                    if isinstance(value, dict):
                        usage, cost, served_model = provider_values(value, usage, cost, served_model)
                except ValueError:
                    pass
            yield f"{line}\n".encode()
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


def _tool_auth(context: _Context, supplied: str | None) -> None:
    expected = os.getenv("WISEMAN_MCP_TOKEN", os.getenv("WISEMAN_PROVIDER_TOKEN", context.token))
    if expected and not hmac.compare_digest(supplied or "", f"Bearer {expected}"):
        raise HTTPException(401, "invalid tool token")


def _register_tools(app: FastAPI, context: _Context) -> None:
    async def describe_image(payload: dict[str, object], authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _tool_auth(context, authorization)
        url = normalize_image_url(payload.get("url"))
        if not url:
            raise HTTPException(422, "image URL must use HTTP or HTTPS")
        attachment_id = str(payload.get("attachment_id") or url)
        try:
            result = await context.clients.provider.describe(url, str(payload.get("question") or "")[:2_000])
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise HTTPException(502, "Vision provider unavailable") from exc
        result["attachments"] = [attachment_id]
        trace = str(payload.get("thread_id") or f"vision-tool-{hashlib.sha256(url.encode()).hexdigest()[:16]}")
        await _external_record(context, trace, "vision_tool", **result)
        return result

    async def set_reactions(payload: dict[str, object], authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _tool_auth(context, authorization)
        try:
            values = {phase: str(payload[phase]) for phase in DEFAULT_REACTION_EMOJIS if phase in payload}
            configured = context.engine.set_reaction_emojis(values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        context.bot.persist_profile()
        return {"reaction_emojis": configured}

    async def set_profile(payload: dict[str, object], authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _tool_auth(context, authorization)
        if os.getenv("WISEMAN_ALLOW_PROFILE_EDITS", "0") != "1":
            raise HTTPException(403, "profile edits are disabled")
        username, avatar = _profile_values(payload)
        try:
            username = await context.clients.discord.set_profile(username, avatar)
        except RuntimeError as exc:
            raise HTTPException(503, "Discord profile is unavailable") from exc
        return {"status": "updated", "username": username}

    async def send_file(payload: dict[str, object], authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _tool_auth(context, authorization)
        thread_id, filename, data = _file_values(payload)
        try:
            receipt = await context.clients.discord.send_file(thread_id, Upload(filename, data), str(payload.get("caption") or "")[:2_000])
        except (ValueError, discord.DiscordException) as exc:
            raise HTTPException(404, "Discord thread was not found") from exc
        except TypeError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"status": "sent", "message_id": receipt.message_id, "url": receipt.url}

    app.add_api_route("/v1/tools/describe-image", describe_image, methods=["POST"])
    app.add_api_route("/v1/tools/set-reactions", set_reactions, methods=["POST"])
    app.add_api_route("/v1/tools/set-profile", set_profile, methods=["POST"])
    app.add_api_route("/v1/tools/send-file", send_file, methods=["POST"])


def _profile_values(payload: dict[str, object]) -> tuple[str | None, bytes | None]:
    username = payload.get("username")
    if username is not None and (not isinstance(username, str) or not MIN_DISCORD_USERNAME_LENGTH <= len(username) <= MAX_DISCORD_USERNAME_LENGTH):
        raise HTTPException(422, "username must be 2-32 characters")
    avatar = payload.get("avatar_base64")
    data: bytes | None = None
    if avatar is not None:
        if not isinstance(avatar, str):
            raise HTTPException(422, "avatar_base64 must be a string")
        try:
            data = base64.b64decode(avatar, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "avatar_base64 is invalid") from exc
        if not data or len(data) > MAX_DISCORD_UPLOAD_BYTES:
            raise HTTPException(422, "avatar exceeds the 8 MiB limit")
    if username is None and data is None:
        raise HTTPException(422, "provide username or avatar")
    return username, data


def _file_values(payload: dict[str, object]) -> tuple[str, str, bytes]:
    thread_id = str(payload.get("thread_id") or "")
    filename = Path(str(payload.get("filename") or "")).name
    encoded = payload.get("data_base64")
    if not thread_id or not filename or filename in {".", ".."} or not isinstance(encoded, str):
        raise HTTPException(422, "thread_id, filename, and data_base64 are required")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "data_base64 is invalid") from exc
    if not data or len(data) > MAX_DISCORD_UPLOAD_BYTES:
        raise HTTPException(422, "file must be non-empty and no larger than 8 MiB")
    return thread_id, filename, data
