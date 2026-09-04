# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import discord
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.clients.client_interfaces import (
        DiscordClient,
        PhoenixClient,
        PromptClient,
        RunnerClient,
        TemporalClient,
    )
    from app.models import Event
    from app.types import JsonObject

from app.admission import normalize_event
from app.clients import ClientContainer, ClientMode, ClientSettings, build_clients
from app.clients.real_clients import RealDependencies, real_services
from app.engine import Engine, EngineConfig
from app.gateway import Gateway
from app.phoenix import json_text, provider_values
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_UPLOAD_BYTES,
    MAX_DISCORD_USERNAME_LENGTH,
    MIN_DISCORD_USERNAME_LENGTH,
    describe_images,
    normalize_image_url,
)
from app.temporal_runtime import TemporalRuntime, configure_engine


@dataclass(slots=True)
class _Context:
    engine: Engine
    bot: Gateway
    clients: ClientContainer
    temporal: TemporalRuntime | None
    token: str
    discord_token: str
    replay_state: dict[str, JsonObject] = field(default_factory=dict)
    discord_task: asyncio.Task[None] | None = None


class ProfileInputError(ValueError):
    def __init__(self, field: str) -> None:
        super().__init__(
            {
                "username": "username must be 2-32 characters",
                "avatar_type": "avatar_base64 must be a string",
                "avatar_encoding": "avatar_base64 is invalid",
                "avatar_size": "avatar exceeds the 8 MiB limit",
                "missing": "provide username or avatar",
            }[field]
        )


def create_app(
    engine: Engine | None = None,
    token: str = "",
    discord_token: str = "",
    clients: ClientContainer | None = None,
) -> FastAPI:
    context = _context(engine, token, discord_token, clients)
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


def _context(engine: Engine | None, token: str, discord_token: str, clients: ClientContainer | None) -> _Context:
    settings = clients.settings if clients is not None else ClientSettings.from_env()
    token = token or settings.runner_token
    discord_token = discord_token or settings.discord_token
    services = real_services(settings) if clients is None else None
    if engine is None:
        if clients is not None:
            engine = Engine(clients=clients)
        else:
            assert services is not None
            engine = Engine(EngineConfig(services[0], services[2], services[1]))
    configure_engine(engine)
    allowlist = {int(value) for value in os.getenv("WISEMAN_DISCORD_ALLOWLIST", "").split(",") if value}
    activity_file = os.getenv("WISEMAN_ACTIVITY_FILE")
    profile_file = os.getenv("WISEMAN_PROFILE_FILE") or (
        str(Path(activity_file).with_name("profile.json")) if activity_file else None
    )
    bot = Gateway(engine, allowlist, activity_file, profile_file)
    temporal = (
        TemporalRuntime(settings.temporal_address, settings.temporal_queue) if settings.temporal_address else None
    )
    bot.temporal = temporal
    if clients is None:
        services = services or real_services(settings)
        clients = build_clients(
            ClientMode.REAL,
            settings,
            RealDependencies(
                discord=cast("DiscordClient", bot),
                temporal=cast("TemporalClient", temporal),
                phoenix=cast("PhoenixClient", engine.config.phoenix),
                prompts=cast("PromptClient", engine.config.prompts),
                runner=cast("RunnerClient", engine.config.runner),
            ),
        )
    engine.config = EngineConfig(clients.phoenix, clients.runner, clients.prompts, engine.config.context)
    assert clients is not None
    return _Context(engine, bot, clients, temporal, token, discord_token)


async def _start(context: _Context) -> None:
    if context.temporal is not None:
        await context.temporal.start()
    if context.discord_token:
        context.discord_task = asyncio.create_task(context.bot.run_forever(context.discord_token))


async def _stop(context: _Context) -> None:
    if context.discord_task is not None:
        context.discord_task.cancel()
    await context.bot.close()
    if context.temporal is not None:
        await context.temporal.close()


@asynccontextmanager
async def _lifespan(context: _Context, _app: FastAPI) -> AsyncIterator[None]:
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

    async def events() -> list[dict[str, object]]:
        return context.engine.config.phoenix.records

    app.add_api_route("/healthz", health, methods=["GET"])
    app.add_api_route("/readyz", ready, methods=["GET"])
    app.add_api_route("/metrics", metrics, methods=["GET"])
    app.add_api_route("/v1/phoenix/events", events, methods=["GET"])


def _replay_auth(context: _Context, supplied: str | None) -> None:
    expected = os.getenv("WISEMAN_REPLAY_TOKEN", context.token)
    if expected and not hmac.compare_digest(supplied or "", expected):
        raise HTTPException(401, "invalid replay token")


async def _replay_local(context: _Context, event: Event) -> dict[str, object]:
    key = event.trigger.thread_id or event.trigger.channel_id
    result = dict(await context.engine.handle(event, state_data=context.replay_state.get(key, {})))
    if isinstance(result.get("state"), dict):
        context.replay_state[key] = cast("JsonObject", result["state"])
    return result


def _register_replay(app: FastAPI, context: _Context) -> None:
    async def events_audit(audit_id: str, x_replay_token: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _replay_auth(context, x_replay_token)
        value = context.engine.config.phoenix.audit(audit_id)
        if value is None:
            raise HTTPException(404, "Phoenix admission audit was not found")
        return value

    async def replay(
        payload: dict[str, object], x_replay_token: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _replay_auth(context, x_replay_token)
        event = _event(payload, "invalid Discord event")
        if context.temporal is not None:
            await context.temporal.submit(event.model_dump(mode="json"))
            return {"status": "queued", "message_id": event.trigger.id}
        return await _replay_local(context, event)

    async def replay_audit(audit_id: str, x_replay_token: Annotated[str | None, Header()] = None) -> dict[str, object]:
        _replay_auth(context, x_replay_token)
        artifact = context.engine.config.phoenix.audit(audit_id)
        if artifact is None:
            raise HTTPException(404, "Phoenix admission audit was not found")
        payload = artifact.get("raw_request")
        if not isinstance(payload, dict):
            raise HTTPException(422, "Phoenix audit has no replayable raw request")
        event = _event(payload, "Phoenix audit contains an invalid Discord event")
        if context.temporal is not None:
            await context.temporal.submit(event.model_dump(mode="json"))
            return {"status": "queued", "message_id": event.trigger.id, "audit_id": audit_id}
        return {**(await _replay_local(context, event)), "audit_id": audit_id}

    app.add_api_route("/v1/phoenix/audits/{audit_id}", events_audit, methods=["GET"])
    app.add_api_route("/v1/replay/discord", replay, methods=["POST"])
    app.add_api_route("/v1/replay/phoenix/{audit_id}", replay_audit, methods=["POST"])
    app.add_api_route("/v1/discord/events", replay, methods=["POST"])


def _event(payload: dict[str, object], detail: str) -> Event:
    try:
        return normalize_event(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, detail) from exc


def _register_provider(app: FastAPI, context: _Context) -> None:
    async def responses(request: Request, authorization: Annotated[str | None, Header()] = None) -> StreamingResponse:
        expected = os.getenv("WISEMAN_PROVIDER_TOKEN", context.token)
        if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
            raise HTTPException(401, "invalid provider token")
        payload = await request.json()
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise HTTPException(503, "OpenRouter is not configured")
        trace = f"provider-{hashlib.sha256(json_text(payload).encode()).hexdigest()[:16]}"
        return StreamingResponse(_provider_stream(context, payload, key, trace), media_type="text/event-stream")

    app.add_api_route("/v1/responses", responses, methods=["POST"])


async def _provider_stream(context: _Context, payload: object, key: str, trace: str) -> AsyncIterator[bytes]:
    usage: object = None
    cost: object = None
    served_model: object = None
    async with (
        httpx.AsyncClient(timeout=300) as client,
        client.stream(
            "POST",
            f"{os.getenv('OPENROUTER_URL', 'https://openrouter.ai')}/api/v1/responses",
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            json=payload,
        ) as response,
    ):
        if response.is_error:
            detail = (await response.aread()).decode(errors="replace")[:1_000]
            message = f"OpenRouter returned HTTP {response.status_code}: {detail}"
            raise RuntimeError(message)
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                try:
                    value: object = json.loads(line[5:].strip())
                    if isinstance(value, dict):
                        usage, cost, served_model = provider_values(value, usage, cost, served_model)
                except ValueError:
                    pass
            yield f"{line}\n".encode()
    await context.engine.config.phoenix.record(
        trace,
        "provider",
        request=payload,
        requested_model=payload.get("model") if isinstance(payload, dict) else None,
        served_model=served_model,
        usage=usage,
        cost=cost,
    )


def _tool_auth(context: _Context, supplied: str | None) -> None:
    expected = os.getenv("WISEMAN_MCP_TOKEN", os.getenv("WISEMAN_PROVIDER_TOKEN", context.token))
    if expected and not hmac.compare_digest(supplied or "", f"Bearer {expected}"):
        raise HTTPException(401, "invalid tool token")


def _register_tools(app: FastAPI, context: _Context) -> None:
    async def describe_image(
        payload: dict[str, object], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _tool_auth(context, authorization)
        url = normalize_image_url(payload.get("url"))
        if not url:
            raise HTTPException(422, "image URL must use HTTP or HTTPS")
        attachment_id = str(payload.get("attachment_id") or url.rstrip("/").split("/")[-2])
        result = await describe_images(
            [{"attachments": [{"id": attachment_id, "content_type": "image/*", "url": url}]}],
            str(payload.get("question") or "")[:2_000],
        )
        trace = str(payload.get("thread_id") or f"vision-tool-{hashlib.sha256(url.encode()).hexdigest()[:16]}")
        await context.engine.config.phoenix.record(trace, "vision_tool", **result)
        return result

    async def set_reactions(
        payload: dict[str, object], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _tool_auth(context, authorization)
        try:
            values = {phase: str(payload[phase]) for phase in DEFAULT_REACTION_EMOJIS if phase in payload}
            configured = context.engine.set_reaction_emojis(values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        context.bot.persist_profile()
        return {"reaction_emojis": configured}

    async def set_profile(
        payload: dict[str, object], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _tool_auth(context, authorization)
        if os.getenv("WISEMAN_ALLOW_PROFILE_EDITS", "0") != "1":
            raise HTTPException(403, "profile edits are disabled")
        try:
            username, avatar = _profile_values(payload)
        except ProfileInputError as exc:
            raise HTTPException(422, str(exc)) from exc
        if context.bot.user is None:
            raise HTTPException(503, "Discord gateway is not ready")
        if username is not None and avatar is not None:
            await context.bot.user.edit(username=username, avatar=avatar)
        elif username is not None:
            await context.bot.user.edit(username=username)
        else:
            await context.bot.user.edit(avatar=avatar)
        return {"status": "updated", "username": getattr(context.bot.user, "name", None)}

    async def send_file(
        payload: dict[str, object], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _tool_auth(context, authorization)
        thread_id, filename, data = _file_values(payload)
        try:
            channel = await context.bot.fetch_channel(int(thread_id))
        except (ValueError, discord.DiscordException) as exc:
            raise HTTPException(404, "Discord thread was not found") from exc
        if not isinstance(channel, discord.Thread):
            raise HTTPException(422, "file delivery requires a Discord thread")
        message = await channel.send(
            content=str(payload.get("caption") or "")[:2_000],
            file=discord.File(io.BytesIO(data), filename=filename),
        )
        return {"status": "sent", "message_id": str(message.id), "url": str(message.jump_url)}

    app.add_api_route("/v1/tools/describe-image", describe_image, methods=["POST"])
    app.add_api_route("/v1/tools/set-reactions", set_reactions, methods=["POST"])
    app.add_api_route("/v1/tools/set-profile", set_profile, methods=["POST"])
    app.add_api_route("/v1/tools/send-file", send_file, methods=["POST"])


def _profile_values(payload: dict[str, object]) -> tuple[str | None, bytes | None]:
    username = payload.get("username")
    if username is not None and (
        not isinstance(username, str) or not MIN_DISCORD_USERNAME_LENGTH <= len(username) <= MAX_DISCORD_USERNAME_LENGTH
    ):
        raise ProfileInputError("username")
    avatar = payload.get("avatar_base64")
    data: bytes | None = None
    if avatar is not None:
        if not isinstance(avatar, str):
            raise ProfileInputError("avatar_type")
        try:
            data = base64.b64decode(avatar, validate=True)
        except (ValueError, TypeError) as exc:
            raise ProfileInputError("avatar_encoding") from exc
        if not data or len(data) > MAX_DISCORD_UPLOAD_BYTES:
            raise ProfileInputError("avatar_size")
    if username is None and data is None:
        raise ProfileInputError("missing")
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
