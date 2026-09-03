# Copyright (c) 2026 Nick van der Merwe
"""FastAPI application and provider/tool endpoints."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import discord
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from app.admission import normalize_event
from app.engine import Engine
from app.gateway import Gateway
from app.phoenix import Phoenix, PromptHub
from app.phoenix import json_text as _json
from app.phoenix import provider_values as _provider_values
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_UPLOAD_BYTES,
    MAX_DISCORD_USERNAME_LENGTH,
    MIN_DISCORD_USERNAME_LENGTH,
)
from app.presentation import (
    describe_images as _describe_images,
)
from app.presentation import (
    normalize_image_url as _normalize_image_url,
)
from app.runner import FakeRunner, HttpRunner
from app.temporal_runtime import TemporalRuntime

describe_images = _describe_images

UPSTREAM_RETRY_ATTEMPTS = 3
UPSTREAM_RETRY_STATUSES = frozenset({404, 408, 425, 429})
UPSTREAM_SERVER_ERROR = 500
HTTP_NOT_FOUND = 404


def create_app(  # noqa: C901, PLR0915
    engine: Engine | None = None, token: str = "", discord_token: str = ""
) -> FastAPI:
    """Create health, raw Discord replay, and Phoenix inspection endpoints."""
    engine = engine or Engine(
        Phoenix(
            os.getenv("PHOENIX_OTLP_ENDPOINT", ""),
            os.getenv("PHOENIX_API_KEY", ""),
            os.getenv("PHOENIX_PROJECT", "wiseman-v2"),
            os.getenv("WISEMAN_AUDIT_DIR"),
        ),
        HttpRunner(os.getenv("WISEMAN_RUNNER_URL", ""), token)
        if os.getenv("WISEMAN_RUNNER_URL")
        else FakeRunner(),
        PromptHub(os.getenv("PHOENIX_PROMPT_HUB_URL", ""), os.getenv("PHOENIX_API_KEY", "")),
    )
    app = FastAPI(title="wiseman-v2", docs_url=None, redoc_url=None)
    allowlist = {
        int(value) for value in os.getenv("WISEMAN_DISCORD_ALLOWLIST", "").split(",") if value
    }
    activity_file = os.getenv("WISEMAN_ACTIVITY_FILE")
    profile_file = os.getenv("WISEMAN_PROFILE_FILE")
    if profile_file is None and activity_file:
        profile_file = str(Path(activity_file).with_name("profile.json"))
    bot = Gateway(engine, allowlist, activity_file, profile_file)
    app.state.gateway = bot
    temporal = (
        TemporalRuntime(os.environ["TEMPORAL_ADDRESS"], os.getenv("TEMPORAL_TASK_QUEUE", "wiseman"))
        if os.getenv("TEMPORAL_ADDRESS")
        else None
    )
    bot.temporal = temporal
    task: asyncio.Task[None] | None = None

    @app.on_event("startup")
    async def start_discord() -> None:
        nonlocal task
        if temporal is not None:
            await temporal.start()
        if discord_token:
            task = asyncio.create_task(bot.run_forever(discord_token))

    @app.on_event("shutdown")
    async def stop_discord() -> None:
        if task is not None:
            task.cancel()
        await bot.close()
        if temporal is not None:
            await temporal.close()

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready() -> dict[str, str]:
        if discord_token and not bot.is_ready():
            raise HTTPException(503, "Discord gateway is not ready")
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/phoenix/events")
    async def events() -> list[dict[str, Any]]:
        return engine.phoenix.records

    def replay_authorized(x_replay_token: str | None) -> None:
        expected = os.getenv("WISEMAN_REPLAY_TOKEN", token)
        if expected and not hmac.compare_digest(x_replay_token or "", expected):
            raise HTTPException(401, "invalid replay token")

    @app.get("/v1/phoenix/audits/{audit_id}")
    async def audit(
        audit_id: str, x_replay_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        replay_authorized(x_replay_token)
        value = engine.phoenix.audit(audit_id)
        if value is None:
            raise HTTPException(404, "Phoenix admission audit was not found")
        return value

    @app.post("/v1/responses")
    async def responses(  # noqa: C901
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        expected = os.getenv("WISEMAN_PROVIDER_TOKEN", token)
        if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
            raise HTTPException(401, "invalid provider token")
        payload = await request.json()
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise HTTPException(503, "OpenRouter is not configured")
        trace = f"provider-{hashlib.sha256(_json(payload).encode()).hexdigest()[:16]}"

        async def stream() -> AsyncIterator[bytes]:
            usage: object = None
            cost: object = None
            served_model: object = None
            async with (
                httpx.AsyncClient(timeout=300) as client,
            ):
                stream_started = False
                for attempt in range(UPSTREAM_RETRY_ATTEMPTS):
                    try:
                        async with client.stream(
                            "POST",
                            f"{os.getenv('OPENROUTER_URL', 'https://openrouter.ai')}/api/v1/responses",
                            headers={
                                "authorization": f"Bearer {key}",
                                "content-type": "application/json",
                            },
                            json=payload,
                        ) as response:
                            if response.is_error:
                                detail = (await response.aread()).decode(errors="replace")[:1_000]
                                retryable = (
                                    response.status_code in UPSTREAM_RETRY_STATUSES
                                    or response.status_code >= UPSTREAM_SERVER_ERROR
                                ) and not (
                                    response.status_code == HTTP_NOT_FOUND
                                    and "No endpoints found that support image input" in detail
                                )
                                if retryable and attempt + 1 < UPSTREAM_RETRY_ATTEMPTS:
                                    await asyncio.sleep(0.5 * (attempt + 1))
                                    continue
                                error = f"OpenRouter returned HTTP {response.status_code}: {detail}"
                                raise RuntimeError(error)
                            async for line in response.aiter_lines():
                                stream_started = True
                                if line.startswith("data:"):
                                    try:
                                        body = json.loads(line[5:].strip())
                                        if isinstance(body, dict):
                                            usage, cost, served_model = _provider_values(
                                                body, usage, cost, served_model
                                            )
                                    except ValueError:
                                        pass
                                yield f"{line}\n".encode()
                            break
                    except httpx.HTTPError:
                        if stream_started or attempt + 1 >= UPSTREAM_RETRY_ATTEMPTS:
                            raise
                        await asyncio.sleep(0.5 * (attempt + 1))
            await engine.phoenix.record(
                trace,
                "provider",
                request=payload,
                requested_model=payload.get("model"),
                served_model=served_model,
                usage=usage,
                cost=cost,
            )

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/tools/describe-image")
    async def describe_image(
        payload: dict[str, Any],
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        expected = os.getenv("WISEMAN_PROVIDER_TOKEN", token)
        if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
            raise HTTPException(401, "invalid tool token")
        url = _normalize_image_url(payload.get("url"))
        if not url:
            raise HTTPException(422, "image URL must use HTTP or HTTPS")
        attachment_id = str(payload.get("attachment_id") or url.rstrip("/").split("/")[-2])
        result = await describe_images(
            [{"attachments": [{"id": attachment_id, "content_type": "image/*", "url": url}]}],
            str(payload.get("question") or "")[:2_000],
        )
        trace = str(
            payload.get("thread_id")
            or f"vision-tool-{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        )
        await engine.phoenix.record(trace, "vision_tool", **result)
        return result

    def tool_authorized(authorization: str | None) -> None:
        expected = os.getenv("WISEMAN_MCP_TOKEN", os.getenv("WISEMAN_PROVIDER_TOKEN", token))
        if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
            raise HTTPException(401, "invalid tool token")

    @app.post("/v1/tools/set-reactions")
    async def set_reactions(
        payload: dict[str, Any], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        tool_authorized(authorization)
        try:
            values = {
                phase: str(payload[phase]) for phase in DEFAULT_REACTION_EMOJIS if phase in payload
            }
            configured = engine.set_reaction_emojis(values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        bot.persist_profile()
        return {"reaction_emojis": configured}

    @app.post("/v1/tools/set-profile")
    async def set_profile(
        payload: dict[str, Any], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        tool_authorized(authorization)
        if os.getenv("WISEMAN_ALLOW_PROFILE_EDITS", "0") != "1":
            raise HTTPException(403, "profile edits are disabled")
        username = payload.get("username")
        if username is not None and (
            not isinstance(username, str)
            or not MIN_DISCORD_USERNAME_LENGTH <= len(username) <= MAX_DISCORD_USERNAME_LENGTH
        ):
            raise HTTPException(422, "username must be 2-32 characters")
        kwargs: dict[str, Any] = {}
        if username is not None:
            kwargs["username"] = username
        avatar = payload.get("avatar_base64")
        if avatar is not None:
            if not isinstance(avatar, str):
                raise HTTPException(422, "avatar_base64 must be a string")
            try:
                data = base64.b64decode(avatar, validate=True)
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, "avatar_base64 is invalid") from exc
            if not data or len(data) > MAX_DISCORD_UPLOAD_BYTES:
                raise HTTPException(422, "avatar exceeds the 8 MiB limit")
            kwargs["avatar"] = data
        if not kwargs:
            raise HTTPException(422, "provide username or avatar")
        if bot.user is None:
            raise HTTPException(503, "Discord gateway is not ready")
        await bot.user.edit(**kwargs)
        return {"status": "updated", "username": getattr(bot.user, "name", None)}

    @app.post("/v1/tools/send-file")
    async def send_file(
        payload: dict[str, Any], authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        tool_authorized(authorization)
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
        try:
            channel = await bot.fetch_channel(int(thread_id))
        except (ValueError, discord.DiscordException) as exc:
            raise HTTPException(404, "Discord thread was not found") from exc
        if not isinstance(channel, discord.Thread):
            raise HTTPException(422, "file delivery requires a Discord thread")
        content = str(payload.get("caption") or "")[:2_000]
        message = await channel.send(
            content=content, file=discord.File(io.BytesIO(data), filename=filename)
        )
        return {"status": "sent", "message_id": str(message.id), "url": str(message.jump_url)}

    @app.post("/v1/replay/discord")
    async def replay(
        payload: dict[str, Any], x_replay_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        replay_authorized(x_replay_token)
        try:
            event = normalize_event(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, "invalid Discord event") from exc
        if temporal is not None:
            await temporal.submit(event.model_dump(mode="json"))
            return {"status": "queued", "message_id": event.trigger.id}
        return await engine.handle(event)

    @app.post("/v1/replay/phoenix/{audit_id}")
    async def replay_audit(
        audit_id: str, x_replay_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        """Replay the exact raw request captured by the Phoenix admission audit."""
        replay_authorized(x_replay_token)
        artifact = engine.phoenix.audit(audit_id)
        if artifact is None:
            raise HTTPException(404, "Phoenix admission audit was not found")
        payload = artifact.get("raw_request")
        if not isinstance(payload, dict):
            raise HTTPException(422, "Phoenix audit has no replayable raw request")
        try:
            event = normalize_event(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, "Phoenix audit contains an invalid Discord event") from exc
        if temporal is not None:
            await temporal.submit(event.model_dump(mode="json"))
            return {"status": "queued", "message_id": event.trigger.id, "audit_id": audit_id}
        result = await engine.handle(event)
        return {**result, "audit_id": audit_id}

    app.add_api_route("/v1/discord/events", replay, methods=["POST"])

    return app
