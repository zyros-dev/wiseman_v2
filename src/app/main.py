# Copyright (c) 2026 Nick van der Merwe
"""Wiseman application entrypoint and compatibility exports."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import discord
import httpx

from app import http_api as _http_api
from app.admission import (
    _message,
    context,
    normalize_event,
)
from app.admission import (
    event_data as _event_data,
)
from app.admission import (
    image_tool_instruction as _image_tool_instruction,
)
from app.admission import (
    render_grammar as _grammar,
)
from app.engine import Engine, _state_data
from app.gateway import DISCORD_CONNECTED, Gateway, _history
from app.models import ActiveTurn, Event, Message, Messageable, State
from app.phoenix import (
    Phoenix,
    PromptHub,
)
from app.phoenix import (
    json_text as _json,
)
from app.phoenix import (
    provider_values as _provider_values,
)
from app.phoenix import (
    route_info as _route_info,
)
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_CONTENT_LENGTH,
    MAX_DISCORD_UPLOAD_BYTES,
    MAX_DISCORD_USERNAME_LENGTH,
    MAX_REACTION_LENGTH,
    MIN_DISCORD_USERNAME_LENGTH,
    THREAD_AUTO_ARCHIVE_MINUTES,
)
from app.presentation import (
    banner as _banner,
)
from app.presentation import (
    deliver_content as _deliver_content,
)
from app.presentation import (
    describe_images as _describe_images,
)
from app.presentation import (
    edit_delivery as _edit_delivery,
)
from app.presentation import (
    normalize_image_url as _normalize_image_url,
)
from app.presentation import (
    render_progress as _render_progress,
)
from app.presentation import (
    split_discord_content as _split_discord_content,
)
from app.presentation import (
    startup_embed as _startup_embed,
)
from app.presentation import (
    thread_name as _thread_name,
)
from app.runner import FakeRunner, HttpRunner, LifecycleRunner, Runner, RunnerError, SteerableRunner

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = [
    "DEFAULT_REACTION_EMOJIS",
    "DISCORD_CONNECTED",
    "MAX_DISCORD_CONTENT_LENGTH",
    "MAX_DISCORD_UPLOAD_BYTES",
    "MAX_DISCORD_USERNAME_LENGTH",
    "MAX_REACTION_LENGTH",
    "MIN_DISCORD_USERNAME_LENGTH",
    "THREAD_AUTO_ARCHIVE_MINUTES",
    "ActiveTurn",
    "Engine",
    "Event",
    "FakeRunner",
    "Gateway",
    "HttpRunner",
    "LifecycleRunner",
    "Message",
    "Messageable",
    "Phoenix",
    "PromptHub",
    "Runner",
    "RunnerError",
    "State",
    "SteerableRunner",
    "_banner",
    "_deliver_content",
    "_describe_images",
    "_edit_delivery",
    "_event_data",
    "_grammar",
    "_history",
    "_image_tool_instruction",
    "_json",
    "_message",
    "_normalize_image_url",
    "_provider_values",
    "_render_progress",
    "_route_info",
    "_split_discord_content",
    "_startup_embed",
    "_state_data",
    "_thread_name",
    "asyncio",
    "context",
    "create_app",
    "discord",
    "httpx",
    "normalize_event",
]


def create_app(engine: Engine | None = None, token: str = "", discord_token: str = "") -> FastAPI:
    """Build the HTTP app while preserving the historical test patch points."""
    _http_api.describe_images = _describe_images
    return _http_api.create_app(engine, token, discord_token)


phoenix = Phoenix(
    os.getenv("PHOENIX_OTLP_ENDPOINT", ""),
    os.getenv("PHOENIX_API_KEY", ""),
    os.getenv("PHOENIX_PROJECT", "wiseman-v2"),
    os.getenv("WISEMAN_AUDIT_DIR"),
)
runner: Runner = (
    HttpRunner(os.environ["WISEMAN_RUNNER_URL"], os.getenv("WISEMAN_RUNNER_API_TOKEN", ""))
    if os.getenv("WISEMAN_RUNNER_URL")
    else FakeRunner()
)
engine = Engine(phoenix, runner)
app = create_app(engine, os.getenv("WISEMAN_REPLAY_TOKEN", ""), os.getenv("DISCORD_BOT_TOKEN", ""))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
