# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

import discord
import httpx
from jinja2 import Environment, StrictUndefined

from app.models import is_image_attachment
from app.phoenix import route_info

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.models import Messageable

DEFAULT_REACTION_EMOJIS = {"processing": "👀", "success": "✅", "failure": "❌"}
MAX_DISCORD_CONTENT_LENGTH = 2_000
MAX_DISCORD_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_REACTION_LENGTH = 32
MIN_DISCORD_USERNAME_LENGTH = 2
MAX_DISCORD_USERNAME_LENGTH = 32


def banner() -> str:
    info = route_info()
    model = info.get("requested_model") or "configured route"
    provider = info.get("provider")
    line = f"⚡ Route: `{model}`" + (f" via {provider}." if provider else ".")
    details = []
    for label, key in (
        ("Fallbacks", "fallback_models"),
        ("Modalities", "modalities"),
        ("Context", "context_window"),
        ("Input", "input_price"),
        ("Output", "output_price"),
        ("Cached", "cached_price"),
        ("Vision assist", "vision_assist_model"),
    ):
        if key in info:
            details.append(f"{label}: `{info[key]}`")
    return "⚡ **Wiseman thread startup**\n" + line + ("\n" + " · ".join(details) if details else "")


def thread_name(number: int) -> str:
    if number < 1:
        raise ValueError("thread number must be positive")
    return f"Gurt {number}"


def startup_embed() -> discord.Embed:
    title, _, description = banner().partition("\n")
    return discord.Embed(title=title.replace("**", ""), description=description, colour=0x57F287)


async def describe_images(messages: Sequence[Mapping[str, object]], question: str = "") -> dict[str, object]:
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for message in messages:
        for value in _sequence(message.get("attachments")):
            attachment = _mapping(value)
            url = str(attachment.get("url") or attachment.get("proxy_url") or "")
            attachment_id = str(attachment.get("id") or url)
            if is_image_attachment(attachment) and url and attachment_id not in seen:
                images.append({"id": attachment_id, "url": url})
                seen.add(attachment_id)
    if not images:
        return {"text": "", "attachments": [], "question": question or None}
    model = os.getenv("WISEMAN_VISION_MODEL", "z-ai/glm-5.3-flash")
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return {
            "text": "[Image description unavailable: vision provider is not configured.]",
            "model": model,
            "attachments": [item["id"] for item in images],
            "question": question or None,
        }
    content: list[dict[str, object]] = [
        {"type": "text", "text": _vision_question(question)},
        *({"type": "image_url", "image_url": {"url": item["url"]}} for item in images[:4]),
    ]
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{os.getenv('OPENROUTER_URL', 'https://openrouter.ai')}/api/v1/chat/completions",
                headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 500,
                },
            )
            response.raise_for_status()
            value: object = response.json()
        data = _mapping(value)
        choices = _sequence(data.get("choices"))
        first = _mapping(choices[0]) if choices else {}
        answer = _mapping(first.get("message")).get("content", "")
        if isinstance(answer, list):
            answer = "".join(str(item.get("text", "")) for item in answer if isinstance(item, dict))
        usage = data.get("usage")
        return {
            "text": str(answer),
            "model": data.get("model", model),
            "usage": usage,
            "cost": _mapping(usage).get("cost") if isinstance(usage, dict) else data.get("cost"),
            "attachments": [item["id"] for item in images],
            "question": question or None,
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return {
            "text": f"[Image description unavailable: {type(exc).__name__}].",
            "model": model,
            "attachments": [item["id"] for item in images],
            "question": question or None,
        }


def _contract(name: str) -> str:
    return (Path(__file__).parents[2] / "contracts" / name).read_text(encoding="utf-8")


def _vision_question(question: str) -> str:
    template = Environment(autoescape=True, undefined=StrictUndefined).from_string(_contract("vision-question.j2"))
    return template.render(question=question)


async def edit_delivery(message: object | None, content: str) -> bool:
    edit = getattr(message, "edit", None)
    if not callable(edit):
        return False
    await cast("Callable[..., Awaitable[object]]", edit)(content=content)
    return True


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def normalize_image_url(value: object) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip().strip("<>\"'")
    parsed = urlsplit(candidate)
    return candidate if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def split_discord_content(content: str) -> list[str]:
    if len(content) <= MAX_DISCORD_CONTENT_LENGTH:
        return [content]
    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= MAX_DISCORD_CONTENT_LENGTH:
            chunks.append(remaining)
            break
        boundary = remaining.rfind("\n", 0, MAX_DISCORD_CONTENT_LENGTH + 1)
        if boundary < MAX_DISCORD_CONTENT_LENGTH // 2:
            boundary = remaining.rfind(" ", 0, MAX_DISCORD_CONTENT_LENGTH + 1)
        if boundary <= 0:
            boundary = MAX_DISCORD_CONTENT_LENGTH
        chunks.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    return chunks


def render_progress(steps: list[str], turn_number: int | None = None) -> str:
    count = turn_number if turn_number is not None else 0
    visible = "\n".join(steps).splitlines()[-8:]
    header = f"⏳ Working · Gurt {count}" if count else "⏳ Working"
    return "\n".join([header, *visible])


async def deliver_content(message: object | None, channel: Messageable, content: str) -> None:
    chunks = split_discord_content(content)
    edited = False
    if message is not None:
        with suppress(discord.DiscordException):
            edited = await edit_delivery(message, chunks[0])
    with suppress(discord.DiscordException):
        if not edited:
            await channel.send(chunks[0])
        for chunk in chunks[1:]:
            await channel.send(chunk)
