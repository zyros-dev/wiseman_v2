# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from urllib.parse import urlsplit

import discord

from app.phoenix import route_info

DEFAULT_REACTION_EMOJIS = {"processing": "👀", "success": "✅", "failure": "❌"}
MAX_DISCORD_CONTENT_LENGTH = 2_000
MAX_PROGRESS_PREVIEW_LENGTH = 200
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


def thread_name(content: str, bot_id: str = "") -> str:
    return (" ".join(content.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>", "").split()).strip(" -") or "Wiseman thread")[:100]


def startup_embed() -> discord.Embed:
    title, _, description = banner().partition("\n")
    return discord.Embed(title=title.replace("**", ""), description=description, colour=0x57F287)


def normalize_image_url(value: object) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip().strip("<>\"'")
    parsed = urlsplit(candidate)
    return candidate if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def render_progress(steps: list[str], turn_number: int | None = None) -> str:
    count = turn_number if turn_number is not None else 0
    visible = [
        line if len(line) <= MAX_PROGRESS_PREVIEW_LENGTH else line[: MAX_PROGRESS_PREVIEW_LENGTH - 3] + "..." for line in "\n".join(steps).splitlines()[-8:]
    ]
    header = f"⏳ Working · Turn {count}" if count else "⏳ Working"
    return "\n".join([header, *visible])
