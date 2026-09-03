# Copyright (c) 2026 Nick van der Merwe
"""Normalize Discord requests and build bounded model context."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from jinja2 import Environment, StrictUndefined, TemplateError
from jsonschema import validate

from app.models import Event, Message

if TYPE_CHECKING:
    from app.types import JsonObject

MAX_ANCESTORS = 12


def _message(value: Mapping[str, object], thread_id: str | None = None) -> Message:
    """Normalize either a canonical message or Discord REST JSON."""
    if "author_id" in value:
        return Message.model_validate(dict(value))
    author = _mapping(value.get("author"))
    reference = _mapping(value.get("message_reference") or value.get("reference"))
    mentions = [
        str(item.get("id", "")) if isinstance(item, dict) else str(item)
        for item in _sequence(value.get("mentions"))
    ]
    return Message(
        id=str(value["id"]),
        author_id=str(author.get("id", "unknown")),
        author_name=str(author.get("global_name") or author.get("username") or "unknown"),
        bot=bool(author.get("bot", False)),
        content=str(value.get("content", "")),
        timestamp=str(value.get("timestamp", "")),
        channel_id=str(value.get("channel_id", "")),
        thread_id=_string(value.get("thread_id")) or thread_id,
        reply_to=str(reference["message_id"]) if reference.get("message_id") else None,
        mentions=mentions,
        attachments=[
            cast(
                "JsonObject",
                {
                    "id": str(_mapping(item).get("id", "")),
                    "filename": _mapping(item).get("filename"),
                    "url": _mapping(item).get("url"),
                    "content_type": _mapping(item).get("content_type"),
                    "size": _mapping(item).get("size"),
                },
            )
            for item in _sequence(value.get("attachments"))
        ],
    )


def normalize_event(value: object) -> Event:
    """Use one admission normalizer for raw Discord fixtures and canonical events."""
    raw = _mapping(value)
    if isinstance(raw.get("d"), dict) and raw.get("t") == "MESSAGE_CREATE":
        value = {
            **_mapping(raw["d"]),
            **{
                key: raw[key]
                for key in ("kind", "parent_messages", "thread_messages")
                if key in raw
            },
        }
    normalized = _mapping(value)
    if "trigger" in normalized:
        trigger = _message(_mapping(normalized["trigger"]), _string(normalized.get("thread_id")))
        parent = _sequence(normalized.get("parent_messages"))
        thread = _sequence(normalized.get("thread_messages"))
    else:
        trigger = _message(normalized, _string(normalized.get("thread_id")))
        parent = _sequence(normalized.get("parent_messages"))
        thread = _sequence(normalized.get("thread_messages"))
    return Event(
        trigger=trigger,
        kind=_string(normalized.get("kind")),
        parent_messages=[_message(_mapping(item)) for item in parent],
        thread_messages=[_message(_mapping(item)) for item in thread],
        seen_ids=[str(item) for item in _sequence(normalized.get("seen_ids"))],
        anchor_id=_string(normalized.get("anchor_id")),
        raw_payload=cast("JsonObject", raw),
    )


def event_data(event: Event) -> dict[str, object]:
    return cast("dict[str, object]", event.model_dump(mode="json", exclude={"raw_payload"}))


def render_grammar(name: str, source: str, raw: object, **values: object) -> dict[str, object]:
    """Produce the inspectable raw, normalized, source, and rendered grammar layers."""
    normalized = {"messages": values.get("messages", []), "mode": values.get("mode", "")}
    try:
        rendered = (
            Environment(undefined=StrictUndefined, autoescape=True)
            .from_string(source)
            .render(**normalized)
        )
    except TemplateError as exc:
        raise ValueError from exc
    parsed = json.loads(rendered)
    if not isinstance(parsed, dict):
        raise TypeError("grammar must render an object")  # noqa: TRY003
    return cast(
        "dict[str, object]",
        {
            "name": name,
            "version": hashlib.sha256(source.encode()).hexdigest()[:12],
            "source": source,
            "raw": raw,
            "normalized": normalized,
            "rendered": rendered,
            "parsed": parsed,
        },
    )


def context(event: Event) -> dict[str, object]:
    """Select startup history or only unseen follow-up history, capped at 100 messages."""
    trigger = event.trigger
    startup = event.kind in (None, "startup")
    pool = event.parent_messages if startup else event.parent_messages + event.thread_messages
    seen = set(event.seen_ids)
    if not startup:
        pool = [m for m in pool if m.id not in seen and m.id != trigger.id]
    pool = sorted({m.id: m for m in pool}.values(), key=lambda m: (m.timestamp, m.id))[-100:]
    by_id = {m.id: m for m in (*event.parent_messages, *event.thread_messages, trigger)}
    ancestors: list[Message] = []
    parent = trigger.reply_to
    while parent and parent in by_id and len(ancestors) < MAX_ANCESTORS:
        if parent in {m.id for m in ancestors}:
            break
        item = by_id[parent]
        ancestors.append(item)
        parent = item.reply_to
    selected = list({m.id: m for m in (*ancestors[::-1], *pool, trigger)}.values())
    selected.sort(key=lambda item: (item.timestamp, item.id))
    messages = [item.model_dump(mode="json") for item in selected]
    result = {
        "schema": "wiseman.discord_context.v2",
        "trigger": trigger.model_dump(mode="json"),
        "messages": messages,
        "reply_ancestors": [m.model_dump(mode="json") for m in ancestors[::-1]],
        "surrounding": [m.model_dump(mode="json") for m in pool],
        "selected_ids": [item["id"] for item in messages],
        "raw_count": len(pool),
        "startup": startup,
    }
    schema = json.loads(
        (Path(__file__).parents[2] / "contracts" / "discord-context.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate(result, schema)
    return cast("dict[str, object]", result)


IMAGE_REFERENCE_WORDS = re.compile(
    r"\b(image|photo|picture|screenshot|attachment|chart|graph|diagram|visual)\b",
    re.IGNORECASE,
)


def image_tool_instruction(
    trigger: Mapping[str, object], reply_ancestors: Sequence[Mapping[str, object]] | None = None
) -> str:
    """Select only the current or explicitly referenced Discord image."""
    messages = [trigger]
    if not trigger.get("attachments") and IMAGE_REFERENCE_WORDS.search(
        str(trigger.get("content") or "")
    ):
        messages.extend(
            message
            for message in reversed(reply_ancestors or [])
            if any(
                str(_mapping(attachment).get("content_type") or "").startswith("image/")
                for attachment in _sequence(_mapping(message).get("attachments"))
            )
        )
        messages = messages[:2]
    images: list[str] = []
    seen: set[str] = set()
    for message in messages:
        for value in _sequence(message.get("attachments")):
            attachment = _mapping(value)
            content_type = str(attachment.get("content_type") or "")
            url = str(attachment.get("url") or attachment.get("proxy_url") or "")
            if content_type.startswith("image/") and url and url not in seen:
                images.append(url)
                seen.add(url)
    if not images:
        return ""
    source = (Path(__file__).parents[2] / "contracts" / "image-tool-instruction.j2").read_text(
        encoding="utf-8"
    )
    template = Environment(autoescape=True, undefined=StrictUndefined).from_string(source)
    return template.render(urls=images[:4]).strip()


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
