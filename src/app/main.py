# Copyright (c) 2026 Nick van der Merwe
"""Small, real Wiseman runtime: Discord admission, durable turn semantics, and Phoenix evidence."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

import discord
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from jinja2 import Environment, StrictUndefined, TemplateError
from jsonschema import validate
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_span_in_context
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from app.temporal_runtime import TemporalRuntime

LOGGER = logging.getLogger("wiseman")

DEFAULT_REACTION_EMOJIS = {"processing": "👀", "success": "✅", "failure": "❌"}
MAX_DISCORD_CONTENT_LENGTH = 2_000
MAX_DISCORD_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_REACTION_LENGTH = 32
MIN_DISCORD_USERNAME_LENGTH = 2
MAX_DISCORD_USERNAME_LENGTH = 32


class Message(BaseModel):
    """The bounded message shape shared by Discord and replay admission."""

    model_config = ConfigDict(extra="ignore")
    id: str
    author_id: str
    author_name: str = "unknown"
    bot: bool = False
    content: str = Field(default="", max_length=4000)
    timestamp: str = ""
    channel_id: str
    thread_id: str | None = None
    reply_to: str | None = None
    mentions: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class Event(BaseModel):
    """Raw Discord-shaped event accepted by the HTTP replay harness."""

    model_config = ConfigDict(extra="ignore")
    trigger: Message
    kind: str | None = None
    parent_messages: list[Message] = Field(default_factory=list)
    thread_messages: list[Message] = Field(default_factory=list)
    seen_ids: list[str] = Field(default_factory=list)
    anchor_id: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class Messageable(Protocol):
    """Minimal Discord channel surface needed for turn delivery."""

    async def send(self, content: str) -> object: ...


def _message(value: dict[str, Any], thread_id: str | None = None) -> Message:
    """Normalize either a canonical message or Discord REST JSON."""
    if "author_id" in value:
        return Message.model_validate(value)
    author = value.get("author") or {}
    reference = value.get("message_reference") or value.get("reference") or {}
    mentions = [
        str(item.get("id", "")) if isinstance(item, dict) else str(item)
        for item in value.get("mentions", [])
    ]
    return Message(
        id=str(value["id"]),
        author_id=str(author.get("id", "unknown")),
        author_name=str(author.get("global_name") or author.get("username") or "unknown"),
        bot=bool(author.get("bot", False)),
        content=str(value.get("content", "")),
        timestamp=str(value.get("timestamp", "")),
        channel_id=str(value.get("channel_id", "")),
        thread_id=value.get("thread_id", thread_id),
        reply_to=str(reference["message_id"]) if reference.get("message_id") else None,
        mentions=mentions,
        attachments=[
            {
                "id": str(item.get("id", "")),
                "filename": item.get("filename"),
                "url": item.get("url"),
                "content_type": item.get("content_type"),
                "size": item.get("size"),
            }
            for item in value.get("attachments", [])
        ],
    )


def normalize_event(value: dict[str, Any]) -> Event:
    """Use one admission normalizer for raw Discord fixtures and canonical events."""
    raw = value
    if isinstance(value.get("d"), dict) and value.get("t") == "MESSAGE_CREATE":
        value = {
            **value["d"],
            **{
                key: value[key]
                for key in ("kind", "parent_messages", "thread_messages")
                if key in value
            },
        }
    if "trigger" in value:
        trigger = _message(value["trigger"], value.get("thread_id"))
        parent = value.get("parent_messages", [])
        thread = value.get("thread_messages", [])
    else:
        trigger = _message(value, value.get("thread_id"))
        parent = value.get("parent_messages", [])
        thread = value.get("thread_messages", [])
    return Event(
        trigger=trigger,
        kind=value.get("kind"),
        parent_messages=[_message(item) for item in parent],
        thread_messages=[_message(item) for item in thread],
        seen_ids=[str(item) for item in value.get("seen_ids", [])],
        anchor_id=value.get("anchor_id"),
        raw_payload=raw,
    )


@dataclass
class State:
    """The small durable state that Temporal would persist between activities."""

    codex_thread: str | None = None
    seen: set[str] = field(default_factory=set)
    processed: set[str] = field(default_factory=set)
    turn: int = 0
    last_activity: float = 0.0
    closed: bool = False


@dataclass
class ActiveTurn:
    """The Discord delivery that can receive steering while Codex is running."""

    trigger_id: str
    delivery_id: str | None = None


class Phoenix:
    """Record every semantic layer immediately and optionally forward it to Phoenix."""

    def __init__(
        self,
        endpoint: str = "",
        key: str = "",
        project: str = "",
        audit_dir: str | Path | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.records: list[dict[str, Any]] = []
        self.audits: dict[str, dict[str, Any]] = {}
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.roots: dict[str, Any] = {}
        self.contexts: dict[str, Any] = {}
        self.provider = TracerProvider(
            resource=Resource.create(
                {"service.name": "wiseman-v2", "openinference.project.name": project}
            )
        )
        if endpoint:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            if project:
                headers["x-project-name"] = project
            self.provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
            )
        self.tracer = self.provider.get_tracer("wiseman-v2")

    async def record(self, trace: str, node: str, **data: object) -> None:
        item = {"trace": trace, "node": node, **{k: v for k, v in data.items() if v is not None}}
        self.records.append(item)
        if node == "admission" and isinstance(data.get("audit_id"), str):
            audit_id = data["audit_id"]
            artifact = {
                "schema": "wiseman.admission.audit.v1",
                "audit_id": audit_id,
                "trace": trace,
                "captured_at": time.time(),
                **item,
            }
            self.audits[audit_id] = artifact
            self._persist_audit(audit_id, artifact)
        if trace not in self.roots:
            root = self.tracer.start_span("wiseman.turn")
            self.roots[trace] = root
            self.contexts[trace] = set_span_in_context(root)
        with self.tracer.start_as_current_span(node, context=self.contexts[trace]) as span:
            span.set_attribute("openinference.session.id", trace)
            for key, value in item.items():
                span.set_attribute(
                    f"wiseman.{key}", _json(value) if not isinstance(value, str) else value
                )
        terminal = node in {"failure", "provider", "vision_tool"} or (
            node == "reaction" and "✅" in _json(data.get("operations", []))
        )
        if terminal:
            self.roots.pop(trace).end()
            self.contexts.pop(trace)

    def audit(self, audit_id: str) -> dict[str, Any] | None:
        """Return the admission artifact, including one recovered after a restart."""
        value = self.audits.get(audit_id)
        if value is None and self.audit_dir is not None:
            try:
                loaded = json.loads(self._audit_path(audit_id).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("audit_id") == audit_id:
                value = loaded
                self.audits[audit_id] = loaded
        return dict(value) if value is not None else None

    def _audit_path(self, audit_id: str) -> Path:
        digest = hashlib.sha256(audit_id.encode()).hexdigest()
        return (self.audit_dir or Path()) / f"{digest}.json"

    def _persist_audit(self, audit_id: str, artifact: dict[str, Any]) -> None:
        if self.audit_dir is None:
            return
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            target = self._audit_path(audit_id)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(_json(artifact), encoding="utf-8")
            temporary.replace(target)
        except OSError:
            LOGGER.exception("Could not persist Phoenix audit %s", audit_id)


class PromptHub:
    """Fetch the active Phoenix prompt source, with a deterministic local default."""

    def __init__(self, url: str = "", key: str = "") -> None:
        self.url, self.key = url.rstrip("/"), key

    async def source(self, kind: str) -> str:
        local_kind = {"startup": "startup-context", "followup": "followup-context"}.get(kind, kind)
        default_path = Path(__file__).parents[2] / "contracts" / f"{local_kind}.json"
        fallback = {
            "wiseman-soul": os.getenv("WISEMAN_SOUL", ""),
            "wiseman-runtime": os.getenv("WISEMAN_RUNTIME", ""),
        }
        default = (
            default_path.read_text(encoding="utf-8")
            if default_path.exists()
            else fallback.get(kind, "")
        )
        if not self.url:
            return default
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    f"{self.url}/v1/prompts/{kind}/latest",
                    headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                )
                response.raise_for_status()
                value = response.json()
                data = value.get("data", value)
                if isinstance(data, dict):
                    if isinstance(data.get("source"), str):
                        return data["source"][:100_000]
                    if data.get("template") is not None:
                        return _json(
                            {
                                "model": data.get("model_name"),
                                "template": data["template"],
                            }
                        )[:100_000]
                return default
        except (httpx.HTTPError, ValueError, AttributeError):
            return default


MAX_ANCESTORS = 12
THREAD_AUTO_ARCHIVE_MINUTES = 60
THREAD_NAME_LIMIT = 100
UPSTREAM_RETRY_ATTEMPTS = 3
UPSTREAM_RETRY_STATUSES = frozenset({404, 408, 425, 429})
UPSTREAM_SERVER_ERROR = 500
HTTP_NOT_FOUND = 404
TURN_TOTAL = Counter("wiseman_turns_total", "Accepted Discord turns")
TURN_FAILURES = Counter("wiseman_turn_failures_total", "Failed Discord turns")
DISCORD_CONNECTED = Gauge("wiseman_discord_connected", "Discord gateway connection state")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _provider_values(
    body: dict[str, Any], usage: object, cost: object, model: object
) -> tuple[object, object, object]:
    nested = body.get("response") or body.get("data") or body
    if not isinstance(nested, dict):
        return usage, cost, model
    next_usage = nested.get("usage", body.get("usage", usage))
    next_cost = nested.get("cost", body.get("cost", cost))
    if next_cost is None and isinstance(next_usage, dict):
        next_cost = next_usage.get("cost")
    return next_usage, next_cost, nested.get("model", body.get("model", model))


def _event_data(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="json", exclude={"raw_payload"})


def _grammar(name: str, source: str, raw: object, **values: object) -> dict[str, Any]:
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
    return {
        "name": name,
        "version": hashlib.sha256(source.encode()).hexdigest()[:12],
        "source": source,
        "raw": raw,
        "normalized": normalized,
        "rendered": rendered,
        "parsed": parsed,
    }


def context(event: Event) -> dict[str, Any]:
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
    return result


def _image_tool_instruction(messages: list[dict[str, Any]]) -> str:
    """Make image handling explicit when the model cannot see Discord attachments natively."""
    images: list[str] = []
    seen: set[str] = set()
    for message in messages:
        for attachment in message.get("attachments", []):
            content_type = str(attachment.get("content_type") or "")
            url = str(attachment.get("url") or attachment.get("proxy_url") or "")
            if content_type.startswith("image/") and url and url not in seen:
                images.append(url)
                seen.add(url)
    if not images:
        return ""
    urls = "\n".join(f"- {url}" for url in images[:4])
    return (
        "This turn includes Discord image attachment(s). Before answering, you MUST run "
        "`/usr/local/bin/wiseman-image` once for each relevant image URL below. Use `--question` "
        "when the user asks about a visual detail; otherwise request a generic description. "
        "Do not claim to have seen an image until the command returns.\n"
        f"Image URLs:\n{urls}"
    )


def _route_info() -> dict[str, Any]:
    """Return only configured provider facts; absent catalog values stay absent."""
    try:
        value = json.loads(os.getenv("WISEMAN_ROUTE_INFO", "{}"))
    except ValueError:
        value = {}
    info = value if isinstance(value, dict) else {}
    defaults = {
        "requested_model": os.getenv("WISEMAN_MODEL"),
        "provider": os.getenv("WISEMAN_PROVIDER"),
        "context_window": os.getenv("WISEMAN_CONTEXT_WINDOW"),
        "fallback_models": os.getenv("WISEMAN_FALLBACK_MODELS"),
        "modalities": os.getenv("WISEMAN_MODALITIES"),
        "input_price": os.getenv("WISEMAN_INPUT_PRICE"),
        "output_price": os.getenv("WISEMAN_OUTPUT_PRICE"),
        "cached_price": os.getenv("WISEMAN_CACHED_PRICE"),
        "vision_assist_model": os.getenv("WISEMAN_VISION_MODEL"),
    }
    return {
        key: info.get(key, value) for key, value in defaults.items() if info.get(key, value)
    } | {key: value for key, value in info.items() if value not in (None, "")}


def _banner() -> str:
    info = _route_info()
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
    return (
        "⚡ **Wiseman thread startup**\n" + line + ("\n" + " · ".join(details) if details else "")
    )


def _thread_name(content: str, attachment_count: int = 0) -> str:
    """Turn the triggering message into a valid, readable Discord thread name."""
    name = re.sub(r"<@!?\d+>", "", content)
    name = " ".join(name.split()).strip()
    if not name:
        name = "image" if attachment_count else "wiseman"
    if len(name) > THREAD_NAME_LIMIT:
        name = name[: THREAD_NAME_LIMIT - 1].rstrip() + "…"
    return name


def _startup_embed() -> discord.Embed:
    """Render the old one-time green thread-start status as a Discord embed."""
    banner = _banner()
    title, _, description = banner.partition("\n")
    return discord.Embed(
        title=title.replace("**", ""),
        description=description,
        colour=0x57F287,
    )


async def _describe_images(messages: list[dict[str, Any]], question: str = "") -> dict[str, Any]:
    """Ask the configured vision model to answer about bounded Discord image attachments."""
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for message in messages:
        for attachment in message.get("attachments", []):
            content_type = str(attachment.get("content_type") or "")
            url = str(attachment.get("url") or attachment.get("proxy_url") or "")
            attachment_id = str(attachment.get("id") or url)
            if content_type.startswith("image/") and url and attachment_id not in seen:
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
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": question
            or "Please describe this image generally in one concise factual paragraph. Read "
            "visible text and report relevant objects, quantities, prices, and layout. Do not "
            "answer only None or guess details that are not visible.",
        },
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
            data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(answer, list):
            answer = "".join(str(item.get("text", "")) for item in answer if isinstance(item, dict))
        usage = data.get("usage")
        return {
            "text": str(answer),
            "model": data.get("model", model),
            "usage": usage,
            "cost": usage.get("cost") if isinstance(usage, dict) else data.get("cost"),
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


async def _edit_delivery(message: object | None, content: str) -> bool:
    """Edit a sent Discord message when the adapter exposes the native method."""
    edit = getattr(message, "edit", None)
    if not callable(edit):
        return False
    await cast("Any", edit)(content=content)
    return True


def _normalize_image_url(value: object) -> str:
    """Accept plain URLs and the angle-bracket form used in model output."""
    if not isinstance(value, str):
        return ""
    candidate = value.strip().strip("<>\"'")
    parsed = urlsplit(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return candidate
    return ""


def _split_discord_content(content: str) -> list[str]:
    """Split an answer at readable boundaries within Discord's content limit."""
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


def _render_progress(steps: list[str]) -> str:
    """Render a bounded rolling trace while the final answer is still pending."""
    turns = [int(match) for step in steps for match in re.findall(r"\bGurt (\d+)\b", step)]
    count = max(turns, default=0)
    visible = steps[-8:]
    header = f"⏳ Working · Gurt {count}" if count else "⏳ Working"
    return "\n".join([header, *visible])


async def _deliver_content(message: object | None, channel: Messageable, content: str) -> None:
    """Edit the progress message and send overflow chunks as normal messages."""
    chunks = _split_discord_content(content)
    edited = False
    if message is not None:
        try:
            edited = await _edit_delivery(message, chunks[0])
        except discord.DiscordException:
            LOGGER.warning("Could not edit Discord delivery message")
    if not edited:
        await channel.send(chunks[0])
    for chunk in chunks[1:]:
        await channel.send(chunk)


class Runner(Protocol):
    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
    ) -> tuple[str, str, dict[str, object]]: ...


class SteerableRunner(Protocol):
    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool: ...


class LifecycleRunner(Runner, Protocol):
    async def acquire(self, user: str, workspace: str) -> None: ...

    async def start(self, thread: str, user: str, workspace: str = "") -> str: ...


class RunnerError(RuntimeError):
    """Raised when the runner returns an invalid or failed response."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"runner returned HTTP {status}: {detail}")


class HttpRunner:
    """Thin authenticated client for the warm Codex runner."""

    def __init__(self, url: str, token: str = "") -> None:
        self.url, self.token = url.rstrip("/"), token

    async def _post(
        self, path: str, payload: dict[str, object], request_timeout: float = 30
    ) -> dict[str, Any]:
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.post(f"{self.url}{path}", headers=headers, json=payload)
        if response.is_error:
            detail = response.text[:1_000]
            raise RunnerError(response.status_code, detail)
        value = response.json()
        if not isinstance(value, dict):
            raise RunnerError(response.status_code, "runner returned a non-object response")
        return value

    async def acquire(self, user: str, workspace: str) -> None:
        await self._post("/acquire", {"thread_id": workspace, "user_id": user, "input": ""})

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        data = await self._post(
            "/start",
            {
                "thread_id": workspace or thread or f"thread-{user}",
                "codex_thread_id": thread or None,
                "user_id": user,
                "input": "",
            },
        )
        return str(data.get("thread_id", thread))

    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        if progress is not None:
            return await self._run_with_progress(thread, prompt, user, workspace, progress)
        return await self._post_turn(thread, prompt, user, workspace)

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        """Send a reply directly to the active Codex turn instead of queueing it."""
        data = await self._post(
            "/steer",
            {
                "thread_id": workspace or thread or f"thread-{user}",
                "codex_thread_id": thread or None,
                "user_id": user,
                "input": prompt,
            },
            request_timeout=30,
        )
        return bool(data.get("steered", False))

    async def _post_turn(
        self, thread: str, prompt: str, user: str, workspace: str
    ) -> tuple[str, str, dict[str, object]]:
        data = await self._post(
            "/turn",
            {
                "thread_id": workspace or thread or f"thread-{user}",
                "codex_thread_id": thread or None,
                "user_id": user,
                "input": prompt,
            },
            request_timeout=300,
        )
        billing = {key: data[key] for key in ("model", "cost", "usage") if key in data}
        return str(data.get("thread_id", thread)), str(data.get("output", "")), billing

    async def _run_with_progress(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str,
        progress: Callable[[str], Awaitable[None]],
    ) -> tuple[str, str, dict[str, object]]:
        payload = {
            "thread_id": workspace or thread or f"thread-{user}",
            "codex_thread_id": thread or None,
            "user_id": user,
            "input": prompt,
        }
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=600) as client:
            request = asyncio.create_task(
                client.post(f"{self.url}/turn", headers=headers, json=payload)
            )
            latest = ""
            while not request.done():
                try:
                    status = await client.get(
                        f"{self.url}/progress/{payload['thread_id']}",
                        headers=headers,
                        timeout=5,
                    )
                    if not status.is_error:
                        value = status.json()
                        message = value.get("message") if isinstance(value, dict) else None
                        if isinstance(message, str) and message != latest:
                            latest = message
                            await progress(message)
                except httpx.HTTPError:
                    pass
                if not request.done():
                    await asyncio.sleep(0.75)
            response = await request
        if response.is_error:
            raise RunnerError(response.status_code, response.text[:1_000])
        value = response.json()
        if not isinstance(value, dict):
            raise RunnerError(response.status_code, "runner returned a non-object response")
        billing = {key: value[key] for key in ("model", "cost", "usage") if key in value}
        return str(value.get("thread_id", thread)), str(value.get("output", "")), billing


class FakeRunner:
    """Deterministic local runner used only when no sandbox URL is configured."""

    async def acquire(self, user: str, workspace: str) -> None:
        del user, workspace

    async def start(self, thread: str, user: str, workspace: str = "") -> str:
        del workspace
        return thread or f"codex-{user}"

    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        del workspace, progress
        return (
            thread or f"codex-{user}",
            f"Codex received: {prompt[:1000]}",
            {"model": os.getenv("WISEMAN_MODEL", "local-fake")},
        )


class Engine:
    """One turn state machine; production adapters can place each method in a Temporal Activity."""

    def __init__(self, phoenix: Phoenix, runner: Runner, prompts: PromptHub | None = None) -> None:
        self.phoenix, self.runner, self.prompts = phoenix, runner, prompts or PromptHub()
        self.states: dict[str, State] = defaultdict(State)
        self.reactions: dict[str, list[str]] = defaultdict(list)
        self.progress: dict[str, list[str]] = defaultdict(list)
        self.locks: dict[str, asyncio.Lock] = {}
        self.reaction_user: object | None = None
        self.reaction_emojis = dict(DEFAULT_REACTION_EMOJIS)
        self.working_reactions: dict[str, str] = {}
        self.active_turns: dict[str, ActiveTurn] = {}
        self.lookup: Callable[[Event], Awaitable[discord.Message | None]] | None = None
        self.lookup_channel: Callable[[Event], Awaitable[Messageable | None]] | None = None

    async def handle(
        self,
        event: Event,
        live: discord.Message | None = None,
        delivery_channel: Messageable | None = None,
        state_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = event.trigger.thread_id or event.trigger.channel_id
        async with self.locks.setdefault(key, asyncio.Lock()):
            if live is None and self.lookup is not None:
                live = await self.lookup(event)
            if delivery_channel is None:
                if self.lookup_channel is not None:
                    delivery_channel = await self.lookup_channel(event)
                elif live is not None:
                    delivery_channel = live.channel
            return await self._handle(event, live, delivery_channel, state_data)

    async def _handle(  # noqa: C901, PLR0912, PLR0915
        self,
        event: Event,
        live: discord.Message | None = None,
        delivery_channel: Messageable | None = None,
        state_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trigger = event.trigger
        key = trigger.thread_id or trigger.channel_id
        state = (
            self.states[key]
            if state_data is None
            else State(
                codex_thread=state_data.get("codex_thread"),
                seen=set(state_data.get("seen", [])),
                processed=set(state_data.get("processed", [])),
                turn=int(state_data.get("turn", 0)),
            )
        )
        if trigger.id in state.processed:
            return {
                "trace": f"discord-{trigger.id}",
                "status": "duplicate",
                "state": _state_data(state),
            }
        state.processed.add(trigger.id)
        TURN_TOTAL.inc()
        if state.closed:
            return {"trace": f"discord-{trigger.id}", "error": "thread is closed"}
        state.last_activity = time.time()
        event.seen_ids = sorted(set(event.seen_ids) | state.seen)
        kind = event.kind or ("followup" if state.turn else "startup")
        trace = f"discord-{trigger.id}"
        raw_request = event.raw_payload or _event_data(event)
        await self.phoenix.record(
            trace,
            "admission",
            audit_id=trace,
            raw_request=raw_request,
            normalized_request=_event_data(event),
            normalizer="normalize_event:v2",
        )
        await self.phoenix.record(
            trace,
            "turn",
            thread_id=trigger.thread_id,
            message_id=trigger.id,
            kind=kind,
            route=_route_info(),
            input=event.raw_payload or _event_data(event),
        )
        processing_emoji = self.reaction_emojis["processing"]
        self.working_reactions[trigger.id] = processing_emoji
        added = self._react(trigger.id, processing_emoji)
        if live is not None and added:
            await live.add_reaction(processing_emoji)
        await self.phoenix.record(trace, "reaction", operations=[f"add:{processing_emoji}"])
        current = context(event)
        state.seen.update(current["selected_ids"])
        await self.phoenix.record(
            trace,
            "context",
            raw=_json(event.raw_payload or _event_data(event)),
            normalized=current,
            selected_ids=current["selected_ids"],
        )
        grammar_name = "startup-context" if kind == "startup" else "followup-context"
        source = await self.prompts.source(grammar_name)
        grammar = _grammar(
            grammar_name,
            source,
            event.raw_payload or _event_data(event),
            mode=kind,
            messages=current["messages"],
        )
        await self.phoenix.record(trace, "grammar", **grammar)
        parts = {
            "soul": await self.prompts.source("wiseman-soul"),
            "runtime": await self.prompts.source("wiseman-runtime"),
            "memories": os.getenv("WISEMAN_MEMORIES", ""),
            "context": grammar["rendered"],
            "user": "\n\n".join(
                part
                for part in (
                    trigger.content,
                    _image_tool_instruction(
                        [trigger.model_dump(mode="json"), *current["reply_ancestors"]]
                    ),
                )
                if part
            ),
        }
        prompt = _json(parts)
        await self.phoenix.record(trace, "prompt", parts=parts, final_input=prompt)
        progress_message: object | None = None
        if delivery_channel is not None and kind == "startup":
            await cast("Any", delivery_channel).send(embed=_startup_embed())
        startup = kind == "startup"
        progress = "🤖 Codex starting..." if startup else "⏳ Working..."
        phase = "codex starting" if startup else "working"
        self.progress[trigger.id].append(phase)
        await self.phoenix.record(trace, "progress", phase=phase)
        if delivery_channel is not None:
            progress_message = await delivery_channel.send(_render_progress([progress]))
        self.active_turns[key] = ActiveTurn(
            trigger_id=trigger.id,
            delivery_id=str(getattr(progress_message, "id", "")) or None,
        )

        async def report(message: str) -> None:
            if self.progress[trigger.id] and self.progress[trigger.id][-1] == message:
                return
            self.progress[trigger.id].append(message)
            await self.phoenix.record(trace, "progress", phase=message)
            if progress_message is not None:
                try:
                    await _edit_delivery(
                        progress_message, _render_progress(self.progress[trigger.id])
                    )
                except discord.DiscordException:
                    LOGGER.warning("Could not update progress message for %s", trigger.id)

        try:
            if isinstance(self.runner, HttpRunner):
                state.codex_thread, output, billing = await self.runner.run(
                    state.codex_thread or "",
                    prompt,
                    trigger.author_id,
                    trigger.thread_id or trigger.channel_id,
                    progress=report,
                )
            else:
                state.codex_thread, output, billing = await self.runner.run(
                    state.codex_thread or "",
                    prompt,
                    trigger.author_id,
                    trigger.thread_id or trigger.channel_id,
                )
        except Exception as exc:  # noqa: BLE001 - visible turn failure, thread survives
            TURN_FAILURES.inc()
            await self.phoenix.record(trace, "failure", error=str(exc))
            failure_emoji = self.reaction_emojis["failure"]
            if self._react(trigger.id, failure_emoji):
                if live is not None:
                    await live.add_reaction(failure_emoji)
                if delivery_channel is not None:
                    await _deliver_content(
                        progress_message, delivery_channel, f"Codex failed: {exc}"
                    )
                await self._remove_working_reaction(trigger.id, live)
            await self.phoenix.record(
                trace,
                "reaction",
                operations=[f"add:{failure_emoji}", f"remove:{processing_emoji}"],
            )
            return {
                "trace": trace,
                "kind": kind,
                "error": str(exc),
                "reactions": self.reactions[trigger.id],
                "state": _state_data(state),
            }
        finally:
            self.active_turns.pop(key, None)
        state.turn += 1
        self.progress[trigger.id].append("finalizing")
        await self.phoenix.record(trace, "progress", phase="finalizing")
        await self.phoenix.record(
            trace,
            "codex",
            input=prompt,
            thread_id=state.codex_thread,
            output=output,
            **billing,
        )
        await self.phoenix.record(trace, "delivery", output=output)
        if delivery_channel is not None:
            await _deliver_content(progress_message, delivery_channel, output)
        success_emoji = self.reaction_emojis["success"]
        if self._react(trigger.id, success_emoji):
            if live is not None:
                await live.add_reaction(success_emoji)
            await self._remove_working_reaction(trigger.id, live)
        await self.phoenix.record(
            trace,
            "reaction",
            operations=[f"add:{success_emoji}", f"remove:{processing_emoji}"],
        )
        return {
            "trace": trace,
            "kind": kind,
            "output": output,
            "selected_ids": current["selected_ids"],
            "reactions": self.reactions[trigger.id],
            "progress": self.progress[trigger.id],
            "state": _state_data(state),
        }

    async def steer_if_active(
        self, thread_id: str, message_id: str, prompt: str, user: str
    ) -> bool:
        """Route a reply to the visible working message into the live Codex turn."""
        active = self.active_turns.get(thread_id)
        if active is None or active.delivery_id != message_id:
            return False
        steer = getattr(self.runner, "steer", None)
        if not callable(steer):
            return False
        accepted = await cast("SteerableRunner", self.runner).steer(
            "", prompt, user, workspace=thread_id
        )
        if accepted:
            await self.phoenix.record(
                f"discord-{message_id}",
                "steer",
                thread_id=thread_id,
                message_id=message_id,
                input=prompt,
            )
        return accepted

    def _react(self, message_id: str, emoji: str) -> bool:
        if emoji not in self.reactions[message_id]:
            self.reactions[message_id].append(emoji)
            return True
        return False

    def set_reaction_emojis(self, values: dict[str, str]) -> dict[str, str]:
        """Set future lifecycle reactions; in-flight turns retain their original emoji."""
        updated = dict(self.reaction_emojis)
        for phase in DEFAULT_REACTION_EMOJIS:
            value = values.get(phase)
            if value is not None:
                if not value.strip() or len(value) > MAX_REACTION_LENGTH:
                    reason = f"invalid {phase} reaction"
                    raise ValueError(reason)
                updated[phase] = value
        self.reaction_emojis = updated
        return dict(updated)

    async def _remove_working_reaction(self, message_id: str, live: discord.Message | None) -> None:
        emoji = self.working_reactions.pop(message_id, self.reaction_emojis["processing"])
        if emoji not in self.reactions[message_id]:
            return
        self.reactions[message_id].remove(emoji)
        remove = getattr(live, "remove_reaction", None)
        if callable(remove) and self.reaction_user is not None:
            await cast("Any", remove)(emoji, self.reaction_user)


def _state_data(state: State) -> dict[str, Any]:
    return {
        "codex_thread": state.codex_thread,
        "seen": sorted(state.seen),
        "processed": sorted(state.processed),
        "turn": state.turn,
    }


class Gateway(discord.Client):
    """Direct Discord gateway adapter; it delegates to the same admission used by raw replay."""

    def __init__(
        self,
        engine: Engine,
        allowlist: set[int],
        activity_path: str | Path | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.engine, self.allowlist = engine, allowlist
        self.temporal: TemporalRuntime | None = None
        self.activity_path = Path(activity_path) if activity_path else None
        self.profile_path = Path(profile_path) if profile_path else None
        self.thread_activity = self._load_thread_activity()
        self._load_profile()
        self.expiry_task: asyncio.Task[None] | None = None
        engine.lookup = self.resolve
        engine.lookup_channel = self.resolve_channel

    async def setup_hook(self) -> None:
        self.expiry_task = asyncio.create_task(self._expire_threads())

    async def on_ready(self) -> None:
        DISCORD_CONNECTED.set(1)
        LOGGER.info("Discord gateway ready as %s", self.user)

    async def on_disconnect(self) -> None:
        DISCORD_CONNECTED.set(0)
        LOGGER.warning("Discord gateway disconnected; discord.py will reconnect")

    async def on_resumed(self) -> None:
        DISCORD_CONNECTED.set(1)
        LOGGER.info("Discord gateway session resumed")

    def _load_thread_activity(self) -> dict[str, float]:
        if self.activity_path is None or not self.activity_path.exists():
            return {}
        try:
            value = json.loads(self.activity_path.read_text(encoding="utf-8"))
            return {str(key): float(timestamp) for key, timestamp in value.items()}
        except (OSError, TypeError, ValueError, AttributeError):
            LOGGER.warning("Ignoring invalid Wiseman thread activity state")
            return {}

    def _persist_thread_activity(self) -> None:
        if self.activity_path is None:
            return
        try:
            self.activity_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.activity_path.with_name(f".{self.activity_path.name}.tmp")
            temporary.write_text(json.dumps(self.thread_activity, sort_keys=True), encoding="utf-8")
            temporary.replace(self.activity_path)
        except OSError:
            LOGGER.exception("Could not persist Wiseman thread activity state")

    def _touch_thread(self, thread_id: str, timestamp: float | None = None) -> None:
        self.thread_activity[thread_id] = timestamp if timestamp is not None else time.time()
        self._persist_thread_activity()

    def _forget_thread(self, thread_id: str) -> None:
        if thread_id in self.thread_activity:
            self.thread_activity.pop(thread_id)
            self._persist_thread_activity()

    def _load_profile(self) -> None:
        if self.profile_path is None or not self.profile_path.exists():
            return
        try:
            value = json.loads(self.profile_path.read_text(encoding="utf-8"))
            reactions = value.get("reaction_emojis", {})
            if isinstance(reactions, dict):
                self.engine.set_reaction_emojis(
                    {str(key): str(item) for key, item in reactions.items()}
                )
        except (OSError, TypeError, ValueError, AttributeError):
            LOGGER.warning("Ignoring invalid Wiseman profile state")

    def persist_profile(self) -> None:
        if self.profile_path is None:
            return
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.profile_path.with_name(f".{self.profile_path.name}.tmp")
            temporary.write_text(
                json.dumps({"reaction_emojis": self.engine.reaction_emojis}, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.profile_path)
        except OSError:
            LOGGER.exception("Could not persist Wiseman profile state")

    async def run_forever(self, token: str) -> None:
        """Keep the gateway supervised when Discord returns a fatal session error."""
        delay = 1.0
        while not self.is_closed():
            try:
                await self.start(token, reconnect=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                DISCORD_CONNECTED.set(0)
                LOGGER.exception("Discord gateway session failed; retrying")
            else:
                if not self.is_closed():
                    LOGGER.warning("Discord gateway stopped unexpectedly; retrying")
            if self.is_closed():
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

    async def _expire_threads(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self._expire_once()

    async def _expire_once(self, now: float | None = None) -> None:
        """Forget local tracking after Discord's native archive window expires."""
        cutoff = (time.time() if now is None else now) - (THREAD_AUTO_ARCHIVE_MINUTES * 60)
        for thread_id, last_activity in list(self.thread_activity.items()):
            if last_activity > cutoff:
                continue
            self._forget_thread(thread_id)

    async def close(self) -> None:
        DISCORD_CONNECTED.set(0)
        if self.expiry_task is not None:
            self.expiry_task.cancel()
            await asyncio.gather(self.expiry_task, return_exceptions=True)
        await super().close()

    async def resolve(self, event: Event) -> discord.Message | None:
        """Recover a live Discord message for a Temporal activity delivery."""
        channel_id = (
            event.trigger.channel_id
            if event.kind == "startup"
            else event.trigger.thread_id or event.trigger.channel_id
        )
        try:
            channel = await self.fetch_channel(int(channel_id))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                self.engine.reaction_user = self.user
                return await channel.fetch_message(int(event.trigger.id))
            return None  # noqa: TRY300
        except (discord.DiscordException, ValueError):
            return None

    async def resolve_channel(self, event: Event) -> Messageable | None:
        """Resolve the managed thread used for progress and answer delivery."""
        channel_id = event.trigger.thread_id or event.trigger.channel_id
        try:
            channel = await self.fetch_channel(int(channel_id))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                return channel
            return None  # noqa: TRY300
        except (discord.DiscordException, ValueError):
            return None

    async def on_message(self, message: discord.Message) -> None:
        if (
            isinstance(message.channel, discord.Thread)
            and not message.author.bot
            and str(message.channel.id) in self.thread_activity
        ):
            self._touch_thread(str(message.channel.id))
        if message.author.bot:
            return
        channel = message.channel
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        if self.allowlist and guild_id not in self.allowlist:
            return
        if isinstance(channel, discord.Thread):
            reference_id = getattr(message.reference, "message_id", None)
            if reference_id is not None and await self.engine.steer_if_active(
                str(channel.id), str(reference_id), message.content, str(message.author.id)
            ):
                return
        if self.user not in message.mentions:
            return
        if isinstance(channel, discord.Thread):
            thread_id, parent_id, kind = str(channel.id), str(channel.parent_id), "followup"
            delivery_channel = channel
            self._touch_thread(thread_id)
            old = self.engine.states[thread_id]
            thread_messages = await _history(channel, 100)
            parent_messages = await _history(channel.parent, 100) if channel.parent else []
        else:
            thread = await message.create_thread(
                name=_thread_name(message.content, len(message.attachments)),
                auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES,
            )
            thread_id, parent_id, kind = str(thread.id), str(channel.id), "startup"
            delivery_channel = thread
            self._touch_thread(thread_id)
            old = self.engine.states[thread_id]
            thread_messages, parent_messages = [], await _history(channel, 100, before=message)
        trigger = Message(
            id=str(message.id),
            author_id=str(message.author.id),
            author_name=message.author.name,
            content=message.content,
            channel_id=parent_id,
            thread_id=thread_id,
            timestamp=message.created_at.isoformat(),
            reply_to=str(message.reference.message_id) if message.reference else None,
            mentions=[str(user.id) for user in message.mentions],
            attachments=[
                {
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "url": attachment.url,
                    "content_type": attachment.content_type,
                    "size": attachment.size,
                }
                for attachment in message.attachments
            ],
        )
        event = Event(
            trigger=trigger,
            kind=kind,
            parent_messages=parent_messages,
            thread_messages=thread_messages,
            seen_ids=list(old.seen),
            raw_payload={
                "trigger": trigger.model_dump(mode="json"),
                "kind": kind,
                "parent_messages": [item.model_dump(mode="json") for item in parent_messages],
                "thread_messages": [item.model_dump(mode="json") for item in thread_messages],
            },
        )
        if self.temporal is not None:
            await self.temporal.submit(event.model_dump(mode="json"))
        else:
            self.engine.reaction_user = self.user
            await self.engine.handle(event, message, delivery_channel=delivery_channel)


async def _history(
    channel: Any,  # noqa: ANN401 - Discord's channel union shares the history protocol
    limit: int,
    before: discord.Message | None = None,
) -> list[Message]:
    """Normalize bounded Discord history for the same grammar path as replay."""
    if channel is None:
        return []
    kwargs: dict[str, Any] = {"limit": limit}
    if before is not None:
        kwargs["before"] = before
    thread_id = str(channel.id) if isinstance(channel, discord.Thread) else None
    channel_id = str(getattr(channel, "parent_id", channel.id))
    return [
        Message(
            id=str(item.id),
            author_id=str(item.author.id),
            author_name=item.author.name,
            bot=item.author.bot,
            content=item.content,
            channel_id=channel_id,
            thread_id=thread_id,
            timestamp=item.created_at.isoformat(),
            reply_to=str(item.reference.message_id) if item.reference else None,
            mentions=[str(user.id) for user in item.mentions],
            attachments=[
                {
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "url": attachment.url,
                    "content_type": attachment.content_type,
                    "size": attachment.size,
                }
                for attachment in item.attachments
            ],
        )
        async for item in channel.history(**kwargs)
    ]


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
        result = await _describe_images(
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


phoenix = Phoenix(
    os.getenv("PHOENIX_OTLP_ENDPOINT", ""),
    os.getenv("PHOENIX_API_KEY", ""),
    os.getenv("PHOENIX_PROJECT", "wiseman-v2"),
)
runner: Runner = (
    HttpRunner(os.environ["WISEMAN_RUNNER_URL"], os.getenv("WISEMAN_RUNNER_API_TOKEN", ""))
    if os.getenv("WISEMAN_RUNNER_URL")
    else FakeRunner()
)
engine = Engine(phoenix, runner)
app = create_app(
    engine,
    os.getenv("WISEMAN_REPLAY_TOKEN", ""),
    os.getenv("DISCORD_BOT_TOKEN", ""),
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
