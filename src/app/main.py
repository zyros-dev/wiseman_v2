# Copyright (c) 2026 Nick van der Merwe
"""Small, real Wiseman runtime: Discord admission, durable turn semantics, and Phoenix evidence."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

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


class Phoenix:
    """Record every semantic layer immediately and optionally forward it to Phoenix."""

    def __init__(self, endpoint: str = "", key: str = "", project: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.records: list[dict[str, Any]] = []
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
THREAD_IDLE_SECONDS = 7200
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
        return {"text": "", "attachments": []}
    model = os.getenv("WISEMAN_VISION_MODEL", "z-ai/glm-5.3-flash")
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return {
            "text": "[Image description unavailable: vision provider is not configured.]",
            "model": model,
            "attachments": [item["id"] for item in images],
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
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return {
            "text": f"[Image description unavailable: {type(exc).__name__}].",
            "model": model,
            "attachments": [item["id"] for item in images],
        }


async def _edit_delivery(message: object | None, content: str) -> bool:
    """Edit a sent Discord message when the adapter exposes the native method."""
    edit = getattr(message, "edit", None)
    if not callable(edit):
        return False
    await cast("Any", edit)(content=content)
    return True


class Runner(Protocol):
    async def run(
        self, thread: str, prompt: str, user: str, workspace: str = ""
    ) -> tuple[str, str, dict[str, object]]: ...


class HttpRunner:
    """Thin authenticated client for the warm Codex runner."""

    def __init__(self, url: str, token: str = "") -> None:
        self.url, self.token = url.rstrip("/"), token

    async def run(
        self, thread: str, prompt: str, user: str, workspace: str = ""
    ) -> tuple[str, str, dict[str, object]]:
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{self.url}/turn",
                headers=headers,
                json={
                    "thread_id": workspace or thread or f"thread-{user}",
                    "codex_thread_id": thread or None,
                    "user_id": user,
                    "input": prompt,
                },
            )
        response.raise_for_status()
        data = response.json()
        billing = {key: data[key] for key in ("model", "cost", "usage") if key in data}
        return str(data.get("thread_id", thread)), str(data.get("output", "")), billing


class FakeRunner:
    """Deterministic local runner used only when no sandbox URL is configured."""

    async def run(
        self, thread: str, prompt: str, user: str, workspace: str = ""
    ) -> tuple[str, str, dict[str, object]]:
        del workspace
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
        await self.phoenix.record(
            trace,
            "turn",
            thread_id=trigger.thread_id,
            message_id=trigger.id,
            kind=kind,
            route=_route_info(),
            input=event.raw_payload or _event_data(event),
        )
        added = self._react(trigger.id, "👀")
        if live is not None and added:
            await live.add_reaction("👀")
        await self.phoenix.record(trace, "reaction", operations=["add:👀"])
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
            "user": trigger.content,
        }
        prompt = _json(parts)
        await self.phoenix.record(trace, "prompt", parts=parts, final_input=prompt)
        progress_message: object | None = None
        if delivery_channel is not None and kind == "startup":
            await cast("Any", delivery_channel).send(embed=_startup_embed())
        startup = kind == "startup"
        progress = (
            "🛠️ Workspace provisioning...\n🤖 Codex starting..." if startup else "⏳ Working..."
        )
        phase = "workspace provisioning" if startup else "working"
        self.progress[trigger.id].append(phase)
        await self.phoenix.record(trace, "progress", phase=phase)
        if delivery_channel is not None:
            progress_message = await delivery_channel.send(progress)
        if startup:
            self.progress[trigger.id].append("codex started")
            await self.phoenix.record(trace, "progress", phase="codex started")
        try:
            state.codex_thread, output, billing = await self.runner.run(
                state.codex_thread or "",
                prompt,
                trigger.author_id,
                trigger.thread_id or trigger.channel_id,
            )
        except Exception as exc:  # noqa: BLE001 - visible turn failure, thread survives
            TURN_FAILURES.inc()
            await self.phoenix.record(trace, "failure", error=str(exc))
            if self._react(trigger.id, "❌"):
                if live is not None:
                    await live.add_reaction("❌")
                if delivery_channel is not None and not await _edit_delivery(
                    progress_message, f"Codex failed: {exc}"
                ):
                    await delivery_channel.send(f"Codex failed: {exc}")
                await self._remove_working_reaction(trigger.id, live)
            await self.phoenix.record(trace, "reaction", operations=["add:❌", "remove:👀"])
            return {
                "trace": trace,
                "kind": kind,
                "error": str(exc),
                "reactions": self.reactions[trigger.id],
                "state": _state_data(state),
            }
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
        if delivery_channel is not None and not await _edit_delivery(progress_message, output):
            await delivery_channel.send(output)
        if self._react(trigger.id, "✅"):
            if live is not None:
                await live.add_reaction("✅")
            await self._remove_working_reaction(trigger.id, live)
        await self.phoenix.record(trace, "reaction", operations=["add:✅", "remove:👀"])
        return {
            "trace": trace,
            "kind": kind,
            "output": output,
            "selected_ids": current["selected_ids"],
            "reactions": self.reactions[trigger.id],
            "progress": self.progress[trigger.id],
            "state": _state_data(state),
        }

    def _react(self, message_id: str, emoji: str) -> bool:
        if emoji not in self.reactions[message_id]:
            self.reactions[message_id].append(emoji)
            return True
        return False

    async def _remove_working_reaction(self, message_id: str, live: discord.Message | None) -> None:
        if "👀" not in self.reactions[message_id]:
            return
        self.reactions[message_id].remove("👀")
        remove = getattr(live, "remove_reaction", None)
        if callable(remove) and self.reaction_user is not None:
            await cast("Any", remove)("👀", self.reaction_user)


def _state_data(state: State) -> dict[str, Any]:
    return {
        "codex_thread": state.codex_thread,
        "seen": sorted(state.seen),
        "processed": sorted(state.processed),
        "turn": state.turn,
    }


class Gateway(discord.Client):
    """Direct Discord gateway adapter; it delegates to the same admission used by raw replay."""

    def __init__(self, engine: Engine, allowlist: set[int]) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.engine, self.allowlist = engine, allowlist
        self.temporal: TemporalRuntime | None = None
        self.thread_activity: dict[str, float] = {}
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
            cutoff = time.time() - THREAD_IDLE_SECONDS
            for thread_id, last_activity in list(self.thread_activity.items()):
                if last_activity >= cutoff:
                    continue
                try:
                    channel = await self.fetch_channel(int(thread_id))
                    if isinstance(channel, discord.Thread):
                        await channel.edit(archived=True, locked=True)
                except (discord.DiscordException, ValueError):
                    continue
                self.thread_activity.pop(thread_id, None)

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
        if isinstance(message.channel, discord.Thread) and not message.author.bot:
            self.thread_activity[str(message.channel.id)] = time.time()
        if message.author.bot or self.user not in message.mentions:
            return
        channel = message.channel
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        if self.allowlist and guild_id not in self.allowlist:
            return
        if isinstance(channel, discord.Thread):
            thread_id, parent_id, kind = str(channel.id), str(channel.parent_id), "followup"
            delivery_channel = channel
            old = self.engine.states[thread_id]
            if old.last_activity and time.time() - old.last_activity > THREAD_IDLE_SECONDS:
                old.closed = True
                await channel.edit(archived=True, locked=True)
                return
            thread_messages = await _history(channel, 100)
            parent_messages = await _history(channel.parent, 100) if channel.parent else []
        else:
            thread = await message.create_thread(name="wiseman")
            thread_id, parent_id, kind = str(thread.id), str(channel.id), "startup"
            delivery_channel = thread
            self.thread_activity[thread_id] = time.time()
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
    bot = Gateway(engine, allowlist)
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

    @app.post("/v1/responses")
    async def responses(
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
                client.stream(
                    "POST",
                    f"{os.getenv('OPENROUTER_URL', 'https://openrouter.ai')}/api/v1/responses",
                    headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                    json=payload,
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
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
        url = str(payload.get("url") or "")
        if not url.startswith(("https://", "http://")):
            raise HTTPException(422, "image URL must use HTTP or HTTPS")
        attachment_id = str(payload.get("attachment_id") or url)
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

    @app.post("/v1/replay/discord")
    async def replay(
        payload: dict[str, Any], x_replay_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        expected = os.getenv("WISEMAN_REPLAY_TOKEN", token)
        if expected and not hmac.compare_digest(x_replay_token or "", expected):
            raise HTTPException(401, "invalid replay token")
        try:
            event = normalize_event(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, "invalid Discord event") from exc
        if temporal is not None:
            await temporal.submit(event.model_dump(mode="json"))
            return {"status": "queued", "message_id": event.trigger.id}
        return await engine.handle(event)

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
