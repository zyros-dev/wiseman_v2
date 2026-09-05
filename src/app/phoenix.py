# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Context, Span, set_span_in_context
from phoenix.client import Client

if TYPE_CHECKING:
    from app.types import JsonObject

LOGGER = logging.getLogger("wiseman")


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def provider_values(body: JsonObject, usage: object, cost: object, model: object) -> tuple[object, object, object]:
    nested = body.get("response") or body.get("data") or body
    if not isinstance(nested, dict):
        return usage, cost, model
    next_usage = nested.get("usage", body.get("usage", usage))
    next_cost = nested.get("cost", body.get("cost", cost))
    if next_cost is None and isinstance(next_usage, dict):
        next_cost = next_usage.get("cost")
    return next_usage, next_cost, nested.get("model", body.get("model", model))


class Phoenix:
    def __init__(
        self,
        endpoint: str = "",
        key: str = "",
        project: str = "",
        audit_dir: str | Path | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.records: list[dict[str, object]] = []
        self.audits: dict[str, dict[str, object]] = {}
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.roots: dict[str, Span] = {}
        self.contexts: dict[str, Context] = {}
        self.provider = TracerProvider(resource=Resource.create({"service.name": "wiseman-v2", "openinference.project.name": project}))
        if endpoint:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            if project:
                headers["x-project-name"] = project
            self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
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
                span.set_attribute(f"wiseman.{key}", json_text(value) if not isinstance(value, str) else value)
        terminal = node in {"completed", "failure", "provider", "vision_tool"} or (node == "reaction" and "remove:" in json_text(data.get("operations", [])))
        if terminal:
            self.roots.pop(trace).end()
            self.contexts.pop(trace)

    def audit(self, audit_id: str) -> dict[str, object] | None:
        value = self.audits.get(audit_id)
        if value is None and self.audit_dir is not None:
            try:
                loaded: object = json.loads(self._audit_path(audit_id).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("audit_id") == audit_id:
                value = loaded
                self.audits[audit_id] = cast("dict[str, object]", loaded)
        return dict(value) if value is not None else None

    def _audit_path(self, audit_id: str) -> Path:
        digest = hashlib.sha256(audit_id.encode()).hexdigest()
        return (self.audit_dir or Path()) / f"{digest}.json"

    def _persist_audit(self, audit_id: str, artifact: dict[str, object]) -> None:
        if self.audit_dir is None:
            return
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            target = self._audit_path(audit_id)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json_text(artifact), encoding="utf-8")
            temporary.replace(target)
        except OSError:
            LOGGER.exception("Could not persist Phoenix audit %s", audit_id)


class PromptHub:
    def __init__(self, url: str = "", key: str = "") -> None:
        self.client = Client(base_url=url.rstrip("/"), api_key=key) if url else None
        try:
            value: object = json.loads(os.getenv("WISEMAN_PROMPT_VERSION_IDS", "{}"))
        except ValueError:
            value = {}
        self.version_ids = {str(name): str(identifier) for name, identifier in value.items()} if isinstance(value, dict) else {}

    async def source(self, kind: str) -> str:
        local_kind = {"startup": "startup-context", "followup": "followup-context"}.get(kind, kind)
        contracts = Path(__file__).parents[2] / "contracts"
        candidates = (
            contracts / f"{local_kind}.j2",
            contracts / f"{local_kind}.json",
        )
        default = next((path.read_text(encoding="utf-8") for path in candidates if path.exists()), "")
        identifier = self.version_ids.get(local_kind)
        if self.client is None or not identifier:
            return default
        try:
            version = await asyncio.to_thread(self.client.prompts.get, prompt_version_id=identifier)
            formatted = version.format(variables={}, sdk="openai")
            messages = getattr(formatted, "messages", ())
            source = "\n\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
            return source[:100_000] or default
        except (OSError, RuntimeError, TypeError, ValueError):
            return default


def route_info() -> dict[str, object]:
    try:
        value = json.loads(os.getenv("WISEMAN_ROUTE_INFO", "{}"))
    except ValueError:
        value = {}
    info = cast("dict[str, object]", value) if isinstance(value, dict) else {}
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
    return {key: info.get(key, value) for key, value in defaults.items() if info.get(key, value)} | {
        key: value for key, value in info.items() if value not in (None, "")
    }
