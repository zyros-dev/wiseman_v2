# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import discord
from httpx import TransportError
from prometheus_client import Counter

from app.admission import ContextConfig, context, render_grammar
from app.engine_support import PromptRequest, build_prompt
from app.models import Event, Messageable, State, TurnWork
from app.phoenix import json_text, route_info
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_CONTENT_LENGTH,
    MAX_REACTION_LENGTH,
    edit_delivery,
    render_progress,
    startup_embed,
)
from app.runner import MESSAGE_ID, TURN_NUMBER, RunnerError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from app.clients.client_interfaces import ClientContainer, PhoenixClient, PromptClient, RunnerClient
    from app.types import EngineResult, JsonObject, StateData

TURN_TOTAL = Counter("wiseman_turns_total", "Accepted Discord turns")
TURN_FAILURES = Counter("wiseman_turn_failures_total", "Failed Discord turns")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    phoenix: PhoenixClient
    runner: RunnerClient
    prompts: PromptClient
    context: ContextConfig = field(default_factory=ContextConfig.from_env)


@dataclass(slots=True)
class DeliveryIO:
    live: discord.Message | None = None
    channel: Messageable | None = None
    message: object | None = None


class Engine:
    def __init__(self, config: EngineConfig | None = None, *, clients: ClientContainer | None = None) -> None:
        if clients is not None:
            if config is not None:
                raise ValueError("choose config or clients")
            config = EngineConfig(clients.phoenix, clients.runner, clients.prompts)
        if config is None:
            raise ValueError("engine config is required")
        self.config = config
        self.reaction_user: object | None = None
        self.reaction_emojis = dict(DEFAULT_REACTION_EMOJIS)
        self.lookup: Callable[[Event], Awaitable[discord.Message | None]] | None = None
        self.lookup_channel: Callable[[Event], Awaitable[Messageable | None]] | None = None
        self.lookup_delivery: Callable[[Event, str], Awaitable[object | None]] | None = None

    async def prepare_context(self, work: TurnWork) -> TurnWork:
        normalized = cast("JsonObject", work.event.model_dump(mode="json", exclude={"raw_payload"}))
        raw = work.event.raw_payload or normalized
        await self.config.phoenix.record(
            work.trace,
            "admission",
            audit_id=work.trace,
            raw_request=raw,
            normalized_request=normalized,
            normalizer="normalize_event:v2",
        )
        await self.config.phoenix.record(
            work.trace,
            "turn",
            thread_id=work.event.trigger.thread_id,
            message_id=work.event.trigger.id,
            kind=work.kind,
            route=route_info(),
            input=raw,
        )
        work.event.seen_ids = sorted(set(work.event.seen_ids) | work.state.seen)
        work.event.kind = work.kind
        work.current = context(work.event, self.config.context)
        work.state.seen.update(cast("list[str]", work.current["selected_ids"]))
        await self.config.phoenix.record(
            work.trace,
            "context",
            raw=json_text(raw),
            normalized=work.current,
            selected_ids=work.current["selected_ids"],
        )
        return work

    async def prepare_prompt(self, work: TurnWork) -> TurnWork:
        grammar_name = "startup-context" if work.kind == "startup" else "followup-context"
        source = await self.config.prompts.source(grammar_name)
        work.grammar = render_grammar(
            grammar_name,
            source,
            work.event.raw_payload or work.event.model_dump(mode="json", exclude={"raw_payload"}),
            mode=work.kind,
            messages=cast("list[dict[str, object]]", work.current["messages"]),
        )
        await self.config.phoenix.record(work.trace, "grammar", **work.grammar)
        work.prompt = await build_prompt(
            PromptRequest(
                self.config.phoenix, self.config.prompts, work.trace, work.event.trigger, work.current, work.grammar
            )
        )
        return work

    async def execute(self, work: TurnWork, report: Callable[[str], Awaitable[None]]) -> TurnWork:
        token, message_token = TURN_NUMBER.set(work.state.turn + 1), MESSAGE_ID.set(work.event.trigger.id)
        try:
            work.state.codex_thread, work.output, work.billing = await self.config.runner.run(
                work.state.codex_thread or "",
                work.prompt,
                work.state.owner_id or work.event.trigger.author_id,
                work.event.trigger.thread_id or work.event.trigger.channel_id,
                progress=report,
            )
        finally:
            TURN_NUMBER.reset(token)
            MESSAGE_ID.reset(message_token)
        return work

    async def bindings(self, work: TurnWork, bound: DeliveryIO | None = None) -> DeliveryIO:
        bound = bound or DeliveryIO()
        if bound.live is None and self.lookup is not None:
            bound.live = await self.lookup(work.event)
        if bound.channel is None and self.lookup_channel is not None:
            bound.channel = await self.lookup_channel(work.event)
        if bound.channel is None and bound.live is not None:
            bound.channel = bound.live.channel
        if bound.message is None and work.state.delivery_id and self.lookup_delivery is not None:
            bound.message = await self.lookup_delivery(work.event, work.state.delivery_id)
        return bound

    async def render(self, work: TurnWork, bound: DeliveryIO | None = None) -> TurnWork:
        bound = await self.bindings(work, bound)
        if bound.channel is None:
            raise RuntimeError("Discord answer channel is unavailable")
        send = cast("Callable[..., Awaitable[object]]", bound.channel.send)
        if not work.state.banner_sent:
            await send(embed=startup_embed(), nonce=f"b:{work.event.trigger.id}")
            work.state.banner_sent = True
        content = render_progress(work.state.progress, work.state.turn + 1)
        if not work.state.delivery_id:
            bound.message = await send(content, nonce=work.event.trigger.id)
            work.state.delivery_id = str(getattr(bound.message, "id", "")) or None
        elif not await edit_delivery(bound.message, content):
            raise RuntimeError("Recorded Discord answer is unavailable")
        return work

    async def deliver(self, work: TurnWork, bound: DeliveryIO | None = None) -> TurnWork:
        bound = await self.bindings(work, bound)
        if bound.channel is None or bound.message is None:
            raise RuntimeError("Recorded Discord answer is unavailable")
        content = f"Codex failed: {work.error}" if work.error else work.output
        if len(content) > MAX_DISCORD_CONTENT_LENGTH:
            edit = getattr(bound.message, "edit", None)
            if not callable(edit):
                raise RuntimeError("Discord answer cannot be edited")
            await cast("Callable[..., Awaitable[object]]", edit)(
                content=content[:1800] + "\n\nFull response attached.",
                attachments=[discord.File(io.BytesIO(content.encode()), filename="response.md")],
            )
        elif not await edit_delivery(bound.message, content):
            raise RuntimeError("Discord answer cannot be edited")
        return work

    async def reconcile(self, work: TurnWork, bound: DeliveryIO | None = None) -> TurnWork:
        bound = await self.bindings(work, bound)
        if bound.live is None:
            raise RuntimeError("Discord trigger is unavailable")
        await bound.live.add_reaction(work.terminal_emoji or work.processing_emoji)
        if work.terminal_emoji:
            if self.reaction_user is None:
                raise RuntimeError("Discord bot identity is unavailable")
            await bound.live.remove_reaction(work.processing_emoji, cast("discord.User", self.reaction_user))
        return work

    async def handle(
        self,
        event: Event,
        live: discord.Message | None = None,
        delivery_channel: Messageable | None = None,
        state_data: JsonObject | None = None,
        *,
        retry_transport: bool = False,
    ) -> EngineResult:
        work = TurnWork(event=event, state=State.model_validate(state_data or {}))
        if event.trigger.id in work.state.processed:
            return {"trace": work.trace, "status": "duplicate", "state": _state_data(work.state)}
        if work.state.closed:
            return {"trace": work.trace, "error": "thread is closed", "state": _state_data(work.state)}
        work.state.owner_id = work.state.owner_id or event.trigger.author_id
        work.processing_emoji = self.reaction_emojis["processing"]
        bound = await self.bindings(work, DeliveryIO(live, delivery_channel))
        TURN_TOTAL.inc()
        await self.prepare_context(work)
        await self.prepare_prompt(work)
        if bound.live is not None:
            await self.reconcile(work, bound)
        if bound.channel is not None:
            await self.render(work, bound)

        async def report(message: str) -> None:
            work.state.progress = [*work.state.progress[-31:], message]
            await self.config.phoenix.record(work.trace, "progress", phase=message)
            if bound.channel is not None:
                await self.render(work, bound)

        try:
            await self.execute(work, report)
        except Exception as exc:
            if retry_transport and _transport_error(exc):
                raise
            work.error = str(exc) or type(exc).__name__
        return await self.finish(work, bound)

    async def finish(self, work: TurnWork, bound: DeliveryIO | None = None) -> EngineResult:
        bound = await self.bindings(work, bound)
        if work.error:
            TURN_FAILURES.inc()
            await self.config.phoenix.record(work.trace, "failure", error=work.error)
        else:
            await self.config.phoenix.record(
                work.trace,
                "codex",
                input=work.prompt,
                thread_id=work.state.codex_thread,
                output=work.output,
                **work.billing,
            )
        if bound.channel is not None:
            if bound.message is None:
                await self.render(work, bound)
            await self.deliver(work, bound)
        work.terminal_emoji = self.reaction_emojis["failure" if work.error else "success"]
        if bound.live is not None:
            await self.reconcile(work, bound)
        await self.config.phoenix.record(
            work.trace, "reaction", operations=[f"add:{work.terminal_emoji}", f"remove:{work.processing_emoji}"]
        )
        work.state.processed.add(work.event.trigger.id)
        work.state.turn += int(not work.error)
        result: EngineResult = {
            "trace": work.trace,
            "kind": work.event.kind or work.kind,
            "reactions": [work.terminal_emoji],
            "selected_ids": cast("list[str]", work.current.get("selected_ids", [])),
            "progress": work.state.progress,
            "state": _state_data(work.state, finished=True),
        }
        if work.error:
            result["error"] = work.error
        else:
            result["output"] = work.output
        return result

    async def preflight(self, event: Event, phase: str, state_data: Mapping[str, object] | None = None) -> StateData:
        work = TurnWork(event=event, state=State.model_validate(state_data or {}))
        if self.lookup_channel is None:
            return _state_data(work.state)
        work.state.progress = [*work.state.progress, phase]
        await self.render(work)
        return _state_data(work.state)

    async def fail(self, event: Event, error: str, state_data: JsonObject | None = None) -> EngineResult:
        return await self.finish(
            TurnWork(
                event=event,
                state=State.model_validate(state_data or {}),
                error=error,
                processing_emoji=self.reaction_emojis["processing"],
            )
        )

    def set_reaction_emojis(self, values: dict[str, str]) -> dict[str, str]:
        updated = dict(self.reaction_emojis)
        for phase in DEFAULT_REACTION_EMOJIS:
            if (value := values.get(phase)) is not None:
                if not value.strip() or len(value) > MAX_REACTION_LENGTH:
                    reason = f"invalid {phase} reaction"
                    raise ValueError(reason)
                updated[phase] = value
        self.reaction_emojis = updated
        return dict(updated)


def _state_data(state: State, *, finished: bool = False) -> StateData:
    data = state.model_dump(mode="json")
    data.update(seen=sorted(state.seen), processed=sorted(state.processed))
    if finished:
        data.update(delivery_id=None, progress=[])
    return cast("StateData", data)


def _transport_error(error: Exception) -> bool:
    return isinstance(error, (RunnerError, TransportError)) or any(
        marker in str(error).lower() for marker in ("disconnect", "transport error", "http 5")
    )
