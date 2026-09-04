# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import discord
from prometheus_client import Counter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.clients.client_interfaces import ClientContainer, PhoenixClient, PromptClient
    from app.types import JsonObject

from app.admission import ContextConfig, context, event_data, image_tool_instruction, render_grammar
from app.models import ActiveTurn, Event, Message, Messageable, State
from app.phoenix import Phoenix, PromptHub, json_text, route_info
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_REACTION_LENGTH,
    deliver_content,
    edit_delivery,
    render_progress,
    startup_embed,
)
from app.runner import TURN_NUMBER, Runner, RunnerError
from app.types import EngineResult, StateData  # noqa: TC001 - public result types are re-exported

LOGGER = logging.getLogger("wiseman")
TURN_TOTAL = Counter("wiseman_turns_total", "Accepted Discord turns")
TURN_FAILURES = Counter("wiseman_turn_failures_total", "Failed Discord turns")


@dataclass(slots=True)
class _Preparation:
    event: Event
    live: discord.Message | None
    channel: Messageable | None
    state: State
    kind: str
    trace: str


@dataclass(slots=True)
class _Lifecycle:
    trigger: Message
    live: discord.Message | None
    channel: Messageable | None
    progress_message: object | None
    processing_emoji: str
    trace: str
    kind: str
    state: State


@dataclass(slots=True)
class _Prepared:
    prompt: str
    current: dict[str, object]
    progress_message: object | None
    processing_emoji: str


@dataclass(slots=True)
class _Success:
    lifecycle: _Lifecycle
    current: dict[str, object]
    prompt: str
    output: str
    billing: dict[str, object]


class Engine:
    def __init__(
        self,
        phoenix: Phoenix | None = None,
        runner: Runner | None = None,
        prompts: PromptHub | None = None,
        context_config: ContextConfig | None = None,
        clients: ClientContainer | None = None,
    ) -> None:
        if clients is None and (phoenix is None or runner is None):
            raise ValueError("phoenix and runner are required without clients")  # noqa: TRY003
        self.phoenix: PhoenixClient = (
            clients.phoenix if clients is not None else cast("PhoenixClient", phoenix)
        )
        self.runner: Runner = clients.runner if clients is not None else cast("Runner", runner)
        self.prompts: PromptClient = (
            clients.prompts if clients is not None else prompts or PromptHub()
        )
        self.clients = clients
        self.context_config = context_config or ContextConfig.from_env()
        self.states: dict[str, State] = defaultdict(State)
        self.reactions: dict[str, list[str]] = defaultdict(list)
        self.progress: dict[str, list[str]] = defaultdict(list)
        self.locks: dict[str, asyncio.Lock] = {}
        self.reaction_user: object | None = None
        self.reaction_emojis = dict(DEFAULT_REACTION_EMOJIS)
        self.working_reactions: dict[str, str] = {}
        self.active_turns: dict[str, ActiveTurn] = {}
        self.deliveries: dict[str, object] = {}
        self.lookup: Callable[[Event], Awaitable[discord.Message | None]] | None = None
        self.lookup_channel: Callable[[Event], Awaitable[Messageable | None]] | None = None

    def bind_clients(self, clients: ClientContainer) -> None:
        self.clients = clients
        self.phoenix = clients.phoenix
        self.runner = clients.runner
        self.prompts = clients.prompts

    async def handle(
        self,
        event: Event,
        live: discord.Message | None = None,
        delivery_channel: Messageable | None = None,
        state_data: JsonObject | None = None,
        *,
        retry_transport: bool = False,
    ) -> EngineResult:
        key = event.trigger.thread_id or event.trigger.channel_id
        async with self.locks.setdefault(key, asyncio.Lock()):
            if live is None and self.lookup is not None:
                live = await self.lookup(event)
            if delivery_channel is None:
                if self.lookup_channel is not None:
                    delivery_channel = await self.lookup_channel(event)
                elif live is not None:
                    delivery_channel = live.channel
            return await self._handle(
                event,
                live,
                delivery_channel,
                state_data,
                retry_transport=retry_transport,
            )

    async def _handle(
        self,
        event: Event,
        live: discord.Message | None = None,
        delivery_channel: Messageable | None = None,
        state_data: JsonObject | None = None,
        *,
        retry_transport: bool = False,
    ) -> EngineResult:
        trigger = event.trigger
        key = trigger.thread_id or trigger.channel_id
        state = self._state(key, state_data)
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
        event.seen_ids = sorted(set(event.seen_ids) | state.seen)
        kind = event.kind or ("followup" if state.turn else "startup")
        trace = f"discord-{trigger.id}"
        prepared = await self._prepare(
            _Preparation(
                event=event,
                live=live,
                channel=delivery_channel,
                state=state,
                kind=kind,
                trace=trace,
            )
        )
        lifecycle = _Lifecycle(
            trigger=trigger,
            live=live,
            channel=delivery_channel,
            progress_message=prepared.progress_message,
            processing_emoji=prepared.processing_emoji,
            trace=trace,
            kind=kind,
            state=state,
        )

        async def report(message: str) -> None:
            if self.progress[trigger.id] and self.progress[trigger.id][-1] == message:
                return
            self.progress[trigger.id].append(message)
            await self.phoenix.record(trace, "progress", phase=message)
            if lifecycle.progress_message is not None:
                try:
                    await edit_delivery(
                        lifecycle.progress_message,
                        render_progress(self.progress[trigger.id], state.turn + 1),
                    )
                except discord.DiscordException:
                    LOGGER.warning("Could not update progress message for %s", trigger.id)

        try:
            state.codex_thread, output, billing = await self._run_codex(
                state=state,
                trigger=trigger,
                prompt=prepared.prompt,
                report=report,
            )
        except Exception as exc:
            if retry_transport and _transport_error(exc):
                raise
            return await self._failure(lifecycle, exc)
        finally:
            self.active_turns.pop(key, None)
        return await self._success(
            _Success(
                lifecycle=lifecycle,
                current=prepared.current,
                prompt=prepared.prompt,
                output=output,
                billing=billing,
            )
        )

    def _state(self, key: str, data: JsonObject | None) -> State:
        if data is None:
            return self.states[key]
        return State(
            codex_thread=cast("str | None", data.get("codex_thread")),
            seen=_strings(data.get("seen", [])),
            processed=_strings(data.get("processed", [])),
            turn=int(cast("int", data.get("turn", 0))),
        )

    async def preflight(self, event: Event, phase: str, state_data: JsonObject | None = None) -> None:  # fmt: skip  # noqa: E501
        trigger = event.trigger
        if self.lookup_channel is None or (channel := await self.lookup_channel(event)) is None:
            return
        steps = self.progress[trigger.id]
        if event.kind == "startup" and not steps:
            with suppress(discord.DiscordException):
                await cast("Callable[..., Awaitable[object]]", channel.send)(embed=startup_embed())
        steps.extend(() if phase in steps else (phase,))
        message = self.deliveries.get(trigger.id)
        content = render_progress(steps, self._state(trigger.thread_id or trigger.channel_id, state_data).turn + 1)  # fmt: skip  # noqa: E501
        if message is None:
            with suppress(discord.DiscordException):
                message = await channel.send(content)
            self.deliveries[trigger.id] = message
        else:
            with suppress(discord.DiscordException):
                await edit_delivery(message, content)

    async def fail(self, event: Event, error: str, state_data: JsonObject | None = None) -> EngineResult:  # fmt: skip  # noqa: E501
        key = event.trigger.thread_id or event.trigger.channel_id
        live = await self.lookup(event) if self.lookup is not None else None
        channel = await self.lookup_channel(event) if self.lookup_channel is not None else None
        lifecycle = _Lifecycle(event.trigger, live, channel, self.deliveries.get(event.trigger.id), self.working_reactions.get(event.trigger.id, self.reaction_emojis["processing"]), f"discord-{event.trigger.id}", event.kind or "startup", self._state(key, state_data))  # fmt: skip  # noqa: E501
        return await self._failure(lifecycle, RuntimeError(error))

    async def _prepare(self, request: _Preparation) -> _Prepared:
        r = request
        trigger = r.event.trigger
        raw = r.event.raw_payload or event_data(r.event)
        await self.phoenix.record(
            r.trace,
            "admission",
            audit_id=r.trace,
            raw_request=raw,
            normalized_request=event_data(r.event),
            normalizer="normalize_event:v2",
        )
        await self.phoenix.record(
            r.trace,
            "turn",
            thread_id=trigger.thread_id,
            message_id=trigger.id,
            kind=r.kind,
            route=route_info(),
            input=raw,
        )
        processing_emoji = self.reaction_emojis["processing"]
        self.working_reactions[trigger.id] = processing_emoji
        self._react(trigger.id, processing_emoji)
        await self._ensure_live_reaction(r.live, processing_emoji)
        await self.phoenix.record(r.trace, "reaction", operations=[f"add:{processing_emoji}"])
        current = context(r.event, self.context_config)
        r.state.seen.update(cast("list[str]", current["selected_ids"]))
        await self.phoenix.record(
            r.trace,
            "context",
            raw=json_text(raw),
            normalized=current,
            selected_ids=current["selected_ids"],
        )
        grammar_name = "startup-context" if r.kind == "startup" else "followup-context"
        source = await self.prompts.source(grammar_name)
        grammar = render_grammar(
            grammar_name,
            source,
            raw,
            mode=r.kind,
            messages=cast("list[dict[str, object]]", current["messages"]),
        )
        await self.phoenix.record(r.trace, "grammar", **grammar)
        parts = {
            "soul": await self.prompts.source("wiseman-soul"),
            "runtime": await self.prompts.source("wiseman-runtime"),
            "memories": os.getenv("WISEMAN_MEMORIES", ""),
            "context": cast("str", grammar["rendered"]),
            "user": "\n\n".join(
                part
                for part in (
                    trigger.content,
                    image_tool_instruction(
                        trigger.model_dump(mode="json"),
                        cast("list[dict[str, object]]", current["reply_ancestors"]),
                    ),
                )
                if part
            ),
        }
        prompt = json_text(parts)
        await self.phoenix.record(r.trace, "prompt", parts=parts, final_input=prompt)
        progress_message = self.deliveries.get(trigger.id)
        if r.channel is not None and r.kind == "startup" and not self.progress[trigger.id]:
            with suppress(discord.DiscordException):
                await cast("Callable[..., Awaitable[object]]", r.channel.send)(
                    embed=startup_embed()
                )
        phase = "codex starting" if r.kind == "startup" else "working"
        progress = "🤖 Codex starting..." if r.kind == "startup" else "⏳ Working..."
        if not self.progress[trigger.id]:
            self.progress[trigger.id].append(phase)
        await self.phoenix.record(r.trace, "progress", phase=phase)
        if progress_message is None and r.channel is not None:
            with suppress(discord.DiscordException):
                progress_message = await r.channel.send(render_progress([progress], r.state.turn + 1))  # fmt: skip  # noqa: E501
            self.deliveries[trigger.id] = progress_message
        key = trigger.thread_id or trigger.channel_id
        self.active_turns[key] = ActiveTurn(trigger.id, str(getattr(progress_message, "id", "")) or None)  # fmt: skip  # noqa: E501
        return _Prepared(prompt, current, progress_message, processing_emoji)

    async def _run_codex(
        self,
        *,
        state: State,
        trigger: Message,
        prompt: str,
        report: Callable[[str], Awaitable[None]],
    ) -> tuple[str, str, dict[str, object]]:
        workspace = trigger.thread_id or trigger.channel_id
        token = TURN_NUMBER.set(state.turn + 1)
        try:
            return await self.runner.run(
                state.codex_thread or "", prompt, trigger.author_id, workspace, progress=report
            )
        finally:
            TURN_NUMBER.reset(token)

    async def _failure(self, lifecycle: _Lifecycle, error: Exception) -> EngineResult:
        trigger = lifecycle.trigger
        TURN_FAILURES.inc()
        await self.phoenix.record(lifecycle.trace, "failure", error=str(error))
        failure_emoji = self.reaction_emojis["failure"]
        if self._react(trigger.id, failure_emoji):
            await self._ensure_live_reaction(lifecycle.live, failure_emoji)
            if lifecycle.channel is not None:
                await deliver_content(
                    lifecycle.progress_message, lifecycle.channel, f"Codex failed: {error}"
                )
            await self._remove_working_reaction(trigger.id, lifecycle.live)
        await self.phoenix.record(
            lifecycle.trace,
            "reaction",
            operations=[f"add:{failure_emoji}", f"remove:{lifecycle.processing_emoji}"],
        )
        self.deliveries.pop(trigger.id, None)
        return {
            "trace": lifecycle.trace,
            "kind": lifecycle.kind,
            "error": str(error),
            "reactions": self.reactions[trigger.id],
            "state": _state_data(lifecycle.state),
        }

    async def _success(self, result: _Success) -> EngineResult:
        lifecycle = result.lifecycle
        trigger = lifecycle.trigger
        state = lifecycle.state
        state.turn += 1
        self.progress[trigger.id].append("✍️ Writing response...")
        await self.phoenix.record(lifecycle.trace, "progress", phase="finalizing")
        await self.phoenix.record(
            lifecycle.trace,
            "codex",
            input=result.prompt,
            thread_id=state.codex_thread,
            output=result.output,
            **result.billing,
        )
        await self.phoenix.record(lifecycle.trace, "delivery", output=result.output)
        if lifecycle.channel is not None:
            await deliver_content(lifecycle.progress_message, lifecycle.channel, result.output)
        success_emoji = self.reaction_emojis["success"]
        if self._react(trigger.id, success_emoji):
            await self._ensure_live_reaction(lifecycle.live, success_emoji)
            await self._remove_working_reaction(trigger.id, lifecycle.live)
        await self.phoenix.record(
            lifecycle.trace,
            "reaction",
            operations=[f"add:{success_emoji}", f"remove:{lifecycle.processing_emoji}"],
        )
        self.deliveries.pop(trigger.id, None)
        return {
            "trace": lifecycle.trace,
            "kind": lifecycle.kind,
            "output": result.output,
            "selected_ids": cast("list[str]", result.current["selected_ids"]),
            "reactions": self.reactions[trigger.id],
            "progress": self.progress[trigger.id],
            "state": _state_data(state),
        }

    async def steer_if_active(
        self, thread_id: str, message_id: str, prompt: str, user: str
    ) -> bool:
        active = self.active_turns.get(thread_id)
        if active is None or active.delivery_id != message_id:
            return False
        steer = getattr(self.runner, "steer", None)
        if not callable(steer):
            return False
        accepted = await cast("Callable[..., Awaitable[bool]]", steer)(
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
        if live is not None and self.reaction_user is not None:
            with suppress(discord.DiscordException):
                await live.remove_reaction(emoji, cast("discord.User", self.reaction_user))

    @staticmethod
    async def _ensure_live_reaction(live: discord.Message | None, emoji: str) -> None:
        if live is not None:
            with suppress(discord.DiscordException):
                await live.add_reaction(emoji)


def _state_data(state: State) -> StateData:
    return {
        "codex_thread": state.codex_thread,
        "seen": sorted(state.seen),
        "processed": sorted(state.processed),
        "turn": state.turn,
    }


def _strings(value: object) -> set[str]:
    return set() if not isinstance(value, list) else {str(item) for item in value}


def _transport_error(error: Exception) -> bool:
    return isinstance(error, RunnerError) or any(marker in str(error).lower() for marker in ("disconnect", "transport error", "http 5"))  # fmt: skip  # noqa: E501
