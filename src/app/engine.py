# Copyright (c) 2026 Nick van der Merwe
"""Turn lifecycle coordination and reaction state."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import discord
from prometheus_client import Counter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from app.admission import context
from app.admission import event_data as _event_data
from app.admission import image_tool_instruction as _image_tool_instruction
from app.admission import render_grammar as _grammar
from app.models import ActiveTurn, Event, Message, Messageable, State
from app.phoenix import Phoenix, PromptHub
from app.phoenix import json_text as _json
from app.phoenix import route_info as _route_info
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_REACTION_LENGTH,
)
from app.presentation import (
    deliver_content as _deliver_content,
)
from app.presentation import (
    edit_delivery as _edit_delivery,
)
from app.presentation import (
    render_progress as _render_progress,
)
from app.presentation import (
    startup_embed as _startup_embed,
)
from app.runner import HttpRunner, Runner, SteerableRunner

LOGGER = logging.getLogger("wiseman")
TURN_TOTAL = Counter("wiseman_turns_total", "Accepted Discord turns")
TURN_FAILURES = Counter("wiseman_turn_failures_total", "Failed Discord turns")


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

    async def _handle(
        self,
        event: Event,
        live: discord.Message | None = None,
        delivery_channel: Messageable | None = None,
        state_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        state.last_activity = time.time()
        event.seen_ids = sorted(set(event.seen_ids) | state.seen)
        kind = event.kind or ("followup" if state.turn else "startup")
        trace = f"discord-{trigger.id}"
        prompt, current, progress_message, processing_emoji = await self._prepare(
            event=event,
            live=live,
            channel=delivery_channel,
            state=state,
            kind=kind,
            trace=trace,
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
            state.codex_thread, output, billing = await self._run_codex(
                state=state,
                trigger=trigger,
                prompt=prompt,
                report=report,
            )
        except Exception as exc:  # noqa: BLE001
            return await self._failure(
                trigger=trigger,
                live=live,
                channel=delivery_channel,
                progress_message=progress_message,
                processing_emoji=processing_emoji,
                trace=trace,
                kind=kind,
                state=state,
                error=exc,
            )
        finally:
            self.active_turns.pop(key, None)
        return await self._success(
            trigger=trigger,
            live=live,
            channel=delivery_channel,
            progress_message=progress_message,
            processing_emoji=processing_emoji,
            trace=trace,
            kind=kind,
            state=state,
            current=current,
            prompt=prompt,
            output=output,
            billing=billing,
        )

    def _state(self, key: str, data: dict[str, Any] | None) -> State:
        if data is None:
            return self.states[key]
        return State(
            codex_thread=data.get("codex_thread"),
            seen=set(data.get("seen", [])),
            processed=set(data.get("processed", [])),
            turn=int(data.get("turn", 0)),
        )

    async def _prepare(
        self,
        *,
        event: Event,
        live: discord.Message | None,
        channel: Messageable | None,
        state: State,
        kind: str,
        trace: str,
    ) -> tuple[str, dict[str, Any], object | None, str]:
        trigger = event.trigger
        raw = event.raw_payload or _event_data(event)
        await self.phoenix.record(
            trace,
            "admission",
            audit_id=trace,
            raw_request=raw,
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
            input=raw,
        )
        processing_emoji = self.reaction_emojis["processing"]
        self.working_reactions[trigger.id] = processing_emoji
        if self._react(trigger.id, processing_emoji) and live is not None:
            await live.add_reaction(processing_emoji)
        await self.phoenix.record(trace, "reaction", operations=[f"add:{processing_emoji}"])
        current = context(event)
        state.seen.update(current["selected_ids"])
        await self.phoenix.record(
            trace,
            "context",
            raw=_json(raw),
            normalized=current,
            selected_ids=current["selected_ids"],
        )
        grammar_name = "startup-context" if kind == "startup" else "followup-context"
        source = await self.prompts.source(grammar_name)
        grammar = _grammar(grammar_name, source, raw, mode=kind, messages=current["messages"])
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
        if channel is not None and kind == "startup":
            await cast("Any", channel).send(embed=_startup_embed())
        phase = "codex starting" if kind == "startup" else "working"
        progress = "🤖 Codex starting..." if kind == "startup" else "⏳ Working..."
        self.progress[trigger.id].append(phase)
        await self.phoenix.record(trace, "progress", phase=phase)
        if channel is not None:
            progress_message = await channel.send(_render_progress([progress]))
        key = trigger.thread_id or trigger.channel_id
        self.active_turns[key] = ActiveTurn(
            trigger_id=trigger.id,
            delivery_id=str(getattr(progress_message, "id", "")) or None,
        )
        return prompt, current, progress_message, processing_emoji

    async def _run_codex(
        self,
        *,
        state: State,
        trigger: Message,
        prompt: str,
        report: Callable[[str], Awaitable[None]],
    ) -> tuple[str, str, dict[str, object]]:
        workspace = trigger.thread_id or trigger.channel_id
        if isinstance(self.runner, HttpRunner):
            return await self.runner.run(
                state.codex_thread or "", prompt, trigger.author_id, workspace, progress=report
            )
        return await self.runner.run(state.codex_thread or "", prompt, trigger.author_id, workspace)

    async def _failure(
        self,
        *,
        trigger: Message,
        live: discord.Message | None,
        channel: Messageable | None,
        progress_message: object | None,
        processing_emoji: str,
        trace: str,
        kind: str,
        state: State,
        error: Exception,
    ) -> dict[str, Any]:
        TURN_FAILURES.inc()
        await self.phoenix.record(trace, "failure", error=str(error))
        failure_emoji = self.reaction_emojis["failure"]
        if self._react(trigger.id, failure_emoji):
            if live is not None:
                await live.add_reaction(failure_emoji)
            if channel is not None:
                await _deliver_content(progress_message, channel, f"Codex failed: {error}")
            await self._remove_working_reaction(trigger.id, live)
        await self.phoenix.record(
            trace,
            "reaction",
            operations=[f"add:{failure_emoji}", f"remove:{processing_emoji}"],
        )
        return {
            "trace": trace,
            "kind": kind,
            "error": str(error),
            "reactions": self.reactions[trigger.id],
            "state": _state_data(state),
        }

    async def _success(
        self,
        *,
        trigger: Message,
        live: discord.Message | None,
        channel: Messageable | None,
        progress_message: object | None,
        processing_emoji: str,
        trace: str,
        kind: str,
        state: State,
        current: dict[str, Any],
        prompt: str,
        output: str,
        billing: dict[str, object],
    ) -> dict[str, Any]:
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
        if channel is not None:
            await _deliver_content(progress_message, channel, output)
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
