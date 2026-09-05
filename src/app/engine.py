# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from prometheus_client import Counter

from app.admission import ContextConfig, context, image_tool_instruction, render_grammar
from app.models import Event, MessageRef, State, TurnWork, Upload
from app.phoenix import json_text, route_info
from app.presentation import (
    DEFAULT_REACTION_EMOJIS,
    MAX_DISCORD_CONTENT_LENGTH,
    MAX_REACTION_LENGTH,
    render_progress,
    startup_embed,
)
from app.runner import MESSAGE_ID, TURN_NUMBER

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from app.clients.client_interfaces import DiscordClient, PhoenixClient, PromptClient, RunnerClient
    from app.types import JsonObject, StateData

TURN_FAILURES = Counter("wiseman_turn_failures_total", "Failed Discord turns")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    phoenix: PhoenixClient
    runner: RunnerClient
    prompts: PromptClient
    context: ContextConfig = field(default_factory=ContextConfig.from_env)
    discord: DiscordClient | None = None


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.reaction_emojis = dict(DEFAULT_REACTION_EMOJIS)

    @property
    def discord(self) -> DiscordClient:
        return cast("DiscordClient", self.config.discord)

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
        parts = {
            "soul": await self.config.prompts.source("wiseman-soul"),
            "runtime": await self.config.prompts.source("wiseman-runtime"),
            "memories": os.getenv("WISEMAN_MEMORIES", ""),
            "context": cast("str", work.grammar["rendered"]),
            "user": "\n\n".join(
                part
                for part in (
                    work.event.trigger.content,
                    image_tool_instruction(
                        work.event.trigger.model_dump(),
                        cast("list[dict[str, object]]", work.current["reply_ancestors"]),
                    ),
                )
                if part
            ),
        }
        work.prompt = json_text(parts)
        await self.config.phoenix.record(work.trace, "prompt", parts=parts, final_input=work.prompt)
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

    async def render(self, work: TurnWork) -> TurnWork:
        channel = work.event.trigger.thread_id or work.event.trigger.channel_id
        if not work.state.banner_sent:
            await self.discord.send(channel, embed=cast("JsonObject", startup_embed().to_dict()), nonce=f"b:{work.event.trigger.id}")
            work.state.banner_sent = True
        content = render_progress(work.state.progress, work.state.turn + 1)
        if work.state.delivery_id:
            await self.discord.edit(MessageRef(channel, work.state.delivery_id), content)
        else:
            work.state.delivery_id = await self.discord.send(channel, content, nonce=work.event.trigger.id)
        return work

    async def deliver(self, work: TurnWork) -> TurnWork:
        content = f"Codex failed: {work.error}" if work.error else work.output
        if not work.state.delivery_id:
            raise RuntimeError("Recorded Discord answer is unavailable")
        ref = MessageRef(work.event.trigger.thread_id or work.event.trigger.channel_id, work.state.delivery_id)
        upload = Upload("response.md", content.encode()) if len(content) > MAX_DISCORD_CONTENT_LENGTH else None
        await self.discord.edit(ref, content[:1800] + "\n\nFull response attached." if upload else content, upload=upload)
        return work

    async def reconcile(self, work: TurnWork) -> TurnWork:
        ref = MessageRef(work.event.trigger.channel_id, work.event.trigger.id)
        await self.discord.add_reaction(ref, work.terminal_emoji or work.processing_emoji)
        if work.terminal_emoji:
            await self.discord.remove_reaction(ref, work.processing_emoji)
        return work

    async def finish(self, work: TurnWork) -> dict[str, object]:
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
        if not work.state.delivery_id:
            await self.render(work)
        await self.deliver(work)
        work.terminal_emoji = self.reaction_emojis["failure" if work.error else "success"]
        await self.reconcile(work)
        await self.config.phoenix.record(work.trace, "reaction", operations=[f"add:{work.terminal_emoji}", f"remove:{work.processing_emoji}"])
        work.state.processed.add(work.event.trigger.id)
        work.state.turn += int(not work.error)
        return {"state": _state_data(work.state, finished=True), **({"error": work.error} if work.error else {"output": work.output})}

    async def preflight(self, event: Event, phase: str, state_data: Mapping[str, object] | None = None) -> StateData:
        work = TurnWork(event=event, state=State.model_validate(state_data or {}))
        work.state.progress = [*work.state.progress, phase]
        await self.render(work)
        return _state_data(work.state)

    async def fail(self, event: Event, error: str, state_data: JsonObject | None = None) -> dict[str, object]:
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
