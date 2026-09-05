# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio import activity
from temporalio.exceptions import ApplicationError

from app.models import Event, TurnWork
from app.runner import MESSAGE_ID, STOPPED_STATUS, RunnerError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from temporalio.client import Client

    from app.engine import Engine


class TurnActivities:
    def __init__(self, engine: Engine, client: Client) -> None:
        self.engine, self.client = engine, client

    async def _work(self, method: Callable[[TurnWork], Awaitable[TurnWork]], payload: dict) -> dict:
        return (await method(TurnWork.model_validate(payload))).model_dump(mode="json")

    @activity.defn(name="wiseman.context")
    async def context(self, payload: dict) -> dict:
        return await self._work(self.engine.prepare_context, payload)

    @activity.defn(name="wiseman.prompt")
    async def prompt(self, payload: dict) -> dict:
        return await self._work(self.engine.prepare_prompt, payload)

    @activity.defn(name="wiseman.render")
    async def render(self, payload: dict) -> dict:
        work = TurnWork.model_validate(payload)
        work.processing_emoji = work.processing_emoji or self.engine.reaction_emojis["processing"]
        return (await self.engine.render(work)).model_dump(mode="json")

    @activity.defn(name="wiseman.infer")
    async def infer(self, payload: dict) -> dict:
        workflow_id = activity.info().workflow_id
        assert workflow_id is not None
        handle = self.client.get_workflow_handle(workflow_id, run_id=activity.info().workflow_run_id)

        async def report(message: str) -> None:
            await handle.signal("progress", message)

        try:
            return (await self.engine.execute(TurnWork.model_validate(payload), report)).model_dump(mode="json")
        except RunnerError as exc:
            raise ApplicationError(str(exc), non_retryable=exc.status == STOPPED_STATUS) from exc

    @activity.defn(name="wiseman.deliver")
    async def deliver(self, payload: dict) -> dict:
        return await self._work(self.engine.deliver, payload)

    @activity.defn(name="wiseman.react")
    async def react(self, payload: dict) -> dict:
        work = TurnWork.model_validate(payload)
        if work.output or work.error:
            work.terminal_emoji = work.terminal_emoji or self.engine.reaction_emojis["failure" if work.error else "success"]
        return (await self.engine.reconcile(work)).model_dump(mode="json")

    @activity.defn(name="wiseman.observe")
    async def observe(self, payload: dict) -> dict:
        work = TurnWork.model_validate(payload)
        await self.engine.config.phoenix.record(
            work.trace,
            "completed",
            thread_id=work.event.trigger.thread_id,
            output=work.output,
            error=work.error,
            input=work.prompt,
            **work.billing,
        )
        return payload

    @activity.defn(name="wiseman.steer")
    async def steer(self, payload: dict) -> dict:
        work, event = TurnWork.model_validate(payload["work"]), Event.model_validate(payload["event"])
        token = MESSAGE_ID.set(event.trigger.id)
        try:
            accepted = await self.engine.config.runner.steer(
                work.state.codex_thread or "",
                event.trigger.content,
                work.state.owner_id,
                work.event.trigger.thread_id or work.event.trigger.channel_id,
            )
            return {"accepted": accepted}
        finally:
            MESSAGE_ID.reset(token)

    @activity.defn(name="wiseman.stop")
    async def stop(self, payload: dict) -> dict:
        work = TurnWork.model_validate(payload["work"])
        event = Event.model_validate(payload["event"])
        token = MESSAGE_ID.set(event.trigger.id)
        try:
            stopped = await self.engine.config.runner.stop(
                work.state.codex_thread or "",
                work.state.owner_id,
                work.event.trigger.thread_id or work.event.trigger.channel_id,
                work.event.trigger.id,
                event.trigger.id,
            )
            return {"accepted": stopped}
        finally:
            MESSAGE_ID.reset(token)

    def registered(self) -> list:
        return [self.context, self.prompt, self.render, self.infer, self.deliver, self.react, self.observe, self.steer, self.stop]
