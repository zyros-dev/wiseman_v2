# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

import httpx
from temporalio import activity

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.types import JsonObject


class Runner(Protocol):
    async def run(
        self,
        thread: str,
        prompt: str,
        user: str,
        workspace: str = "",
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, dict[str, object]]: ...


TURN_NUMBER: ContextVar[int] = ContextVar("wiseman_turn_number", default=0)
MESSAGE_ID: ContextVar[str] = ContextVar("wiseman_message_id", default="")


class LifecycleRunner(Runner, Protocol):
    async def acquire(self, user: str, workspace: str) -> None: ...

    async def start(self, thread: str, user: str, workspace: str = "") -> str: ...


class RunnerError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"runner returned HTTP {status}: {detail}")


class HttpRunner:
    def __init__(self, url: str, token: str = "") -> None:
        self.url, self.token = url.rstrip("/"), token

    async def _post(self, path: str, payload: dict[str, object]) -> JsonObject:
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.url}{path}", headers=headers, json=payload)
        if response.is_error:
            raise RunnerError(response.status_code, response.text[:1_000])
        value: object = response.json()
        if not isinstance(value, dict):
            raise RunnerError(response.status_code, "runner returned a non-object response")
        return cast("JsonObject", value)

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
        payload: dict[str, object] = {
            "thread_id": workspace or thread or f"thread-{user}",
            "codex_thread_id": thread or None,
            "user_id": user,
            "input": prompt,
            "turn_number": TURN_NUMBER.get(),
            "message_id": MESSAGE_ID.get() or uuid4().hex,
        }
        value = await self._post("/turn", payload)
        cursor = 0
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=30) as client:
            while value.get("status") == "running":
                if activity.in_activity():
                    activity.heartbeat()
                response = await client.get(f"{self.url}/jobs/{payload['message_id']}", headers=headers)
                response.raise_for_status()
                value = response.json()
                steps = value.get("steps", [])
                if progress is not None and isinstance(steps, list):
                    if messages := [str(message) for message in steps[cursor:][-8:]]:
                        await progress("\n".join(messages))
                    cursor = len(steps)
                if value.get("status") == "running":
                    await asyncio.sleep(0.75)
        if value.get("status") == "failed":
            raise RunnerError(503, str(value.get("error")))
        result = value.get("result", value)
        if not isinstance(result, dict):
            raise RunnerError(502, "runner returned a non-object result")
        billing = {key: result[key] for key in ("model", "cost", "usage") if key in result}
        return str(result.get("thread_id", thread)), str(result.get("output", "")), billing

    async def steer(self, thread: str, prompt: str, user: str, workspace: str = "") -> bool:
        data = await self._post(
            "/steer",
            {
                "thread_id": workspace or thread or f"thread-{user}",
                "codex_thread_id": thread or None,
                "user_id": user,
                "input": prompt,
            },
        )
        return bool(data.get("steered", False))


class FakeRunner:
    async def acquire(self, user: str, workspace: str) -> None: ...

    async def start(self, thread: str, user: str, _workspace: str = "") -> str:
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
