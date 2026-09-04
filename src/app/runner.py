# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, cast

import httpx

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


class LifecycleRunner(Runner, Protocol):
    async def acquire(self, user: str, workspace: str) -> None: ...

    async def start(self, thread: str, user: str, workspace: str = "") -> str: ...


class RunnerError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"runner returned HTTP {status}: {detail}")


class HttpRunner:
    def __init__(self, url: str, token: str = "") -> None:
        self.url, self.token = url.rstrip("/"), token

    async def _post(self, path: str, payload: dict[str, object], request_timeout: float = 30) -> JsonObject:
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=request_timeout) as client:
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
        if progress is not None:
            return await self._run_with_progress(thread, prompt, user, workspace, progress)
        return await self._post_turn(thread, prompt, user, workspace)

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
                "turn_number": TURN_NUMBER.get(),
            },
            request_timeout=300,
        )
        billing: dict[str, object] = {key: data[key] for key in ("model", "cost", "usage") if key in data}
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
            "turn_number": TURN_NUMBER.get(),
        }
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=600) as client:
            request = asyncio.create_task(client.post(f"{self.url}/turn", headers=headers, json=payload))
            seen_steps: set[str] = set()
            while not request.done():
                try:
                    status = await client.get(
                        f"{self.url}/progress/{payload['thread_id']}",
                        headers=headers,
                        timeout=5,
                    )
                    if not status.is_error:
                        value = status.json()
                        if isinstance(value, dict):
                            steps = value.get("steps", [])
                            messages = steps if isinstance(steps, list) and steps else [value.get("message")]
                            for message in messages:
                                if isinstance(message, str) and message not in seen_steps:
                                    seen_steps.add(message)
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
        billing: dict[str, object] = {key: value[key] for key in ("model", "cost", "usage") if key in value}
        return str(value.get("thread_id", thread)), str(value.get("output", "")), billing


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
