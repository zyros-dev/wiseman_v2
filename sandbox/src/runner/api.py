# Copyright (c) 2026 Nick van der Merwe
"""Authenticated warm Codex runner with per-user shared and private workspaces."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import pwd
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    ItemCompletedNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openai_codex.models import Notification

CODEX_TEXT_ONLY_OVERRIDES = (
    "features.view_image=false",
    "features.image_generation=false",
)


class Turn(BaseModel):
    thread_id: str = Field(min_length=1)
    codex_thread_id: str | None = None
    user_id: str = Field(min_length=1)
    input: str = Field(max_length=100_000)


class WorkspaceError(ValueError):
    """Raised when a workspace path or link is unsafe."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class Workspace:
    """Materialize an owner tree and reject symlink escapes before every run."""

    def __init__(self, root: str = "/workspaces") -> None:
        self.root = Path(root).resolve()

    @staticmethod
    def username(user: str) -> str:
        return f"wsm_{hashlib.sha256(user.encode()).hexdigest()[:20]}"

    @staticmethod
    def _admin(args: list[str]) -> None:
        subprocess.run(args, check=True)  # noqa: S603 - arguments are generated below

    def _ensure_account(self, user: str) -> str:
        name = self.username(user)
        if os.getenv("WISEMAN_MANAGE_ACCOUNTS") != "1":
            return ""
        try:
            pwd.getpwnam(name)
        except KeyError:
            self._admin(["/usr/sbin/groupadd", "--system", name])
            self._admin(
                [
                    "/usr/sbin/useradd",
                    "--system",
                    "--gid",
                    name,
                    "--groups",
                    "wsm_sudo",
                    "--home-dir",
                    str(self.root / "users" / user),
                    "--shell",
                    "/bin/bash",
                    name,
                ]
            )
        return name

    @staticmethod
    def _own_tree(path: Path, user: str) -> None:
        # Existing Codex SQLite files may have been created by a stale account
        # after an interrupted provisioning race. Repair the complete local
        # tree, but never follow symlinks into another workspace.
        if not os.path.lexists(path):
            return
        try:
            shutil.chown(path, user=user, group=user, follow_symlinks=False)
            for root, directories, files in os.walk(path, followlinks=False):
                children = [Path(root) / name for name in (*directories, *files)]
                for child in children:
                    try:
                        shutil.chown(child, user=user, group=user, follow_symlinks=False)
                    except FileNotFoundError:
                        # Codex cleanup can remove a file between walk and chown.
                        continue
        except FileNotFoundError:
            # A top-level entry may still be removed concurrently.
            return

    def thread(self, user: str, thread: str) -> Path:
        users = (self.root / "users").resolve()
        users.mkdir(parents=True, exist_ok=True)
        base = (users / user).resolve()
        if base.parent != users:
            raise WorkspaceError("workspace owner escapes root")  # noqa: TRY003
        account = self._ensure_account(user)
        threads = base / "threads"
        threads.mkdir(parents=True, exist_ok=True)
        if threads.resolve().parent != base:
            raise WorkspaceError("thread root escapes owner")  # noqa: TRY003
        path = (threads / thread).resolve()
        if path.parent != threads.resolve():
            raise WorkspaceError("workspace path escapes owner")  # noqa: TRY003
        shared = base / "shared"
        if shared.is_symlink() and shared.resolve() != shared:
            raise WorkspaceError("shared directory escapes owner")  # noqa: TRY003
        path.mkdir(parents=True, exist_ok=True)
        (shared / "skills").mkdir(parents=True, exist_ok=True)
        (shared / "memories.md").touch(exist_ok=True)
        (shared / "AGENTS.md").touch(exist_ok=True)
        base.chmod(0o700)
        shared.chmod(0o700)
        (shared / "skills").chmod(0o700)
        (shared / "memories.md").chmod(0o600)
        (shared / "AGENTS.md").chmod(0o600)
        link = path / "shared"
        if link.is_symlink() and link.resolve() != shared:
            raise WorkspaceError("invalid shared link")  # noqa: TRY003
        if not link.exists():
            link.symlink_to(shared, target_is_directory=True)
        (path / ".codex").mkdir(exist_ok=True)
        path.chmod(0o700)
        (path / ".codex").chmod(0o700)
        relay = os.getenv("WISEMAN_RELAY_URL", "")
        if relay:
            (path / ".codex" / "config.toml").write_text(
                'model_provider = "wiseman-relay"\n'
                "[model_providers.wiseman-relay]\n"
                'name = "Wiseman relay"\n'
                f"base_url = {json.dumps(relay)}\n"
                'env_key = "OPENAI_API_KEY"\n'
                'wire_api = "responses"\n',
                encoding="utf-8",
            )
        (path / "AGENTS.md").write_text(
            "Read shared/AGENTS.md and shared/memories.md before acting.\n",
            encoding="utf-8",
        )
        (path / "AGENTS.md").chmod(0o600)
        if account:
            try:
                shutil.chown(base, user=account, group=account, follow_symlinks=False)
            except FileNotFoundError:
                return path
            self._own_tree(shared, account)
            self._own_tree(path, account)
        return path

    def cleanup(self, idle_seconds: int = 259200) -> int:
        """Remove thread state after three idle days while retaining shared state."""
        removed = 0
        users = self.root / "users"
        for path in users.glob("*/threads/*"):
            if path.is_dir() and time.time() - path.stat().st_mtime >= idle_seconds:
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_symlink() or child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
                removed += 1
        return removed


def _auth(got: str | None, expected: str) -> None:
    if expected and not hmac.compare_digest(got or "", f"Bearer {expected}"):
        raise HTTPException(401, "unauthorized")


class CodexRunner:
    """Use one SDK client and resumable Codex thread per warm runner."""

    def __init__(self) -> None:
        self.codex: dict[str, AsyncCodex] = {}
        self.threads: dict[str, object] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.progress: dict[str, str] = {}
        limit = max(1, int(os.getenv("WISEMAN_MAX_CONCURRENT_TURNS", "1")))
        self.capacity = asyncio.Semaphore(limit)

    async def start(self, turn: Turn, path: Path, account: str = "") -> dict[str, object]:
        lock = self.locks.setdefault(turn.thread_id, asyncio.Lock())
        async with self.capacity, lock:
            return await self._start_locked(turn, path, account)

    async def _start_locked(self, turn: Turn, path: Path, account: str) -> dict[str, object]:
        existing = self.threads.get(turn.thread_id)
        if not turn.codex_thread_id and existing is not None:
            return {"thread_id": getattr(existing, "id", "")}
        client = self._client(turn, path, account)
        if turn.codex_thread_id:
            thread = await client.thread_resume(
                turn.codex_thread_id,
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.full_access,
                cwd=str(path),
                model=os.getenv("WISEMAN_MODEL") or None,
                model_provider="wiseman-relay" if os.getenv("WISEMAN_RELAY_URL") else None,
            )
        else:
            thread = await client.thread_start(
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.full_access,
                cwd=str(path),
                developer_instructions=self._developer_instructions(),
                model=os.getenv("WISEMAN_MODEL") or None,
                model_provider="wiseman-relay" if os.getenv("WISEMAN_RELAY_URL") else None,
            )
        self.threads[turn.thread_id] = thread
        return {"thread_id": thread.id}

    def _client(self, turn: Turn, path: Path, account: str) -> AsyncCodex:
        key = turn.thread_id
        if key not in self.codex:
            env = dict(os.environ)
            if relay := os.getenv("WISEMAN_RELAY_URL"):
                env["OPENAI_BASE_URL"] = relay
            env["OPENAI_API_KEY"] = os.getenv("WISEMAN_PROVIDER_TOKEN", "")
            env["HOME"], env["CODEX_HOME"] = str(path), str(path / ".codex")
            env["WISEMAN_EXEC_USER"] = account
            env["WISEMAN_THREAD_ID"] = turn.thread_id
            self.codex[key] = AsyncCodex(
                CodexConfig(
                    codex_bin=os.getenv("WISEMAN_CODEX_BIN") or None,
                    config_overrides=CODEX_TEXT_ONLY_OVERRIDES,
                    cwd=str(path),
                    env=env,
                )
            )
        return self.codex[key]

    @staticmethod
    def _developer_instructions() -> str:
        return (
            "Use shared/AGENTS.md and shared/memories.md. "
            "For current or external facts, use available network tools. "
            "When Discord context includes an image attachment, use "
            "wiseman-image with its attachment URL. Add --question for a "
            "specific question, or omit it for a generic description. "
            "Do not claim visual details until the command returns a result. "
            "To send a file or image from this thread workspace into the Discord "
            "thread, run wiseman-discord send-file PATH --caption 'optional caption'. "
            "When the user explicitly requests it, use wiseman-discord set-profile "
            "or set-reactions to update the bot presentation. "
            "This trusted runner permits package installation: use sudo -n apt-get "
            "update and sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install. "
            "For large builds, use the available disk and keep parallelism modest. "
            "Never claim to have searched unless a command returned usable results; "
            "if a web command fails, say so plainly."
        )

    async def run(self, turn: Turn, path: Path, account: str = "") -> dict[str, object]:
        lock = self.locks.setdefault(turn.thread_id, asyncio.Lock())
        async with self.capacity, lock:
            thread = await self._thread(turn, path, account)
            self.progress[turn.thread_id] = "🤖 Codex turn started..."
            return await self._run_thread(thread, turn, path)

    async def _thread(self, turn: Turn, path: Path, account: str) -> object:
        # A client owns its cwd and Codex session environment, so it is private
        # to a Discord thread even when the user has many threads.
        key = turn.thread_id
        client = self._client(turn, path, account)
        thread = self.threads.get(key)
        if thread is None or (
            turn.codex_thread_id and getattr(thread, "id", None) != turn.codex_thread_id
        ):
            if not turn.codex_thread_id:
                started = await self._start_locked(turn, path, account)
                thread = self.threads[key]
                assert started["thread_id"] == thread.id
            else:
                thread = await client.thread_resume(
                    turn.codex_thread_id,
                    approval_mode=ApprovalMode.deny_all,
                    sandbox=Sandbox.full_access,
                    cwd=str(path),
                    model=os.getenv("WISEMAN_MODEL") or None,
                    model_provider="wiseman-relay" if os.getenv("WISEMAN_RELAY_URL") else None,
                )
                self.threads[key] = thread
        assert thread is not None
        return thread

    async def _run_thread(self, thread: object, turn: Turn, path: Path) -> dict[str, object]:
        if not hasattr(thread, "turn"):
            result = await thread.run(
                turn.input,
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.full_access,
                cwd=str(path),
            )
            return {
                "thread_id": thread.id,
                "output": result.final_response or "",
                "model": os.getenv("WISEMAN_MODEL", "codex"),
                "usage": result.usage.model_dump(mode="json") if result.usage else None,
            }
        active_turn = await thread.turn(
            turn.input,
            approval_mode=ApprovalMode.deny_all,
            sandbox=Sandbox.full_access,
            cwd=str(path),
        )
        items: list[object] = []
        usage: object = None
        completed: TurnCompletedNotification | None = None
        async for event in active_turn.stream():
            if message := _progress_message(event):
                self.progress[turn.thread_id] = message
            payload = event.payload
            if isinstance(payload, ItemCompletedNotification):
                items.append(payload.item)
            elif isinstance(payload, ThreadTokenUsageUpdatedNotification):
                usage = payload.token_usage
            elif isinstance(payload, TurnCompletedNotification):
                completed = payload
        if completed is None:
            raise RuntimeError("turn completed event not received")  # noqa: TRY003
        if completed.turn.error is not None:
            raise RuntimeError(completed.turn.error.message or "Codex turn failed")
        final_response = _final_response(items)
        return {
            "thread_id": thread.id,
            "output": final_response,
            "model": os.getenv("WISEMAN_MODEL", "codex"),
            "usage": usage.model_dump(mode="json") if usage is not None else None,
        }


def _final_response(items: list[object]) -> str:
    for item in reversed(items):
        root = getattr(item, "root", item)
        if isinstance(root, AgentMessageThreadItem) and root.text:
            return root.text
    return ""


def _progress_message(event: Notification) -> str | None:  # noqa: PLR0911
    """Map SDK lifecycle notifications to compact user-visible phases."""
    if event.method == "turn/started":
        return "🤖 Codex turn started..."
    if event.method == "item/agentMessage/delta":
        return "✍️ Writing response..."
    if event.method == "item/commandExecution/outputDelta":
        return "⚙️ Running command..."
    if event.method == "item/fileChange/outputDelta":
        return "📝 Editing files..."
    if event.method == "item/mcpToolCall/progress":
        return "🔌 Using a tool..."
    if event.method == "item/started":
        item = getattr(event.payload, "item", None)
        name = type(getattr(item, "root", item)).__name__
        return {
            "CommandExecutionThreadItem": "⚙️ Running command...",
            "FileChangeThreadItem": "📝 Editing files...",
            "McpToolCallThreadItem": "🔌 Using a tool...",
            "AgentMessageThreadItem": "✍️ Writing response...",
            "ReasoningThreadItem": "🧠 Reasoning...",
        }.get(name, "🤖 Codex working...")
    return None


def create_app() -> FastAPI:  # noqa: C901
    root, secret = (
        os.getenv("WISEMAN_WORKSPACE_ROOT", "/workspaces"),
        os.getenv("WISEMAN_RUNNER_API_TOKEN", ""),
    )
    workspaces = Workspace(root)
    codex = CodexRunner()
    app = FastAPI(title="wiseman-sandbox", docs_url=None, redoc_url=None)
    cleaner: asyncio.Task[None] | None = None

    async def sweep() -> None:
        while True:
            workspaces.cleanup()
            await asyncio.sleep(3600)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.on_event("startup")
    async def start_cleanup() -> None:
        nonlocal cleaner
        cleaner = asyncio.create_task(sweep())

    @app.on_event("shutdown")
    async def stop_cleanup() -> None:
        if cleaner is not None:
            cleaner.cancel()

    @app.post("/cleanup")
    async def cleanup(authorization: Annotated[str | None, Header()] = None) -> dict[str, int]:
        _auth(authorization, secret)
        return {"removed": workspaces.cleanup()}

    @app.post("/acquire")
    async def acquire(
        turn: Turn, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _auth(authorization, secret)
        path = workspaces.thread(turn.user_id, turn.thread_id)
        return {"thread_id": turn.thread_id, "path": str(path), "shared": str(path / "shared")}

    @app.post("/start")
    async def start(
        turn: Turn, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _auth(authorization, secret)
        path = workspaces.thread(turn.user_id, turn.thread_id)
        try:
            return await codex.start(turn, path, workspaces.username(turn.user_id))
        except Exception as exc:
            raise HTTPException(503, f"codex startup unavailable: {exc}") from exc

    @app.post("/turn")
    async def run(
        turn: Turn, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, object]:
        _auth(authorization, secret)
        path = workspaces.thread(turn.user_id, turn.thread_id)
        try:
            data = await codex.run(turn, path, workspaces.username(turn.user_id))
        except Exception as exc:
            raise HTTPException(503, f"codex unavailable: {exc}") from exc
        return data

    @app.get("/progress/{thread_id}")
    async def progress(
        thread_id: str, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, str]:
        _auth(authorization, secret)
        return {"message": codex.progress.get(thread_id, "🤖 Codex starting...")}

    return app


app = create_app()
