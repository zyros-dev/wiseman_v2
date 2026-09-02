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
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
from pydantic import BaseModel, Field


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
                    "--home-dir",
                    str(self.root / "users" / user),
                    name,
                ]
            )
        return name

    @staticmethod
    def _own_tree(path: Path, user: str) -> None:
        # Codex owns its live descendants. Walking them here races with plugin
        # clones and cleanup, so only repair the stable workspace surface.
        stable = (
            path,
            path / "AGENTS.md",
            path / ".codex",
            path / "files",
            path / "shared",
            path / "shared" / "AGENTS.md",
            path / "shared" / "memories.md",
            path / "shared" / "skills",
            path / "threads",
        )
        for child in stable:
            if child.is_symlink() or not os.path.lexists(child):
                continue
            try:
                shutil.chown(child, user=user, group=user, follow_symlinks=False)
            except FileNotFoundError:
                # A top-level entry may still be removed concurrently.
                continue

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
            self._own_tree(base, account)
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
        self.locks: dict[str, asyncio.Lock] = {}

    async def run(self, turn: Turn, path: Path, account: str = "") -> dict[str, object]:
        lock = self.locks.setdefault(turn.thread_id, asyncio.Lock())
        async with lock:
            key = account or turn.user_id
            if key not in self.codex:
                env = dict(os.environ)
                if relay := os.getenv("WISEMAN_RELAY_URL"):
                    env["OPENAI_BASE_URL"] = relay
                env["OPENAI_API_KEY"] = os.getenv("WISEMAN_PROVIDER_TOKEN", "")
                env["HOME"], env["CODEX_HOME"] = str(path), str(path / ".codex")
                env["WISEMAN_EXEC_USER"] = account
                self.codex[key] = AsyncCodex(
                    CodexConfig(
                        codex_bin=os.getenv("WISEMAN_CODEX_BIN") or None,
                        cwd=str(path),
                        env=env,
                    )
                )
            client = self.codex[key]
            provider = "wiseman-relay" if os.getenv("WISEMAN_RELAY_URL") else None
            model = os.getenv("WISEMAN_MODEL") or None
            if not turn.codex_thread_id:
                thread = await client.thread_start(
                    approval_mode=ApprovalMode.deny_all,
                    sandbox=Sandbox.full_access,
                    cwd=str(path),
                    developer_instructions="Use shared/AGENTS.md and shared/memories.md.",
                    model=model,
                    model_provider=provider,
                )
            else:
                thread = await client.thread_resume(
                    turn.codex_thread_id,
                    approval_mode=ApprovalMode.deny_all,
                    sandbox=Sandbox.full_access,
                    cwd=str(path),
                    model=model,
                    model_provider=provider,
                )
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

    return app


app = create_app()
