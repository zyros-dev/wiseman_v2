# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class Receipt(BaseModel):
    workspace: str
    status: Literal["running", "completed", "failed"] = "running"
    result: dict[str, object] = Field(default_factory=dict)
    error: str = ""


class Jobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def _path(self, key: str) -> Path:
        return self.root / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _save(self, key: str, receipt: Receipt) -> None:
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(receipt.model_dump_json())
        temporary.replace(path)

    def get(self, key: str) -> Receipt:
        receipt = Receipt.model_validate_json(self._path(key).read_text())
        if receipt.status == "running" and key not in self.tasks:
            receipt.status, receipt.error = "failed", "Runner restarted during this turn; execution outcome is unknown."
            self._save(key, receipt)
        return receipt

    def submit(self, key: str, workspace: str, work: Callable[[], Awaitable[dict[str, object]]]) -> Receipt:
        if self._path(key).exists():
            receipt = self.get(key)
            if receipt.workspace != workspace:
                raise ValueError("message ID belongs to a different workspace")
            return receipt
        receipt = Receipt(workspace=workspace)
        self._save(key, receipt)
        self.tasks[key] = asyncio.create_task(self._execute(key, receipt, work))
        return receipt

    async def _execute(self, key: str, receipt: Receipt, work: Callable[[], Awaitable[dict[str, object]]]) -> None:
        try:
            receipt.result = await work()
            receipt.status = "completed"
        except (Exception, asyncio.CancelledError) as exc:
            logging.getLogger(__name__).exception("Runner job %s failed", key)
            receipt.status, receipt.error = "failed", str(exc) or type(exc).__name__
        finally:
            self._save(key, receipt)
            self.tasks.pop(key, None)

    async def close(self) -> None:
        tasks = tuple(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
