# Copyright (c) 2026 Nick van der Merwe
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from runner.api import CodexRunner, Turn


def test_codex_launcher_does_not_inherit_service_credentials(tmp_path):
    launcher = Path(__file__).parents[1] / "sandbox" / "codex-as-user"
    secret_names = (
        "DISCORD_BOT_TOKEN",
        "OPENROUTER_API_KEY",
        "PHOENIX_API_KEY",
        "WISEMAN_RUNNER_API_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    )
    env = {**os.environ, **dict.fromkeys(secret_names, "never-expose-this")}
    env.update(
        WISEMAN_EXEC_USER="",
        WISEMAN_CODEX_REAL_BIN=shutil.which("env") or "/usr/bin/env",
        HOME=str(tmp_path),
        CODEX_HOME=str(tmp_path / ".codex"),
        OPENAI_API_KEY="relay-capability",
        OPENAI_BASE_URL="http://gateway/v1",
        WISEMAN_RELAY_URL="http://gateway/v1",
        WISEMAN_MCP_TOKEN="tool-capability",
        WISEMAN_THREAD_ID="123",
    )
    child = subprocess.run(["/bin/sh", str(launcher)], env=env, text=True, capture_output=True, check=True)  # noqa: S603
    values = dict(line.split("=", 1) for line in child.stdout.splitlines())
    assert not set(secret_names) & values.keys()
    assert "never-expose-this" not in child.stdout
    assert values["HOME"] == str(tmp_path)
    assert values["CODEX_HOME"] == str(tmp_path / ".codex")
    assert values["OPENAI_API_KEY"] == "relay-capability"
    assert values["WISEMAN_MCP_TOKEN"] == env["WISEMAN_MCP_TOKEN"]
    assert values["WISEMAN_THREAD_ID"] == "123"


@pytest.mark.asyncio
async def test_sdk_client_uses_launcher_and_only_explicit_capabilities(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setenv("WISEMAN_RUNNER_API_TOKEN", "runner-secret")
    monkeypatch.setenv("PHOENIX_API_KEY", "phoenix-secret")
    monkeypatch.setenv("WISEMAN_PROVIDER_TOKEN", "relay-capability")
    monkeypatch.setenv("WISEMAN_RELAY_URL", "http://gateway/v1")
    monkeypatch.delenv("WISEMAN_CODEX_BIN", raising=False)
    monkeypatch.setattr("runner.api.AsyncCodex", captured.append)
    CodexRunner()._client(Turn(thread_id="123", user_id="alice", input="hi"), tmp_path, "alice")
    config = captured[0]
    assert config.codex_bin.endswith("codex-as-user")
    assert "features.multi_agent=false" in config.config_overrides
    assert "WISEMAN_RUNNER_API_TOKEN" not in config.env
    assert "PHOENIX_API_KEY" not in config.env
    assert config.env["OPENAI_API_KEY"] == config.env["WISEMAN_MCP_TOKEN"] == "relay-capability"
    assert config.env["WISEMAN_EXEC_USER"] == "alice"
