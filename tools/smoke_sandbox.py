# Copyright (c) 2026 Nick van der Merwe
import asyncio
import json
import os
import subprocess

from runner.api import CODEX_RUNTIME_OVERRIDES, CodexRunner, Turn, Workspace

PROBE = """
import os, subprocess
from pathlib import Path
assert os.getuid() != 0
secrets = {'WISEMAN_RUNNER_API_TOKEN', 'DISCORD_BOT_TOKEN', 'PHOENIX_API_KEY', 'OPENROUTER_API_KEY'}
assert not secrets & os.environ.keys()
assert subprocess.check_output(['sudo', '-n', 'id', '-u'], text=True).strip() == '0'
Path('files').mkdir(exist_ok=True)
Path('files/permission-check.txt').write_text('ok')
print('owner shell, workspace write, sudo and environment isolation passed')
"""


async def main() -> None:
    workspaces = Workspace("/workspaces")
    path = workspaces.thread("smoke-owner", "smoke-thread")
    turn = Turn(user_id="smoke-owner", thread_id="smoke-thread", input="")
    runner = CodexRunner()
    probe_env = {
        **os.environ,
        "HOME": str(path),
        "CODEX_HOME": str(path / ".codex"),
        "WISEMAN_EXEC_USER": workspaces.username(turn.user_id),
        "WISEMAN_CODEX_REAL_BIN": "/app/.venv/bin/python",
    }
    probe_env.update(
        dict.fromkeys(
            ("DISCORD_BOT_TOKEN", "OPENROUTER_API_KEY", "PHOENIX_API_KEY", "WISEMAN_RUNNER_API_TOKEN"),
            "fixture-not-a-secret",
        )
    )
    await asyncio.to_thread(
        subprocess.run, ["/usr/local/bin/codex-as-user", "-c", PROBE], env=probe_env, cwd=path, check=True, timeout=15
    )
    assert (path / "files/permission-check.txt").read_text() == "ok"
    assert not (path / ".codex/auth.json").exists()
    del probe_env["WISEMAN_CODEX_REAL_BIN"]
    arguments = [item for value in CODEX_RUNTIME_OVERRIDES for item in ("--config", value)]
    feature_list = await asyncio.to_thread(
        subprocess.run,
        ["/usr/local/bin/codex-as-user", *arguments, "features", "list"],
        env=probe_env,
        cwd=path,
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    )
    features = {columns[0]: columns[-1] for line in feature_list.stdout.splitlines() if (columns := line.split())}
    assert features["multi_agent"] == features["multi_agent_v2"] == "false"
    assert features["shell_tool"] == "true"
    async with asyncio.timeout(45):
        started = await runner.start(turn, path, workspaces.username(turn.user_id))
        assert started["thread_id"]
        await runner.close(turn.thread_id)
    print(  # noqa: T201
        json.dumps(
            {
                "workspace": "passed",
                "sdk_startup": "passed",
                "delegation": "disabled",
                "provider_inference": "not exercised",
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
