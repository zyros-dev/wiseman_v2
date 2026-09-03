# Copyright (c) 2026 Nick van der Merwe
"""Wiseman application entrypoint."""

from __future__ import annotations

import os

from app.engine import Engine
from app.http_api import create_app
from app.phoenix import Phoenix
from app.runner import FakeRunner, HttpRunner, Runner

phoenix = Phoenix(
    os.getenv("PHOENIX_OTLP_ENDPOINT", ""),
    os.getenv("PHOENIX_API_KEY", ""),
    os.getenv("PHOENIX_PROJECT", "wiseman-v2"),
    os.getenv("WISEMAN_AUDIT_DIR"),
)
runner: Runner = (
    HttpRunner(os.environ["WISEMAN_RUNNER_URL"], os.getenv("WISEMAN_RUNNER_API_TOKEN", ""))
    if os.getenv("WISEMAN_RUNNER_URL")
    else FakeRunner()
)
engine = Engine(phoenix, runner)
app = create_app(engine, os.getenv("WISEMAN_REPLAY_TOKEN", ""), os.getenv("DISCORD_BOT_TOKEN", ""))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
