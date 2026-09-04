# Wiseman v2

Wiseman v2 is a compact Discord-facing Codex runtime. Discord admission is
handled by `discord.py`, prompt and turn evidence is recorded in Phoenix, and
Codex runs through the official async SDK in one warm trusted sandbox
container with per-user workspaces.

## Layout

```text
src/app/http_api.py         canonical Discord/raw-event turn path and Phoenix evidence
sandbox/src/runner/api.py   authenticated warm runner and workspace materialization
contracts/                  context schema plus startup/follow-up grammar sources
docs/                       design and observability contracts
.gitea/workflows/           strict CI
```

## Local checks

```sh
uv sync --group dev
uv run pytest tests -q
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run vulture src sandbox/src tests --min-confidence 100
```

For a local container configuration, copy `.env.example` into the deployment
secret manager and provide the required values. Phoenix and Midgard are
operated dependencies; the Compose file does not create duplicate platform
services. `POST /v1/discord/events` accepts a raw Discord-shaped event for
startup and follow-up testing.

The sandbox is intentionally trusted: it permits outbound HTTP/HTTPS, package
installation, and full Codex access inside the container. It does not receive
Discord, Phoenix, OpenRouter, Kubernetes, host-mount, or platform credentials.
