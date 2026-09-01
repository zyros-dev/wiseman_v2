# Validation Record

## Local evidence

- `uv run pytest tests -q --cov=app --cov=runner --cov-fail-under=80`: 25 passed,
  83.72% coverage.
- Ruff, Pyrefly, Vulture, pip-audit, `compileall`, shell syntax, `uv lock
  --check`, and `docker compose config` all pass.
- Python source and tests total 1,932 physical lines.
- A live Uvicorn process accepted raw Discord-shaped HTTP startup and follow-up
  payloads. Startup selected `old2,start2` and returned turn 1; follow-up
  selected only reply ancestor `start2` plus `newp2,newt2,follow2` and returned
  turn 2. Both returned `👀,✅`.
- The Phoenix inspection endpoint showed the ordered trace nodes for both
  requests, including context, grammar, prompt, Codex, delivery, and reactions.

## External validation blocker

The current environment has no Discord, Phoenix, OpenRouter, Temporal, or
runner credentials configured. A three-second `docker info` probe also timed
out because the Docker daemon is unavailable. Therefore live Discord gateway,
Phoenix OTLP export, Temporal worker execution, OpenRouter billing, and the
real Codex container remain unexercised. No production completion claim is made
until those dependencies are available.
