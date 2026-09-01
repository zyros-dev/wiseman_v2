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

## Deployment attempt: 2026-09-02

- Commit `8ce586d` is now the local root revision.
- Gateway and sandbox images both built successfully on Thor with
  `mg-cli dev thor run`; the gateway build includes the pinned Codex CLI
  bundle.
- The committed gateway accepted raw Discord-shaped `MESSAGE_CREATE` payloads
  over HTTP. Startup returned turn 1 with `👀,✅`; follow-up returned turn 2
  with `👀,✅`.
- Phoenix inspection for that run recorded `turn`, `context`, `grammar`,
  `prompt`, `codex`, `delivery`, and `reaction` nodes for both traces. Startup
  selected `parent-old,raw-start`; follow-up selected only
  `raw-start,parent-new,raw-follow`.
- Deployment did not proceed. The new Gitea repository cannot be created with
  the managed token because it lacks `write:user`; Docker Desktop is also
  unavailable locally. The old `hermes-discord-gateway` and
  `hermes-discord-runner` deployments remain healthy and were not stopped.

## Cutover preparation: 2026-09-02

- Release wiring is committed at `4a2710e` and the registry memory correction
  at `8ce586d`.
- Gateway and sandbox images built from `4a2710e` on Thor and were pushed to
  `registry.odin.home:5000` as tag `4a2710e`.
- Midgard app-registry trusted-sandbox support is implemented on local branch
  `42d033d`; its app-registry target and inline tests pass in the Midgard dev
  container. Generated V2 manifests validate with root UID/GID, writable root
  filesystem, and privilege escalation enabled only for the trusted sandbox.
- The V2 GitOps workflow is ready but cannot run until `zel/wiseman_v2` exists
  in Gitea and the dedicated Vault paths `k8s/wiseman-v2/{gateway}` are
  provisioned. The managed Tea token currently lacks `write:organization`, and
  Gitea has push-to-create disabled.
- The old Hermes deployments remain running because V2 has not reached a
  verified Discord/Phoenix acceptance state. No cutover claim is made.

- A fresh Uvicorn process at the exact release revision accepted a raw
  `MESSAGE_CREATE` envelope and a follow-up over HTTP. Startup selected 100
  parent messages plus the trigger; follow-up selected only the reply
  ancestor, two new messages, and the follow-up trigger. The follow-up did not
  replay the older parent window. Both returned turn 1/2 and exactly
  `👀,✅`.
- Phoenix inspection for that process showed, for each trace, `turn`,
  `reaction`, `context`, `grammar`, `prompt`, three progress nodes, `codex`,
  `delivery`, and terminal `reaction`.
