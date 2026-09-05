# Validation Record

## Local gate

The compact suite covers the normalized Discord admission path, startup and
follow-up context selection, Engine delivery and reactions, provider SSE
forwarding, image/file/profile tools, the Temporal test server, duplicate
messages, session retention, and retirement. The current run is 5 tests with
56.18% application coverage and a checked-in 55% floor.

Ruff check and format, strict Pyrefly, Vulture, compileall, `uv lock --check`,
and `git diff --check` pass. The counted source budget is `2,997/3,000`; no
production file exceeds 500 counted lines and application code contains no
`Any` annotations.

GraphWalker CLI 4.3.3 is checksum-pinned in CI. Two seeded native traversals
run on Thor and are replayed through the authenticated HTTP admission boundary;
both paths passed and covered every modeled edge. The model includes queued
questions, background context, steering, `/stop`, retries, restart recovery,
delivery failure, unknown cancellation, and three-day retirement states.

Both coordinator and sandbox images build on Thor. The sandbox returns
`{"status":"ok"}` from `/healthz` and exposes its low-cardinality Prometheus
endpoint at `/metrics`.

The deployed sandbox image was exercised on Thor as a managed non-root user:
it installed `build-essential` with passwordless `sudo` and successfully ran
GCC and Make. The toolchain is therefore available to agent workloads without
being baked into the image.

## Live evidence

On 2026-09-05, deployed revision `d0472eb` was verified through the raw
Discord REST API and live Temporal histories. The real parent-channel mention
`1545705046547898388` created thread `1545705046547898388`, produced one
startup banner and `INITIAL_OK` (`1545705063014735903`), then the real
in-thread mention `1545705897458671686` produced `FOLLOWUP_OK`
(`1545705908926029854`) without repeating the startup banner. A plain
non-ping remained silent, and the authenticated HTTP replay path also returned
`ignored` for an ordinary message. The gateway and sandbox pods are both ready
on image tag `d0472ebd1f345ee9557c0716c2d44e665b63eea9`, with zero restarts.

## Remaining acceptance evidence

Live Discord credentials and the deployment target are still required for the
real initial/follow-up conversation, multi-question background context,
steering, `/stop`, concurrency/restart recovery, image/file exchange, and
agent-installed Linux build workload. Completion also requires recording the
correlated Discord, Temporal, Phoenix, gateway, runner, and provider IDs.
Synthetic HTTP, mock clients, and container builds do not substitute for that
evidence.
