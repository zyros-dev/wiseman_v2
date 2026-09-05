# Validation Record

## Local gate

The compact suite covers the normalized Discord admission path, startup and
follow-up context selection, Engine delivery and reactions, provider SSE
forwarding, image/file/profile tools, the Temporal test server, duplicate
messages, session retention, and retirement. The current run is 5 tests with
55.93% application coverage and a checked-in 55% floor.

Ruff check and format, strict Pyrefly, Vulture, compileall, `uv lock --check`,
and `git diff --check` pass. The counted source budget is `2,996/3,000`; no
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

## Live evidence

On 2026-09-05, deployed revision `cd21c16` created Discord thread `Gurt 7`
(`1545705046547898388`) from a real parent-channel mention. It produced one
startup banner and one edited `INITIAL_OK` answer, then one edited
`FOLLOWUP_OK` answer for a real in-thread mention without repeating the
startup banner. A plain non-ping remained silent.

## Remaining acceptance evidence

Live Discord credentials and the deployment target are still required for the
real initial/follow-up conversation, multi-question background context,
steering, `/stop`, concurrency/restart recovery, image/file exchange, and
agent-installed Linux build workload. Completion also requires recording the
correlated Discord, Temporal, Phoenix, gateway, runner, and provider IDs.
Synthetic HTTP, mock clients, and container builds do not substitute for that
evidence.
