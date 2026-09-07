# Validation Record

## Local gate

The compact suite covers the normalized Discord admission path, startup and
follow-up context selection, Engine delivery and reactions, provider SSE
forwarding, image/file/profile tools, the Temporal test server, duplicate
messages, session retention, and retirement. The current run is 3 tests with
60.79% application coverage and a checked-in 55% floor.

## Step 1 test reduction

The standalone admission fuzzer was removed because it duplicated the
admission assertions without exercising the required GraphWalker model. The
retained tests have separate responsibilities: the Temporal boundary test
proves workflow execution and cancellation, the client boundary test proves
tool/provider contracts, and the GraphWalker test proves native path replay
through `/v1/replay/discord`. The retained GraphWalker model is the only
lifecycle traversal harness.

Step 2 evidence: the pinned GraphWalker 4.3.3 CLI generated one edge-coverage
path and sixteen parallel 100-transition seeded walks on lappy2. Each path was
replayed through `/v1/replay/discord`; the three retained tests passed for all
seventeen traversals. Duplicate questions now receive `duplicate`, background
messages and idle stops receive `ignored`, and admitted questions receive
`queued`.

The provider relay boundary is covered by the same client test. Run `3369`
passed the original CRLF SSE bytes, including comment, `id`, and `retry`
fields, through `/v1/responses` while Phoenix recorded the served model and
`transport_complete=true`.

Ruff check and format, Pyrefly, Pyright, Vulture, compileall, `uv lock --check`,
and `git diff --check` pass. The combined counted source budget is `5,607/6,000`;
the separate 1,000-counted-line per-file gate remains enabled, and application
code contains no `Any` annotations.

The initialization stop race and the mock runner's persistent stop flag were
fixed in commit `0f066f3`; the sandbox instruction literal was replaced by
the Jinja contract in `ca17c69`. CI run `3354` for `ca17c69` passed on
`k8s-thor` in 3 minutes, including the edge-coverage path and the seeded
100-transition GraphWalker walks.

GraphWalker CLI 4.3.3 is checksum-pinned in CI. Sixteen seeded native
traversals run in parallel on lappy2 and are replayed through the authenticated
HTTP admission boundary; all paths passed. The separate edge-coverage path
covered every modeled edge. The model includes queued
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

On 2026-09-05, deployed revision `3bc1b84` was verified through the raw
Discord REST API and live Temporal histories. The real parent-channel mention
`1545705046547898388` created thread `1545705046547898388`, produced one
startup banner and `INITIAL_OK` (`1545705063014735903`), then the real
in-thread mention `1545705897458671686` produced `FOLLOWUP_OK`
(`1545705908926029854`) without repeating the startup banner. A plain
non-ping remained silent, and the authenticated HTTP replay path also returned
`ignored` for an ordinary message. The gateway and sandbox pods are both ready
on image tag `3bc1b848bbe9f38cb372840b6020c79c7eec4ade`, with zero restarts.

The rollout retained the live Temporal session records for threads
`1545760413390606336` and `1545760674452606996`; both report one completed turn,
an empty active message, and a pinned Codex thread after the restart.

## Remaining acceptance evidence

Live Discord credentials and the deployment target are still required for the
real initial/follow-up conversation, multi-question background context,
steering, `/stop`, concurrency/restart recovery, image/file exchange, and
agent-installed Linux build workload. Completion also requires recording the
correlated Discord, Temporal, Phoenix, gateway, runner, and provider IDs.
Synthetic HTTP, mock clients, and container builds do not substitute for that
evidence.
