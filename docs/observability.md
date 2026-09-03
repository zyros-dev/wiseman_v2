# Observability Contract

Each inbound message receives one trace key, `discord-<message-id>`, and one
Phoenix root span with child node spans. Phoenix events are appended during
execution, not only at the end:

```text
turn -> reaction(add:👀) -> context -> grammar -> progress -> codex -> delivery -> reaction(add:✅|❌, remove:👀)
```

The first `admission` child is a replay artifact. It stores the exact raw
request accepted by the gateway, the normalized event passed to the shared
admission function, and the `normalize_event:v2` revision. The artifact is
available in Phoenix span attributes as `wiseman.raw_request` and through the
authenticated `GET /v1/phoenix/audits/<trace-id>` endpoint. In a non-production
environment, `POST /v1/replay/phoenix/<trace-id>` feeds that raw request back
through the same normalizer and Temporal submission path; it does not invent a
second test-only request format.

The `context` event contains raw Discord JSON, normalized messages, selected
IDs, authors, mentions, reply ancestry, and attachment metadata. The
`grammar` event contains the grammar name, content hash, source, raw input,
normalized input, and rendered output. The `codex` event contains the final
input and any provider model, usage, and cost returned by the runner. Values
that are unavailable are omitted rather than reported as zero.

When Codex invokes `wiseman-image`, the gateway records a `vision_tool` event
with bounded attachment IDs, the configured vision model, question,
description, usage, and cost; image bytes and provider credentials are never
recorded.

`GET /v1/phoenix/events` is a local inspection endpoint. Setting
`PHOENIX_OTLP_ENDPOINT` forwards each event to the configured Phoenix ingress;
telemetry failure is swallowed so it cannot turn a successful Discord answer
into a failure. Prompt sources are fetched from Phoenix's versioned
`/v1/prompts/<name>/latest` endpoint and their source hash is recorded with
the grammar event.

The provider span also contains the redacted relay request plus requested and
served model, usage, and cost. Nested `response.completed` payloads are
normalized so billing is not lost because the provider wraps metadata.

Midgard receives only low-cardinality operational metrics through its existing
Telegraf-to-TimescaleDB path. Wiseman does not maintain a second metrics
database or put message IDs, prompt content, raw requests, or secrets in
metrics labels. `GET /metrics` exposes turn and failure counters for Telegraf;
Phoenix remains the source for the detailed semantic lifecycle.

Temporal exposes the durable startup boundaries separately: `wiseman.workspace`
materializes the user/thread workspace, `wiseman.codex_start` creates or
resumes the SDK thread, and `wiseman.turn` performs context construction, the
model turn, delivery, and terminal reaction. Follow-up turns skip the first
two Activities and execute only `wiseman.turn`. Activity schedule, start, and
completion timestamps provide the startup-versus-model latency breakdown.
