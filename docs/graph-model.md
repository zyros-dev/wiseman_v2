# Wiseman V2 Graph Model

GraphWalker owns traversal. The Python model in `tests/graphwalker/model.py`
supplies the native GraphWalker document and owns all graph definitions:
vertices, edge names, edge topology, edge records, and timeouts. `vertices.py`
supplies observable state conditions, `edges.py` supplies one action function
for every graph edge, and `graph_utils.py` owns runtime model state and harness
contracts. The HTTP harness executes those
actions through `/v1/replay/discord`; it does not call `Engine` or mutate
Temporal state directly.

```mermaid
flowchart TD
    I["Idle"] -->|admit-question| P["Preparing"]
    I -->|background-chatter, duplicate-question, idle-stop| I
    P -->|context-ready| R["Running"]
    P -->|preparation-failed| E["Delivering error"]
    P -->|stop-preparing| X["Cancelling"]
    R -->|background chatter, queued question, progress, steering| R
    R -->|worker restart; reattach| R
    R -->|inference-complete| D["Delivering"]
    R -->|transient-failure| C["Recovering"]
    R -->|permanent-failure| E
    R -->|execution-uncertain| U["Outcome unknown"]
    C -->|resume-session| R
    C -->|retry-exhausted| E
    C -->|stop-recovering| X
    R -->|stop-running| X
    D -->|delivery-retry or worker restart| D
    D -->|stop-delivering| X
    X -->|duplicate-stop or worker restart| X
    X -->|stop-confirmed| I
    X -->|completion-race| D
    X -->|cancellation-unknown| U
    U -->|outcome-established; report without blind retry| E
    D -->|answer-finalized| I
    E -->|worker restart| E
    E -->|error-finalized| I
    I -->|three idle days; cleanup succeeds| T["Retired"]
    T -.->|fixture-reset: fresh namespace| I
```

## Model data and invariants

- `pending_questions` is ordered and deduplicated by Discord message ID.
- `background_context` records ordinary thread discussion without advancing
  the consumed-context cursor. `context-ready` consumes it for the next
  admitted question, which covers Q1, background discussion, Q2, and Q3.
- `active_question` remains in `pending_questions` until a terminal answer,
  stopped answer, or error is finalized.
- Steering IDs and stop command IDs are tracked independently. A duplicate or
  delayed stop cannot target a later active question.
- `outcome-unknown` has no retry edge: the harness must establish the runner
  receipt before the model can report an error and continue.
- Retiring requires an idle conversation with no active or queued work.
