# C2 — Correct Queueing, GPU Admission, and Deadline Semantics

## Defects

Temporary resource scarcity must not become an execution failure. Real concurrency must not admit two tasks against the same unreserved VRAM. Batch tasks must receive fair independent deadlines.

## Required scheduler model

Separate:

- queued request;
- execution slot;
- GPU reservation;
- live/resumable worktree;
- durable result.

Required states should map cleanly to the existing schema and distinguish at least queued, running, resumable/interrupted, succeeded, failed, and cancelled.

## Capacity behavior

When all execution slots or safe GPU capacity are temporarily occupied:

- keep the request queued;
- do not allocate a worktree before admission;
- do not consume model tokens while queued;
- wake through condition/event signaling, not model-mediated polling;
- support caller cancellation and bounded queue wait;
- preserve FIFO unless an existing explicit priority rule is already authoritative;
- only return a capacity error for queue-full, deadline expiry, cancellation, or a real non-transient resource/security condition.

A temporary GPU probe failure or insufficient current free memory must not immediately become terminal `GPU_BLOCKED`.

## GPU reservation

For real concurrency:

- admission must atomically account for already-reserved VRAM;
- two tasks may not both consume the same free-memory observation;
- release reservations on every terminal/cancelled/failed startup path;
- reconcile reservations after restart/crash;
- retain a conservative headroom margin;
- keep real concurrency opt-in until certified.

Do not infer exact safe concurrency values. Use existing measured serving-profile limits and bounded stress tests.

## Full vs --nocodev topology

Keep worktree safety quota independent from model topology.

Allow the agent execution-slot policy to differ between:
- full runtime: Qwen + CodeV;
- `--nocodev`: Qwen + Operator.

Only configure a higher `--nocodev` execution capacity if measured stress evidence proves it stable.

## Deadlines

Each admitted task must receive its own execution deadline based on its own admission/start time. Batch position must not silently shorten another task's execution budget.

Queue-wait deadline and execution wall budget must be distinct.

## Tests

Cover:
- saturation queues then automatically runs;
- queued task owns no worktree;
- queued task uses no model calls/tokens before admission;
- cancellation while queued;
- queue timeout;
- queue full;
- FIFO/fairness;
- atomic reservation under simultaneous admission;
- reservation release on success/failure/cancel;
- restart reconciliation;
- full runtime;
- `--nocodev`;
- independent per-task execution deadlines;
- no OOM under certified stress;
- no worktree/resource leaks.

Update status output to expose effective topology, execution slots, running/queued counts, live worktrees, and reservation state without leaking secrets.
