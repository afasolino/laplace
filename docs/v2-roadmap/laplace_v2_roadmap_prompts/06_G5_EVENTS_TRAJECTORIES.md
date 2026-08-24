# G5 — Event Store, Trajectories and Provenance

Make append-only trajectory history authoritative; checkpoints become derived snapshots.

## Execute

1. Inspect current OpenHands software-agent SDK and mini-SWE-agent event/trajectory designs; use Open Science provenance concepts only where concretely useful. Record commits/licenses.
2. Define typed append-only events for task start/resume/cancel, retrieval/memory access, model calls, actions/tools, edits, verification, compaction, checkpoint, failure and completion.
3. Carry owner/project/session identity and sufficient provenance for deterministic reconstruction.
4. Keep checkpoints as acceleration snapshots; replayable events remain authoritative.
5. Define versioning/migration and duplicate/idempotency behavior before enabling writes.

## Gate

Inject crashes during append and between event/checkpoint, partial/corrupt/duplicate events, corrupt checkpoint with intact trajectory, restart/resume, cancellation and cross-owner access.

Require deterministic replay to the same exact task state or explicit fail-closed recovery. Certify and commit before G6.
