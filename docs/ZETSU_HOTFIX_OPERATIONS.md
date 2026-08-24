# Zetsu lifecycle hotfix operations

This hotfix adds automatic worktree-database schema migration, stale-executor
reconciliation, durable paged task results, process-tree runtime ownership, a
bounded FIFO agent scheduler, and an optional CodeV topology. It does not change
repository grants or source repositories.

Operational commands:

```console
laplace zetsu worktrees --json
laplace zetsu gc --dry-run --json
laplace zetsu gc --json
laplace zetsu start                 # CodeV + Qwen + Operator
laplace zetsu start --nocodev       # Qwen + Operator
laplace zetsu stop
```

The collector releases only terminal records with a matching ownership proof,
registered Git worktree identity, repository identity, and intact durable result
artifacts. Dry-run and collection are bounded. Normal, foreign, ambiguous, and
uncaptured dirty worktrees are reported as protected.

Agent admission now occurs before worktree allocation. The persistent scheduler
has separate execution-slot, live/resumable-worktree, and pending-queue bounds.
Queued tasks allocate no worktree and invoke no model. They wake by condition
notification or an exact deadline, support owner-scoped cancellation, and are
FIFO. Orphaned queue payloads are cancelled on restart rather than replayed.
`agent_task_status` and `cancel_agent_task` expose the owner-authorized lifecycle;
Zetsu status exposes topology, running/queued counts, and both limits.

The deployable 8 GB baseline is one Qwen agent execution slot for both `full` and
`nocodev`, with a pending queue capacity of 16 and the existing per-user live
worktree safety limit of 8. The available certification host was an RTX A6000
(49,140 MiB), not the target 8 GB GPU. The probe used the protected production
model checkout read-only while keeping all runtime state and logs isolated. Both
topologies completed direct Qwen concurrency levels 1 through 4 without an OOM
or abnormal generation stop. Full-runtime samples used 44,940--44,943 MiB;
`nocodev` samples used 38,249--38,255 MiB. Each owned runtime stopped with zero
survivors. These 49 GB measurements are not transferable to the target laptop,
so no higher `nocodev` production capacity is claimed without matching 8 GB
evidence.

The runtime record schema changes from supervisor PID records to a boot-bound,
unguessable runtime identity inherited by supervisors and workers. A dormant
legacy record is replaced automatically on the next start. A legacy record with
a responding endpoint is rejected without signalling any process; stop the old
runtime before deploying this hotfix and verify its model endpoints are down.

Minimal reload:

```console
laplace zetsu stop
# deploy/install the hotfix commit in the production checkout
laplace zetsu start                 # or: laplace zetsu start --nocodev
laplace zetsu status --json
laplace zetsu gc --dry-run --json
# Review the bounded report, then explicitly run: laplace zetsu gc --json
```

If the pre-hotfix stop leaves a model endpoint alive, use the existing production
service owner (or a controlled host restart) to stop it. The hotfix intentionally
will not adopt or kill a process that lacks its new runtime identity.
