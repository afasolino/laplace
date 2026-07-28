# Agent worktrees

## Authorization model

Account activation and the `agent` capability do not grant repository access. An administrator first registers a Git root under a logical repository ID, then grants that ID and an exact base commit to a stable user ID in both the repository authorization database and registered-user registry. The client never submits a filesystem path.

After a grant changes, the user's sessions are revoked. The user signs in again and confirms that the logical repository appears in **Agent → Authorized repository**. If none is available, Laplace shows:

> No repository is authorized for this account.
> Ask an administrator to register and grant one.

## User lifecycle

Selecting **Start isolated agent** creates a detached worktree below `<state_root>/tiered_serving/worktrees/`. The record persists in SQLite and includes session/user/logical repository IDs, base commit, grant revision, timestamps and expiry, task title/instruction digest, state, lane/model display name, tool policy, command count, changed logical paths, diff hash, verification summary, export state, and high-level events.

Users can create multiple worktrees within the default quota of eight per user and 64 globally. They can list/inspect their own records, start or resume work, cancel, close a clean worktree, preserve a dirty/failed worktree, request export or promotion review, download a patch, inspect history, and discard with the exact `discard:<session_id>` confirmation. Worktree creation is idempotent across safe client retries.

Clean close uses `git worktree remove`. Dirty or failed close never silently destroys changes. It records a diff hash and retains the directory for the default 30-day policy. Export requests do not merge, push, or promote automatically. Operator force cleanup is explicit and audited.

## Isolation

The server revalidates repository device/inode identity, grant revision, Git root, and base commit. Paths are checked against traversal, absolute paths, symlinks, hardlinks, mounts, nested repositories, sibling worktrees, and submodules. The fixed tool set is `read_file`, `apply_patch`, and `run_validation`; environment variables are allowlisted. The declared worktree network policy is denied and no network tool is exposed to the Agent.

Personal-corpus retrieval, when selected, is supplied as read-only model context with logical citations. It is not mounted in the worktree and copying it into repository files is denied by policy. Canonical repository/worktree paths appear only in Operator inspection responses.

## States and recovery

Typical states are `ACTIVE`, `RUNNING`, `DIRTY`, `FAILED`, `CANCELLED_DIRTY`, `STALE_GRANT`, `CLOSED_CLEAN`, and `DISCARDED`. On restart Laplace reloads persistent records, revalidates live directories and grants, and marks unavailable/stale records instead of rebinding silently. A stale grant requires a fresh authorized worktree.

If quota is reached, release a clean worktree or explicitly export and discard a retained one. Do not remove worktree directories manually: that bypasses Git metadata and lifecycle audit. Operators use the sanitized inventory first and canonical paths only when their independent Operator capability permits inspection.

## Backup and shutdown

Back up the worktree registry and retained directories only when recovery of unexported dirty work is required. Use a consistent snapshot and encrypted storage; worktrees can contain repository secrets. A normal code backup should instead preserve exported patches and lifecycle records.

During shutdown, stop accepting Agent work, stop the Operator service, and leave dirty worktrees in place. Do not run broad `git worktree prune`, `rm`, or process-kill commands. Model servers are separate and may be stopped only after their Laplace ownership records match live processes.
