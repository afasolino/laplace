# Desktop repository synchronization

v7 provides a complete CPU-tested protocol and reference desktop client/server
transport. External SSH/HTTPS integration remains a staged rollout; no external host
is contacted by v7 certification.

## Safety model

The user selects a Git top-level and a logical Laplace repository ID. The inspector
shows branch, HEAD, dirty state, sanitized remotes and proposed tracked-file changes.
It rejects arbitrary folders, a nested selection, symlinks, hardlinks, mounts,
submodules, nested repositories, rename/copy patches, binary patches, traversal and
oversized change sets.

Untracked files appear in the snapshot with `included=false` and are never silently
added. Add and commit them explicitly with normal Git tools if they should become
tracked. Laplace never force-pushes or stores Git credentials.

Every upload plan is bound to the base HEAD and patch SHA-256. Transfer requires the
exact `confirm:<plan_id>` string, uses resumable offsets, detects replays/conflicting
base revisions and records owner-scoped operation/audit events in SQLite. Ordinary
records contain only the logical repository ID; canonical server paths stay inside
the reference service.

SSH policy requires host-key verification. HTTPS policy requires TLS verification.
Credentials remain with the user's SSH agent or system credential helper and are not
persisted by Laplace.

## Inspect and dry-run

```bash
PYTHONPATH=src python scripts/sync_repository.py \
  --repository /path/to/local/git-top-level \
  --repository-id authorized-logical-id \
  --action inspect

PYTHONPATH=src python scripts/sync_repository.py \
  --repository /path/to/local/git-top-level \
  --repository-id authorized-logical-id \
  --action plan
```

These commands make no network request and change no repository file. The plan
reports the exact confirmation string.

## Patch fallback

Re-run the deterministic plan and supply its confirmation:

```bash
PYTHONPATH=src python scripts/sync_repository.py \
  --repository /path/to/local/git-top-level \
  --repository-id authorized-logical-id \
  --action export \
  --patch-output /new/path/change.patch \
  --confirmation confirm:sync-plan-REPLACE_WITH_PLAN_ID
```

The destination must not exist. Applying an incoming patch is a separate
confirmation-bound operation: base HEAD, patch paths and hash are revalidated, then
`git apply --check` runs before `git apply`. A mismatch is a conflict; no force
operation or automatic merge occurs.

The reference `FixtureSyncService` is used for deterministic CPU, replay, resume,
authorization and cross-user tests. Deploying a real transport requires an approved
server endpoint, authentication integration and a separate live certification.
