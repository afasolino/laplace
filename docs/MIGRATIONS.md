# State migrations

Laplace v7 inventories project SQLite, global registry, registered users, sessions,
conversations, repository authorization, worktrees, personal corpus, artifact
registry, research/jobs and audit stores. Production state is never auto-discovered.
Every command requires an explicit state root and the `state_id` already recorded in
its manifest.

## Preflight and dry-run

```bash
PYTHONPATH=src python scripts/check_migrations.py \
  --state-root /path/to/copied-fixture-state \
  --state-id fixture-v7-old

laplace-migrate \
  --state-root /path/to/copied-fixture-state \
  --state-id fixture-v7-old \
  --dry-run
```

Preflight rejects incomplete inventories, missing/corrupt/oversized stores, hash
mismatches, unsafe permissions, symlinks, duplicate IDs/paths, traversal and unknown
schema transitions. SQLite uses `PRAGMA quick_check`.

## Apply

Stop writers and create a tested external backup before preparing a deployment
change. First copy and sanitize the state into an isolated test location. Only after
preflight evidence and an approved maintenance window:

```bash
laplace-migrate \
  --state-root /explicit/state/root \
  --state-id fixture-or-deployment-id \
  --apply --yes
```

The engine acquires an exclusive mode-0600 lock, creates an automatic file/hash
backup, writes a restart journal, applies ordered 0→1 migrations, verifies integrity,
atomically replaces the manifest and appends a migration audit event. SQLite changes
use `BEGIN IMMEDIATE`.

## Recovery and rollback

```bash
laplace-migrate \
  --state-root /explicit/state/root \
  --state-id fixture-or-deployment-id \
  --recover

laplace-migrate \
  --state-root /explicit/state/root \
  --state-id fixture-or-deployment-id \
  --rollback-backup-id 20260728T120000000000Z --yes
```

Recovery accepts only the backup recorded in the matching journal. Rollback accepts
only a validated backup ID below the state-specific backup root, rechecks every hash,
restores atomically and runs preflight. No migration tool deletes historical backups.

The v7 implementation and certification run these commands only on synthetic
fixtures created under temporary/output roots. Production state is untouched.

## v8 copied-state rehearsal

The release-candidate rehearsal runs disposable sanitized copies labelled v5,
v6, and v7 across all 11 store classes. It requires dry-run, backup, apply,
fresh-process reopen, record preservation, rollback hash equality, reapply,
idempotency, and interruption recovery. See
[MIGRATION_REHEARSAL_V8.md](MIGRATION_REHEARSAL_V8.md). The harness accepts only
`fixture-*` identities and therefore cannot target production state accidentally.
