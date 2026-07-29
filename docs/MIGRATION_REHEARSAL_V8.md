# v8 migration rehearsal

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v8_migration_rehearsal.py \
  --output outputs/release_candidate_v8_migration_rehearsal_<UTC>/migration_rehearsal.json
```

The command creates disposable copied-state fixtures labelled v5, v6, and v7.
Each contains all 11 required store classes: project metadata, global registry,
users, sessions, conversations, repository grants, worktrees, personal corpus,
artifacts, research jobs, and audit/provenance.

For each generation the rehearsal validates identity, permissions, manifest
hashes, schema and store inventory; performs a non-mutating dry run; creates a
backup; migrates; reopens state in a fresh Python process; verifies records;
rolls back and compares exact source hashes; reapplies; and proves the current
state is idempotent. A separate injected interruption proves automatic rollback
and restart-safe recovery.

Fixtures contain only synthetic identifiers and records. Production state is
never an argument. No irreversible transformation is authorized; the current
schema metadata changes are reversible through the identity-bound backup.

