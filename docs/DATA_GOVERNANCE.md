# Data governance

Laplace applies one versioned governance vocabulary to conversations, drafts,
attachments, corpora, extracted text, indexes, worktrees, artifacts, audit events,
sessions and backup records. The fixture implementation is intentionally rooted at an
explicit absolute path; the v7 tools never discover or default to production state.

Admission checks the account state, per-user quota, global quota and remaining disk
space before bytes are accepted. Operator summaries contain aggregate counts and byte
totals only. Storage directories use keyed owner namespaces, never email addresses.
Content equality is owner-scoped with a keyed digest, so another user cannot use an
ingestion response as a cross-user hash oracle.

Retention is configured per asset category. `purge_state.py` is dry-run by default.
Execution requires the literal `PURGE_FIXTURE_STATE`; records first become tombstones,
and physical deletion is a separate explicit step. Conversations, drafts, corpora and
artifacts require an export receipt before tombstoning. Tombstones keep the provenance
identifier and export receipt after content deletion.

```bash
PYTHONPATH=src python scripts/storage_report.py --fixture-state /tmp/laplace-fixture
PYTHONPATH=src python scripts/purge_state.py --fixture-state /tmp/laplace-fixture
```

Accounts can be disabled immediately or marked for deletion. Ownership transfer is
restricted by category, requires an enabled destination account, moves bytes within
the governed root, recalculates the owner-scoped digest and preserves provenance.
Audit events store keyed owner namespaces rather than raw account identifiers.

The scripts in this document are fixture-state tools. Production governance requires
an operator-selected deployment configuration and a separately reviewed migration.

