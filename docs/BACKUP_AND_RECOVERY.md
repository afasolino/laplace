# Backup and recovery

The v7 backup boundary passes an external key reference to an encryption provider; it
never accepts or stores encryption key material. Export requires the exact
`EXPORT_FIXTURE_BACKUP` confirmation. A provider must return a receipt and validate
the encrypted destination before Laplace emits a manifest.

Each strict manifest records a schema version, backup identity, provider identity, key
reference, safe logical paths, byte counts, SHA-256 digests and the provenance chain.
Logical paths must be relative and cannot contain `..`. Account email addresses and
server filesystem paths are not included.

Restore is staged into an empty operator-selected directory. Decryption is owned by
the configured backup provider. Before import, validate every restored entry:

```bash
PYTHONPATH=src python scripts/verify_backup.py \
  --manifest /tmp/backup-manifest.json \
  --restored-root /tmp/restored-fixture
```

Validation rejects missing, duplicate, traversing, size-mismatched and hash-mismatched
entries. Only after validation should an operator run a state migration, integrity
checks and an application-level ownership/provenance audit. An interrupted export
must not replace the prior successful backup; an interrupted restore directory is
discarded and recreated. Key rotation and off-machine custody remain deployment
operator responsibilities.

