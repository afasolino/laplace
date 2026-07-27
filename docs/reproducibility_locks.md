# Reproducibility locks

Every measured run writes:

```text
skills.lock.json
context.lock.json
corpus.lock.json
models.lock.json
tools.lock.json
run.lock.json
```

The run lock hashes every subordinate lock plus the base revision, experiment
configuration, request, and held-out evaluator identity. Evaluator identity is
read only from its manifest; held-out test content is excluded.

Canonical JSON uses sorted keys, compact separators for hashing, UTF-8, and
normalized line endings. `run_lock_sha256` is copied into the terminal result.
Research locks are separate and cannot update a measured run lock. Promotion
creates a new corpus snapshot for future selections; an already locked run
continues to reference its original snapshot.

