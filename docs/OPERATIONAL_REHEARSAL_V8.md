# v8 operational rehearsal

Run the complete fixture rehearsal on isolated loopback ports and disposable
state:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v8_operational_rehearsal.py \
  --output-dir outputs/release_candidate_v8_operational_rehearsal_<UTC>
```

The rehearsal covers bootstrap and activation, opaque sessions, capabilities,
no-repository users, repository grants, worktree lifecycle, accepted and rejected
personal-folder files, indexing, grounded retrieval and citations, cross-user
denial, Markdown/code rendering, composer/progress behavior, backup/restore,
purge planning and execution, disk pressure, restart semantics, SSH policy,
fixture TLS reverse proxy, and degraded providers.

Screenshots use synthetic documents and accounts. Their manifest records SHA-256,
size, and `fixture_data_only=true`; credentials and private paths are absent.
Every listener binds to an alternate localhost port and is shut down by its
fixture context. No production state, repository, model endpoint, or external
network is used.

