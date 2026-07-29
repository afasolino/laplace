# v8 release-candidate runbook

1. Verify branch `feature/release-candidate-review-v8`, certified-base ancestry,
   clean status, and unchanged clean stable checkout.
2. Run migration, CI, package, operational, desktop-sync, documentation,
   security, offline-evaluation, failure, and CPU-soak gates.
3. Commit only demonstrated fixes and rerun affected gates.
4. Push only the dedicated v8 branch if remote matrix execution is required.
5. Decide whether the guarded live gate is eligible.
6. Run the final certifier and verify `certification.tar.gz` against
   `manifest.json`.

Final CPU/fixture command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_release_candidate_v8_certification.py \
  --output-root outputs
```

Never merge, tag, publish, modify production state, or modify the stable checkout
as part of this runbook.

