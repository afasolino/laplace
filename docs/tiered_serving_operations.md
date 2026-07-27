# Tiered serving operations

Resolve profiles without launching a model:

```bash
.venv/bin/python scripts/resolve_serving_profiles.py \
  --output outputs/profile_resolution/resolved_profiles.json
```

Run the full CPU/GPU certification:

```bash
.venv/bin/python scripts/run_tiered_serving_experiments.py
```

Run the real mixed-user HTTP/GUI certification while keeping both owned routes alive
and guaranteeing release:

```bash
LAPLACE_VLLM_EXECUTABLE=.venv-vllm-cu129/bin/vllm \
  .venv/bin/python scripts/run_tiered_live_integration.py \
  --certification-root outputs/tiered_serving_<timestamp> \
  --live-api-name live_api_v4
```

Run final deterministic checks and rebuild the certification archive:

```bash
.venv/bin/python scripts/run_tiered_final_checks.py \
  --output-root outputs/tiered_serving_<timestamp>/tests_final
.venv/bin/python scripts/finalize_tiered_serving_certification.py \
  --certification-root outputs/tiered_serving_<timestamp>
```

Use `--smoke-only` only for a diagnostic pass; it is not full certification. The
runner writes a fresh `outputs/tiered_serving_<timestamp>` tree and never rewrites a
historical result.

Monitor one runtime state directory without mutation:

```bash
.venv/bin/python scripts/monitor_tiered_serving_experiments.py \
  --state-root outputs/tiered_serving_<timestamp>/profiles/P0_baseline/runtime
```

The profile runtime starts one process group, stores a mode-0600 ownership record,
checks exact model identity, and releases only the PID whose `/proc` start ticks and
process group still match that record. PID reuse or process-group drift fails closed.
Pre-existing compute PIDs are recorded before launch and compared after release.

Operators manage users and repositories through authenticated, CSRF-protected admin
endpoints. A repository grant should use a stable commit. Revocation invalidates
existing sessions on their next action. Basic and Plus credentials cannot access the
Operator Plane, model lifecycle, global audits, artifacts, or research administration.
The Operator server loads the active lane routes and scheduler limits from
`configs/selected_serving_profiles.json`; changing a profile JSON alone does not
silently change deployment.

If a profile fails:

- `unsupported_profile`: inspect its recorded missing flags; do not translate it to a
  similar option.
- `gpu_admission_blocked`: wait for the unrelated owner to finish; never stop it.
- `profile_process_exited`: inspect the preserved server log.
- `profile_startup_timeout`: release the owned process and retain the exact timeout
  evidence.
- `response_validation_failed`: one lower-lane escalation is allowed; quality failure
  remains a structured failure.
- dirty agent worktree: preserve it and ask the owner to archive or clean it.

Dual-route admission rejects any pre-existing compute PID. Its empirical memory gate
compares the measured workload footprint to free memory and the footprint plus
residual to total device capacity, avoiding double-counting idle driver memory. Safe
release treats EngineCore descendants as part of the recorded launcher process tree
and never signals a process outside that tree.
