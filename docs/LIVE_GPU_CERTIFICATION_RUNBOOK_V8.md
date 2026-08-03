# v8 live-GPU certification runbook

Live work starts only after all non-GPU gates pass. Run a preflight into a new
output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_live_production_gpu_certification.py \
  --output-root outputs/live_gpu_v8_<UTC> --preflight-only
```

Preflight requires a clean stable checkout, verified existing local model
artifacts, free target ports, valid runtime paths, successful GPU and compute-PID
queries, and `GPU_CLEAR`. It never downloads a model.

For a live run, omit `--preflight-only`. P1 and CodeV run sequentially; one main
generative route is resident at a time. The runner distinguishes `/health`
liveness from exact `/v1/models` readiness and requires real vLLM SSE
time-to-first-token evidence. P1 must complete owner-private retrieval with
file/chunk citations and a Python Agent patch verified by pytest and Ruff. CodeV
must complete chat, strict structured patching, and a SystemVerilog Agent patch
verified by Verilator, Icarus/vvp, and Yosys. The same isolated browser run checks
cross-user corpus isolation, Basic capability denial, queue metadata, bounded
cancellation, provider failure, credential storage, and synthetic identities.
Before and after each model group it reclassifies compute ownership.

A live PASS includes five synthetic screenshots. The final certifier accepts
them only when each path is contained beneath the live screenshot directory and
its SHA-256 matches `run_manifest.json`; it then copies them into the final
archive.

An existing output root is rejected. `--resume` accepts only a terminal
coordination-blocked record with no ownership file. Never resume a partial model
run. On SpecDec appearance, the runner releases only Laplace-owned process groups,
records `YIELDED_TO_SPECDEC`, and does not reacquire the GPU.
