# Qwen3.8 production migration

## Current certification state (2026-08-17)

Qwen3.8 is **not promoted**. The requested upstream identifier
`Qwen/Qwen3.8-27B` and a revision-pinned official quantized artifact were not
available from the official sources checked on this host. Consequently no
artifact was downloaded, no hash or tokenizer revision can truthfully be
recorded, and model load, inference, 32K context, VRAM, and MTP gates have not
run. The active selector remains the certified Qwen3.6 rollback selector.

The machine-readable record is
`configs/model_manifests/qwen38_27b_a6000.json`. Null provenance fields and
`promotion_allowed: false` are intentional fail-closed state, not placeholders
that may be bypassed.

## Prepared candidate

- Base request: `Qwen/Qwen3.8-27B`; revision unavailable.
- Quantization candidate: compressed-tensors W4A16, group size 128, symmetric,
  with a Marlin-compatible Ampere kernel. This is an intended configuration,
  not a certified artifact.
- Standard candidate: `P6_qwen38_w4a16`, 32K context, loopback port 8206,
  `qwen3` reasoning parser, and `qwen3_coder` tool parser.
- MTP candidate: `P7_qwen38_w4a16_mtp`, identical except native MTP with three
  speculative tokens on loopback port 8207.
- Installed CLI compatibility passed for both profiles against vLLM
  `0.25.0+cu129`; this checked flags only and did not load a model.

The target runtime observed on the A6000 host is vLLM `0.25.0+cu129`, PyTorch
`2.11.0+cu129`, Transformers `5.14.1`, compressed-tensors `0.17.0`, CUDA runtime
12.9, and NVIDIA driver 550.135. The GPU reported 49,140 MiB total. These are
runtime observations, not Qwen3.8 performance measurements.

## Certification and promotion

After an official artifact exists, record exact base, artifact, and tokenizer
revisions plus a reproducible whole-artifact SHA-256. Resolve the profile:

```bash
PYTHONPATH=src .venv/bin/python scripts/resolve_serving_profiles.py \
  --profile-root configs/serving_profile_candidates \
  --output /var/lib/laplace/certification/qwen38-resolved.json
```

Certify P6 first using the existing live-GPU gates: exact model identity,
quantized load, inference, streaming, reasoning, structured tool calls,
multi-turn behavior, cancellation, 32K context, measured VRAM/headroom,
Operator Ask/Search/Write/Research, persistent conversations, owner isolation,
Python, and CodeV RTL routing. Then repeat every inference gate for P7. Keep MTP
only if every gate and headroom check still passes.

Only after the manifest says `certification_status: PASSED`, has all provenance
fields, and sets `promotion_allowed: true` may an operator run:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_production_model.py qwen38
sudo systemctl restart laplace-model-servers laplace-operator
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

The selector refuses Qwen3.8 while any required evidence is absent.
It selects P7 only when `mtp.status` is exactly `PASSED`; otherwise it selects
the certified non-MTP P6 profile, so an MTP blocker never blocks the migration.

## Routing and rollback

After certification, quality and standard both route to Qwen3.8; economy remains
CodeV. CodeV receives only bounded policy-eligible RTL implementation or repair
and deterministic verification returns its result to the Qwen supervisor.
Unresolved architecture never routes to CodeV.

Rollback requires configuration selection and service restart only:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_production_model.py qwen36
sudo systemctl restart laplace-model-servers laplace-operator
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

This restores the recorded Qwen3.6 quality/standard routes and retains the same
CodeV economy route without source-code reversion.
