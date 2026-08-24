# Qwen3.8 production migration

## Current state

Qwen3.8 P7 is production-active for Quality and Standard with native MTP at the
certified three-token workpoint. Economy remains on CodeV, and the previously
certified Qwen3.6 configuration remains the immediate rollback. Promotion followed
the mandatory P6 131,072-token profile and co-resident production gates, followed
by independent P7 profile and co-resident gates.

Machine-readable migration state is in
`configs/model_manifests/qwen38_27b_a6000.json`.

The measured 2026-08-24 production workpoint uses Qwen GPU utilization `0.77`
and CodeV GPU utilization `0.134` with an 8,192-token CodeV context. Its
co-resident gate recorded 45,109 MiB peak allocation and 4,031 MiB minimum free
headroom without weakening the 2,048 MiB admission floor. The retained P7 evidence
is under `outputs/qwen38_certification/20260824-primary-e6b4/`.

## Artifact selection and provenance

Production preparation is pre-quantized-checkpoint first:

1. `barrydeen/Qwen3.8-27B-AWQ-4bit`;
2. `soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ` only after a demonstrated
   checkpoint-specific incompatibility of the primary;
3. another demonstrably better compatible published pre-quantized checkpoint
   only with recorded justification;
4. repository-local W4A16 quantization only as the final fallback.

Use `scripts/prepare_qwen38_prequantized.py`. It resolves the requested
Hugging Face reference to an immutable revision before download and records that
revision and artifact hashes in the local provenance. Production selection must
never depend on a floating `main` revision.

Checkpoint validation is based on configuration and tensor metadata. Native MTP
may be stored inside ordinary `model-*.safetensors` shards; an `mtp*` filename is
not required. A candidate claiming preserved MTP must expose the corresponding
configuration and actual MTP tensor keys, including nested `.mtp.` module segments.

The official Qwen3.8 architecture advertises a larger native context. Laplace's
initial production target for this migration is 131,072 tokens. P6/P7 candidate
profiles and certification use that configured context rather than a fixed 32K
assumption.

The primary published checkpoint is mixed precision by design: packed 4-bit text
decoder `Linear` modules are quantized, while protected vision, Gated DeltaNet, and
MTP components remain BF16. Provenance must preserve that distinction.

## Serving environment

Try the existing isolated serving environment first. Do not broadly upgrade
Laplace dependencies to match a checkpoint author's environment. If live evidence
shows a real serving incompatibility, change only the serving environment and use
the smallest justified version change; record it with the certification output.

The RTX A6000 remains the target device for these profiles. Artifact preparation
and static validation are not substitutes for real-weight serving, measured
context, co-residency, or release checks.

## Certification and promotion

P6 is the mandatory non-MTP profile. P7 adds native Qwen3.8 MTP with the measured
three-token co-resident workpoint and is optional for promotion. Both are loopback-only.
The primary checkpoint serving recipe uses the `qwen3` reasoning parser and
`qwen3_xml` tool-call parser; the checked-in P6/P7 candidates use the same parser
pair and must be validated against the installed serving environment.
Run each gate into a fresh output directory and retain its exact repository,
artifact, profile, command, and runtime provenance.

A typical sequence after resolving and preparing the immutable checkpoint is:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="outputs/qwen38_certification/${stamp}"

PYTHONPATH=src .venv/bin/python scripts/prepare_qwen38_prequantized.py

PYTHONPATH=src .venv/bin/python scripts/certify_qwen38_profile.py \
  P6_qwen38_w4a16 --output-root "${root}/p6"

PYTHONPATH=src .venv/bin/python scripts/certify_qwen38_production.py \
  P6_qwen38_w4a16 \
  --profile-certification "${root}/p6/certification.json" \
  --output-root "${root}/p6-production"
```

The profile gate must exercise the exact served model identity, normal/SSE
inference, reasoning, structured tool calls, multi-turn behavior, cancellation,
the configured 131,072-token context gate, runtime stability, packed-kernel
evidence, GPU sampling, and owned-process release. The production gate must also
exercise Quality/Standard routing and co-resident CodeV RTL routing, enforce
residual headroom, and release only owned processes.

If P6 passes, Qwen3.8 may be promoted without MTP. P7 is run independently with
its measured MTP configuration. A failed or unavailable P7
must leave MTP disabled while preserving a valid P6 promotion.

Promotion remains fail-closed through `scripts/finalize_qwen38_certification.py`:

```bash
PYTHONPATH=src .venv/bin/python scripts/finalize_qwen38_certification.py \
  --p6-certification "${root}/p6/certification.json" \
  --p6-production-gate "${root}/p6-production/production_gate.json" \
  --promote
```

Add the P7 certification and production-gate arguments only when both P7 gates
passed. Do not mark a gate passed from architecture probes, static tests, or model
self-report.

## Routing and Zetsu

After successful Qwen3.8 promotion:

- Quality → Qwen3.8;
- Standard → Qwen3.8;
- Economy → CodeV for policy-eligible bounded RTL/SystemVerilog work.

Qwen3.8 is also the Quality/Standard model used by Zetsu `delegate` and
`agent_task`. CodeV remains on the dedicated `rtl_task` route and is not replaced
by the generic Qwen repository agent.

Long Qwen agent tasks use persistent checkpoints and rolling semantic compaction
near 80% of the active context. Exact task/repository/validation state remains
outside the generated summary; failed compaction fails closed. See
[ZETSU.md](ZETSU.md).

## Rollback

Qwen3.6 is retained as the immediate configuration rollback. After any future
Qwen3.8 promotion:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_production_model.py qwen36
sudo systemctl restart laplace-model-servers laplace-operator
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

No source reversion or artifact deletion is required.
