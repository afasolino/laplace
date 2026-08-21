# Qwen3.8 production migration

## Current state

The official source and local Ampere artifact are pinned and byte-verified, but
Qwen3.8 is not promoted. The host's external-execution allowance was exhausted
immediately after artifact creation, so the real-weight P6, co-resident CodeV,
and P7 MTP gates have not run. `configs/selected_serving_profiles.json` therefore
still selects the certified Qwen3.6 profile. This is the required fail-closed
state; architecture-probe results are not inference certification.

Machine-readable provenance and blocker evidence are in
`configs/model_manifests/qwen38_27b_a6000.json`.

## Pinned upstream and architecture

- Base and tokenizer: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- License: Apache-2.0.
- Architecture: `Qwen3_5ForConditionalGeneration`, with a `qwen3_5_text`
  language model, 64 text layers (48 gated-delta-net linear-attention and 16
  full-attention layers), and one native MTP layer.
- Tokenizer: `Qwen2Tokenizer`; processor: `Qwen3VLProcessor`.
- Native advertised context: 262,144 tokens. Laplace production remains 32,768
  tokens until measured evidence supports changing it.
- Source: 32 pinned files, 55,562,855,904 weight bytes, source-tree SHA-256
  `05a9101111a68cba02877e56283246cba67578ef2da02d6a604be530f0fbe3e9`.

The official FP8 checkpoint was not selected. RTX A6000 is Ampere SM86 and does
not provide native FP8 tensor-core execution, while the FP8 checkpoint is not a
memory-appropriate substitute for an Ampere W4A16 deployment. No official
Qwen3.8-27B INT4/AWQ/GPTQ checkpoint was available at the pinned inspection
time.

Before the quantization environment was changed, installed vLLM
`0.25.0+cu129` resolved `Qwen3_5ForConditionalGeneration`, initialized its GDN
and attention kernels with dummy weights, and served the exact probe identity.
That architecture-only run observed 40,910 MiB used and 7,761 MiB free after a
34.07 GiB dummy GPU load plus 16.15 GiB CPU offload. It did not test official
weights or inference.

## Local W4A16 artifact

The production candidate is generated locally from the pinned official weights:

- path: `.models/Qwen3.8-27B-W4A16-AWQ`;
- revision: `local-awq:315b9b21b0a085b7e2f4a65f0c5db1f1c950aff037ad777382c8ed9bd1181808`;
- canonical artifact SHA-256:
  `315b9b21b0a085b7e2f4a65f0c5db1f1c950aff037ad777382c8ed9bd1181808`;
- size: 19,473,078,044 bytes across 16 manifested files;
- format: compressed-tensors packed W4A16 AWQ, symmetric group size 128,
  memoryless-minmax observer, 20-point AWQ grid, Marlin target for SM86;
- recipe: `configs/quantization/qwen38_27b_w4a16_awq.json`, SHA-256
  `5c5ad69c371abb8bc82a0c90bbd9e7105602b1c564dda502ccff1a7a39e2a393`;
- calibration: `HuggingFaceH4/ultrachat_200k` revision
  `8049631c405ae6576f93f445c6b8166f76f5505a`, `train_sft`, 128 samples,
  2,048-token limit, deterministic seed 42;
- native MTP shard: SHA-256
  `1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe`;
  all 15 BF16 MTP tensors (849,398,784 tensor bytes) were copied losslessly and
  compared byte-for-byte with the pinned source.

Quantization used a 20 GiB GPU weight-placement budget and 205 GiB CPU budget.
The recorded 38 GiB and 30 GiB placement attempts failed with CUDA OOM; they
are retained in the recipe as reproducibility evidence. The artifact verifier
hashes every packed output file, rejects file-set drift and symlinks, checks the
recipe/base binding, and compares native MTP tensors with the source.

## Runtime pins

Serving uses Python 3.11.15, vLLM `0.25.0+cu129`, PyTorch `2.11.0+cu129`,
Transformers 5.14.1, compressed-tensors 0.17.0, CUDA runtime 12.9, and NVIDIA
driver 550.135. Quantization used Python 3.11.15, llmcompressor 0.12.0,
compressed-tensors 0.17.1, Transformers 5.10.1, PyTorch `2.12.0+cu126`, and CUDA
runtime 12.6.

## Certification and promotion

P6 is the mandatory non-MTP profile. P7 adds native MTP with exactly three
speculative tokens. Both bind only to loopback; P6 is port 8206 and P7 is port
8207. They use Qwen3 reasoning and Qwen3-Coder structured-tool parsers.

Use fresh output directories and run in this order after GPU-visible execution
is available:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="outputs/qwen38_certification/${stamp}"

PYTHONPATH=src .venv-quantization/bin/python \
  scripts/quantize_qwen38.py --verify-artifact

PYTHONPATH=src .venv/bin/python scripts/certify_qwen38_profile.py \
  P6_qwen38_w4a16 --output-root "${root}/p6"

PYTHONPATH=src .venv/bin/python scripts/certify_qwen38_production.py \
  P6_qwen38_w4a16 \
  --profile-certification "${root}/p6/certification.json" \
  --output-root "${root}/p6-production"

PYTHONPATH=src .venv/bin/python scripts/certify_qwen38_profile.py \
  P7_qwen38_w4a16_mtp --output-root "${root}/p7"

PYTHONPATH=src .venv/bin/python scripts/certify_qwen38_production.py \
  P7_qwen38_w4a16_mtp \
  --profile-certification "${root}/p7/certification.json" \
  --output-root "${root}/p7-production"
```

Each profile gate runs exact `/v1/models` identity, normal and SSE inference,
reasoning, structured tool calls, multi-turn memory, cancellation, a measured
32K prompt, runtime stability, packed-kernel log evidence, 250 ms GPU sampling,
and owned-process release. P7 additionally requires nonzero native-MTP draft
metrics and runtime evidence for method `mtp` with three tokens.

Each production gate starts the already-certified profile directly from its
staged configuration, starts CodeV first for deterministic co-resident memory
profiling, exercises quality, standard, and economy/RTL routes, measures at
least 2,048 MiB residual headroom, and releases only its owned processes. It
does not read or modify the active selector.

If P6 passes, promotion may proceed without MTP. Include P7 arguments only when
both P7 commands passed:

```bash
PYTHONPATH=src .venv/bin/python scripts/finalize_qwen38_certification.py \
  --p6-certification "${root}/p6/certification.json" \
  --p6-production-gate "${root}/p6-production/production_gate.json" \
  --p7-certification "${root}/p7/certification.json" \
  --p7-production-gate "${root}/p7-production/production_gate.json" \
  --promote
```

For a P6-only promotion, omit both P7 arguments. The finalizer re-verifies all
artifact bytes, profile hashes, repository revision, gate completeness,
headroom, CodeV coexistence, and process release. It atomically writes each
configuration file and restores both prior files on any selection failure.
Normal production startup re-hashes the artifact and refuses a
certified selector if any artifact or recipe byte has drifted.

After promotion, restart the existing persistent model and operator services
and check readiness. Administrator privileges are required on hosts where the
units are system-owned:

```bash
sudo systemctl restart laplace-model-servers laplace-operator
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

## Routing, MTP, and rollback

After P6 promotion, quality and standard route to Qwen3.8 while economy remains
CodeV. CodeV remains limited to policy-eligible bounded RTL implementation or
repair followed by deterministic verification; unresolved RTL architecture
stays with Qwen3.8. P7 is selected only when its independent profile and
co-resident gates both pass. Missing or failed P7 evidence records MTP as not
certified and does not block P6.

Qwen3.6 is retained as a configuration-only rollback. It is already active in
the current blocked state. After any future Qwen3.8 promotion, rollback is:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_production_model.py qwen36
sudo systemctl restart laplace-model-servers laplace-operator
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

No source reversion or artifact deletion is required.
