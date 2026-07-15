# Multilanguage dual-model ablation v1

This directory prepares one controlled local experiment. It does not download
models, start serving processes, expose held-out tests, or permit CPU inference.

## Arms

- Arm A: `Qwen2.5-Coder-32B-Instruct-AWQ` performs every model role.
- Arm B: `Qwen3.6-35B-A3B-INT4` performs every model role.
- Arm C: the identical Arm B main profile performs supervision, retrieval
  interpretation, general implementation, RTL contract generation, integration,
  failure diagnosis and review. `CodeV-R1-RL-Qwen-7B-INT4` may implement or
  repair only a metadata-eligible, single synthesizable RTL module.

The arms form one logical experiment but run in two resumable serving phases.
Phase 1 contains only Arm A, so neither Qwen3.6 nor CodeV is required. Phase 2
contains Arms B and C, requires a compatible completed Phase 1, and never
repeats an already evaluated Arm-A pair. The merger rejects different base
commits, task manifests, corpus snapshots, experiment configurations or
evaluation settings.

The profiles contain exact official source revisions and loopback serving
identities. Phase 1 uses the installed upstream Qwen AWQ artifact at revision
`1ed0a6145da0ce550c628e8e8b678f51e695995d`. Phase 2 pins
`Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`
and `zhuyaoyu/CodeV-R1-RL-Qwen-7B@286cf433f596f1b8525529c1163eb81c19425c22`.
No arbitrary community quantization is accepted. The two Phase-2 artifacts
remain unavailable until the pinned local W4A16 recipe is explicitly run and
its complete file-hash manifest and compressed-tensors metadata validate.
Arm B and Arm C retain identical main-model settings.

## Safety boundaries

Routing is deterministic. The main model creates a strict hash-bound RTL
contract. The worker receives that contract, not repository or tool authority,
and can return only a complete replacement for the one permitted file. Existing
stale-hash, malformed JSON, duplicate path, scope and no-op checks remain the
only path to a local patch. Invalid contracts or worker outputs use bounded
retries and then fall back to the main model.

Held-out material belongs in a separate evaluator-owned directory named by
`LAPLACE_ABLATION_HELD_OUT_ROOT`. Its `manifest.json` must satisfy
`codex_a6000/templates/held_out_ablation_manifest.schema.json` and contain exact
SHA-256 hashes. The runner reads it only after an implementation lane stops,
applies the lane patch in a new evaluation worktree and runs fixed
language-specific gates. No held-out path or content is added to an
implementation prompt or worktree.

`LAPLACE_ABLATION_BASE_REVISION` must be set only after this implementation is
reviewed and committed. The runner requires a clean worktree, resolves the
exact commit, and proves that the experiment, model configurations, task
manifest, fixtures and public tests are present in that commit.

## Commands

From the repository root:

```bash
.venv/bin/python -m research_workspace.multilanguage_ablation validate-config
.venv/bin/python -m research_workspace.multilanguage_ablation validate-phase1
.venv/bin/python -m research_workspace.multilanguage_ablation validate-manifest
.venv/bin/python -m research_workspace.multilanguage_ablation validate-corpus
.venv/bin/python -m research_workspace.multilanguage_ablation preflight
.venv/bin/python -m research_workspace.multilanguage_ablation plan-only
.venv/bin/python scripts/manage_multilanguage_models.py validate-metadata
.venv/bin/python scripts/manage_multilanguage_models.py check
scripts/bootstrap_multilanguage_tools.sh report
scripts/bootstrap_multilanguage_tools.sh probe
```

The deterministic tool bootstrap is a separate, potentially substantial user
action. It installs only into `.tools/multilanguage` and never uses sudo:

```bash
scripts/bootstrap_multilanguage_tools.sh install
export PATH="$PWD/.tools/multilanguage/bin:$PATH"
```

The complete pinned download and quantization commands are emitted without
executing them by:

```bash
.venv/bin/python scripts/manage_multilanguage_models.py commands
```

That output also includes the hashed lock and sync commands for the isolated
Phase-2 `.venv-vllm` serving environment. Phase 1 deliberately reuses the
validated CUDA 12.4 environment at `.venv-vllm-cu124` with vLLM
`0.8.5.post1`; the server lifecycle script resolves the phase-specific
executable from the governed model-artifact profile.

After the model profiles, held-out pack, exact post-implementation base commit,
CUDA device and separately started loopback servers pass validation:

```bash
export LAPLACE_ABLATION_BASE_REVISION=<reviewed-40-character-commit>
export LAPLACE_ABLATION_HELD_OUT_ROOT=/absolute/evaluator-owned/heldout-pack
.venv/bin/python -m research_workspace.multilanguage_ablation validate-phase1
.venv/bin/python -m research_workspace.multilanguage_ablation preflight --phase phase1
scripts/manage_multilanguage_model_servers.sh start-phase1
.venv/bin/python -m research_workspace.multilanguage_ablation validate-runtime --phase phase1
.venv/bin/python -m research_workspace.multilanguage_ablation run-phase1
.venv/bin/python -m research_workspace.multilanguage_ablation phase-status --phase phase1
scripts/manage_multilanguage_model_servers.sh stop-phase1
scripts/manage_multilanguage_model_servers.sh start-phase2
.venv/bin/python -m research_workspace.multilanguage_ablation validate-runtime --phase phase2
.venv/bin/python -m research_workspace.multilanguage_ablation run-phase2
.venv/bin/python -m research_workspace.multilanguage_ablation phase-status --phase phase2
.venv/bin/python -m research_workspace.multilanguage_ablation merge-report
scripts/manage_multilanguage_model_servers.sh stop-phase2
```

The fail-closed launcher performs those steps in order and preserves a timestamped
log:

```bash
scripts/run_multilanguage_dual_model_ablation.sh phase1
scripts/run_multilanguage_dual_model_ablation.sh phase2
scripts/run_multilanguage_dual_model_ablation.sh status
scripts/run_multilanguage_dual_model_ablation.sh merge
```

Do not infer statistical generality from the 32 tasks. Reports retain paired
per-task differences and deterministic bootstrap intervals only as diagnostic
evidence.
