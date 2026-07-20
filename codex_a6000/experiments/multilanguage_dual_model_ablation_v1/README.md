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

The arms form one logical experiment but run in three resumable serving phases.
Phase 1 contains Arm A, Phase 2 contains Arm B, and Phase 3 contains Arm C.
Phase 2 requires compatible completed Phase-1 results. Phase 3 requires both
prior phases. Every task-arm result carries the complete compatibility
fingerprint, so resume never skips an unbound or stale pair. The merger rejects different base
commits, task manifests, corpus snapshots, experiment configurations or
evaluation settings.

Task-arm result schema 2 applies context-aware generation budgets to every
role. Reviewer output is capped independently, prompts are compacted only by
explicit evidence-preserving policies, and irreducible overflow is recorded as
a typed terminal infrastructure failure. Every attempted pair produces either
a complete evaluated candidate result or a resumable terminal failure record.
Only complete, fingerprint-compatible candidate results are skipped on resume.
Schema-1 Phase-1 evidence remains preserved in the legacy output paths, while
new results are written below a base-revision and `failure-accounting-v2`
result-set directory.

Qwen reasoning and final-answer channels are retained separately. First-pass
implementation and RTL calls and their first response retry keep the model's
default thinking mode. If both Qwen3.6 responses are rejected, the final
bounded serialization retry uses an OpenAI JSON-schema constraint and disables
thinking only for that retry so the completion budget is reserved for the
deterministic replacement object.
Reasoning text is never substituted for final content. Audit records retain
final content, finish reason, token counts, and only a length/hash summary of
reasoning. Because the constrained retry is non-thinking, the managed server
does not require vLLM's `structured_outputs.enable_in_reasoning` startup option.

The profiles contain exact official source revisions and loopback serving
identities. Phase 1 uses the installed upstream Qwen AWQ artifact at revision
`1ed0a6145da0ce550c628e8e8b678f51e695995d`. Phases 2 and 3 pin
`Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`
and `zhuyaoyu/CodeV-R1-RL-Qwen-7B@286cf433f596f1b8525529c1163eb81c19425c22`.
No arbitrary community quantization is accepted. The two new-model artifacts
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
.venv/bin/python -m research_workspace.multilanguage_ablation validate-phase2
.venv/bin/python -m research_workspace.multilanguage_ablation validate-phase3
.venv/bin/python -m research_workspace.multilanguage_ablation validate-manifest
.venv/bin/python -m research_workspace.multilanguage_ablation validate-corpus
.venv/bin/python -m research_workspace.multilanguage_ablation preflight
.venv/bin/python -m research_workspace.multilanguage_ablation plan-only
.venv/bin/python scripts/manage_multilanguage_models.py validate-metadata
.venv/bin/python scripts/manage_multilanguage_models.py check
.venv/bin/python scripts/manage_multilanguage_models.py validate-quantization-lock \
  --lock .models/quantization_requirements.lock
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

The quantization lock pins `llmcompressor==0.12.0`,
`compressed-tensors==0.17.1`, `transformers==5.10.1`, `datasets==5.0.0`,
and the backend-compatible Torch range resolved by `uv`. The offline validator
checks those constraints, hashes and required quantization API markers. It does
not claim that either Phase-2 source model or quantized artifact exists; source
`config.json` architecture validation remains fail-closed until the user later
authorizes the pinned source downloads.

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
.venv/bin/python -m research_workspace.multilanguage_ablation smoke-runtime --phase phase2
.venv/bin/python -m research_workspace.multilanguage_ablation selective-retry-plan --phase phase2
.venv/bin/python -m research_workspace.multilanguage_ablation run-phase2
.venv/bin/python -m research_workspace.multilanguage_ablation phase-status --phase phase2
scripts/manage_multilanguage_model_servers.sh stop-phase2
scripts/manage_multilanguage_model_servers.sh start-phase3
.venv/bin/python -m research_workspace.multilanguage_ablation validate-runtime --phase phase3
.venv/bin/python -m research_workspace.multilanguage_ablation run-phase3
.venv/bin/python -m research_workspace.multilanguage_ablation phase-status --phase phase3
scripts/manage_multilanguage_model_servers.sh stop-phase3
.venv/bin/python -m research_workspace.multilanguage_ablation merge-report
```

The fail-closed launcher performs those steps in order and preserves a timestamped
log:

```bash
scripts/run_multilanguage_dual_model_ablation.sh phase1
scripts/run_multilanguage_dual_model_ablation.sh phase2
scripts/run_multilanguage_dual_model_ablation.sh phase3
scripts/run_multilanguage_dual_model_ablation.sh all
scripts/run_multilanguage_dual_model_ablation.sh status
scripts/run_multilanguage_dual_model_ablation.sh merge
```

Individual commands default to externally managed servers. Append `managed`
to let the safe PID/model/address/port/token lifecycle start and stop that
phase. Bare `all` defaults to managed serialized execution and reuses the
identical Qwen3.6 server from Phase 2 while starting CodeV for Phase 3:

```bash
scripts/run_multilanguage_dual_model_ablation.sh phase2 external
scripts/run_multilanguage_dual_model_ablation.sh phase2 managed
scripts/run_multilanguage_dual_model_ablation.sh all managed
scripts/run_multilanguage_dual_model_ablation.sh all external
```

External `all` never changes server state; use the individual external phase
commands when the operator must replace one resident model between phases.

For the CUDA-12-family Phase 2/3 environment and the resumable serial run:

```bash
scripts/bootstrap_vllm_cu129.sh
scripts/run_phase2_phase3_serial.sh
```

The bootstrap installs the official vLLM `0.25.0+cu129` wheel in the separate
`.venv-vllm-cu129` environment. The serial launcher verifies Phase 1, finishes
Phase 2, stops its server, finishes Phase 3, stops both Phase 3 servers, and
merges the experiment. It retains per-invocation launcher and server logs and
never deletes compatible partial pair results.

Do not infer statistical generality from the 32 tasks. Reports retain paired
per-task differences and deterministic bootstrap intervals only as diagnostic
evidence.
