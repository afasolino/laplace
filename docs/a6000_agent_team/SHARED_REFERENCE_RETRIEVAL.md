# Shared governed-reference retrieval

The governed engineering corpus is stored independently from temporary task projects:

```text
<FORMALSCIENCE_ROOT>/Library/
├── Python/
└── SystemVerilog/
```

A task project keeps state, worktrees and reports. It receives only ranked, bounded,
hash-verified content chunks from the shared library. Every evidence packet records the
shared-library root, immutable snapshot hash, manifest count, chunk identifiers, scores,
licences, selected paths and inserted content.

## Register an approved local snapshot

Network cloning and licence acceptance remain outside this command. First check out the
source at an exact 40-character commit, inspect its licence and create a descriptor matching
`codex_a6000/templates/shared_reference_registration.schema.json`.

```bash
source .venv/bin/activate
python scripts/register_shared_reference.py \
  --library-root "$FORMALSCIENCE_ROOT/Library" \
  --domain systemverilog \
  --descriptor /absolute/path/reference.json
```

The operation is idempotent for an identical immutable descriptor. Any changed commit,
licence, selected file or hash requires a new `reference_id`.

## Run the corrected quality evaluation

```bash
python -m research_workspace.quality_improvement \
  --candidate-json outputs/a6000_agent_team/benchmarks/gpu_inference_results.json \
  --output-root outputs/a6000_agent_team/quality_improvement_corrected \
  --run-ablations \
  --shared-reference-root "$FORMALSCIENCE_ROOT/Library"
```

`curated_only` now stops with `BLOCKED_REFERENCE_EMPTY` when no verified content chunk is
retrieved. This prevents a no-context run from being labelled as a curated-reference result.
