# Laplace A6000 engineering-agent integration

The extension remains part of the local research workspace. Shared governed references are
stored independently from ephemeral task projects so every benchmark task sees the same
immutable corpus snapshot.

```text
<FORMALSCIENCE_ROOT>/Library/
├── Python/
│   ├── sources/<reference-id>/<commit>/
│   ├── logical topic folders/
│   ├── 90_manifests/
│   ├── indexes/reference_index.db
│   └── selections/
└── SystemVerilog/
    └── ... same governed layout ...

Laplace task project/
├── Data/Metadata/workspace.db
├── Data/AgentTeam/tasks/<task-id>/
├── Data/AgentTeam/worktrees/
└── Outputs/AgentTeam/
```

Each registered reference is an exact-commit, licence-hashed, read-only snapshot. The derived
reference index stores selected text chunks and provenance metadata. Retrieval returns bounded,
ranked content chunks plus the immutable shared-library snapshot hash. A temporary task project
does not own or duplicate the shared corpus.

The shared root is selected explicitly with `shared_reference_root`,
`LAPLACE_SHARED_REFERENCE_ROOT`, or `FORMALSCIENCE_ROOT/Library`. Project-local reference storage
remains available as a compatibility fallback when no shared root is configured.

The five logical roles have fixed authority:

- Supervisor moves a persisted task through `request → requirements → plan → retrieval →
  implementation → verification → review → [at most two repairs] → final report`.
- Researcher is read-only and writes only an evidence packet. Target-project files precede
  governed shared references, followed by model prior knowledge.
- Implementer may write only a validated patch in an isolated worktree.
- Verifier uses the allowlisted `LocalToolRunner` and writes command evidence and defect reports.
- Reviewer requires explicit verifier evidence and receives the exact governed-reference snapshot
  used by the implementation.

`curated_only` fails closed with `BLOCKED_REFERENCE_EMPTY` when no verified content chunk is
available. This prevents an empty reference corpus from being scored as a curated-reference run.

`agent_mcp.py` exposes the typed engineering operations over local stdio JSON-RPC and provides no
general host shell. `inference.py` requires a fresh CUDA/A6000 proof before local vLLM/SGLang
inference. `paired_benchmark.py` and `quality_improvement.py` keep hidden tests outside every
implementation worktree and score only completed lanes.
