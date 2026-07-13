# Laplace A6000 engineering-agent integration

The extension is deliberately an addition to the local research workspace.
Project knowledge, curated references and verification logs are all evidence
for an engineering task; they are not a separate RAG application.

```text
Laplace project
├── Data/References/{Python,SystemVerilog}/
│   ├── sources/<reference-id>/<commit>/     immutable local snapshots
│   ├── logical topic folders/                immutable selected files
│   ├── 90_manifests/                         licence, hash and provenance records
│   ├── indexes/                              derived ingestion reports
│   └── selections/                           topic-selection evidence
├── Data/Metadata/workspace.db                existing documents/chunks + reference_ingestions
├── Data/AgentTeam/tasks/<task-id>/           persisted bounded state graph
└── Outputs/AgentTeam/                        quality, EDA and final reports
```

The five logical roles have fixed authority:

- Supervisor moves a persisted task through `request → requirements → plan →
  retrieval → implementation → verification → review → [at most two repairs]
  → final report`. It cannot write an implementation patch.
- Researcher is read-only and writes only an evidence packet. Target-project
  files are emitted before private/governed references, which are emitted
  before model prior knowledge.
- Implementer may only write an implementation report and patch manifest for
  an isolated worktree. A future local-model runner must parse and validate a
  patch before applying it; it has no shell tool.
- Verifier only uses `LocalToolRunner`, whose executable and argument
  combinations are fixed. It writes logs and verification reports, not source.
- Reviewer writes a review report and cannot merge or edit source.

`agent_mcp.py` exposes the same typed operations over stdio JSON-RPC for a
project-scoped Codex configuration. It provides no general host-shell tool and
does not change user-level Codex configuration. `inference.py` requires a
fresh CUDA/A6000 proof before it makes a local vLLM/SGLang request. The
candidate manifest records engine, checkpoint revision, quantization, kernel,
context, concurrency, cache, prefill, CUDA-graph and scheduler settings.

`paired_benchmark.py` refuses to start just one lane. It writes an invalid-run
report when CUDA is unavailable, avoiding a CPU substitute or asymmetric
Codex-only score. A valid run requires six prepared tasks, clean worktrees at
the same commit, hidden tests outside both worktrees, equal budgets, positive
overlap, and post-completion scoring.
