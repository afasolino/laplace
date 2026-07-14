# A6000 agent-team baseline reuse map

## Baseline observed

The authoritative implementation is the existing flat `research_workspace`
package. Its user-facing entry points are `laplace` and `research-workspace`.
The A6000 extension does not replace either command, project format, database,
or server.

| Existing component | Reused extension point | Compatibility rule |
| --- | --- | --- |
| `laplace_cli.py` | Added governed-reference, task, quality, EDA, serving and paired-benchmark options | Existing lifecycle and chat options are unchanged. |
| `projects.py` and `.laplace/project.yaml` | Shared governed `Library/{Python,SystemVerilog}` plus project-local `Data/AgentTeam` state | Shared sources remain immutable and task projects store only derived evidence and execution artifacts. |
| `documents.py` and `library.py` | The existing SQLite `documents`/`chunks` index receives hash-verified selected reference text | Reference snapshots and selected files remain separate and read-only. |
| `retrieval.py`, `chat.py`, provenance records | Engineering research packets retain source paths, hashes, licence policy and exact commits | Chat citation validation and rejected-draft/fallback semantics are untouched. |
| `llm.py` | Existing narrow provider shape now has loopback health/model identity plus vLLM/SGLang clients | Ollama remains the fallback and no CPU fallback is treated as GPU inference. |
| `api.py` and `laplace_server.py` | Loopback-only engineering reference and quality endpoints | Existing FastAPI/UI routes and project scope remain intact. |
| `core.py` logging/manifests and project output trees | Immutable command logs and typed JSON reports | Generated evidence is kept under `Outputs` or ignored application `outputs`. |
| Existing strict mypy/Ruff/pytest setup | `LocalToolRunner.run_python_quality_gates` | Only allowlisted executable/argument combinations run. |
| Existing deterministic analysis | `LocalToolRunner.run_eda_flow` | RTL lint, compile, self-checking simulation and synthesis are command evidence, not model review. |

## Baseline evidence and risks

- The control environment is Python 3.11.15, with CUDA 12.4 PyTorch present.
- The current host did not expose CUDA: `torch.cuda.is_available()` was false;
  `nvidia-smi --query-gpu=...` returned code 9 and reported that it could not
  communicate with the driver. No A6000 benchmark or inference claim is made.
- Ruff initially reported formatting drift in existing baseline files. The
  required formatter was run; this is a behavior-preserving normalization.
- The installed FastAPI/Starlette test client has a known local streaming test
  harness failure: even a finite minimal `StreamingResponse` does not close.
  The exact pre-existing affected test and package versions are recorded in
  `outputs/a6000_agent_team/status/phase_00.json`.
- The working tree already contained the supplied overlay as staged additions.
  It is preserved and no baseline commit/merge is attempted.

## Data migration assessment

No existing schema migration is required. Engineering references add one
idempotent `reference_ingestions` table to the project's existing SQLite index
on first reference ingestion. Agent tasks are independent JSON artifacts in
`Data/AgentTeam/tasks/<task-id>/`; they do not modify chat messages, source
documents, registry records, queues, or existing metadata rows.
