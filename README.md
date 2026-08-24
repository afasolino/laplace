# Laplace — FormalScience Local Research Workspace

Laplace is the user-facing, local-only command for private PDF retrieval, structured extraction, engineering-log analysis, and evidence-grounded drafting. It binds services to `127.0.0.1`, keeps one main Ollama generation active by default, and never uploads documents or stores credentials.

The application repository is separate from the user-owned FormalScience library. A Laplace project can live in any safe working directory and contains its own `.laplace/project.yaml`, `Config`, `Data`, `Outputs`, and lifecycle state.

## v0.7 architecture and release status

v0.7 consolidates Desktop/local and Server/multi-user operation behind versioned
domain, provider, storage, provenance, repository, worktree, identity, capability,
job, audit and configuration contracts. It does not add a third GUI. The Operator GUI
loads a frontend-safe provider catalog and routes by provider/model identity; provider
ports and endpoints are not sent to ordinary frontend selectors.

Configuration validation is offline and does not load a provider:

```bash
PYTHONPATH=src python -m research_workspace.laplace_cli \
  --validate-config configs/laplace.example.yaml \
  --configuration-mode desktop \
  --diagnostic-export /tmp/laplace-config-diagnostic.json
```

The release gate is CPU/fixture-only. It covers static analysis, full tests, browser
fixtures, migrations, reproducible packaging, offline evaluation, CPU soak/failure
injection, security and documentation. Its live-model status is exactly
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`; fixture quality is never presented as
live-model quality. See [operating modes](docs/OPERATING_MODES.md),
[product architecture](docs/PRODUCT_ARCHITECTURE.md), [configuration](docs/CONFIGURATION_REFERENCE.md),
[migrations](docs/MIGRATIONS.md), and [release policy](docs/RELEASE_POLICY.md).

### Citation-safe chat revisions

Streaming is revision-aware. The model draft is persisted as an immutable `CITATION_REJECTED` assistant message when its citations do not resolve; a separate `GROUNDED_FALLBACK` message is then generated from retrieved evidence. The browser keeps the rejected draft expandable and shows the fallback as the primary answer, so a late validation event can never overwrite a streamed draft. Evidence supplied to the model uses compact IDs (`E1`, `E2`, …); the audit retains the raw model response and full provenance packet. The terminal equivalent is `laplace --ask "..." --show-rejected-draft` (or `--json`).

## Install

From this repository in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_laplace.ps1
laplace --version
laplace --doctor
```

The installer creates a user launcher at `%LOCALAPPDATA%\Programs\Laplace\laplace.cmd` and adds only that directory to the user PATH. If package build dependencies cannot be downloaded in an offline environment, the launcher can still be created with the repository source path; the script reports the fallback explicitly. Uninstall removes only that launcher and PATH entry:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\uninstall_laplace.ps1
```

The global registry and non-secret configuration are stored under `%USERPROFILE%\.laplace` (`config.yaml`, `projects.json`, `logs/`). Set `LAPLACE_HOME` only when an isolated test registry is required.

## First project and auto-detection

```powershell
mkdir .\SRAMCIMDraft
cd .\SRAMCIMDraft
laplace --init .
laplace --validate
laplace --status
```

From any child directory, Laplace searches the current directory and parents for `.laplace/project.yaml`. Outside a project it prints the exact next command to run. The global registry refuses name/path collisions; `laplace --list` and `laplace --unregister NAME` manage only registry entries.

## Primary chat workflow

For the shared FormalScience layout:

```powershell
cd "C:\Users\andre\OneDrive\Desktop\dottorato\FormalScience\Workspace"
laplace --init MyProject
cd MyProject
laplace --ingest MyWorks
laplace --start
```

The browser opens the project-aware chat at `http://127.0.0.1:8000/chat`. The sidebar holds conversations and collection filters; the center pane supports Ask, Search, Write, Research, and explicitly ungrounded General modes; selecting `[1]`, `[2]`, or another citation opens the right evidence panel with filename, title, page, chunk, source class, availability, score, and quoted passage. `＋` stages PDF/TXT/Markdown/CSV/JSON attachments in the project cache; it never writes to the shared Library.

Conversation history is persisted in `Data/Metadata/laplace.db`, bounded before prompting, and survives server restarts. Rename/archive controls, export, regenerate, Stop, project settings, Library status, Research candidates, Downloads, and the concise `/dashboard` remain available from the sidebar. Settings writes create timestamped backups and never return secrets.

The model response is normalized before display. Nested/fenced JSON is converted to readable Markdown; citation IDs must resolve to exact retrieved evidence. Invalid model citations are kept in the audit record and replaced with `GROUNDED_EXTRACTIVE_FALLBACK`. Full evidence is available through `/api/chat/messages/{message_id}/evidence`, not dumped into normal chat messages.

## Everyday commands

```powershell
laplace --config
laplace --ingest MyWorks --dry-run
laplace --ingest MyWorks
laplace --search "compute-in-memory low-bit quantization"
laplace --ask "Which local evidence supports the latency claim?"
laplace --ask "Which local evidence supports the latency claim?" --json
laplace --search "compute-in-memory low-bit quantization" --json
laplace --write related-work "Draft one concise evidence-grounded paragraph"
laplace --research "speculative decoding hardware"
laplace --web fetch https://arxiv.org/abs/2203.16487
laplace --web search "local retrieval benchmark"
laplace --extract metrics .\run.log
laplace --compare .\run_a.csv .\run_b.csv
laplace --backup
laplace --clean-cache --yes
```

`--ask` and `--write` accept an answer only when every returned citation matches a retrieved filename, page, and chunk ID. Otherwise they write `REVIEW_REQUIRED` and preserve the evidence packet. Routine generation uses non-thinking mode with `qwen3:4b`; the measured benchmark evidence for this machine is in `outputs/model_benchmark.json`.

## Local server and optional paths

```powershell
laplace --start
laplace --stop
laplace --ieee status
laplace --ieee browser-init
laplace --ieee login
laplace --queue
laplace --queue add .\candidate.json
laplace --ieee approve 0 --force
laplace --download 0
laplace --promote document.pdf MyTopics/speculative-decoding --force
```

`--start` validates the current project and exact Ollama models, launches the project-aware FastAPI/UI on `127.0.0.1:8000` in the background, records its PID, and opens `/chat`. It prints the project, model, embeddings, dashboard URL, and chat URL. `--start --foreground` keeps it attached; `--start --no-browser` suppresses browser opening. IEEE login is optional, visible, manual, and never receives credentials from Laplace; subscribed downloads require per-item approval. The AMD Ryzen AI NPU and vision path remain optional and cannot block the GPU/CPU baseline.

The shared library is selected through `FORMALSCIENCE_ROOT` (default `C:\Users\andre\OneDrive\Desktop\dottorato\FormalScience`). Source PDFs remain unchanged; derived text, metadata, downloads, vectors, drafts, and reports stay in the project.

## Verification

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check src tests
& .\.venv\Scripts\python.exe -m mypy src
laplace --doctor
```

A historical, separately certified runtime used local Ollama at `http://127.0.0.1:11434` with `qwen3:4b` and `qwen3-embedding:0.6b`. That preserved benchmark records RTX 5060 Laptop GPU execution, context settings, prompt/generation token counts, TTFT, latency, token/s, CPU RAM, and sampled peak VRAM. It is not evidence for the v0.7 CPU/fixture release, and no GPU result is inferred from API success alone.

The preserved historical continuation rerun measured 0.379 tok/s (short), 32.838 tok/s (4k grounded), 20.798 tok/s (8k grounded), 21.575 tok/s (JSON), and 2.093 embedding texts/s, with observed peak VRAM up to 7795 MiB under a concurrent GPU process. The prior uncontended verified peak was 3832 MiB; see [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) for both historical runs and the safety interpretation.

## Advanced compatibility command

The original low-level command remains supported for reproducible scripts and CI:

```powershell
& .\.venv\Scripts\python.exe -m research_workspace.cli config
& .\.venv\Scripts\python.exe -m research_workspace.cli probe
& .\.venv\Scripts\python.exe -m research_workspace.cli benchmark-model
```

The FormalScience example and detailed security/provenance procedures are documented in [docs/FORMALSCIENCE_OPERATING_GUIDE.md](docs/FORMALSCIENCE_OPERATING_GUIDE.md), [docs/ONLINE_RESEARCH_SECURITY.md](docs/ONLINE_RESEARCH_SECURITY.md), [docs/IEEE_XPLORE_WORKFLOW.md](docs/IEEE_XPLORE_WORKFLOW.md), [docs/PROJECT_LIFECYCLE.md](docs/PROJECT_LIFECYCLE.md), [docs/PROVENANCE_MODEL.md](docs/PROVENANCE_MODEL.md), and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Frozen CodeV execution and Operator Plane

The A6000 workflow now has an append-only, reproducibility-locked execution
core and a separate Research and Operator Plane. The browser and exploratory
research stores cannot influence measured execution.

```bash
laplace-operator --role read status --json
laplace-operator --role read model-servers preflight --json
laplace-operator-server --host 127.0.0.1 --port 8765
research create --request request.json --adapters adapters.json
research run --job RESEARCH_JOB_ID --adapters adapters.json
```

The Operator GUI provides authenticated run, evidence, research, corpus,
hardware, comparison, approval, and diagnostic views. GPU work and corpus
promotion require explicit approvals. Model-server admission uses preserved
A6000 measurements and exact `/v1/models` identities; shutdown signals only
PIDs proven to match Laplace model paths and ports.

Technical contracts are indexed from
[orchestration architecture](docs/orchestration_architecture.md),
[model-server lifecycle](docs/model_server_admission_and_lifecycle.md),
[Research Plane](docs/research_and_knowledge_plane.md), and
[GUI security](docs/gui_security.md). The exact reference revisions and
licence decisions are recorded in
[the external reference audit](docs/external_reference_audit.md).

## v8 release-candidate review

The v8 review starts from certified v7 commit
`a2b0bdf17445012114bbdee8fb3a30a9b4c73680`. Use
[RELEASE_CANDIDATE_RUNBOOK_V8.md](docs/RELEASE_CANDIDATE_RUNBOOK_V8.md) for the
CPU/fixture gates and [LIVE_GPU_CERTIFICATION_RUNBOOK_V8.md](docs/LIVE_GPU_CERTIFICATION_RUNBOOK_V8.md)
for the conditional live gate. SpecDec always has priority; see
[SPECDEC_GPU_COORDINATION.md](docs/SPECDEC_GPU_COORDINATION.md).

## Qwen3.8 and Zetsu production path

The A6000 production migration is documented in
[QWEN38_PRODUCTION_MIGRATION.md](docs/QWEN38_PRODUCTION_MIGRATION.md) and the
Codex integration layer in [ZETSU.md](docs/ZETSU.md). The migration targets a
pinned published W4A16 Qwen3.8 checkpoint and a 131,072-token P6 profile;
Quality/Standard promotion remains contingent on live certification. CodeV keeps
its bounded RTL/SystemVerilog Economy route, and Qwen3.6 remains the rollback.

Zetsu adds compact owner-authorized retrieval, bounded Qwen delegation and
repository-agent tasks, CodeV RTL delegation, and deterministic evidence. Codex
continues to use its own checkout/shell/Git for ordinary local work. The managed
MCP registration lives in the user's Codex configuration while the usage Skill is
project-local; bearer values stay in the process environment.
