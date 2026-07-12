# Laplace — FormalScience Local Research Workspace

Laplace is the user-facing, local-only command for private PDF retrieval, structured extraction, engineering-log analysis, and evidence-grounded drafting. It binds services to `127.0.0.1`, keeps one main Ollama generation active by default, and never uploads documents or stores credentials.

The application repository is separate from the user-owned FormalScience library. A Laplace project can live in any safe working directory and contains its own `.laplace/project.yaml`, `Config`, `Data`, `Outputs`, and lifecycle state.

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

## Everyday commands

```powershell
laplace --config
laplace --ingest MyWorks --dry-run
laplace --ingest MyWorks
laplace --search "compute-in-memory low-bit quantization"
laplace --ask "Which local evidence supports the latency claim?"
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

`--start` launches the existing FastAPI/UI on `127.0.0.1:8000` in the background and records its PID in project state. `--start --foreground` keeps it attached. IEEE login is optional, visible, manual, and never receives credentials from Laplace; subscribed downloads require per-item approval. The AMD Ryzen AI NPU and vision path remain optional and cannot block the GPU/CPU baseline.

The shared library is selected through `FORMALSCIENCE_ROOT` (default `C:\Users\andre\OneDrive\Desktop\dottorato\FormalScience`). Source PDFs remain unchanged; derived text, metadata, downloads, vectors, drafts, and reports stay in the project.

## Verification

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check src tests
& .\.venv\Scripts\python.exe -m mypy src
laplace --doctor
```

The current verified runtime is local Ollama at `http://127.0.0.1:11434` with `qwen3:4b` and `qwen3-embedding:0.6b`. The benchmark records RTX 5060 Laptop GPU execution, context settings, prompt/generation token counts, TTFT, latency, token/s, CPU RAM, and sampled peak VRAM. No GPU result is inferred from API success alone.

The latest continuation rerun measured 0.379 tok/s (short), 32.838 tok/s (4k grounded), 20.798 tok/s (8k grounded), 21.575 tok/s (JSON), and 2.093 embedding texts/s, with observed peak VRAM up to 7795 MiB under a concurrent GPU process. The prior uncontended verified peak was 3832 MiB; see [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) for both runs and the safety interpretation.

## Advanced compatibility command

The original low-level command remains supported for reproducible scripts and CI:

```powershell
& .\.venv\Scripts\python.exe -m research_workspace.cli config
& .\.venv\Scripts\python.exe -m research_workspace.cli probe
& .\.venv\Scripts\python.exe -m research_workspace.cli benchmark-model
```

The FormalScience example and detailed security/provenance procedures are documented in [docs/FORMALSCIENCE_OPERATING_GUIDE.md](docs/FORMALSCIENCE_OPERATING_GUIDE.md), [docs/ONLINE_RESEARCH_SECURITY.md](docs/ONLINE_RESEARCH_SECURITY.md), [docs/IEEE_XPLORE_WORKFLOW.md](docs/IEEE_XPLORE_WORKFLOW.md), [docs/PROJECT_LIFECYCLE.md](docs/PROJECT_LIFECYCLE.md), [docs/PROVENANCE_MODEL.md](docs/PROVENANCE_MODEL.md), and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
