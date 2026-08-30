# Codex master implementation prompt

Act as a senior local-LLM engineer, retrieval-system architect, scientific-document processing engineer, and verification-focused Python developer.

Read `AGENTS.md`, `PROJECT_CONFIG.yaml`, and `PROMPTS.md`. Implement the complete repository sequentially. Do not stop after producing a plan. Do not wait for approval between phases except for credentials, licence acceptance, administrator privileges, destructive user-data operations, or access outside the repository.

The target machine is an ASUS laptop with:
- one NVIDIA GPU with 8 GB VRAM;
- one AMD Ryzen AI NPU rated at 50 TOPS;
- Windows as the primary deployment environment;
- optional Linux/WSL compatibility.

The final product is a fully local research workspace providing:
1. private PDF and project knowledge retrieval;
2. structured scientific information extraction;
3. engineering-log and experimental-result analysis;
4. evidence-grounded academic drafting and revision.

Use a GPU-first baseline. The NPU is optional and must never block the working system.

## Required repository shape
Create a clean implementation resembling:

```text
src/research_workspace/
  api/
  cli/
  config/
  documents/
  retrieval/
  llm/
  extraction/
  analysis/
  drafting/
  provenance/
  security/
  ui/
tests/
configs/
schemas/
prompts/
scripts/
docs/
benchmarks/
data/.gitkeep
outputs/.gitkeep
pyproject.toml
.env.example
README.md
```

Adjust only where justified. Keep modules small and typed.

## Phase 0 — Repository and reproducibility foundation
Implement:
- Python project with `uv` and pip-compatible installation;
- configuration loader with environment overrides;
- structured JSON logging;
- CLI entry point;
- deterministic run IDs and artifact manifests;
- Windows PowerShell and POSIX shell bootstrap scripts;
- localhost-only defaults;
- unit-test, lint, and type-check commands;
- architecture, security, data-flow, and troubleshooting documentation.

Acceptance:
- clean install in a new virtual environment;
- configuration validation tests pass;
- CLI help works;
- no source document or generated index is tracked by Git.

## Phase 1 — Hardware and runtime discovery
Implement a non-destructive diagnostic command that records:
- OS, Python, CPU and RAM;
- NVIDIA GPU name, VRAM, driver, CUDA visibility, power state when available;
- Ollama/LM Studio/llama.cpp availability;
- AMD NPU and Ryzen AI/Windows ML/ONNX Runtime availability;
- free disk space;
- exact commands and raw outputs.

Generate `outputs/system_probe.json` and a human-readable report. Never infer unavailable hardware properties.

Acceptance:
- command works when NVIDIA or NPU tools are absent;
- tests cover missing executables and malformed outputs;
- report distinguishes detected, unavailable, unsupported, and permission-blocked states.

## Phase 2 — Local model server and model-selection benchmark
Implement an abstraction for a localhost model endpoint, preferring Ollama while supporting an OpenAI-compatible local endpoint and llama.cpp as alternatives.

Benchmark candidate 4B-class, 4-bit instruct models configured by the user. Do not silently download a model whose licence has not been recorded. Measure:
- model identifier and file size;
- idle and peak VRAM;
- context configuration;
- time to first token;
- visible output token/s;
- prompt processing rate where exposed;
- deterministic writing, extraction, citation-formatting, and structured-JSON checks;
- OOM and fallback behavior.

Use 8k as the target operational context and concurrency one. Keep a configurable 4k safe mode. Never fabricate measurements.

Select a default only from measured results and record the decision. If no model is installed, finish the software implementation with a clear `MODEL_REQUIRED` status and exact installation instructions.

Acceptance:
- mocked benchmark tests pass;
- real benchmark runs when a local model is available;
- the selected model stays within the configured VRAM safety limit;
- the application remains functional with a mock provider for tests.

## Phase 3 — Document ingestion and provenance
Implement incremental ingestion for PDF, DOCX, Markdown, text, HTML, CSV, JSON, and common log/report files.

For PDFs:
- use Docling as the preferred structured parser;
- preserve page boundaries, sections, captions, tables, references, and extraction confidence where available;
- use a documented fallback parser for recoverable failures;
- invoke OCR only for pages detected as image-only or unusable through native extraction;
- retain extraction warnings and page-level provenance.

Implement:
- SHA-256 identity and duplicate detection;
- idempotent re-indexing;
- document classes from `PROJECT_CONFIG.yaml`;
- immutable source storage and separate derived artifacts;
- metadata editing without changing source hashes;
- deletion and full rebuild commands;
- ingestion manifest and failure quarantine.

Acceptance:
- tests with normal, multi-column, table-containing, malformed, duplicate, and scanned-document fixtures;
- no silent page loss;
- every chunk maps back to source file and page range.

## Phase 4 — Retrieval and citation-grounded answering
Implement local embeddings, persistent vector storage, metadata filtering, lexical fallback, deduplication, optional reranking, and context-budget management.

Required retrieval modes:
- semantic search;
- exact/keyword search;
- hybrid search;
- filter by document class, author, year, project, and file;
- “my work only,” “external literature only,” and comparison modes.

Use an embedded local vector store and SQLite metadata/run tracking. Keep the storage layer replaceable.

Answers must include:
- direct response;
- evidence entries with filename, page, section, and chunk ID;
- uncertainty/conflicts;
- missing evidence;
- a machine-readable evidence packet.

Implement citation validation that rejects page references absent from the retrieved provenance.

Acceptance:
- retrieval tests with a labelled mini-corpus;
- report Recall@k, MRR or nDCG, citation-validity rate, duplicate rate, and latency;
- no answer is labelled grounded when zero valid evidence chunks are available.

## Phase 5 — Structured scientific extraction
Implement schema-driven extraction to JSON and CSV. Include schemas for:
- bibliographic metadata;
- contribution/novelty claims;
- methodology and experimental setup;
- model/dataset/software configuration;
- semiconductor process, circuit, memory, precision and operating conditions;
- area, power, energy, latency, throughput and accuracy metrics with units;
- baselines, limitations, future work and evidence locations.

Every extracted value must include provenance and confidence. Unsupported fields remain null. Validate units and preserve the source wording separately from normalized values.

Provide batch extraction and comparison-table generation.

Acceptance:
- strict schema validation;
- tests for missing fields, conflicting units, duplicated metrics and table-derived values;
- extraction accuracy report on a hand-labelled fixture set;
- no numerical value without provenance.

## Phase 6 — Engineering logs and deterministic result analysis
Implement parsers and adapters for generic logs plus extensible profiles for:
- Python exceptions;
- GCC/Clang and build logs;
- Git;
- Vivado synthesis/timing/utilization reports;
- Cadence/Spectre-style logs and tabular outputs;
- CUDA/PyTorch benchmark logs;
- CSV/JSON experimental summaries.

The deterministic layer extracts metrics and identifies candidate root errors. The LLM receives only structured records and relevant log windows.

Implement run-to-run comparison and known-good versus failed-run analysis. Never execute model-generated commands automatically.

Acceptance:
- fixture-based parser tests;
- first-actionable-error evaluation;
- metric extraction and unit tests;
- generated analysis separates observed facts, interpretation and proposed diagnostics.

## Phase 7 — Evidence-grounded drafting and revision
Implement workflows for:
- paragraph revision preserving technical meaning;
- abstract, introduction, methodology, results, conclusion and reviewer-response drafting;
- terminology and notation consistency checking;
- claim-to-source mapping;
- claim-versus-table consistency checks;
- identification of unsupported statements;
- comparison with the user's previous works without copying external prose;
- evidence packet export for optional use with a stronger external model, without uploading anything automatically.

Add style profiles, with formal IEEE-style English as the default. Store user style examples as retrievable documents rather than fine-tuning data.

Acceptance:
- tests ensure unsupported claims receive `[SOURCE REQUIRED]`;
- citations must resolve to indexed provenance;
- revision preserves numbers, units, symbols and explicit hedging unless the user requests a substantive change;
- outputs distinguish evidence, inference and proposal.

## Phase 8 — Local user interface and API
Create a compact local interface with pages or tabs for:
- system status;
- collections and ingestion;
- grounded chat/search;
- scientific extraction and comparison tables;
- log/result analysis;
- drafting/revision;
- benchmark and provenance inspection.

Use a lightweight local UI framework and a FastAPI backend. Bind to `127.0.0.1`. Support drag-and-drop uploads with strict extension, size, path and MIME validation. Show source citations that open the local derived page view or identify the exact page.

Acceptance:
- API tests;
- basic UI smoke test;
- no network dependency after assets and models are installed;
- no accidental exposure on all interfaces.

## Phase 9 — Optional NPU integration
Only after the GPU/CPU baseline passes, probe current official AMD-supported paths. Implement an optional provider for one bounded auxiliary workload, preferably embeddings, classification or reranking.

Requirements:
- exact hardware/software compatibility check;
- ONNX model and operator compatibility validation;
- CPU/GPU baseline comparison;
- measured latency, throughput, power telemetry only when available, and output-quality equivalence;
- automatic fallback when the NPU is unavailable;
- no unsupported claim that the NPU accelerates the NVIDIA-hosted LLM.

Promote the NPU path to default only when it passes correctness and offers a measured operational benefit. Otherwise record `NPU_OPTIONAL_NOT_BENEFICIAL` or the precise blocker.

## Phase 10 — Optional visual document inspection
Add only after the core system passes. Support rendering selected PDF pages and analysing figures/tables with an on-demand 3B–4B vision model or a supported local vision runtime.

Keep image analysis explicitly secondary to native document extraction. Require user-visible warnings for uncertain small labels or dense schematics. Never estimate plotted values without a dedicated, labelled digitization step.

Ensure the text and vision models do not exceed the 8 GB VRAM budget; unload one before loading the other when required.

## Phase 11 — Final verification and packaging
Run:
- unit, integration and smoke tests;
- linting and type checking;
- offline operation test;
- retrieval benchmark;
- extraction benchmark;
- citation-validity test;
- security checks for path traversal, unsafe uploads and localhost binding;
- real hardware/model benchmark where available.

Create:
- `docs/USER_GUIDE.md`;
- `docs/INSTALL_WINDOWS.md`;
- `docs/INSTALL_LINUX_WSL.md`;
- `docs/NPU_STATUS.md`;
- `docs/BENCHMARK_REPORT.md`;
- `docs/LIMITATIONS.md`;
- reproducible start/stop/index/backup scripts;
- a final implementation manifest.

The final report must state:
- implemented and tested capabilities;
- exact model and runtime used;
- measured VRAM and token/s;
- NPU status;
- retrieval and extraction metrics;
- unresolved limitations;
- exact commands required to reproduce the result.

## Prohibited shortcuts
- no cloud API fallback;
- no fabricated benchmark values;
- no whole-document prompt stuffing presented as retrieval;
- no citations generated from model memory;
- no page numbers without source provenance;
- no numerical analysis performed only by the LLM;
- no automatic execution of commands suggested by the model;
- no mandatory NPU dependency;
- no placeholder-only implementation presented as complete.

Begin by inspecting the repository and writing a short executable plan to `docs/IMPLEMENTATION_PLAN.md`, then immediately implement Phase 0 and continue through all feasible phases. Commit only when tests pass if the repository is already under Git; otherwise leave a clean, reviewable working tree and report all changes.
