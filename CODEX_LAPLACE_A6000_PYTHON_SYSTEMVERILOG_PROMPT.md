# Codex master task — extend Laplace into a high-quality Python/SystemVerilog A6000 software-agent team and compare it directly with Codex

You are running from the root of an already cloned `https://github.com/afasolino/laplace` repository. The Git history, a working repository `.venv`, CUDA-capable PyTorch, and this overlay are already present. Extend Laplace in place. Do not create a replacement repository or a parallel RAG application.

Read, in this order:

1. every applicable `AGENTS.md`;
2. root `README.md`, `CODEX_PROMPT.md`, `PROJECT_CONFIG.yaml`, `PROMPTS.md`, `pyproject.toml`, and `.env.example`;
3. the actual implementation under `src/research_workspace/`, especially current CLI, project, library, document, retrieval, LLM, chat, drafting, benchmark, API, server, UI, acquisition, online, and optional-backend modules;
4. all current tests, schemas, prompts, configs, scripts, documentation, benchmarks, and fixtures;
5. `codex_a6000/AGENT_RULES.md`;
6. `codex_a6000/PROJECT_CONFIG.json`;
7. both reference catalogs in `codex_a6000/reference_sources/`;
8. task and benchmark schemas in `codex_a6000/templates/`;
9. `codex_a6000/benchmarks/paired_task_catalog.yaml`;
10. `codex_a6000/runtime/preflight.json`, when present.

Inspect the local clone before making path or API assumptions. The public repository currently has a flat `src/research_workspace` package, Python 3.11+, strict mypy, Ruff, pytest, the `laplace` and `research-workspace` entry points, local project state, RAG/citation validation, localhost FastAPI/UI, and an Ollama fallback. Preserve working behavior and user data.

Proceed automatically through repository-local work. Ask only for credentials, licence acceptance, administrator privileges, host package installation, external network access, access outside approved roots, destructive operations, Git merge, or an unrecoverable external blocker. Never fabricate measurements, tool availability, citations, source provenance, licences, or test results.

## Primary objective

Build a reusable local software-engineering agent team on one RTX A6000. Its default outputs are production-quality source changes, tests, diffs, verification evidence, and review reports. Python and SystemVerilog are first-class domains. Laplace RAG, provenance, persistence, and service infrastructure must support the engineering workflow rather than become a disconnected subsystem.

Preserve:

- existing CLI entry points and project lifecycle;
- project discovery, registry, SQLite state, queues, and background operations;
- document ingestion, retrieval, evidence IDs, citation validation, immutable rejected drafts, grounded fallback, provenance, and source immutability;
- localhost-only FastAPI/UI behavior;
- current extraction, comparison, research, writing, and benchmark operations;
- current Windows support while targeting Linux/A6000 deployment;
- Ollama as a lightweight fallback.

Add:

- governed Python and SystemVerilog reference libraries;
- a high-quality Python implementation policy and automated quality gates;
- a measured vLLM/SGLang path for a shared 24–32B 4-bit coding/instruct model;
- five explicit logical agents: supervisor, researcher, implementer, verifier, reviewer;
- typed task, evidence, patch, verification, review, and final-report artifacts;
- isolated Git worktrees and narrow tool wrappers;
- Python and SystemVerilog task-normalization and verification workflows;
- localhost-only MCP tools for hosted Codex delegation;
- reproducible latency, throughput, VRAM, task-quality, and end-to-end benchmarks;
- as the final phase, a concurrent paired comparison of Codex-direct implementations against Laplace-team implementations on identical held-out tasks.

## Mandatory phase protocol

For each phase:

1. inspect and reuse existing code first;
2. state narrow scope and acceptance criteria;
3. preserve backward compatibility unless a tested migration is necessary;
4. run existing relevant tests plus new tests;
5. repair focused failures without unrelated refactoring;
6. write `outputs/a6000_agent_team/status/phase_NN.json` or a demonstrably better existing Laplace convention;
7. mark `PASS` only after commands succeed, or `BLOCKED` with exact external evidence;
8. record commands, return codes, log paths, changed files, tests, measurements, assumptions, and unresolved defects;
9. commit passing changes only when Git identity is configured and no unrelated user changes exist.

Planning alone never passes a phase.

## Phase 0 — actual Laplace baseline and reuse map

Run the overlay preflight with the existing `.venv`, then run the repository's actual tests and quality commands. Resolve paths cross-platform. Typical Linux checks are:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/python -m mypy src
```

Inspect the actual command dispatch, project and safe-path logic, library/document ingestion, RAG/citation/provenance behavior, LLM abstraction, FastAPI/UI lifecycle, SQLite usage, queues, benchmark code, configuration, and tests.

Produce:

- `docs/a6000_agent_team/BASELINE_REUSE_MAP.md`;
- `docs/a6000_agent_team/INTEGRATION_ARCHITECTURE.md`;
- `outputs/a6000_agent_team/status/phase_00.json`.

Pass only when current behavior, extension points, baseline defects, compatibility risks, data-migration risks, and exact environment evidence are explicit.

## Phase 1 — governed Python and SystemVerilog reference libraries

Extend Laplace's existing library/document/provenance lifecycle. Do not create an untracked vector store or index entire monorepositories.

Preferred shared logical layouts:

```text
<FORMALSCIENCE_ROOT>/Library/Python/
  00_policies/
  10_language_stdlib/
  20_architecture_patterns/
  30_web_api/
  40_testing/
  50_typing_validation/
  60_tooling_packaging/
  70_agents_mcp/
  80_security_performance/
  90_manifests/

<FORMALSCIENCE_ROOT>/Library/SystemVerilog/
  00_policies/
  10_rtl_patterns/
  20_interfaces/
  30_verification/
  40_tooling/
  50_vendor_flows/
  90_manifests/
```

Provide project-local fallback using existing Laplace `Data` and cache conventions. Separate source clones, selected files, indexes, metadata, and generated output.

Use the supplied catalogs as candidates. Before network access, stop for approval. After approval, for each selected source:

1. resolve exact commit SHA;
2. inspect licence and notices at that commit;
3. select focused files by topic;
4. record URL, commit, selected path, licence identifier and text hash, file SHA-256, topics, attribution, and permitted use;
5. keep reciprocal or uncertain sources `reference_only_no_copy`;
6. make synchronization explicit, resumable, idempotent, and drift-detecting;
7. preserve current citation and provenance validation.

Implement backward-compatible CLI/API operations to initialize, list, synchronize, select, ingest, verify, and report references offline. Use local fixtures for tests; normal test runs must not require network access.

Pass when both libraries can be initialized, fixture references ingested, precedence enforced, hashes verified, licence gates tested, and existing Laplace behavior remains intact.

## Phase 2 — Python quality foundation

Make Python quality a system invariant rather than a prompt preference. Extend the current pyproject and tooling only when necessary and without mass-upgrading unrelated dependencies.

For every new or modified Python path require:

- Python 3.11+ and complete public API typing;
- strict mypy with no unjustified `Any`, `cast`, `type: ignore`, or untyped decorator escape;
- Ruff formatting and linting;
- explicit exception taxonomy and useful error context;
- structured logging without secrets;
- safe paths, atomic output where appropriate, and source immutability;
- explicit subprocess allowlists, timeouts, termination, and captured output;
- transaction-safe SQLite changes with rollback and migration tests;
- cancellation-safe async behavior and no blocking I/O on the event loop;
- backward-compatible CLI/API contracts;
- focused modules and functions, without speculative frameworks or duplicated infrastructure;
- unit and integration tests for success, invalid input, failure, and boundary behavior;
- property-based tests when parser, state-machine, path, or invariant coverage benefits;
- coverage evidence for new modules and critical security/provenance/tool paths.

Implement a reusable `run_python_quality_gates` service/CLI/MCP operation that emits a typed report rather than parsing informal output. It must run only allowlisted commands and preserve complete logs.

Create offline reference-retrieval tests demonstrating that a Python implementation task retrieves target-project conventions before open-source guidance.

Pass when existing and new Python code satisfies the real repository's format, lint, strict type, test, and coverage gates, with documented exceptions limited to explicit external boundaries.

## Phase 3 — inference abstraction and A6000 serving

Refactor the existing Ollama client behind a narrow backend interface while preserving behavior. Add local OpenAI-compatible clients for vLLM and SGLang in isolated environments when dependency compatibility requires it.

Requirements:

- one shared 24–32B-class 4-bit coding/instruct model on the A6000;
- measured candidate comparison, including checkpoint, revision, engine, quantization, kernel, context, concurrency, prefix caching, chunked prefill, CUDA graph mode, and scheduler settings;
- 8k/16k/24k/32k context and concurrency 1/2/4 sweeps within safe VRAM;
- real CUDA proof through PyTorch/NVML or equivalent device evidence;
- no CPU substitution for real inference;
- Ollama fallback retained;
- explicit timeouts, cancellation, health checks, model identity, and structured errors;
- no claim of speed or quality without recorded measurements.

Stop with `BLOCKED_GPU` after two focused repair attempts if CUDA inference cannot run. Do not fake a CPU baseline as A6000 evidence.

## Phase 4 — typed requirements normalization

Implement typed normalized task artifacts for generic software, Python, and SystemVerilog. Validate Python and SystemVerilog tasks against the supplied schemas.

A task must resolve objective, allowed paths, public interfaces, functional requirements, edge/error behavior, compatibility, security, performance constraints, references, verification commands, deliverables, assumptions, and out-of-scope items before implementation.

Ambiguity resolution order:

1. explicit user requirement;
2. target repository behavior and tests;
3. target project documentation/conventions;
4. private curated references;
5. governed open-source references;
6. model prior knowledge.

The supervisor may request clarification only when an unresolved ambiguity changes public behavior, safety, or acceptance criteria.

## Phase 5 — bounded five-agent state graph

Implement an explicit, persisted state machine:

```text
request
→ requirements
→ plan
→ retrieval
→ implementation
→ verification
→ review
→ at most two focused correction loops
→ final report
```

Roles and permissions:

- supervisor: state transitions, budgets, escalation; no direct implementation edits;
- researcher: read-only retrieval and evidence packet;
- implementer: writes only in an isolated task worktree;
- verifier: runs approved tools and reports; cannot silently edit implementation;
- reviewer: requirements/diff/evidence review; cannot merge;
- no recursive delegation.

Use typed persisted artifacts, resumable task IDs, bounded prompt/context budgets, deterministic status transitions, and failure recovery. Reuse current Laplace project state and database patterns where safe.

## Phase 6 — high-quality Python implementation workflow

Implement a Python-specialized workflow using `python_task_spec.schema.json`.

Required sequence:

```text
normalize task
→ inspect target code/tests
→ retrieve target-project and curated references
→ plan narrow changes
→ implement in worktree
→ Ruff format/check
→ strict mypy
→ pytest unit/integration/property tests
→ coverage and security/resource checks
→ reviewer assessment
→ bounded correction
```

Verification must detect at least:

- API/CLI compatibility regressions;
- unsafe path handling and partial writes;
- missing transaction rollback or migration behavior;
- async cancellation/resource leaks and blocking event-loop I/O;
- broad exception swallowing and silent fallback;
- permissive Pydantic coercion in sensitive boundaries;
- mutable global state and nondeterminism;
- missing invalid-input, failure, and boundary tests;
- unjustified typing escapes;
- generated code that merely satisfies public tests through special cases.

Every final Python report must include exact references used, assumptions, changed files, test/type/lint/coverage evidence, residual risks, and commands to reproduce.

## Phase 7 — SystemVerilog implementation and verification workflow

Implement the equivalent workflow using `systemverilog_task_spec.schema.json`.

Generation requirements include explicit clock/reset semantics, width/signedness handling, parameter validity, synthesizability, ready/valid stability, backpressure, WSTRB/response behavior for AXI-Lite, W1C semantics, IRQ behavior, and explicit CDC/RDC boundaries.

Verification order:

1. formatter/lint when available;
2. compile/elaboration;
3. self-checking deterministic and randomized simulation;
4. assertions/formal checks for protocols and stateful invariants when practical;
5. synthesis check and report inspection when available;
6. independent diff/evidence review.

Waveform inspection or model review alone never establishes correctness.

## Phase 8 — secure tools, worktrees, and Codex MCP integration

Implement narrow wrappers for Git, Python quality gates, pytest, Ruff, mypy, coverage, selected deterministic scripts, Verilator/iverilog/Yosys/formal tools, and optional vendor EDA tools discovered on the host.

Each wrapper requires typed inputs, path validation, explicit allowed executables/arguments, timeout and process-tree cleanup, bounded output capture, environment allowlist, and immutable logs. No general host-shell MCP tool.

Expose localhost-only MCP through STDIO and/or Streamable HTTP, following current supported protocol behavior. Include typed tools at minimum for task normalization, research, implementation, verification, review, Python gates, EDA flow, reference status, local model benchmarking, and paired quality benchmarking.

Create project-scoped Codex configuration examples, but do not modify user-level Codex configuration without approval. Verify tool discovery and one harmless read-only call from Codex.

## Phase 9 — performance tuning and end-to-end local-team validation

Benchmark real representative software workflows, not only synthetic token generation. Measure:

- TTFT, inter-token latency, output and aggregate token/s;
- prefill and decode behavior;
- peak/steady VRAM and GPU utilization;
- complete-task time;
- build/test/type/lint/simulation success;
- correction loops, interventions, and error rate;
- concurrency 1/2/4;
- routing between optional routine model and main model;
- prefix caching, chunked prefill, CUDA graphs, and speculation where supported.

Run at least one complete Python task and one SystemVerilog task through the local five-agent workflow before the comparative phase. Fix integration defects first. Produce operational documentation, launch commands, security model, benchmark data, and known limitations.

## Phase 10 — final concurrent Codex-direct versus Laplace-team comparison

This is the final phase. Do not perform another implementation phase afterward.

Use `codex_a6000/benchmarks/paired_task_catalog.yaml` and the paired manifest schema. Create a reproducible local benchmark with at least six valid tasks: at least four Python tasks and two SystemVerilog tasks. Use deterministic fixture repositories or isolated fixture subprojects. Include realistic requirements, seeded defects or missing features, public tests, and held-out scoring tests.

For every task:

1. create two clean Git worktrees from the same exact base commit;
2. provide both lanes the identical normalized task specification, public fixtures, time limit, and approved tools;
3. keep held-out tests outside both implementation worktrees until scoring;
4. prohibit either lane from reading the other lane's files, logs, or patch;
5. record the hosted Codex model, reasoning setting, CLI version, local model/engine/revision, prompts, tool permissions, and hardware state;
6. start both lanes concurrently:
   - **Codex-direct lane:** invoke a separate non-interactive `codex exec` process from its worktree using the runtime-supported flags shown by `codex exec --help`;
   - **Laplace-team lane:** invoke the local team through its MCP/CLI API from its own worktree;
7. launch both before awaiting either result, using an asynchronous orchestrator; record monotonic and wall-clock start/end timestamps and prove positive overlap;
8. enforce equal wall-time budgets and terminate process trees on timeout;
9. score both patches only after completion using the same held-out tests and tool versions;
10. anonymize lane identities as A/B before qualitative review;
11. make deterministic evidence the primary score and disclose subjective reviewer disagreement.

Python score, 100 points:

- held-out functional tests: 35;
- edge/failure/invariant tests: 15;
- strict typing: 10;
- Ruff format/lint: 10;
- security, concurrency, and resource safety: 10;
- maintainability and API compatibility: 10;
- performance/resource behavior: 5;
- scope discipline: 5.

SystemVerilog score, 100 points:

- held-out functional simulation: 35;
- protocol assertions and corner cases: 20;
- lint/compile/elaboration: 10;
- synthesis portability: 10;
- reset/width/signedness/CDC correctness: 10;
- self-checking test quality: 10;
- scope discipline: 5.

A failed build/elaboration or invalid run must be reported and cannot receive an unrestricted subjective score. Do not force a winner. Fewer than six valid tasks, unequal resources, missing execution overlap, exposed held-out tests, or lane contamination invalidate the aggregate conclusion.

Produce:

- `outputs/a6000_agent_team/comparison/codex_vs_laplace_results.json`;
- `outputs/a6000_agent_team/comparison/codex_vs_laplace_results.csv`;
- `outputs/a6000_agent_team/comparison/codex_vs_laplace_summary.md`;
- per-task task specs, timings, logs, patches, command manifests, objective score breakdowns, anonymized reviews, and invalid-run evidence;
- win/tie/loss, pass rate, mean/median score, median wall time, score per minute, defect categories, interventions, and confidence limitations;
- a final conclusion on relative code quality and implementation quality separately for Python and SystemVerilog.

The final user-facing report must state what was actually measured, what remains uncertain, which tasks favored each lane, and whether Laplace is ready for routine use, requires Codex supervision, or should be limited to narrower task classes.
