# Changelog

All notable Laplace changes are documented here. Versions follow Semantic
Versioning; the 0.7.0 entry below is retained as historical release context.

## 2.0.0 — 2026-08-25

### Added

- standalone neutral `LaplaceCore` services with an optional authenticated Zetsu adapter;
- repository readiness, exact committed-revision synchronization, dirty-state diagnostics,
  bounded worktree lifecycle reconciliation, quota recovery, resume binding, and session inspection;
- bounded/paged logical results, GPU admission queueing, independent execution deadlines,
  continuation-safe ACI, fail-closed memory schema negotiation, explicit parser contract,
  cooperative hook timeout semantics, and bounded retention;
- machine-readable corrective certifications C0–C8.

### Changed

- release metadata now reports v2.0.0;
- the normal development/runtime topology is Qwen quality/standard with CodeV optional and
  explicitly disabled by `--nocodev` when not required;
- public documentation describes the actual v2 control plane and immutable production/development boundary.

### Deferred

- the C8 SiliconMind-vs-CodeV RTL A/B remains explicitly `BLOCKED` until the candidate
  checkpoint and certified full-topology vLLM executable are available locally.

## 0.7.0 — 2026-07-28

### Added

- versioned provider-neutral architecture and presentation contracts;
- deterministic model, embedding and service fixtures;
- strict layered configuration with provenance and redacted diagnostics;
- explicit local Ollama and vLLM adapters without lifecycle or download authority;
- identity-bound migration preflight, backup, lock, recovery, rollback and audit;
- reproducible wheel/sdist and isolated install smoke;
- desktop repository sync, data governance, offline evaluation, reliability,
  security and CPU/fixture certification tooling.

### Changed

- application and Operator version reporting now includes the Git build revision;
- independent capabilities, private corpus and worktree lifecycle from v6 remain
  compatibility requirements.

### Deferred

- all GPU and live-model quality gates:
  `BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.
