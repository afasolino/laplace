# Changelog

All notable Laplace changes are documented here. Versions follow Semantic
Versioning while the public API remains pre-1.0.

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
