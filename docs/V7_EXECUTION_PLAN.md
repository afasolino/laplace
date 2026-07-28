# v7 execution plan

The implementation is incremental and remains on
`feature/architecture-release-hardening-v7`.

## Reviewable commit sequence

1. **Audit and architecture records** — this audit, execution plan, terminology,
   operating-mode and decision records.
2. **Provider and configuration contracts** — versioned neutral schemas, protocols,
   fixture providers, Ollama/vLLM adapters, strict configuration merging and
   diagnostics.
3. **Versioning, packaging and migrations** — semantic version/build metadata,
   state inventory, safe fixture migration engine, lock and reproducible build
   inputs.
4. **CI and desktop sync** — least-privilege pinned workflows plus a safe,
   confirm-before-transfer Git synchronization protocol and fixture transport.
5. **Governance and recovery** — accounting, retention/purge plans, backup
   manifests, validation and CLI scripts over fixtures.
6. **Evaluation and reliability** — frozen offline cases, deterministic provider
   scoring, bounded CPU soak and failure matrix.
7. **Security and documentation** — adversarial tests, threat model, residual
   risks, cross-document command/link validation and user/admin guidance.
8. **Certification** — one non-GPU release-gate runner, manifest verification,
   secret/path scan, final evidence archive and post-commit rerun.

## Gate policy

Each step receives unit and integration coverage before its commit. The final runner
executes compileall, full pytest including browser fixtures, Ruff, strict mypy,
Bandit, migration tests, package build/install smoke, offline evaluation, CPU soak,
failure testing, security tests, documentation checks, dependency/license inventory,
secret scanning, whitespace checks and manifest verification.

GPU and live-model checks are never executed in this task. Their result is the
expected non-failing status:

```text
BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE
```

No production state is copied, opened or migrated. No branch is pushed or merged.

