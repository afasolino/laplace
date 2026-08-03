# Laplace v8 release-candidate review

## Scope and baseline

This review starts at certified v7 commit
`a2b0bdf17445012114bbdee8fb3a30a9b4c73680` and is performed only on
`feature/release-candidate-review-v8`. The adjacent stable checkout, production
state, user corpora, model configuration, and unrelated processes are read-only.
No merge, tag, or release is part of this review.

The independent review covers architecture protocols and fixtures, provider
abstraction, deterministic configuration, migrations, CI, packaging, storage
governance, offline evaluation, reliability, desktop synchronization, operator
GUI behavior, personal corpus ingestion, Agent worktrees, security, and the
conditional live-GPU runner.

## Review method

Claims require either source evidence plus a reproducer, a deterministic fixture
test, or an explicitly classified environment limitation. A finding is fixed only
after reproduction. Every fix has a regression test and a dedicated commit in
[DEFECT_REGISTER_V8.md](DEFECT_REGISTER_V8.md).

The release review deliberately distinguishes these boundaries:

- production-capable local interfaces and fixture-certified implementations;
- reference-only transport or provider implementations;
- staged external integration requiring native CI or a deployment endpoint;
- live-model behavior, which is certified only by the guarded GPU gate.

## Independent results

- Architecture and provider boundaries remain local-only and inject model,
  embedding, retrieval, corpus, conversation, artifact, provenance, identity,
  capability, repository, worktree, job, audit, and configuration services.
- Configuration is strict, layered, provenance-aware, path-validated, and
  redacted in diagnostics.
- Copied v5/v6/v7 fixture state rehearses all 11 persistent-store classes without
  touching production state.
- Package construction is byte-reproducible and now exercises the installed
  console entry point in a dependency-empty environment.
- Desktop patch apply now requires exact base commit, branch, clean target,
  changed paths, size, and SHA-256.
- GPU observation and SpecDec arbitration now fail closed when either GPU or
  compute-process ownership evidence is incomplete.
- Clean-clone tests no longer consume ignored models, historical GPU probes, a
  repository-local virtual environment, or optional EDA tools.
- Portable execution-record locks use the host-native primitive, and corpus
  parser isolation imports without POSIX-only modules on Windows.
- The guarded live gate now requires real SSE evidence, personal-corpus
  retrieval/citations, model-backed Python and SystemVerilog patches with
  deterministic verification, isolation/capability checks, cancellation,
  provider-failure behavior, and hash-bound screenshot evidence.
- SSH/HTTPS sync transports remain staged policy boundaries; the fixture
  transport is a reference implementation, not a production transport.
- Current vulnerability-database lookup and Windows-native execution remain
  environment-dependent and must not be reported as locally executed.

Final certification evidence is produced by
`scripts/run_release_candidate_v8_certification.py`. The final decision is made
only against [GO_NO_GO_CRITERIA_V8.md](GO_NO_GO_CRITERIA_V8.md).
