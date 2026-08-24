# G1 — Shared Standalone Laplace Core

Refactor only where necessary so standalone Laplace and Zetsu call the same core services.

## Target

Shared Laplace Core must own retrieval/Personal Corpus access, model routing, Qwen agent execution, RTL-specialist execution, deterministic verification and project/task state. Zetsu remains an adapter, not a parallel implementation.

## Execute

1. Map current standalone and Zetsu call paths and identify real duplication/divergence.
2. Introduce the smallest shared service boundaries required.
3. Preserve authorization, routing, verifier semantics, budgets, cancellation, telemetry and error behavior.
4. With Zetsu completely disabled, prove standalone UI/API can perform retrieval, one bounded Qwen repository task, one RTL-specialist task and deterministic verification.
5. Run equivalent standalone and Zetsu tasks from the frozen G1 set.

## Gate

Require equivalent routing/authorization/verification behavior, no new privilege path, no standalone UI/API regression, no Zetsu-only dependency for core capabilities, and passing previous regression.

Certify and commit before G2.
