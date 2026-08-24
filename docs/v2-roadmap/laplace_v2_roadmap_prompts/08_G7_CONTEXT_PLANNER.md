# G7 — Shared ContextPlanner, Condensation and Compact Handoffs

Move context planning and compaction into Laplace Core.

## Execute

1. Inspect OpenHands condenser/context mechanisms; record exact commit/license.
2. Deterministically combine policy, objective, exact task state, project rules, relevant memory, RepoMap, Personal Corpus/RAG evidence, recent trajectory and semantic summaries.
3. Exact state, authorization and verification state must never depend on LLM summarization.
4. Preserve the current ~80% compaction concept unless measured evidence supports another threshold.
5. Support repeated compaction and resume across it.
6. A/B compact structured handoff versus current handoff. Caveman-style terseness is allowed only for narration/summary fields; never compress exact code, commands, policy, state, errors, provenance or verifier evidence.
7. Optionally evaluate task-aware reasoning/generation budgets in shadow mode after the deterministic planner is stable; promote only from measured gains.

## Gate

Use controlled histories exceeding 128K logical history. Verify correct trigger, preservation of objective/policy/exact state/provenance, bounded repeated compaction, restart/resume, no summary-induced privilege/verifier change and successful completion.

For compact handoffs require no correctness loss and measurable Codex-context/token reduction.

Certify and commit before G8.
