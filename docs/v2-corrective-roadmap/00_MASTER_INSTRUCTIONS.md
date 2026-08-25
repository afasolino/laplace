# Laplace v2 Corrective Roadmap — Master Instructions

You are revising `afasolino/laplace` on branch `feature/laplace-v2` in the dedicated development worktree:

`/home/giando/work/laplace-v2`

The stable production checkout is:

`/home/giando/work/laplace`

Do not modify the production checkout. Do not use `/tmp` for worktrees, generated studies, certification artifacts, or persistent outputs.

## Objective

Perform a surgical corrective pass over the current v2 implementation. Do not redesign the system wholesale. Preserve all behavior that is already correct and certified. Fix the concrete source-level gaps identified by the audit, then run an integrated certification campaign.

The corrective pass is complete only when the code, tests, documentation, and certification evidence agree on the same behavior.

## Operating rules

1. Read this file once.
2. Execute the phase prompts in numeric order.
3. For each phase, read only:
   - this master file;
   - the current phase prompt;
   - the previous phase certification;
   - repository files needed for that phase.
4. Do not preload later phase prompts.
5. Use Laplace/Zetsu for substantial repository reconnaissance and bounded implementation where the production control plane is healthy. Codex remains the orchestrator, deterministic verifier, and final reviewer.
6. Do not use CodeV unless the current phase explicitly requires RTL-specialist evaluation. The normal development runtime may stay in `--nocodev`.
7. Do not silently fall back from a failing Zetsu repository path to an unrelated execution path. Diagnose real infrastructure failures.
8. Avoid broad refactors. Change the minimum set of modules necessary to make behavior correct and internally consistent.
9. Reuse existing abstractions before introducing new ones.
10. Every new persistent schema, queue, result store, or lifecycle state must have explicit migration/recovery semantics.
11. Never weaken:
    - repository/owner isolation;
    - bounded ACI policy;
    - deterministic verification;
    - human approval requirements;
    - fail-closed handling of ambiguous ownership/state;
    - production rollback capability.

## Phase lifecycle

For every phase:

`inspect -> define exact defect -> implement minimally -> focused tests -> negative/security tests -> prior-phase regression -> diff review -> certification artifact -> commit -> push -> next phase`

Do not advance if a deterministic gate fails.

Each phase must produce:

`docs/v2-corrective-roadmap/certifications/<PHASE>.json`

with at least:

- phase;
- status: `FAILED | EXPERIMENTAL | SHADOW | CERTIFIED`;
- base_commit;
- resulting_commit;
- changed_paths;
- tests;
- negative_tests;
- regressions;
- unresolved_items;
- evidence;
- timestamp_utc.

## Git policy

Trusted remote:

`git@github.com:afasolino/laplace.git`

Normal non-force commits and pushes to `feature/laplace-v2` are pre-authorized.

Never:
- push directly to `production/v1`;
- push directly to `main`;
- force-push;
- rewrite published history.

## Global acceptance criteria

At the end of this roadmap:

- successful large work is never converted into failure only because of response size;
- temporary execution/GPU saturation queues instead of failing;
- logical subagents and repository agents use compatible lifecycle/result semantics;
- GPU admission is race-safe under real concurrency;
- completed scheduler/result state is bounded or durably externalized;
- per-task deadlines are fair and independent;
- ACI supports bounded continuation for large reads/writes;
- memory schema handling fails closed on unsupported versions;
- repository intelligence has an explicit supported-language contract and production-grade parser path for complex C/C++/RTL, or a documented reduced contract backed by tests;
- hook timeout semantics are honest and safe;
- the standalone core is not architecturally dependent on Zetsu-specific private APIs;
- the unresolved CodeV-vs-SiliconMind experiment is either completed or explicitly waived with evidence and no false promotion claim;
- release metadata/documentation describe the actual v2 system;
- the full integrated campaign passes without modifying production.
