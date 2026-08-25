# C10 — Final Integrated Corrective Certification

## Rule

No new features in this phase. Fix only defects uncovered by the integrated campaign.

## Static gates

Run:
- full relevant pytest suite;
- Ruff;
- mypy;
- `git diff --check`;
- security-focused tests;
- packaging/import smoke;
- CLI help/smoke.

## Mandatory integration scenarios

1. Standalone core without Zetsu.
2. Zetsu adapter using the same core services.
3. `laplace zetsu start --nocodev`:
   - READY;
   - Qwen quality/standard usable;
   - CodeV explicitly disabled;
   - no degraded false failure.
4. Full runtime:
   - Qwen + CodeV + Operator;
   - READY.
5. `laplace zetsu stop`:
   - all owned model workers terminated;
   - no owned GPU-process leak;
   - foreign processes untouched.
6. Sequential repository-agent tasks well beyond historical worktree quota:
   - no stale worktree accumulation.
7. Concurrent saturation:
   - excess request queues;
   - no immediate false quota/GPU failure;
   - queued request executes after slot release.
8. Real-concurrent admission:
   - reservation accounting prevents over-admission;
   - no OOM in certified configuration.
9. Oversized repository-agent result:
   - SUCCESS;
   - exact paged retrieval.
10. Oversized logical-subagent result:
    - same semantics as repository agent.
11. Large-file ACI read/write continuation.
12. Memory schema migration/newer-version fail-closed.
13. Repository-intelligence complex-language fixtures.
14. Hook timeout/cancellation semantics.
15. Idle consolidation remains shadow/human-governed.
16. Skills cannot self-promote without required governance.
17. Rollback path remains intact.
18. Production checkout remains untouched throughout v2 certification.

## Performance/regression evidence

Compare against the pre-corrective baseline where meaningful:
- Qwen throughput;
- queue latency;
- repository-agent wall time;
- token usage;
- memory overhead;
- worktree count;
- GPU headroom;
- large-result caller-context size.

Do not require every metric to improve. Require no unexplained material regression.

## Final verdict

Emit exactly one:

- `NOT_READY`
- `RESEARCH_READY`
- `CONTROLLED_SINGLE_USER_PRODUCTION_READY`
- `GENERAL_PRODUCTION_READY`

Do not use `GENERAL_PRODUCTION_READY` unless multi-user/concurrency/security claims are actually covered by evidence.

Create:
`docs/v2-corrective-roadmap/certifications/FINAL.json`

Then:
- review final diff/status;
- commit;
- push `feature/laplace-v2`;
- provide a concise final report listing resolved defects, remaining limitations, exact commit, and promotion recommendation.
