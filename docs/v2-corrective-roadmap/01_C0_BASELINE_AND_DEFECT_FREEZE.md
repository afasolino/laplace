# C0 — Freeze the Corrective Baseline

## Goal

Establish an exact baseline of the current `feature/laplace-v2` branch and prove the audited defects still exist before changing code.

## Required inspection

Record the current commit, branch, remote, worktree list, dirty state, and existing G0-G11 certification artifacts.

Inspect the current implementations and tests for:

- `src/research_workspace/logical_subagents.py`
- `src/research_workspace/zetsu_agent.py`
- `src/research_workspace/agent_sandbox.py`
- `src/research_workspace/bounded_aci.py`
- `src/research_workspace/memory.py`
- `src/research_workspace/repository_context.py`
- `src/research_workspace/hooks.py`
- `src/research_workspace/laplace_core.py`
- `src/research_workspace/zetsu_mcp.py`
- `src/research_workspace/skills.py`
- `src/research_workspace/idle_consolidation.py`
- `pyproject.toml`
- `README.md`
- relevant G1-G11 tests.

## Confirm or falsify each audit item

1. Logical-subagent result size can cause terminal failure.
2. Temporary GPU scarcity can produce terminal `GPU_BLOCKED`.
3. Real-concurrent GPU admission lacks reservation accounting sufficient to prevent double-admission on the same observed free memory.
4. Completed logical-subagent outcomes are retained without a clear bound/durable externalization policy.
5. Batch execution shares a common deadline that unfairly reduces time available to later tasks.
6. Bounded ACI cannot continue large reads/writes through cursor/chunk semantics.
7. Memory initialization does not fail closed on a database schema newer than the code.
8. Repository intelligence for complex C/C++/SystemVerilog is not using a production-grade parser and depends on conservative parsing/regex.
9. Hook timeout does not terminate an already-running callback.
10. `LaplaceCore` depends directly on Zetsu-specific coordinator/private verification APIs.
11. CodeV-vs-SiliconMind mandatory evaluation remains unresolved or blocked.
12. README/package version/release surface is inconsistent with v2.

Do not assume the audit is correct. Verify from current source.

## Deliverable

Create `C0-baseline.json` containing:
- exact commit;
- confirmation/falsification for all 12 items;
- exact symbols and paths;
- existing tests covering each;
- missing tests;
- no code changes except the certification artifact.

Run:
- `git diff --check`
- a focused smoke subset proving the branch is testable before correction.

Commit and push C0 only if the baseline is reproducible.
