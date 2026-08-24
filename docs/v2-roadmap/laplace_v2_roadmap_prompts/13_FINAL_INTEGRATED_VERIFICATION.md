# Final Integrated Verification Campaign

Freeze the v2 candidate after G0–G11. Add no features during this campaign; only repair demonstrated defects.

## 1. Standalone Laplace — Zetsu disabled

Verify UI/API, episodic/semantic/project memory, Personal Corpus/RAG, RepoMap, Qwen agent, production RTL specialist plus rollback specialist, checkpoint/resume, >128K logical history with compaction, project/global/path rules, event replay/recovery, approved skills, idle consolidation in its approved lifecycle state, scheduling and rollback to certified v1.

## 2. Zetsu/Codex

Verify retrieval, Qwen/RTL delegation, blocking timeout/wait behavior without short model-mediated polling, cancellation/orphan prevention, deterministic verification, compact handoffs, exact Codex versus local-model token accounting and one controlled task where Codex uses certified Laplace to modify Laplace v2 itself.

## 3. Reliability and security

Re-test owner/project isolation, memory contradiction/update/delete, corrupt storage recovery, traversal/symlink/`.git`/network restrictions, malformed verifier, verifier-induced mutation, cancellation/timeout, concurrency/races, worktree drift and malformed checkpoint/event state.

Inject failures during memory/event/checkpoint writes, compaction, mutation, verification, restart and idle maintenance. Require explicit recovery or fail-closed behavior.

## 4. Performance and endurance versus certified v1

Compare frozen workloads for task correctness, Qwen throughput, RTL-specialist throughput/correctness, MTP if enabled, memory/RAG latency, RepoMap cost, ContextPlanner overhead, Codex-token savings, GPU/VRAM and repeated task/session/maintenance stability including process, VRAM, locks, file descriptors and disk growth.

Run the frozen final evaluation suite exactly once after freeze and report every failure.

## 5. Final report

Report G0–G11 results/commits, upstream commits/licenses/reuse modes, standalone capabilities, memory/provenance, repository intelligence, agent reliability/security, rules/skills/events/hooks/consolidation, RTL A/B and final routing, scheduling, optional OpenCodex/Unsloth/Nemotron results if executed, Zetsu/Codex-token impact, performance/endurance versus v1, regressions/limitations and exact rollback.

Choose exactly one verdict:

- `NOT_READY`
- `RESEARCH_READY`
- `CONTROLLED_SINGLE_USER_PRODUCTION_READY`
- `GENERAL_PRODUCTION_READY`

Base the verdict only on executed acceptance, security, persistence, endurance and integrated verification evidence.
