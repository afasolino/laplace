# Laplace v2 — Master Execution Instructions

You are Codex Terra with `xhigh` reasoning, continuing development of `afasolino/laplace` after the Qwen3.8/Zetsu v1 production certification has passed.

Read this file once. Then execute the phase prompts in numerical order, reading only the current phase prompt plus the previous phase certification/report and any repository documentation needed for that phase.

## Certified v1 baseline

Assume v1 already has:

- standalone Laplace UI/API;
- Qwen3.8 Quality/Standard;
- CodeV Economy/RTL;
- Qwen3.6 rollback;
- 131072 context;
- Personal Corpus/RAG;
- bounded Qwen repository agent;
- deterministic verification;
- checkpoint/resume;
- context compaction;
- cancellation, budgets and owner/session isolation;
- Zetsu MCP;
- Codex-token benchmarking;
- an executed MTP enable/disable decision;
- final production regression.

Treat the exact certified v1 commit, configuration and certification artifacts as authoritative.

## Operating model

Use the certified Laplace/Zetsu stack as the default local engineering substrate for substantial work, including Laplace development itself.

- Codex: architecture, decomposition, verifier selection, gate coordination, trivial Git/status operations, final review.
- Laplace/Zetsu: local retrieval, repository analysis, bounded inspect/edit/test loops, Qwen agent work, RTL-specialist work and compact evidence return.
- Do not repeatedly read large repository regions in Codex when Laplace can retrieve or summarize them locally.
- Prefer one blocking delegation for long work. Avoid model-mediated polling; if background execution is unavoidable, use the longest appropriate blocking wait and then exponential backoff.
- Keep handoffs compact: `status`, `changed_paths`, `verification`, `unresolved_failures`, `evidence`, `next_action`, `telemetry`.
- Never compress exact code, commands, policy, deterministic state, errors, provenance or verification evidence.

## Control-plane safety

Never modify the same live Laplace instance controlling development.

Keep the certified v1 instance available as the stable control plane. Develop v2 in a separate branch/worktree under the repository/project directory tree. Never use `/tmp` for worktrees, persistent outputs, caches intended for certification, or generated study artifacts.

Preserve immediate rollback to the previous certified state at every phase.

## Architecture

The invariant is:

**Laplace Core = complete standalone local system**

**Zetsu = optional optimized Codex ↔ Laplace adapter**

Any capability that is useful without Codex belongs in shared Laplace Core. Zetsu must not become the only implementation path for memory, retrieval, agent execution, repository intelligence, context planning, skills, events, hooks or scheduling.

## Reuse before reimplementation

Before implementing a mechanism from scratch, inspect the current upstream implementation, exact commit and license. Prefer a thin adapter, isolated reuse or architectural reference over copying an entire framework.

Candidates:

- `mem0ai/mem0`: episodic/semantic memory backend.
- existing Laplace `PersonalCorpusStore`: source-grounded document/literature RAG.
- `Aider-AI/aider`: RepoMap, symbols, relevance/dependency ranking, token-budgeted repository summaries.
- `xberg-io/tree-sitter-language-pack`: parser/code-intelligence support.
- `SWE-agent/mini-swe-agent` and SWE-agent ACI: simple replayable loops and bounded model-friendly repository tools.
- `OpenHands/software-agent-sdk`: events, trajectories, replay/resume, context condensation.
- `NousResearch/Hermes-Agent`: skills, memory consolidation, subagents, maintenance.
- `cline/cline`: global/project/conditional rules and lifecycle boundaries.
- `ai4s-research/open-science` / reusable scientific skills: provenance and research-skill reference where useful.
- `Yaxin9Luo/AutoDesign`: shadow meta-harness optimization reference for controlled self-improvement.
- `lidge-jun/opencodex`: isolated compatibility/baseline experiment only.
- SiliconMind V1/V1.2 RTL models: candidate replacement for CodeV after local A/B validation.
- Unsloth Dynamic v3 GGUF: optional post-stability model/backend experiment.
- Nemotron/OpenCode/ACE-RTL: external architectural or benchmark references, not required production dependencies.

For every reused mechanism record: upstream repository, immutable commit SHA, license, files/API used, reuse mode, local modifications and rollback/removal path.

## Feature lifecycle and gates

Every substantial feature moves:

`OFF → EXPERIMENTAL → SHADOW → CERTIFICATION → ENABLED`

Do not advance a phase after a failed deterministic test. Diagnose the failure; never retry until it happens to pass.

Every phase gate runs, in this order:

1. focused development tests;
2. frozen phase acceptance tests;
3. negative/security tests;
4. previous-phase regression;
5. final diff/status inspection;
6. machine-readable certification evidence.

Do not alter frozen tasks after observing results.

Use the repository's existing artifact conventions. If no convention exists, keep all v2 certification evidence inside the project tree and outside `/tmp`.

Each phase certification artifact must include at least:

- phase and lifecycle state;
- base and candidate commits;
- upstream dependencies/commits/licenses;
- executed tests and commands;
- deterministic pass/fail results;
- security/negative results;
- persistence/restart results where applicable;
- metrics;
- limitations;
- rollback target;
- exactly one phase status: `FAILED | EXPERIMENTAL | SHADOW | CERTIFIED`.

Commit a passing phase before advancing.

## Continuous frozen evaluation

Maintain one frozen v2 evaluation suite spanning memory, Personal Corpus/RAG, repository understanding, coding/debugging, long context, restart/resume, compaction, owner/project isolation, RTL, skills, consolidation and scheduling.

Measure where relevant: deterministic correctness, verifier result, model tokens, tool/model calls, files read, latency, throughput, GPU/VRAM and human intervention.

## Optimization policy

Do not optimize from intuition alone. Establish a working baseline first, then A/B one change at a time.

The following are experimental until measured:

- compact/Caveman-style handoff formatting;
- adaptive reasoning effort or task-specific generation budgets;
- chunked large-file generation;
- OpenCodex as a local-model Codex compatibility layer;
- Unsloth Dynamic v3 versus the certified AWQ serving path;
- replacing CodeV with SiliconMind;
- multi/subagent concurrency;
- AutoDesign-style harness optimization.

A production change must preserve deterministic correctness and security and show a material benefit on frozen tasks.

## Stop conditions

Stop the current phase and report rather than advancing if a security boundary is weakened, isolation is uncertain, deterministic verification cannot establish correctness, persistence/replay semantics are ambiguous, a license is incompatible, rollback/control-plane safety is threatened, or the phase requires unrelated infrastructure redesign.

## Final campaign

After G0–G11, freeze the candidate and execute `13_FINAL_INTEGRATED_VERIFICATION.md`. No new features are allowed during that campaign except defect fixes.

The intended outcome is a standalone, persistent local engineering/research system in Laplace Core, with Zetsu remaining an optional optimized bridge to Codex.
