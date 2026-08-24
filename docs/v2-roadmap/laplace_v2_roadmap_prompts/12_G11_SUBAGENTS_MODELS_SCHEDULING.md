# G11 — Subagents, RTL Specialist Modernization and GPU-Aware Scheduling

Run only after the single-agent path is stable.

The mandatory core is safe GPU-aware scheduling. Model/harness alternatives are isolated A/B experiments and become production changes only if they pass their own gates.

## A. Logical subagents and scheduling

1. Inspect Hermes/OpenHands subagent patterns; reuse architecture, not unrestricted frameworks.
2. Start with one GPU-aware queue and logical subagents.
3. Compare serial execution, queued logical concurrency and real concurrent serving where feasible.
4. Measure correctness, queue latency, throughput, VRAM, OOM/recovery, cancellation and fairness.
5. Enable real concurrency only if aggregate throughput improves without unacceptable correctness/latency/VRAM regression.

## B. RTL specialist A/B — mandatory experiment

Keep CodeV-R1-7B BF16 as control.

Primary candidate: `SiliconMind-V1.2-Qwen3-4B-T-2507` BF16. Verify exact checkpoint/config/tensors locally; do not trust Hub metadata blindly.

Use the same frozen 12–18 RTL tasks for both models:

- 4–6 spec→RTL;
- 4–6 repair/debug tasks with real simulator/synthesis feedback;
- 4–6 bounded repository-level RTL modifications where Qwen3.8 selects context and delegates.

Use identical prompts, context, generation/repair limits and deterministic Verilator/Icarus/Yosys verification. Allow initial generation plus at most two repairs.

Record first-attempt pass, pass-by-3, repair count, generated tokens, wall time, decode tok/s, peak/residual VRAM and Qwen3.8 co-residency stability.

Promote SiliconMind-4B only if it is at least as correct as CodeV and materially improves headroom or latency. Keep CodeV rollback.

Test SiliconMind 8B only if 4B is meaningfully worse or ambiguous. If V1.2 behaves unexpectedly, use published V1 4B as a diagnostic control.

## C. Optional serving/model experiments

These do not block G11 unless adopted.

### Unsloth Dynamic v3

A/B certified Qwen3.8 AWQ/vLLM versus `UD-Q4_K_XL` in an isolated backend. Test agent correctness, tool semantics, 128K behavior, VRAM, throughput, specialist co-residency and speculative-decoding availability. Consider Q6 only if Q4 quality is insufficient. Do not replace the certified path if vLLM/MTP/Zetsu semantics regress.

### OpenCodex

In an isolated Codex configuration, test Qwen3.8 through OpenCodex as a native Codex-model baseline. Do not modify production Codex/Zetsu config. Compare at least one frozen task across Codex alone, Codex+Zetsu/Qwen and Codex harness+local Qwen through OpenCodex. Use the result to decide whether generic agent-loop functionality should be reused/simplified.

### Nemotron/OpenCode/ACE-RTL

Use only as external benchmark/architecture references. A hosted comparison may be run for difficult orchestration or RTL if available, but v2 production must remain functional without hosted services.

## Gate

G11 core passes only with stable single-GPU scheduling, cancellation/recovery, no security regression and preserved frozen-suite correctness.

Any adopted model/backend replacement requires separate machine-readable A/B evidence and rollback.

Certify and commit before final integrated verification.
