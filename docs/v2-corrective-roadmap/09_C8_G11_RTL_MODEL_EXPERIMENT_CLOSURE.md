# C8 — Close the G11 RTL Specialist Experiment

## Goal

Resolve the outstanding CodeV-vs-SiliconMind evaluation without conflating infrastructure implementation with model promotion.

## Candidates

Baseline:
- current certified CodeV BF16 route.

Primary candidate:
- `SiliconMind-V1.2-Qwen3-4B-T-2507` BF16, if the artifact is accessible and compatible.

Diagnostic fallback:
- prior SiliconMind V1 4B only if V1.2 results are anomalous or artifact support prevents a fair comparison.

Do not test the 8B candidate unless 4B correctness is ambiguous or worse and additional capacity is justified.

## Frozen task set

Use a versioned, hashed 12-18 task set spanning:
- spec-to-RTL;
- deterministic repair/debug;
- bounded repository-level RTL modification.

Use identical prompts/context/repair budget for all candidates:
- initial attempt;
- at most two repair rounds;
- real Verilator/Icarus/Yosys feedback as appropriate.

## Metrics

Record:
- first-pass correctness;
- pass-by-3;
- verifier outcomes;
- total model tokens;
- repair count;
- wall time;
- throughput;
- peak VRAM;
- co-residency/headroom with Qwen;
- failure modes.

## Promotion rule

Promote SiliconMind 4B only if it is at least as correct as CodeV on the frozen campaign and materially improves latency/headroom or another predefined production metric.

If the candidate cannot be fairly executed because of artifact/toolchain incompatibility, mark the experiment `BLOCKED` with exact evidence. Do not falsely claim completion or promotion.

This phase may temporarily require full runtime with CodeV. Restore `--nocodev` after testing if CodeV is not otherwise needed.

Do not change the production model route in this phase without a separate explicit human promotion decision. The phase closes the evidence gap; it does not silently promote.
