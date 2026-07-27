# Tiered serving experiment methodology

## Reproducibility

The resolver captures the installed vLLM version and full CLI help, hashes the help,
validates every required flag, loads strict JSON profiles, verifies local model paths,
and emits exact argv plus a stable resolution SHA-256. Unsupported combinations fail
before a GPU process is created.

Each live profile runs alone. Admission requires enough currently free VRAM for the
profile's configured vLLM budget plus 2 GiB residual. The runtime records the child
PID, process group, `/proc` start ticks, command hash, resolution hash, and log. It
accepts readiness only when `/v1/models` returns the expected ID.

## Workloads

The versioned quality manifest covers general grounding, Python, SystemVerilog, inline
RAG citation location, contradictory research synthesis, and strict JSON. Scoring is
marker- and schema-based. Fabricated location, malformed JSON, silent length
termination, and a failed verification represented as success are hard failures.

Load runs use 20% quality, 60% standard, and 20% economy priority classes at
concurrency 1, 2, 4, 8, and 12. Context probes target 2k, 8k, 16k, 32k, and 64k when
the profile declares the capacity. Each probe puts unique markers near the beginning,
middle, and end and measures actual prompt tokens reported by the server.

Per-request evidence includes status, capability, requested lane, domain, context
target, concurrency, TTFT, end-to-end latency, inter-token latency, input/output
tokens, output throughput, and marker recall. GPU memory, utilization, power, PIDs,
Prometheus snapshots, PCIe topology, and a PCIe traffic sample are retained.

Batch throughput is total output tokens divided by wall time of the highest-concurrency
load arm. Executor wait plus request end-to-end time defines the batch wall clock.
Summing per-request token rates is explicitly not used for selection.

## Selection

P0 defines the observed quality baseline. A quality profile must pass every hard gate
and score at least 99% of P0. Among eligible profiles, the default must increase active
sequence capacity and have p95 no worse than P0; corrected highest-concurrency batch
throughput then selects the point. This chooses P1. The fastest eligible maximum-context
point is separately retained as P4. Standard and economy request routing never changes
Basic or Plus capability. Unsupported, startup-failed, OOM, or incomplete profiles
are excluded rather than assigned estimates.

The initial audit observed an RTX A6000 with 49,140 MiB, driver 550.135, local vLLM
0.25.0, a 19 GiB Qwen MoE artifact, and a 5.2 GiB CodeV artifact. Runtime preflight
rejected the generic CUDA 13 environment against this CUDA 12.4 driver, then selected
the existing `.venv-vllm-cu129` environment already used by the frozen serial launcher;
that environment reports CUDA available on the A6000 and exposes all required vLLM
0.25.0 offload flags. The Qwen weight index
contains names such as
`model.language_model.layers.0.mlp.experts.0.down_proj.weight_packed`, which is the
local evidence for the exact selective `experts` segment.

The real API smoke starts the historically frozen main+CodeV pair only for the
integration window, drives 60 requests from 12 authenticated users with an exact
20/60/20 mix, polls the live scheduler, exercises Plus worktree lifecycle and
cross-user denials, and submits one Basic request through Playwright. The lifecycle
wrapper records launcher and descendant ownership and releases the pair in `finally`.
