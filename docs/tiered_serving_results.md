# Tiered serving results

Implementation and deterministic security evidence are maintained by
`tests/test_tiered_serving.py`. The suite covers profile resolution, tool-free Basic
chat, orthogonal lane routing, bounded escalation, CodeV domain restriction,
per-user worktrees, cross-user denial, immediate revocation, filesystem escape
attempts, mixed concurrency, quality reservation, responsive API policy, and
ownership-safe shutdown.

The authoritative measured run is
`outputs/tiered_serving_20260727T084923Z`. All six profiles completed without an OOM
or request failure, passed hard quality gates at the same deterministic score
(0.7778), and recalled every beginning/middle/end context marker.

| Profile | Max context | Max sequences | Corrected C12 batch output tok/s | p95 E2E ms | Model load GiB | KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| P0 baseline | 32k | 2 | 31.064 | 1447.596 | 18.21 | 519,606 |
| P1 FP8 KV | 32k | 8 | 106.819 | 580.931 | 18.21 | 1,847,080 |
| P2 expert UVA 4 GiB | 32k | 8 | 57.414 | 1049.918 | 14.10 | 1,203,989 |
| P3 expert UVA 8 GiB | 32k | 8 | 39.974 | 1539.753 | 10.23 | 1,382,809 |
| P4 priority/FP8/expert | 64k | 12 | 48.333 | 2367.482 | 10.23 | 2,323,078 |
| P5 P4 plus native KV | 64k | 12 | 46.996 | 2415.745 | 10.23 | 2,324,803 |

P1 is the default: it raises configured active sequences from 2 to 8 and more than
triples corrected concurrency-12 batch throughput while retaining quality. P4 is the
validated opt-in 64k profile. Selective expert offload reduced GPU model-load memory
but did not beat P1 throughput; native KV offload did not beat P4.

The real v4 API/GUI run passed 60/60 requests from 12 users with the exact 12
quality/36 standard/12 economy split. The main model served 48 and CodeV served the
12 SystemVerilog economy requests. The scheduler exposed up to 10 waiting requests;
mean quality wait was 0.020 s versus 0.508 s standard and 2.419 s economy. All lane
scores were 0.7778 with hard gates, Playwright passed at 390×844 without console
errors or horizontal overflow, and all repository isolation, denial, cancellation,
final-result, audit-schema, and no-leakage checks passed.

At implementation time both inherited endpoints were stopped and no Laplace process
was owned. A first probe observed an unrelated 21 GiB compute process and correctly
deferred admission. A later probe showed that process had exited independently,
47,362 MiB was free, and no compute process remained. No unrelated PID was signalled.
During final live integration an unrelated QAT process was observed once and admission
refused without signalling it. It exited independently. The passing lifecycle then
released only the two recorded Laplace launchers and their EngineCore descendants,
verified both endpoints down, preserved unrelated-process policy, and observed no
remaining GPU compute PID.
