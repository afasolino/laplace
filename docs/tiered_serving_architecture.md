# Tiered serving architecture

Laplace treats authorization capability and inference quality as independent axes.
`CapabilityTier` controls which service methods exist for a user; `ModelLane` controls
which local model route and scheduler priority serves an otherwise authorized request.
Neither enum is derived from the other.

## Request path

1. A bearer credential resolves to a server-owned `user_id`, operator role, and claimed
   capability.
2. The current capability is re-read from SQLite. A disabled or changed credential
   fails closed.
3. Basic and Plus chat enter the same scheduler, but the backend is always called with
   an empty tool tuple. No dormant tool schema reaches the model.
4. A Plus agent call must name an already-bound session. The binding contains the
   server registry's repository root, detached base commit, dedicated worktree, grant
   revision, tool policy, no-network policy, environment allowlist, and quotas.
5. Quality, standard, and economy routes are selected only after capability checks.
   Economy SystemVerilog uses CodeV; every other economy domain uses the main model
   with economy limits, so CodeV is never a general-purpose fallback.
6. A deterministic validation failure may cause one lower-lane retry on quality.
   Quality never silently downgrades and no path retries more than once.

The in-process priority scheduler preserves one or two quality slots, reports active
and waiting counts, and applies the configured standard/economy capacity. vLLM
priority scheduling is separately enabled in the selected serving profile. These are
complementary controls: admission protects capacity before a request reaches vLLM,
while vLLM orders admitted sequences.

The production HTTP handlers dispatch blocking local-model and Git operations to the
server thread pool. This is necessary for Laplace-side admission to observe real
concurrency; running the synchronous model client on the ASGI event loop would
serialize requests before they reached the scheduler.

## Serving profiles

P0 is the unmodified-KV baseline. P1 adds FP8 KV and chunked prefill. P2 and P3
selectively offload the exact `experts` parameter segment through UVA at 4 and 8 GiB.
P4 combines the best candidate with FP8 KV, prefix caching, chunked prefill, and
priority scheduling at a 64k limit. P5 adds 8 GiB of native KV offload. Every option is
matched against the installed `vllm serve --help=all` output before launch. The
resolved command, installed-help hash, and resolution hash are immutable evidence.
`configs/selected_serving_profiles.json` is the deployment boundary: it selects P1
for default quality/standard traffic, retains P4 as the explicit 64k option, records
the measured scheduler limits, and keeps the CodeV route separate.

The implementation follows the installed vLLM 0.25.0 CLI, not documentation alone.
Primary references are the [vLLM serve CLI](https://docs.vllm.ai/en/latest/cli/serve/),
[vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/),
and [vLLM offload configuration](https://docs.vllm.ai/en/stable/api/vllm/config/offload/).

## Measurement boundary

Mutable chat history, exploratory research memory, GUI state, and RAG stores are not
inputs to profile resolution or benchmarks. Live quality cases and load requests come
from versioned fixture manifests. Numerical aggregation is performed by deterministic
Python code. The frozen execution, Research, and Operator planes remain separate.
