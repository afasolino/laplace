# Step 3.4 Phase B — paired real-agent A/B

Immutable Laplace baseline: `5d8e463511d84ab1a7ee87b0f717b665f91d3e0f`.

Pinned SWE-agent reference: `3ea751c087f32b16e039a2233dd6eefecef325d5`.

Phase A returned `PROMISING` with governance preserved and all three measured
candidate primitives promoted to Phase B:

- `view`;
- `search`;
- `syntax_preflight`.

Phase B is required before any `ADOPT` decision.

## Isolation

This patch is evaluation-only.

It does not modify `zetsu_agent.py`, production Operator/MCP wiring, serving
profiles, authorization, verification policy, or the production checkout.

Both arms instantiate the current real `ZetsuAgentCoordinator` engine and real
Laplace authorization/sandbox/tiered-serving substrate. The candidate arm
changes only the typed ACI class. The mutation authority rule remains:

```python
allow_mutation = (
    ctx.allow_mutation
    and "apply_patch" in ctx.binding.tool_policy.allowed_tools
)
```

No environment selector is added.

## Matrix

Twelve frozen paired tasks are used:

- 4 long-file/view tasks;
- 4 repository-search tasks;
- 4 governed Python mutation tasks.

Each arm receives an independently generated Git repository with byte-identical
fixture content and deterministic commit metadata. Pair scoring refuses fixture
hash differences.

Arm order alternates per task to reduce systematic warm-cache/order bias.

## Real model/runtime

The runner uses the currently selected production quality lane and
`LocalOpenAIChatBackend`. The selected endpoint must be loopback.

The same model ID must be reported by every baseline and candidate task.

No Codex call is involved.

## Exact telemetry

The scorer requires the coordinator's model-reported per-request telemetry:

- `qwen_input_tokens`;
- `qwen_output_tokens`;
- `qwen_calls`;
- `qwen_usage_reported_calls`;
- `qwen_token_usage_source == "model_reported_per_request"`;
- `qwen_token_usage_complete == true`;
- `tool_calls`.

Missing/incomplete usage makes the decision `BLOCKED`; no char/4 estimate is
substituted in Phase B.

The runner additionally measures elapsed wall time, an independent rerun of
the exact task verifier on the preserved worktree, expected repository-path
coverage, path relevance, model result correctness, and sandbox containment.
A mutation can only be scored correct when the real coordinator reaches
`SUCCESS`, the file oracle matches, and that independent verifier passes.

## Decision

Phase B requires all 12 pairs; the general minimum remains >=10.

`ADOPT` requires:

- complete identical-model telemetry;
- identical fixture hashes per pair;
- no pairwise correctness/failure/verifier/security regression;
- aggregate correctness, failure, verifier success, coverage and relevance
  non-regression (5% tolerance only for coverage/relevance);
- no input-token, output-token, tool-round, or wall-time regression >5%;
- either an aggregate correctness/verifier gain or at least two efficiency
  metrics improving by >=5%.

Otherwise the valid terminal result is `KEEP_LAPLACE`.

Invalid pairing, fixture/model mismatch, or incomplete token telemetry is
`BLOCKED`.

Even `ADOPT` does not modify production automatically. A separate final
Step-3.4 adoption/consolidation patch must wire only the winning measured
primitives into the governed baseline ACI and then rerun certification.
