# Step 3.4 Decision — KEEP_LAPLACE

## Scope and provenance

Laplace immutable baseline:

`5d8e463511d84ab1a7ee87b0f717b665f91d3e0f`

Pinned SWE-agent reference:

`3ea751c087f32b16e039a2233dd6eefecef325d5`

Resident Phase-B quality model:

`laplace-quality-qwen38-mtp`

Phase-A evidence SHA-256:

`823a2c0d5fdfa759b6699612a8ded382a55107fa4f68cfa269522f1255116573`

Step 3.4 evaluated SWE-agent-inspired ACI interaction patterns behind the
existing governed Laplace typed ACI. The experiment did not import SWE-agent's
shell runtime or agent loop and did not modify production Operator/MCP routing.

The Step-3.2 mutation-authority invariant remained mandatory throughout:

```python
allow_mutation = (
    ctx.allow_mutation
    and "apply_patch" in ctx.binding.tool_policy.allowed_tools
)
```

## Phase A

The reproducible primitive-level Phase A was `PROMISING`.

All three measured candidate primitives advanced:

- `view`;
- `search`;
- `syntax_preflight`.

Governance remained safe. Phase A was not an adoption decision.

## Phase B

Phase B executed the frozen 12 paired real-agent tasks against the same resident
quality model for both arms. The only intended arm difference was the typed ACI
provider.

The complete result was:

`KEEP_LAPLACE`

Aggregate metrics:

| Metric | Laplace baseline | SWE-agent-inspired candidate | Candidate / baseline |
|---|---:|---:|---:|
| Correct rate | 0.7500 | 0.8333 | 1.1111 |
| Failure rate | 0.2500 | 0.1667 | 0.6667 |
| Verifier success | 0.2500 | 0.5000 | 2.0000 |
| Repository coverage | 0.8333 | 0.8333 | 1.0000 |
| Relevance | 0.7917 | 0.7500 | 0.9474 |
| Input tokens | 7798.5 | 13650.83 | 1.7504 |
| Output tokens | 129.17 | 183.08 | 1.4174 |
| Tool rounds | 3.4167 | 5.7500 | 1.6829 |
| Wall time | 7.9347 s | 12.3366 s | 1.5548 |

The candidate improved correctness, verifier success, and failure rate, but it
had **zero qualifying efficiency gains** and materially regressed every measured
efficiency dimension:

- input tokens: +75.0%;
- output tokens: +41.7%;
- tool rounds: +68.3%;
- wall time: +55.5%.

Those regressions violate the frozen Phase-B adoption rule.

## Mutation-task observation

Several mutation tasks failed because the resident agent selected
`create_text_file` for an already-existing target. Laplace's governed ACI
correctly rejected those attempts with `aci_create_target_invalid` (or the
coordinator-level equivalent). The benchmark records these as agent-task
failures; the safety rule is not weakened.

This observation does not justify changing `create_text_file` semantics in Step
3.4.

## Final architectural decision

**KEEP_LAPLACE**

Do not integrate the SWE-agent-inspired ACI shadow into the production/default
agent path.

In particular:

- do not add an environment selector to `zetsu_agent.py`;
- do not infer mutation authority from `apply_patch` policy alone;
- do not import SWE-agent's shell/runtime authority;
- do not retain any candidate production wiring;
- do not reinterpret the promising primitive Phase A as an adoption result.

The Phase-A and Phase-B benchmark manifests, harnesses, tests, and this decision
record are retained as reproducible architectural evidence.

Step 3.4 can be promoted to GREEN only after the final deterministic gate
passes, the exact intended tracked scope is committed and pushed cleanly, and
the immutable online commit is inspected.
