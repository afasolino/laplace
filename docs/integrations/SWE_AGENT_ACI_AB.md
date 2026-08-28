# Step 3.4 — SWE-agent ACI A/B

Laplace immutable baseline: `5d8e463511d84ab1a7ee87b0f717b665f91d3e0f`.

SWE-agent reference: `SWE-agent/SWE-agent@3ea751c087f32b16e039a2233dd6eefecef325d5`.

## Reviewed upstream scope

SWE-agent documents four ACI ideas that are relevant here:

- reject syntactically invalid edits with lint/preflight feedback;
- show roughly 100 lines in a file-view observation;
- make repository search succinct by listing matching files;
- make successful empty output explicit.

Laplace evaluates those interaction patterns only. No SWE-agent shell, agent
loop, unrestricted command execution, filesystem authority or network-capable
tool is imported.

## Current Laplace integration boundary

The candidate subclasses `BoundedRepositoryACI`, and an isolated
`SweAgentABCoordinator` overrides only `_typed_aci()`.

Production `ZetsuAgentCoordinator`, Operator, MCP routing, repository
authorization, worktree ownership, mutation accounting, verification and
cancellation are unchanged.

The candidate service is composed through the current neutral
`LaplaceCore(..., repository_agent_service=...)` injection and then supplied to
`ZetsuService`.

The Step-3.2 mutation invariant is mandatory:

```python
allow_mutation = (
    ctx.allow_mutation
    and "apply_patch" in ctx.binding.tool_policy.allowed_tools
)
```

Tool-policy membership alone never grants mutation authority.

## Phase A: reproducible primitive screen

The initial patch does **not** ask an operator to invent JSONL metrics manually.

The tracked 12-task manifest drives a deterministic benchmark over independently
created repo-local Git fixtures. Each baseline/candidate task is repeated five
times.

The matrix contains:

- 3 file-view tasks;
- 4 search tasks, including an explicit no-match case;
- 2 Python edit tasks (valid and syntactically invalid);
- 3 governance tasks (mutation denial, path escape, `.git` denial).

Measured objectively:

- correctness;
- deterministic char/4 observation tokens;
- median primitive wall time;
- harness failure rate;
- deterministic Python/diff verifier success;
- expected repository-path coverage;
- expected-term relevance;
- preservation of security guards.

Phase A can return only:

- `PROMISING`;
- `NOT_PROMISING`;
- `BLOCKED`.

It can never return `ADOPT`.

A primitive is `PROMISING` only if quality/security do not regress and it shows
a measurable quality, token, or timing gain. Governance is a mandatory gate,
never an optimization target.

## Phase B rule

Only if Phase A is `PROMISING` do the named winning primitives advance to a
separate >=10 paired **agent-level** A/B using the resident model/runtime.

Phase B must measure the roadmap metrics that exist only at agent level:

- answer/task correctness;
- model-reported context/input tokens;
- model-reported completion/output tokens;
- tool rounds;
- elapsed time;
- failure rate;
- latest-mutation verifier success;
- repository coverage;
- relevance.

No production wiring is allowed before Phase B. If Phase A is
`NOT_PROMISING`, record `KEEP_LAPLACE` and do not spend live-model budget.

## Why this supersedes the earlier r3

The earlier draft had a 12-task scorer but no reproducible producer for its
paired JSONL records, leaving coverage/relevance and other values dependent on
manual measurement. This revision makes Phase A fully executable and reserves
model-only metrics for the proper agent-level Phase B.
