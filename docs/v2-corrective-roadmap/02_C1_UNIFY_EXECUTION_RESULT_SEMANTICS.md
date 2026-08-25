# C1 — Unify Result Delivery and Execution Semantics

## Defect

The repository-agent path already has durable/compact result handling, while logical subagents may still convert a successful execution into failure when the returned object is too large.

## Required behavior

Execution status and delivery status must be separate.

For all repository-agent and logical-subagent paths:

- successful execution remains successful regardless of presentation size;
- authoritative results are persisted before compact handoff where practical;
- normal caller response stays bounded;
- oversized data is retrieved by stable result identity and deterministic cursor/chunk paging;
- retrieval does not rerun the underlying task;
- owner/repository/session authorization is enforced;
- exact patch/log/result bytes can be reconstructed;
- UTF-8 and structured payload boundaries are preserved;
- no arbitrary path supplied by the model is accepted for result retrieval.

Reuse the existing durable result/evidence mechanisms. Do not create a second incompatible paging protocol unless the existing one cannot satisfy the requirements.

## Logical subagent changes

Remove any behavior equivalent to `subagent_result_too_large -> FAILED`.

A logical-subagent success envelope must contain enough information to retrieve omitted content, using current project naming conventions.

Bound in-memory completed-result retention. Prefer durable result storage plus a small bounded cache. Define explicit retention/eviction behavior.

## Tests

Add deterministic tests for:

- below-limit result unchanged;
- above-limit logical-subagent result remains SUCCESS;
- exact reconstruction over multiple pages;
- oversized patch;
- oversized verifier/log payload;
- repeat retrieval does not rerun work;
- authorization isolation;
- malformed/invalid cursor;
- UTF-8 boundary correctness;
- cache eviction does not destroy durable result;
- result-store restart/recovery.

Run relevant agent, Zetsu, logical-subagent, security, and checkpoint regressions.

Do not change scheduling behavior in this phase except where required to support result persistence.
