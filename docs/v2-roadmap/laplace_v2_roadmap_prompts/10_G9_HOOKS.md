# G9 — Typed Lifecycle Hooks

Add internal lifecycle extensibility without arbitrary shell hooks.

## Execute

Use Cline hook boundaries as reference, but implement typed internal hooks only:

`TASK_START`, `TASK_RESUME`, `TASK_CANCEL`, `PRE/POST_RETRIEVAL`, `PRE/POST_MEMORY_WRITE`, `PRE/POST_MUTATION`, `PRE/POST_VERIFY`, `TASK_COMPLETE`, `TASK_FAILURE`, `IDLE_START`, `IDLE_END`.

Define ordering, timeout, exception, idempotency and cancellation semantics. Security-critical PRE hooks fail closed. Observability-only POST hooks follow an explicit non-security failure policy.

No arbitrary shell execution and no implicit privilege escalation.

## Gate

Test deterministic ordering, disabled hooks, duplicate/replayed events, timeout, exception, cancellation, restart/resume, idempotence, security-hook failure and owner/project isolation.

Certify and commit before G10.
