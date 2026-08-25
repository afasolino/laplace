# C6 — Make Hook Timeout Semantics Safe and Honest

## Defect

A thread-based timeout can return control while the callback continues executing.

## Required approach

First classify hook types by allowed side effects.

If hooks are restricted to trusted, cooperative, in-process callbacks:
- add an explicit cooperative cancellation/deadline context;
- require long-running hooks to check cancellation;
- document that hard termination is not guaranteed for arbitrary Python code;
- prevent security-critical state from assuming that timeout means physical termination.

If hooks may execute untrusted or durable side-effecting code:
- move those hooks to a killable process/isolation boundary.

Choose the minimum design consistent with current actual hook usage. Do not introduce process isolation if all hooks are internal/trusted and cooperative cancellation is sufficient.

## Required invariants

- pre-hook fail-closed behavior remains;
- post-hook observational behavior remains;
- a timed-out hook cannot later mutate authoritative state unless its contract explicitly permits and guards it;
- hook completion after caller timeout must be observable/diagnosable;
- no unbounded thread accumulation.

## Tests

Cover:
- normal hook;
- cooperative timeout;
- callback that ignores cancellation;
- repeated timeout does not leak workers;
- pre-hook failure policy;
- post-hook failure policy;
- no late authoritative mutation after timeout;
- shutdown with active hook.
