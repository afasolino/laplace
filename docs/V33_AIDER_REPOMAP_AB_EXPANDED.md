# Step 3.3 — Expanded Aider RepoMap Phase-A screening

Laplace baseline: `25b2d7264e153e9214ade1cb3e98db81754c8e36`

Aider pin: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`

The original 12-task Phase-A screen completed successfully and returned
`NOT_PROMISING`.  It is retained as evidence and is **not** overwritten.

This expanded screen addresses the principal validity concern in that first
experiment: the initial tasks were dominated by focused/local Python lookups.

## Expanded matrix

Thirty frozen tasks are stratified into:

- direct symbol lookup;
- cross-file dependency/navigation;
- whole-repository orientation;
- ambiguous concepts that occur in multiple modules;
- change-impact identification;
- governance/security navigation.

The matrix contains both `FOCUSED` and `NO_FOCUS` tasks.  No-focus tasks are
important because Aider RepoMap's graph/tree-sitter ranking is intended to
orient a model within a repository without already knowing the target file.

Every task runs at four context budgets:

- 256;
- 512;
- 1000;
- 2000 deterministic char/4 tokens.

That yields 120 task/budget samples per provider.

## Quality metrics

The benchmark records:

- path recall;
- path precision over repository paths actually emitted;
- term recall;
- expected-symbol recall when a frozen symbol set is available;
- aggregate relevance score;
- relevance per 1k context tokens;
- context tokens.

The same deterministic char/4 token estimator is used for both providers.

## Timing

For each task/budget/provider pair:

1. one warm-up call is excluded;
2. five map calls are measured;
3. median and p95 map time are recorded.

For Aider, the warm-up and all five measured calls execute in **one isolated
candidate process**, matching the long-running library integration that Phase B
would use.  Candidate-process startup is recorded separately and excluded from
algorithmic map time.

This distinction matters because the pinned Aider implementation has a
cross-process serialization-order variation in the untagged-file tail: it
collects remaining filenames in a Python `set` and later iterates that set.
Different Python hash seeds can therefore reorder that tail between fresh
processes even though the ranked definitions are sorted.  Such cross-process
ordering is recorded as an upstream reproducibility observation; it is not
misclassified as per-call nondeterminism.

Within one candidate process, identical inputs must still produce byte-stable
maps across the warm-up and five forced-refresh measurements.  Failure of that
stronger same-process invariant aborts the experiment.

The normal Aider tag cache is allowed after warm-up because it is part of the
upstream implementation.

## Snapshot and runtime fairness

Both providers receive independently materialized copies of the same immutable
Git tree at `25b2d7264e153e9214ade1cb3e98db81754c8e36`.

The Aider candidate must execute with the lexical venv launcher under
`.runtime/v33-aider/aider-venv`; dereferencing the venv Python symlink is
forbidden and regression-tested.

Aider must resolve from the pinned checkout and `grep-ast` distribution version
must be exactly `0.9.0`.

## Decision rule

This remains Phase A and cannot return `ADOPT`.

`PROMISING` requires either:

- overall quality non-inferiority within the 5% margin plus a material
  efficiency gain; or
- the same quality/efficiency condition in a meaningful discovery stratum,
  including cross-file, orientation, ambiguous, change-impact, or the complete
  no-focus subset.

If neither condition is met, the expanded result is `NOT_PROMISING` and the
recommended action is to document `KEEP_LAPLACE` for RepoMap rather than build
an Aider shadow adapter.

If a meaningful subdomain is `PROMISING`, only that identified subdomain moves
to paired agent-level Phase B.

The old `0002-v33-agent-shadow-selector` patch remains prohibited because it
predates the Step-3.2 transient mutation-authority contract.
