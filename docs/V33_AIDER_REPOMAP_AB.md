# Step 3.3 — Aider RepoMap A/B

Baseline Laplace revision: `25b2d7264e153e9214ade1cb3e98db81754c8e36`.

Pinned Aider revision: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`.

## Scope

Phase A compares the current Laplace advisory `RepositoryContextService` RepoMap
against Aider's ranked/tree-sitter RepoMap on 12 frozen repository-context
tasks.

This phase does **not** wire Aider into `ZetsuAgentCoordinator`, does not change
the bounded ACI, and cannot return `ADOPT`.

Both providers receive independently materialized copies of the same immutable
Git `HEAD` tree. Dirty or untracked development files therefore cannot bias one
provider.

The same deterministic `chars / 4` estimate is used for context-budget
reporting. Each task also records path recall, term recall and wall time.

## Phase-A outcome

`PROMISING` means only that a separately guarded Phase-B agent-level experiment
is justified. `NOT_PROMISING` means the map evidence does not justify adding an
agent shadow adapter.

## Phase B and final decision

Only Phase B may decide `ADOPT` or `KEEP_LAPLACE`. It must use at least 10 paired
tasks and record the full roadmap metrics:

- task correctness;
- context tokens;
- completion tokens;
- tool rounds;
- wall time;
- failure rate;
- verifier success;
- repository coverage;
- relevance.

Any future Aider shadow adapter must preserve the Step-3.2 invariant:

`allow_mutation = transient_turn_authority AND apply_patch_policy`

and must not infer mutation authority from tool policy alone.

No upstream source is vendored into Laplace. The pinned Aider checkout and its
virtual environment live only under `.runtime/v33-aider/`.
