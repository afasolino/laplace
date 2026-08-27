# Step 3.3 — Aider RepoMap decision

Laplace baseline: `25b2d7264e153e9214ade1cb3e98db81754c8e36`

Pinned Aider revision: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`

Decision: **KEEP_LAPLACE**

## Evidence

Step 3.3 evaluated Aider RepoMap as a candidate replacement or shadow provider
for Laplace repository-context selection.

The initial 12-task screening returned `NOT_PROMISING`.

Because that first screen was dominated by focused/local Python navigation, the
experiment was expanded before making an architectural decision.

The expanded screen used:

- 30 frozen tasks;
- 6 categories:
  - direct symbol;
  - cross-file;
  - repository orientation;
  - ambiguous concepts;
  - change impact;
  - governance;
- both `FOCUSED` and `NO_FOCUS` tasks;
- 4 token budgets: 256, 512, 1000, 2000;
- 120 task/budget samples per provider;
- one warm-up plus five measured map calls for each sample;
- identical immutable Git snapshots for both providers;
- an isolated Aider virtualenv;
- exact Aider source provenance;
- `grep-ast==0.9.0` provenance;
- path recall;
- path precision;
- term recall;
- expected-symbol recall;
- relevance score;
- relevance per 1k context tokens;
- median map-generation time.

The expanded result was:

- assessment: `NOT_PROMISING`;
- overall_promising: `false`;
- advantage_strata: `[]`.

No meaningful discovery stratum justified escalation to paired agent-level
Phase B.

## Aggregate result

Across all 120 expanded task/budget samples:

| Metric | Laplace | Aider |
| --- | ---: | ---: |
| Mean context tokens | 928.5333 | 918.4667 |
| Median map time | 0.040739 s | 0.239363 s |
| Path precision | 0.155818 | 0.051360 |
| Path recall | 0.351389 | 0.259028 |
| Term recall | 0.529861 | 0.295833 |
| Symbol recall | 0.380682 | 0.198864 |
| Relevance score | 0.346221 | 0.196365 |
| Relevance / 1k tokens | 0.641073 | 0.316925 |

Aider's small average context-size reduction does not compensate for the
quality regressions and slower map generation.

## Discovery strata

The candidate did not qualify as promising in:

- `NO_FOCUS`;
- `cross_file`;
- `orientation`;
- `ambiguous`;
- `change_impact`.

Those strata were added specifically to test the repository-discovery and
graph-ranking use cases where Aider RepoMap had the strongest plausible
architectural advantage.

## Architectural conclusion

Laplace keeps its existing repository-context implementation.

Aider RepoMap is **not** integrated into the production agent path.

The old `0002-v33-agent-shadow-selector` patch is obsolete and must not be
applied. In addition to being unnecessary after the A/B result, it predates the
Step-3.2 transient mutation-authority contract and must not be used as an
integration shortcut.

Step 3.3 therefore concludes with:

`KEEP_LAPLACE`

and no Aider runtime dependency is added to Laplace.

The Aider checkout and benchmark virtualenv remain repo-local experimental
artifacts under `.runtime/v33-aider/` and are not production dependencies.

## Evidence files

The runtime evidence used for the decision is intentionally not committed as
production source:

- `.runtime/v33-aider/results/phase-a-repomap.json`
- `.runtime/v33-aider/results/phase-a-expanded-repomap.json`

The tracked benchmark manifests, harnesses, tests and methodology documents
make the experiment reproducible.
