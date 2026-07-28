# Offline evaluation methodology

The v7 harness evaluates contracts without a network, model server or GPU. It loads a
strict frozen JSON suite and synthetic Python/SystemVerilog repositories. Cases cover
retrieval relevance, citation precision and recall, unsupported claims, corpus
permissions, conversation persistence, safe Markdown/citation presentation, patch
applicability, verification completeness, worktree isolation, provider-capability
routing, cancellation/timeouts and artifact provenance.

Run it from the repository:

```bash
PYTHONPATH=src python scripts/run_offline_evaluation.py \
  --output /tmp/laplace-offline-evaluation.json
```

The report deliberately separates:

- `infrastructure_correctness`: schema, coverage and isolation of the harness;
- `fixture_task_quality`: computed outcomes for deterministic known-answer cases;
- `live_model_quality`: deferred with
  `BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.

Fixture scores are not estimates of live-model accuracy. No production corpus,
repository or user state is sampled. A suite update changes the suite ID, preserves
old fixtures for comparison and receives normal code review. Invalid suites fail on
unknown fields, missing categories, duplicate IDs or unsafe paths.

Retrieval scores use relevant chunk IDs at a fixed cutoff. Citation cases compute
precision and recall from exact IDs. Unsupported-claim cases compare exact claim IDs.
Patch applicability requires a safe in-fixture path and an exact old-text match.
Permissions and routes are evaluated from explicit identities, grants and capability
flags; no frontend ports participate in routing.

