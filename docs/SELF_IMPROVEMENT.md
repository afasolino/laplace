# Governed self-improvement

Laplace does not allow a model, a skill, Chat, Zetsu, or an idle callback to
silently rewrite source, activate a procedure, change model routing, or promote
a new policy. Its self-improvement feature is a local, durable **shadow** loop:

1. provide bounded owner/project trajectory and memory snapshots;
2. produce deterministic candidate summaries, failures, skills, process, and
   code-change proposals;
3. select at most one harness candidate per cycle;
4. record frozen baseline/candidate/development/held-out evidence;
5. require explicit human approval; and
6. leave the result `human_approved_shadow_only`.

Approval records a review decision. It does not execute, activate, or promote
anything. A separately reviewed normal change with its own tests and release
gate remains required for any real implementation change.

## Run a shadow cycle

Use the package CLI from a repository. It writes only its local state below
`.laplace-state/consolidation` unless `--state-root` is supplied.

```bash
laplace-maintenance run \
  --cycle-id maintenance-20260826 \
  --owner-id alice --project-id power-model --session-id session-42 \
  --events-json evidence/trajectory-window.json \
  --memories-json evidence/memory-window.json \
  --window-id weekly
laplace-maintenance proposals --owner-id alice --project-id power-model
```

`events-json` and `memories-json` are JSON arrays. They must already be
owner/project scoped and contain only bounded, non-secret evidence. The CLI
never reads repositories, invokes a model, or runs a command from those files.

## Evaluate one candidate

After reviewing a cycle, create the bounded harness proposal:

```bash
laplace-maintenance propose-harness \
  --cycle-id maintenance-20260826 \
  --owner-id alice --project-id power-model --session-id session-42 \
  --description "Add a deterministic regression for repeated parser timeout." \
  --source-event-id evt-17 --source-event-id evt-23
```

Run baseline and candidate evaluations independently. They must use disjoint
frozen, development, and held-out task IDs. Store only the result hashes and
boolean outcomes in an evidence file:

```json
{
  "baseline_result_sha256": "<64 hex chars>",
  "candidate_result_sha256": "<64 hex chars>",
  "frozen_task_ids": ["frozen-parser-1"],
  "development_task_ids": ["dev-parser-1"],
  "held_out_task_ids": ["heldout-parser-1"],
  "baseline_correct": true,
  "candidate_correct": true,
  "security_regression": false,
  "correctness_regression": false,
  "observed_at_utc": "2026-08-26T12:00:00Z"
}
```

Then record and explicitly approve the shadow result:

```bash
laplace-maintenance record-ab --proposal-id <proposal-id> --evidence-json evidence/ab.json
laplace-maintenance approve --proposal-id <proposal-id> --approver-id alice
laplace-maintenance status
```

If either correctness or security regresses, evidence is rejected. It cannot be
approved. `status` must continue to show `mode: SHADOW`,
`active_production_mutations: false`, and `auto_promotion: false`.

## Related features

- **Skills** are separate versioned procedures with their own approval and
  activation lifecycle; a maintenance candidate does not activate one.
- **Memory, rules, trajectories and compaction** are evidence inputs, not
  authorities to edit code or policies.
- **Quality-improvement evaluation** in `research_workspace.quality_improvement`
  is a separate GPU-enabled benchmark harness. A measured run requires an
  explicit candidate configuration and immutable held-out scoring; it refuses
  to substitute CPU inference for the required A6000. Its `--analysis-only`
  mode does not need a candidate, but it does require the recorded historical
  quality-evidence bundle. Use the command and reference-registration guidance
  in [the quality evaluation guide](a6000_agent_team/SHARED_REFERENCE_RETRIEVAL.md).
