# G8 — Procedural Memory / Skill Registry

Add explicit, versioned, human-governed procedural skills.

## Execute

1. Inspect current Hermes-Agent skills and reusable `ai4s-research/open-science` scientific skills where relevant; record exact commits/licenses.
2. Implement Laplace `SkillRegistry`.
3. Lifecycle: `candidate → validated → A/B-tested → human-approved → active`.
4. Store version, trigger, scope, procedure, required tools/verifiers, provenance, success/failure statistics and rollback/deactivation metadata.
5. No model-generated skill may auto-promote.
6. Skills cannot override policy/rules or grant new authority.

## Gate

Use positive/negative/ambiguous trigger tests, no-skill controls, rollback/deactivation and restart persistence.

Compare frozen tasks with/without each candidate skill. Require deterministic correctness and a justified efficiency/reliability benefit before human approval.

Certify and commit before G9.
