---
name: zetsu
description: Use Laplace for compact knowledge, verified Qwen agent delegation, RTL specialization, and evidence.
---

<!-- managed-by: laplace-zetsu v4 -->
# Zetsu

Use Codex local filesystem, shell, Git, builds and tests directly for simple work in the current checkout.
Use `search`, `project_context` or `experiment_context` for indexed, historical, literature or distributed
project knowledge, then expand only needed IDs with `get_evidence`. Use `delegate` for bounded reasoning.
Use `agent_task` for a coherent self-contained repository task when local Qwen can inspect/edit/verify it
more cheaply than repeated Codex orchestration. Batch related reads/edits and avoid polling or narration.
The exposed MCP schema is authoritative. On the normal path, do not grep Zetsu source/configuration,
runtime audit logs, checkpoints, or result artifacts before calling `agent_task`; those are anomaly-only
evidence surfaces and broad inspection defeats delegation token savings.
For a mutating `agent_task`, Codex must choose the verification contract before delegation. Use
`verification_argv` only for one root-working-directory command. Use `verification_plan` for multiple
independent checks or when a check needs its own contained repository-relative `cwd`; every step still
uses a direct `pytest`, `ruff`, or `mypy` argv, never shell, `python -m`, or environment injection.
The plan is authoritative and bound by Zetsu: Qwen should request `verify` and must not reconstruct,
merge, or weaken the caller-selected commands. Keep that verifier bound on resume and set
`apply_to_repository=true` for authorized edits.
When a returned promotion is applied, its bound verifier passed after the latest mutation, and there are
no unresolved failures, treat the compact handoff as authoritative: do not reread the edits, inspect
result artifacts, or rerun the same verifier absent an anomaly. Stop the delegated path immediately on
that successful receipt. Exact patches/checkpoints remain persistent and `verify` can expand them only
when anomaly evidence requires it.
Use `rtl_task` only for policy-eligible bounded RTL work handled by CodeV. Never request whole repositories,
papers or logs when compact evidence suffices.
