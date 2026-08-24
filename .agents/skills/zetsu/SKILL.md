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
For a mutating `agent_task`, Codex must choose `verification_argv` before delegation and use a direct
`pytest`, `ruff`, or `mypy` executable accepted by the Zetsu verifier policy; never wrap it with
`python -m`. Keep that verifier bound on resume and set `apply_to_repository=true` for authorized edits.
When a returned promotion is applied, its bound verifier passed after the latest mutation, and there are
no unresolved failures, treat the compact handoff as authoritative: do not reread the edits or rerun the
same verifier absent an anomaly. Exact patches/checkpoints remain persistent and `verify` can expand them.
Use `rtl_task` only for policy-eligible bounded RTL work handled by CodeV. Never request whole repositories,
papers or logs when compact evidence suffices.
