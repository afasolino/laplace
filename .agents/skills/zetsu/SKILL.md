---
name: zetsu
description: Use Laplace for compact project context, evidence, delegation, RTL specialization, and verification.
---

<!-- managed-by: laplace-zetsu v2 -->
# Zetsu

Use Zetsu when information is outside the current checkout: literature, uploaded sources,
project history, prior experiments/results, or governed Laplace evidence. Use Codex's local
filesystem, shell, Git, builds, and tests directly for the current checkout.

Start with `project_context`, `experiment_context`, or `search` using a compact budget. Expand
only the evidence IDs needed with `get_evidence`. Prefer bounded `delegate` for self-contained
work that does not need Codex's full local context. Use `rtl_task` only for eligible bounded RTL
implementation/repair; architectural decisions remain with the main agent. Use `verify` for the
existing Laplace verification workflows.

Never request repository dumps or complete papers when compact evidence is sufficient.
