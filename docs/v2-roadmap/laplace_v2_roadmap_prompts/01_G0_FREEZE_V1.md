# G0 — Freeze and Prove the v1 Baseline

Use the certified v1 instance as the immutable control plane for all subsequent phases.

## Execute

1. Record the exact certified commit, branch, configuration, model/checkpoint revisions, routing, serving profiles and certification artifacts.
2. Record whether MTP is enabled or disabled and the measured reason.
3. Re-run only the mandatory v1 production regression needed to prove the baseline is still healthy.
4. Exercise the documented rollback to Qwen3.6/previous certified state, then restore the certified v1 configuration.
5. Create the separate v2 branch/worktree inside the repository/project tree. Do not use `/tmp`.
6. Ensure the live control-plane instance is not the v2 worktree/process.
7. Freeze the initial v2 evaluation tasks and store their hashes/identities before any v2 implementation.

## Gate

PASS only if mandatory v1 regression passes once, rollback works once and restoration succeeds, control-plane and v2 worktree are demonstrably separate, and frozen evaluation definitions are recorded before v2 changes.

Emit the machine-readable G0 certification artifact and stop on any failure.
