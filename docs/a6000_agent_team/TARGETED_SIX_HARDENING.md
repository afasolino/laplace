# Structured repair and targeted-six validation

The implementer returns hash-bound complete file replacements as one JSON object. Laplace validates paths, roles, language, current SHA-256 values, uniqueness, size, and non-empty semantic change. It then creates the unified diff locally and applies it with Git checks.

Malformed, ambiguous, stale, duplicate, out-of-scope, and no-op responses are retried without consuming a meaningful correction loop. The two correction loops are reserved for patches that were applied and then failed deterministic verification or operational review.

The reviewer returns a machine-readable `approve`, `request_changes`, or `block` verdict. In the full five-role workflow, deterministic verification and reviewer approval are both required. Direct mode and the reviewer-invariant ablation retain deterministic verification as the approval authority.

Retrieval expands task queries with domain invariants, records matched terms, limits repeated chunks from one path or reference, and adds project-authored invariant cards for strict Python validation/transactions and SystemVerilog ready-valid/AXI4-Lite/W1C behavior.

The targeted command runs exactly:

- `py_fastapi_strict_endpoint`
- `py_unseen_sqlite_state`
- `sv_ready_valid_buffer`
- `sv_axi_lite_irq_regs`
- `sv_unseen_rv_slot`
- `sv_unseen_w1c_event`

It writes `targeted_six_results.json`, `targeted_six_results.csv`, and `targeted_six_summary.md`, optionally comparing scores against a prior corrected result root.
