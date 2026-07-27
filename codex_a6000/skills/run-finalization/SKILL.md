---
name: run-finalization
description: Finalize a Laplace measured run from immutable locks, events, traces, gate evidence, and grounded review. Use only after all required evidence is present and the terminal result and certification bundle must be produced.
---

# Run finalization

1. Verify subordinate lock hashes and compute the run-lock hash.
2. Confirm the event stream is monotonic and the terminal projection is consistent.
3. Confirm every required gate passes and the reviewer bindings are current.
4. Record the final source hash, trace ID, run-lock hash, and residual risks.
5. Build the certification bundle from preserved artifacts without mutating them.
