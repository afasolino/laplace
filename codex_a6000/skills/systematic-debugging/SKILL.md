---
name: systematic-debugging
description: Diagnose a Laplace engineering failure from deterministic logs and current source state. Use when a public or adversarial gate fails and one minimal root-cause correction must be identified.
---

# Systematic debugging

1. Start from failed return codes, bounded logs, and the current source fingerprint.
2. Separate toolchain, evidence, reviewer, and implementation failures.
3. Identify the earliest causal failure; do not infer held-out behavior.
4. Map the failure to a declared requirement and current source fragment.
5. Produce a bounded defect record with exact gate and log identifiers.
