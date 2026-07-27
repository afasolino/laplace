---
name: verification-before-completion
description: Require complete executable verification evidence before a Laplace task can finish. Use for public simulation, adversarial simulation, Verilator lint and binary execution, Icarus/VVP, and Yosys gate assessment.
---

# Verification before completion

1. Load gate and tool requirements only from the authoritative registry.
2. Execute every required command with a timeout and preserved log.
3. Treat lint and synthesis as non-functional evidence.
4. Require executable simulations and their self-checking success markers.
5. Bind every result to the current source-state fingerprint.
6. Fail completion for missing tools, missing results, non-zero exits, or missing markers.
