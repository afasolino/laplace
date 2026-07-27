---
name: rtl-implementation
description: Implement one bounded portable RTL task from a frozen Laplace contract. Use for hash-bound SystemVerilog or Verilog source replacement with exact interfaces and no shell or test modification authority.
---

# RTL implementation

1. Treat the current source and SHA-256 in the context packet as authoritative.
2. Change only declared editable RTL paths.
3. Preserve the exact module name, ports, parameters, and reset behavior.
4. Implement ready/valid stability, FIFO order, simultaneous events, and boundaries literally.
5. Return only the requested structured replacement; never emit commands or modify tests.
