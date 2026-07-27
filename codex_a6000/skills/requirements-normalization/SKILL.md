---
name: requirements-normalization
description: Normalize a Laplace engineering request into an explicit, deterministic task contract. Use for requirements, interfaces, corner cases, allowed paths, verification gates, and held-out separation before implementation.
---

# Requirements normalization

1. Read only the request, public task material, and declared governed evidence.
2. Preserve exact interfaces, parameters, reset semantics, and allowed paths.
3. Make simultaneous events, boundary behavior, and error behavior explicit.
4. Obtain required gates and tools from the authoritative verification registry.
5. Mark held-out content unavailable to implementation and review.
6. Emit the normalized contract; do not implement source in this role.
