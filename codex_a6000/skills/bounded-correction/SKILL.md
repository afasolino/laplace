---
name: bounded-correction
description: Correct one current, evidence-backed Laplace defect within a bounded retry budget. Use after deterministic verification or grounded review identifies a concrete repair and duplicate prompts must be prevented.
---

# Bounded correction

1. Use only the latest source fingerprint, failed evidence, and grounded verdict.
2. Change only the smallest declared source scope that resolves the defect.
3. Never weaken or edit tests, gates, or reviewer evidence.
4. Never repeat an identical correction prompt.
5. Classify byte-identical output as convergence, reviewer conflict, or no-effect failure from current evidence.
