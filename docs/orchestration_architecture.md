# Orchestration architecture

Laplace has three explicit boundaries:

1. The frozen execution core owns measured task state, model calls, patches, EDA,
   review, locks, and terminal results.
2. The Research Plane owns exploratory source snapshots, claims, reports, and
   promotion requests.
3. The Operator Plane projects immutable artifacts through typed Python
   services, a CLI, and an authenticated localhost API.

Measured execution never reads operator SQLite state, exploratory research
state, personal workspace state, or GUI state. It reads only the configuration,
skills lock, governed-corpus snapshot, model lock, tool lock, and deterministic
context packets declared by its run lock. The GUI calls
`OperatorService`; it does not edit result artifacts.

The core path is:

```text
run identity -> context packet -> model call -> source fingerprint
-> verification registry -> grounded review -> bounded correction
-> reproducibility locks -> terminal projection
```

Only one main generative model is resident by default. GPU VRAM, host RAM, iGPU
memory, and NPU memory are not treated as a shared model-memory pool. Numerical
results come from deterministic tools; the model interprets their structured
outputs.

The compatibility JSON reports remain projections. `events.jsonl`, source
fingerprints, verification evidence, and immutable locks are authoritative.

