# Local observability

`LocalTraceRecorder` emits OpenTelemetry-compatible JSON Lines and
`metrics.json`. Network export is absent by default and no-op mode is
supported.

Spans use random trace/span IDs, timestamps, duration, status, and at most 32
bounded scalar/hash attributes. Attribute names containing prompt, response,
source code, held-out, secret, or token semantics are rejected.

The core span vocabulary includes run pair, corpus preflight, requirements,
retrieval, context building, CodeV generation, patching, each EDA gate, review,
review reconciliation, correction, model-server lifecycle, and finalization.
Research stages use `research_<stage>`. Trace IDs belong in model-call,
verification, review, research, and final projections.

