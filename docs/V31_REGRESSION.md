# Laplace v3.1 regression

Baseline: `feature/laplace-v3` at `d63f33be6e12f0b77d3cabe22f9e3fd60c1c5068`.

The v3.1 gate proves objective-state isolation, deterministic per-message routing,
truthful runtime telemetry (measured or explicitly unavailable), and bounded
owner-authorized corpus introspection.  Live regression must exercise unrelated
messages in the same CLI session and verify that stale repository-agent evidence
never leaks into chat/runtime/corpus answers.
