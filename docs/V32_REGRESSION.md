# Laplace v3.2 upstream integration certification

Baseline: `3ebd752bd6f02d0a6ce1e36ce5bb63bfd7caaaf2`.

Decisions: prompt_toolkit ADOPT; grep-ast ADOPT; Gradio 6 ADOPT; MCP SDK v2 ADOPT;
Codex configuration EXTERNALIZE; Hermes remains Step 3.5.

Core invariant: `verification_argv` is durable session identity. `allow_mutation` is
transient per-turn authority and is not checkpointed. Operator persistent turns and
direct MCP agent_task calls default read-only; mutation requires explicit authority
and a deterministic verifier.

No live result is PASS until the human/live regression in the bundle README succeeds.
