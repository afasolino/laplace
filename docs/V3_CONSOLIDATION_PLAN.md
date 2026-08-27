# Laplace v3 upstream consolidation

## Goal

Laplace v3 narrows Laplace to governed local execution. Commodity interaction,
source-context and protocol plumbing are delegated to mature upstream packages;
Laplace keeps policy-bearing invariants and thin adapters.

## Adopted upstream runtime components

| Upstream | License | Runtime responsibility in v3 |
|---|---|---|
| prompt_toolkit | BSD-3-Clause | multiline terminal editor, bracketed paste, history, fuzzy completion, history suggestions |
| grep-ast 0.9.0 | Apache-2.0 | tree-sitter parsing and scope-aware structural source context |
| Gradio 6 | Apache-2.0 | loopback browser UI components/event state |
| MCP Python SDK v2 | MIT | stdio MCP framing, initialization and protocol compatibility |
| Codex CLI | external executable | MCP registration/status/removal via `codex mcp`; no Laplace-written Codex config parser |
| Hermes Agent | MIT, external | optional MCP frontend; no vendored Hermes runtime |

No source from those projects is copied into Laplace. They are dependencies or
external executables, with thin Laplace adapters around existing governance.

## Laplace-native invariants retained

- authenticated owner/repository/revision authorization;
- isolated owned worktree lifecycle;
- bounded ACI without generic shell/network/.git exposure;
- deterministic latest-mutation verification;
- adaptive bounded continuation/stagnation control;
- durable task/checkpoint/result provenance;
- Qwen/CodeV routing and hardware/runtime certification;
- governed A/B self-improvement and human promotion;
- Zetsu tool semantics and authorization.

## Finite v3 implementation

1. **Terminal UX** — prompt_toolkit owns editing/history/completion. Laplace adds
   only commands, capability rendering and verifier/session policy.
2. **Structural context** — `laplace ast-context` and MCP `ast_context` call
   grep-ast's `TreeContext`; Laplace bounds repository-relative paths, file size,
   regex length and output size.
3. **Browser UX** — `laplace web` builds a Gradio 6 UI over the resident Operator;
   it does not create another model/core/scheduler/sandbox.
4. **Zetsu MCP stdio** — `laplace-zetsu-mcp` uses the official MCP Python SDK and
   proxies the existing authenticated Zetsu backend while preserving its tool
   schemas. `ast_context` is added only when repository-agent capability exists.
5. **Codex integration** — `laplace codex install|status|remove|launch` delegates
   MCP configuration ownership to Codex's own `codex mcp` commands. No bearer
   token is written into Codex configuration.
6. **Hermes interoperability** — Hermes may consume the same stdio MCP bridge;
   its runtime remains external to Laplace.
7. **Regression/certification** — deterministic tests plus human CLI/web/Codex
   checks must pass before committing or promoting v3.

## Explicitly not imported

Laplace does not import Aider's full agent/repo-map runtime, SWE-agent's shell or
agent loop, OpenHands' runtime, or Hermes' agent runtime. Those would duplicate
or weaken already-certified Laplace governance. The useful narrow repository
context primitive is reused directly from grep-ast instead.
