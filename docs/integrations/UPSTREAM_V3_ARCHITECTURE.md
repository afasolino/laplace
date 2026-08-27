# Laplace v3 upstream-consolidated architecture

## What Laplace owns

Laplace continues to own only the policy-bearing surfaces: authenticated owner/repository binding, isolated worktrees, bounded ACI, deterministic post-mutation verification, adaptive continuation, durable task evidence, model/runtime certification, RTL routing, and human-controlled promotion.

## What upstream owns

- `prompt_toolkit`: interactive terminal editing, bracketed paste, persistent history, completion and history suggestions.
- `grep-ast`: tree-sitter parsing and scope-aware source context.
- Gradio 6: browser chat components, event wiring, state and local web presentation.
- Model Context Protocol Python SDK v2: stdio framing, protocol negotiation and compatibility for Codex/Hermes.
- Codex CLI: MCP configuration storage/management through `codex mcp add/get/remove`; Laplace no longer needs to manipulate Codex TOML for the new stdio path.

Hermes Agent remains an optional frontend. It can point its stdio MCP configuration at `laplace-zetsu-mcp`; its core runtime is not vendored into Laplace.

## Data flow

```
terminal: laplace chat ── prompt_toolkit ─┐
                                         │
browser:  laplace web ─── Gradio ────────┼── Operator API ── Laplace governance/runtime
                                         │
Codex/Hermes ─ official MCP SDK stdio ───┴── authenticated Zetsu HTTP backend
                    │
                    └── grep-ast read-only AST context for the bound repository
```

The SDK bridge never gets a generic shell. `ast_context` accepts one repository-relative path and a bounded regex, refuses path escape/symlinks, and caps file/output size.
