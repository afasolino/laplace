# v3 upstream dependencies and license boundary

This patch links to upstream packages; it does not vendor their source.

| Dependency | Purpose | Upstream license | Integration |
|---|---|---|---|
| prompt_toolkit | terminal UX | BSD-3-Clause | Python dependency |
| grep-ast 0.9.0 | tree-sitter structural context | Apache-2.0 | pinned optional `v3` dependency |
| Gradio | browser UI | Apache-2.0 | optional `v3` dependency |
| MCP Python SDK | Codex/Hermes stdio MCP protocol | MIT | optional `v3` dependency |
| Codex CLI | MCP configuration/host | upstream OpenAI project | external executable; no copied code |
| Hermes Agent | optional alternative frontend | MIT | external executable/MCP client; no copied code |

Keep the resolved dependency license metadata with release artifacts. No upstream source file is copied into the Laplace repository by this patch.
