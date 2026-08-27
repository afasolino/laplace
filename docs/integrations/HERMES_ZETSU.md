# Hermes Agent + Laplace Zetsu

Hermes is an optional external frontend. Laplace remains the governed repository,
verification and execution backend.

The preferred v3 boundary is the same official-MCP-SDK stdio bridge used by
Codex:

```bash
laplace-zetsu-mcp --repo <repo> --state-root <state-root> --endpoint http://127.0.0.1:8765/mcp
```

Configure Hermes with its current MCP CLI/configuration to launch that command as
a stdio MCP server. Keep the Laplace bearer token in the existing local Laplace
state or environment; do not write it into project files or a committed Hermes
configuration. The bridge reads the credential itself and forwards calls to the
authenticated loopback Zetsu backend.

Acceptance criteria:

- Hermes discovers the same authorized Zetsu tool list as Codex;
- `ast_context` appears only when repository-agent capability is authorized;
- retrieval/result/cancellation semantics remain Zetsu-owned;
- write tasks still require Laplace's deterministic verifier contract;
- invalid credentials or unauthorized repositories fail closed;
- Hermes never receives direct generic shell or unrestricted filesystem access.

The existing Zetsu HTTP endpoint remains the backend transport. The stdio SDK
bridge is the frontend interoperability boundary and contains no Hermes code.
