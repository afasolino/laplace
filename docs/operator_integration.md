# Operator integration

`OperatorService` is the shared typed service for CLI and HTTP actions. It
stores local metadata and append-only operator events in SQLite while treating
filesystem artifacts as source evidence.

Useful commands:

```bash
laplace-operator --role read status --json
laplace-operator --role read model-servers status --json
laplace-operator --role operate run prepare --config run.json --json
laplace-operator --role operate approval request \
  --action START_GPU_RUN --entity-id RUN_ID --json
laplace-operator --role approve approval decide \
  --approval-id APPROVAL_ID --approve --json
laplace-operator-server
```

Prepared run configuration is immutable. GPU starts, model-server start/stop,
corpus promotion, reviewer override, and bundle publication require an
approval. Repeated compatible actions are idempotent; incompatible actions
return structured conflicts.

The optional ntfy-compatible adapter is disabled by default, accepts only
bounded allowlisted metadata, only targets loopback endpoints, uses a timeout,
and returns a non-terminal operator warning on failure.

A future MCP adapter may expose read-only `status`, `summarize`, bundle
metadata, and model-server status. Write-capable MCP tools are intentionally
not implemented.

