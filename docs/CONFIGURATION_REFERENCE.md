# Configuration reference

Laplace v7 configuration precedence is exactly:

1. secure built-in defaults;
2. versioned repository configuration;
3. external deployment configuration;
4. environment overrides;
5. explicit CLI arguments.

`configuration.py` deep-merges these layers, validates the final schema with unknown
fields forbidden and records the winning source for each leaf setting. It never
loads a provider while validating configuration.

## Schema

The current `schema_version` is `1`. Top-level keys are:

`operating_mode`, `security`, `storage`, `logging`, `governance`, `providers`,
`routes` and `secrets`.

Security always requires local-only operation, loopback binding, no model downloads,
no telemetry and path redaction. Providers are fixture, Ollama or vLLM. Non-fixture
origins must be credential-free loopback HTTP; routes must reference an enabled,
declared provider. Quotas and timeouts are bounded.

`configs/laplace.example.yaml` is a complete example. Its live adapters are disabled
and the placeholder model IDs must be replaced only with already provisioned local
models. `configs/providers.example.yaml` is documentation inventory, not a deployment
file.

## Environment overrides

Only these names are accepted:

```text
LAPLACE_CONFIG_MODE
LAPLACE_CONFIG_BIND_HOST
LAPLACE_CONFIG_STATE_ROOT
LAPLACE_CONFIG_LOG_LEVEL
```

Any other `LAPLACE_CONFIG_*` name fails validation. Provider endpoints and secrets
are not accepted through generic JSON environment blobs.

The `secrets` section contains environment-variable names, never values. Diagnostic
exports redact every secret reference and replace the state path with a path class
and digest. Windows and POSIX absolute paths are classified without interpreting a
Windows path as a Linux path; traversal and NUL are rejected.

External deployment configuration must not be group/world writable on POSIX.
Repository examples contain no credentials.
