# Model provider architecture

`providers.py` contains four CPU-importable adapters:

- `FixtureModelProvider`;
- `FixtureEmbeddingProvider`;
- `OllamaProvider`;
- `VLLMProvider`.

The fixture providers are deterministic and mark every result `fixture=true`.
Ollama and vLLM constructors validate credential-free loopback HTTP origins but make
no request until an explicit method is invoked. Requests have 0.1–30 second timeouts,
an 8 MiB response limit, redirects disabled and strict typed response validation.
Errors expose only categories such as `provider_unavailable`,
`provider_timeout` or `provider_malformed_response`.

Provider descriptors declare identity, type, internal endpoint, lifecycle,
context/output limits, streaming, tools, structured output, embeddings, thinking
control, CPU/GPU requirements and start/stop capability. Ordinary frontend catalogs
replace the endpoint with `configured-local` and never reveal a port.

Invocation adapters cannot start or stop a process, even when the descriptor says an
endpoint is Laplace-owned. Ownership is meaningful to the separate lifecycle
controller only. No provider method downloads or selects a model automatically.

Routes reference `provider_id` and `model_id`; lane is scheduling policy, not a
transport. If a provider is unavailable or malformed, its readiness is deterministic
`DEGRADED` and the route must not silently fall back to an undeclared provider.

The GPU was unavailable during v7. Ollama/vLLM adapters were import-, validation- and
fixture-tested without contacting an endpoint. Live provider quality remains
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.
