# C7 — Remove Zetsu-Specific Private Coupling from LaplaceCore

## Goal

Make Laplace Core independently usable while keeping Zetsu as an adapter.

## Required refactor

Inspect every direct dependency from `LaplaceCore` to:
- `ZetsuAgentCoordinator`;
- Zetsu-specific types;
- private methods such as `_verify_argv`;
- MCP-specific assumptions.

Introduce the smallest neutral abstraction necessary, for example:
- repository-agent service/protocol;
- verification policy/service;
- neutral request/result dataclasses.

Do not duplicate implementations.

Zetsu should adapt MCP calls to neutral core services. Standalone Laplace should invoke the same services directly.

## Constraints

- no behavior regression;
- no change to authorization semantics;
- no change to deterministic verifier selection;
- no duplicated routing/retrieval/agent logic;
- dependency direction must be `adapter -> core`, not `core -> adapter`.

## Tests

Add architecture-level tests proving:
- core can be constructed without importing/initializing Zetsu/MCP;
- standalone repository-agent path works;
- Zetsu path uses the same core service;
- verifier behavior is identical;
- no private Zetsu method is called by core;
- current G1 regressions remain green.
