# G4 — Repository Intelligence / RepoMap

Add token-budgeted structural repository context without replacing exact file reads.

## Execute

1. Inspect current Aider RepoMap and `xberg-io/tree-sitter-language-pack`; record exact commits/licenses.
2. Implement `RepositoryContextService` with symbols, definitions/references where reliable, dependency/import/include/instantiation edges, relevance ranking, token-budgeted RepoMap and deterministic cache invalidation.
3. Support Python, C, C++, Verilog/SystemVerilog and mixed repositories.
4. Add only missing RTL extraction needed for modules/interfaces/packages/instantiations/includes if upstream support is insufficient.
5. RepoMap is advisory context; mutations/verification use real files.

## Gate

Test symbol/edge correctness, edits/renames/add/delete invalidation, mixed-language repos and stale-cache rejection.

Compare frozen tasks with and without RepoMap. Default-enable only if deterministic correctness is preserved and files read, tokens/context, tool calls or latency materially improve.

Certify and commit before G5.
