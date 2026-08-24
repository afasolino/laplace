# G2 — Episodic and Semantic Memory

Add persistent learned memory without conflating it with document RAG.

## Invariant

- Personal Corpus = source-grounded documents/literature with provenance.
- Memory = episodic/semantic learned state.
- Authoritative rules/policy are separate from both.

## Execute

1. Define `MemoryService` with replaceable `MemoryBackend`.
2. Inspect current `mem0ai/mem0`, exact commit and license; evaluate a fully self-hosted adapter first.
3. Support owner/project scope, provenance, add/search/update/supersede/delete and restart persistence.
4. Initial writes must be explicit deterministic writes or human-approved/model-proposed writes; no silent free-form long-term memory.
5. Define contradiction/supersession semantics and preserve history/provenance.
6. Ensure deletion removes searchable/indexed state as intended.

## Gate

Freeze tests for two owners/projects, conflicting updates, supersession/obsolete facts, deletion, restart, malformed/corrupt entries, provenance recovery and no hosted-service dependency.

Any isolation failure is FAIL. Certify and commit before G3.
