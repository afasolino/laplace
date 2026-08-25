# C4 — Harden Memory Persistence and Clarify Backend Capability

## Schema defect

Memory startup must never overwrite or reinterpret an unsupported schema version.

Implement explicit schema negotiation:

- no metadata -> initialize current schema;
- equal version -> open normally;
- older supported version -> run explicit ordered migrations;
- newer version -> fail closed with a clear compatibility error;
- failed migration -> leave a recoverable state and do not silently advance version metadata.

Migration must be transactional.

## Backend contract

The current local lexical backend is acceptable only if its capability is explicit.

Do not add Mem0 merely because it appeared in the roadmap.

Instead:
1. preserve the existing backend interface;
2. document the current backend's exact lexical/non-semantic behavior;
3. inspect whether prior roadmap evidence contains a real Mem0 evaluation;
4. if no evaluation exists, create an EXPERIMENTAL comparison plan/artifact rather than silently claiming semantic-memory completeness;
5. do not promote a new backend without deterministic A/B evidence and migration tests.

Personal Corpus/RAG must remain distinct from learned memory.

## Tests

Cover:
- empty DB initialization;
- same-version reopen;
- each supported migration;
- newer-schema fail-closed;
- interrupted migration;
- rollback/reopen;
- provenance and approval semantics unchanged;
- owner/project isolation;
- lexical backend behavior explicitly tested;
- no model proposal becomes authoritative without required approval.
