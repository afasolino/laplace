# G3 — Project Rules and Context Hierarchy

Add authoritative rules distinct from learned memory and RAG.

## Execute

1. Inspect current Cline rules as reference; record exact commit/license.
2. Implement the smallest internal representation supporting global/user, project and path-conditional rules with deterministic precedence/conflict handling.
3. Define deterministic context assembly so learned memory or retrieved documents can never override authoritative policy/rules.
4. Keep rules inspectable and provenance-bearing.
5. Reject malformed/ambiguous rules fail-closed where authority/security is affected.

## Gate

Test precedence, conflicts, path conditions, project switching, malformed rules, memory contradicting policy and deterministic repeated context assembly.

Require no authorization regression. Certify and commit before G4.
