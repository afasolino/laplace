---
name: reviewer-grounding
description: Review a Laplace change against authoritative current source and hash-bound verification evidence. Use for approve, request-changes, or block decisions that must reject stale code quotations and contradictory evidence.
---

# Reviewer grounding

1. Read the authoritative source before references or historical material.
2. Echo the exact source-state fingerprint and verification-report SHA-256.
3. Quote only fragments present byte-for-byte in the current source.
4. Approve only a complete passing required-gate matrix.
5. Identify concrete failed gates for any non-approval.
6. Return only the strict reviewer JSON schema and never edit source.
