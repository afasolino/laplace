# C3 — Make Bounded ACI Large-Work Capable

## Goal

Preserve the bounded ACI security model while removing hard size cliffs that force avoidable task failure.

## Required read semantics

Large reads must support deterministic continuation.

Add or extend an existing bounded read operation so the caller can retrieve a file/region in pages using an exact cursor or next-line token.

Every page must:
- remain within configured byte/line limits;
- identify path and snapshot/hash information sufficient to detect drift;
- provide a deterministic continuation token;
- fail closed if the file changed in a way that invalidates continuation.

## Required write/create semantics

Support bounded creation/update of files larger than a single ACI payload.

Prefer a transaction-like API using:
- begin/create with expected path and optional expected base state;
- ordered chunks with explicit offsets/sequence and hashes;
- finalize with whole-content hash;
- abort/recovery.

If the existing atomic edit abstraction can be extended more simply, use it.

Requirements:
- retries cannot duplicate chunks;
- out-of-order/missing chunks fail;
- symlinks/path escape remain forbidden;
- final file appears atomically or the previous state remains;
- verifier runs only against finalized state;
- cancellation leaves recoverable/clean state.

Do not expose arbitrary shell or unrestricted filesystem append.

## Tests

Cover:
- multi-page read reconstructs exact file;
- file drift invalidates stale cursor;
- multi-chunk file creation;
- retry of same chunk is idempotent;
- duplicate conflicting chunk fails;
- out-of-order chunk fails;
- finalize hash mismatch fails;
- cancellation/abort;
- path traversal/symlink protections;
- large UTF-8 source file;
- large SystemVerilog/Python source;
- existing small-file behavior unchanged.
