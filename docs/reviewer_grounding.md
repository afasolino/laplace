# Reviewer grounding

The reviewer sees the authoritative source path, source SHA-256, source-state
fingerprint, verification-report SHA-256, gate matrix, and bounded verification
results before any instructions.

The strict v2 verdict contains the verdict, reason, current source fingerprint,
verification hash, quoted source fragments, missing evidence, and violated
requirements. Quotes must occur in the current source. Stale hashes, stale
quotes, malformed fields, and deterministic gate contradictions are rejected.
One retry is allowed; a second invalid verdict fails.

For byte-identical corrections:

- passing gates plus approval => `converged_no_change`;
- a reviewer request for a change already present =>
  `reviewer_state_conflict`;
- a failed required gate => `no_effect_correction` with exact failed evidence.

An identical correction prompt is never repeated. Reviewer output cannot
override deterministic tool evidence.

