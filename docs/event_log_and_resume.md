# Event log and resume

`AppendOnlyEventLog` writes canonical JSON Lines under an inter-process
`flock`. Each append is flushed and `fsync`ed. Sequence numbers are monotonic
and event IDs are deterministic over run identity, transition, attempt, source
fingerprint, and payload hash.

An identical append returns the existing event with `deduplicated=true`.
If a crash leaves a partial final line, the fragment is archived by hash and
the valid prefix is retained before the next event is appended.

`RunIdentityStore` implements:

- same identity plus no terminal result: `RESUME`;
- same identity plus terminal result: `IDEMPOTENT_TERMINAL`, with zero new
  model calls, EDA runs, or events;
- incompatible identity at the same project: `run_identity_conflict`;
- new run ID: a new project root.

Compatibility reports may be regenerated from the event/evidence state. They
must not overwrite the append-only stream.

