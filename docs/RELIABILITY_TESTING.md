# Reliability testing

The v7 reliability runners are bounded, CPU-only and fixture-only. They create a
unique temporary root, open no listener, make no network request, contact no provider
endpoint and remove their state before returning. They do not inspect or manage model
processes. Seeds are recorded in each result.

```bash
PYTHONPATH=src python scripts/run_cpu_soak.py \
  --iterations 32 --max-seconds 30 --output /tmp/soak.json
PYTHONPATH=src python scripts/run_failure_matrix.py \
  --max-seconds 30 --output /tmp/failures.json
```

The CPU soak covers concurrent fixture users, Chat plus Agent scheduling, simultaneous
corpus writes, a bounded large batch, queue backpressure, worktree quota denial,
disk-pressure denial and SQLite lock detection. It records assertion counts and
observed fixture outcomes, not invented throughput or latency.

The failure matrix injects restart journals for upload/indexing/verification, session
expiry, client disconnect/resume, retrieval and Agent cancellation, unavailable,
malformed and timed-out fixture providers, plus transactional interruption of
migration, backup and purge. The production migration tests separately exercise the
real migration journal and rollback implementation against synthetic old state.

No alternate port is needed because these tests open no listener. Any future HTTP
fixture must bind an explicitly allocated alternate loopback port and must never reuse
a production port. A failure leaves its JSON report as evidence but cleanup still runs
in `finally`.

GPU/live-model reliability remains
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`. The fixture provider result is not a
live serving measurement.

