# Model-server admission and lifecycle

Run preflight before Phase 3:

```bash
laplace-operator --role read model-servers preflight --json
```

`model_server_preflight.json` records the A6000 identity and memory, compute
PIDs, owners of ports 8102/8103, exact expected model IDs, empirical threshold,
startup timeout, decision, and failure category.

The threshold is measurement-based. Six preserved successful CodeV probes
showed a maximum dual-server allocation of 43,801 MiB. Repository policy
requires 4 GiB remain free, so a cold start requires 47,897 MiB free. Parameter
count is not used as a fit estimate.

Both exact `/v1/models` identities already healthy yields
`REUSED_HEALTHY_SERVERS`. Wrong identities, unrelated listeners, or partial
inconsistent service state yield `model_server_port_conflict`. Insufficient
memory yields `resource_admission_failure`.

Startup is bounded and records PIDs and logs. Release reads Laplace PID files,
then revalidates model path, loopback host, and port in `/proc/<pid>/cmdline`.
It sends TERM only to proven Laplace PIDs, never force-kills, verifies owned
PIDs are gone and endpoints are down, and checks unrelated compute PIDs remain.

