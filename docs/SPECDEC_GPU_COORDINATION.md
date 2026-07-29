# SpecDec GPU coordination

SpecDec has priority over Laplace certification. Protected evidence includes
process command, parent chain, or working directory associated with
`/home/giando/projects/specdec_ladder`, `specdec_ladder`, speculative decoding,
HumanEval quantization/recovery/adaptation, or Qwen drafter/verifier work.
Uncertain ownership is unavailable ownership.

Before a GPU group, inspect compute PIDs once and resolve sanitized command hash,
executable name, parent chain, and working-directory classification. Never stop,
signal, reconfigure, or otherwise interfere with SpecDec.

When waiting, use shell `sleep`; do not use `watch` or repeated `tail`. Poll no
more frequently than every five minutes. After two unchanged observations use
ten minutes; after four use fifteen minutes. Each observation is one compact
record containing timestamp, PID/state, elapsed time, last log line, GPU memory,
and next sleep. After three active checks, defer the live gate until all CPU work
is finished. If the final check remains active, record
`BLOCKED_BY_SPECDEC_ACTIVE`.

If SpecDec appears during a Laplace group, release only process groups whose PID,
process-group ID, start ticks, and ownership record match the Laplace lifecycle
record. Record `YIELDED_TO_SPECDEC` and do not reacquire the GPU in that
certification run.

