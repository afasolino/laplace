# Synthesizable Verilog-2001 patterns

Release: `multilanguage-corpus-v1`. Licence: MIT. This independently authored
card targets Icarus and Yosys-compatible Verilog-2001.

## Sequential and combinational structure

- Use nonblocking assignments in edge-triggered sequential blocks and blocking
  assignments in combinational blocks. Do not assign one state variable from
  multiple procedural blocks.
- Give every combinational output and next-state value a default before a
  conditional or `case` statement. Cover all branches to prevent unintended
  latch inference. Use a plain `case` with an explicit `default` for portable
  Verilog-2001.
- Size numeric constants and parameters intentionally. Derive counter widths
  from legal parameter ranges outside the RTL worker when `$clog2` is not part
  of the selected Verilog subset.

## Ready/valid, FIFOs, and events

- A producer holds `valid` and its payload stable until a rising clock edge on
  which `ready && valid` is true. A one-entry elastic slot can accept when it is
  empty or when its current item is consumed on the same edge.
- Compute FIFO full and empty from registered occupancy or read/write pointers.
  Specify simultaneous push/pop behavior explicitly and test empty, full,
  overflow attempts, underflow attempts, and pointer wrap.
- Define priority when an event set and a software clear occur together. Reset
  values and reset assertion/deassertion timing are observable protocol rules.

## Portable verification

Use `iverilog -g2001` plus `vvp` for an executable self-checking testbench. Use
Yosys `read_verilog`, `hierarchy -check -top`, and `synth` for the selected top.
Lint and synthesis success do not replace checks for reset, stalls, simultaneous
events, boundaries, stable payloads, and exact cycle latency.
