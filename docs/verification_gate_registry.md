# Verification gate registry

`VerificationGateRegistry` is the only authoritative gate list. The normalized
SystemVerilog final order is:

1. `self_checking_public_simulation`
2. `adversarial_protocol_simulation`
3. `verilator_lint`
4. `verilator_simulation`
5. `iverilog_compile`
6. `vvp_simulation`
7. `yosys_synthesis`

Gate scope and tool mapping come from the registry. Verification fails when a
required result is missing, a required tool is missing, or a gate does not
pass. Lint and synthesis are never treated as functional proof.

Verilator simulation uses `verilator --binary --timing`, a unique build
directory bound to the source fingerprint, and the emitted executable. A zero
compiler result without executable invocation is failure. A zero simulation
result without the required success marker is failure.

The configured Verilator 5.032 binary is resolved from
`LAPLACE_EDA_TOOL_ROOT` or the validated local multilanguage tool root before
`PATH`. Icarus/VVP and Yosys retain their own logs and marker checks.

