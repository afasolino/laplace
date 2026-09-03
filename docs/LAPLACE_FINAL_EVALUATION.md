# Laplace final evaluation

The three-task Zetsu benchmark is only a wiring/methodology smoke test.

Freeze GPT-5.6 Luna at high reasoning and compare:
1. Luna alone.
2. Luna plus Laplace with corpus retrieval disabled.
3. Luna plus Laplace with a pre-qualified frozen corpus.

Pass@1 and deterministic verification are primary. Token savings count only for
passing comparisons.

The internal matrix covers Python, C/C++, SystemVerilog, testbench engineering,
mixed RTL/software work, repository research, corpus retrieval, and long-horizon
tasks. M4 capstones are a complete Pac-Man implementation with hidden headless
tests and a parameterized RTL subsystem with self-checking SystemVerilog
testbench, Python reference model, lint, simulation, and synthesis.

Before corpus-enabled runs, freeze corpus revision and hashes, run held-out exact,
multi-hop, stale/current, and no-evidence retrieval cases, and forbid benchmark
solutions, hidden tests, issue resolutions, or derived benchmark answers in the
corpus.

After freezing the internal configuration, run external suites including
SWE-bench Verified, SWE-bench Multilingual, Multi-SWE-bench, SWE-bench Pro,
Terminal-Bench, TuRTLe, and RTLLM.
