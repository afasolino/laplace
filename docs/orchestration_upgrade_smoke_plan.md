# Orchestration upgrade smoke plan

Run in order and preserve command, return code, stdout/stderr, and artifact
hashes:

1. full pytest, Ruff, mypy, Bandit, and `git diff --check`;
2. deterministic skill/context/event/identity/lock workflow;
3. real local Verilator/Icarus/VVP/Yosys RTL flow and injected failures;
4. stale reviewer and no-effect convergence fixtures;
5. notification and GPU admission/lifecycle fixtures;
6. fixture research, citation, contradiction, export, and promotion;
7. authenticated API, SSE, artifact security, and Playwright desktop/mobile
   accessibility smoke;
8. model-server preflight and exact endpoint identity;
9. fresh live `sv_elastic_buffer2` Arm C run with fallback disabled;
10. same-run duplicate invocation with zero new work;
11. a second fresh run with stable lock inputs and different run/trace IDs;
12. live web research with official and repository/paper evidence;
13. live run and research evidence shown through the GUI;
14. deterministic certification archive creation and hash;
15. release only proven Laplace model-server PIDs, then verify endpoints down,
   owned PIDs absent, unrelated compute processes preserved, and final GPU
   memory recorded.

A smoke passes only with executable evidence. Missing hardware/tooling is
recorded with exact command output; it is not converted into a pass.

