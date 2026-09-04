# Prime Agent + Laplace P8 integration

## Objective

Qualify Prime Agent **v0.9.1** as an optional execution harness for Zetsu
`agent_task` while keeping Laplace's existing assurance boundary authoritative.
The local model remains the selected P8 service:

- provider: `laplace-local-p8`;
- model: `laplace-quality-qwen38-mtp8`;
- endpoint: the selected quality route in `configs/selected_serving_profiles.json`.

The implementation is grounded on Laplace revision
`b026fed7ca4e4385b4db701680a6fc3e55e1e260` and Prime Agent release `v0.9.1`
(commit `81ae3cb34d27d38ee37f9e205a1e73694993b344`). Later Prime releases are not
accepted by this qualification until separately reviewed.

## Architecture

```text
Codex / Luna
    |
    | zetsu.agent_task(agent_backend="prime")
    v
Laplace repository authorization
    v
Laplace-created isolated Git worktree
    v
bubblewrap write confinement
    v
Prime Agent v0.9.1
    v
Qwen3.8-27B P8
    |
    | IPython / compaction / autonomous repair / optional RLM child
    v
Laplace observe_worktree()
    v
Laplace authoritative verifier
    v
candidate assurance -> handoff -> optional promotion
```

Prime does **not** replace repository grants, worktree creation, verifier binding,
result storage, or promotion. A Prime failure does not fall back silently to the
native agent. `native` remains the default backend and the A/B control.

The inner production Prime worker is not given Zetsu `agent_task` or `rtl_task`,
which prevents recursive `agent_task -> Prime -> agent_task` execution. The
separate MCP qualification gate exposes only read/context Zetsu tools.

## Upstream code reused

The integration deliberately delegates generic harness behavior to Prime Agent
rather than reimplementing it:

- persistent IPython control environment;
- automatic compaction;
- autonomous continuation and quality-gate retries;
- RLM child lifecycle and usage attribution;
- custom OpenAI-compatible/vLLM provider support;
- JSON event stream.

Laplace adds only the adapter and the pre-existing assurance boundary around it.

## Security boundary

Prime Agent's Python runtime is not itself a security sandbox. Production Prime
runs therefore require `bubblewrap`.

The sandbox mounts the canonical filesystem read-only, remounts only the
Laplace-authorized candidate worktree and private per-run Prime state writable,
and mounts `.git` read-only. The prepared Prime kernel is mounted read-only.
Laplace state and the canonical repository `.runtime` path are masked at their
original locations.

The host network namespace is intentionally retained because P8 is served on
`127.0.0.1:8207`. Consequently this is strong write confinement, **not network
isolation and not complete read isolation**. Prime is launched `--offline` to
disable its own startup/update network activity, but model-generated Python must
still be treated as code running with the sandbox's network visibility.

The advisory Prime quality gate is run against a disposable repository copy.
Absolute symlinks and relative symlinks escaping the authorized workspace are
rejected before that copy is made. Laplace reruns the bound verifier against the
real candidate afterward; only that second run can create a verification binding.

## Prime kernel prerequisite

Prime lazily bootstraps its Python kernel on first IPython use. The first bootstrap
may require Internet access. Production qualification intentionally runs Prime
offline, so prepare the upstream kernel once before starting Zetsu.

With Prime's normal configuration this is typically:

```bash
prime-agent
```

Run one harmless request that causes an IPython call, then quit. Confirm the
prepared interpreter exists:

```bash
test -x "$HOME/.prime/agent/kernel-venv/bin/python" && echo PRIME_KERNEL_READY
```

Alternatively point Laplace at an already prepared, compatible Prime runtime:

```bash
export LAPLACE_PRIME_AGENT_KERNEL_PYTHON=/absolute/path/to/kernel-venv/bin/python
```

The interpreter must contain the `prime-agent-runtime` version expected by Prime
Agent v0.9.1. The adapter fails closed rather than bootstrapping packages during a
measured run.

## Static qualification

After applying the patch:

```bash
cd /home/giando/work/laplace-v3-refactor

PYTHONPATH="$PWD/src" .venv/bin/python -m pytest \
  tests/test_prime_agent_harness.py \
  tests/test_prime_agent_gate.py \
  tests/test_prime_agent_zetsu_backend.py \
  tests/test_prime_agent_public_contract.py -q

PYTHONPATH="$PWD/src" .venv/bin/ruff check \
  src/research_workspace/prime_agent_harness.py \
  src/research_workspace/prime_agent_gate.py \
  src/research_workspace/zetsu_agent.py \
  src/research_workspace/zetsu_mcp.py \
  src/research_workspace/laplace_core.py \
  src/research_workspace/repository_agent_service.py \
  scripts/qualify_prime_agent_p8.py \
  tests/test_prime_agent_harness.py \
  tests/test_prime_agent_gate.py \
  tests/test_prime_agent_zetsu_backend.py \
  tests/test_prime_agent_public_contract.py

PYTHONPATH="$PWD/src" .venv/bin/mypy \
  src/research_workspace/prime_agent_harness.py \
  src/research_workspace/prime_agent_gate.py \
  src/research_workspace/zetsu_agent.py \
  src/research_workspace/zetsu_mcp.py \
  src/research_workspace/laplace_core.py \
  src/research_workspace/repository_agent_service.py
```

Then run the normal complete hardware-independent pytest suite used for the
current branch before treating the integration as certified.

## Runtime preflight

```bash
cd /home/giando/work/laplace-v3-refactor

git branch --show-current
git rev-parse HEAD
git status --short

prime-agent --version
command -v bwrap
curl -fsS http://127.0.0.1:8207/v1/models

PYTHONPATH="$PWD/src" .venv/bin/laplace zetsu status --repo "$PWD" --json
```

The Prime version must be exactly `0.9.1` for this qualification. Restart the
owned Zetsu operator after applying the code patch so it imports the new backend.
The P8 service may remain the selected quality service.

## Live qualification

Run all four gates:

```bash
cd /home/giando/work/laplace-v3-refactor

PYTHONPATH="$PWD/src" \
  .venv/bin/python scripts/qualify_prime_agent_p8.py \
  --repository-root "$PWD" \
  --prime-agent "$(command -v prime-agent)"
```

The runner loads the local Plus bearer token from the normal Laplace state store
when it is not already present in `LAPLACE_ZETSU_TOKEN`; it never prints the
token.

The decisive result is:

```text
status = PASS
local_p8_repository_repair = PASS
local_p8_recursive_subagent = PASS
prime_to_zetsu_mcp = PASS
zetsu_agent_task_prime_backend = PASS
```

The fourth gate creates a disposable Git repository below the qualification
runtime tree, registers it through `RepositoryAuthorizationStore`, grants the
existing local Plus principal at the exact commit, invokes the **public official
MCP SDK path** through `ZetsuBackend`, requests `agent_backend="prime"`, requires
Laplace verification and promotion, reruns pytest independently, and revokes the
grant afterward. The repository is retained as evidence; the grant is inactive.

Raw Prime events, stderr, final messages, MCP result, fixtures, and `summary.json`
remain under:

```text
.runtime/prime-agent-p8-pilot/qualification/<UTC timestamp>/
```

`--skip-zetsu` and `--skip-agent-task` exist only for diagnosis. A run using either
is not the decisive production qualification.

## Selecting the backend from Codex

The Zetsu `agent_task` schema gains an optional:

```json
{"agent_backend": "prime"}
```

`repo_id` is explicitly a logical registered Laplace repository ID, never a path.
The default remains `native`. An operator-wide default can still be set with
`LAPLACE_ZETSU_AGENT_BACKEND=prime`, but explicit per-call selection is preferred
for qualification and A/B evaluation.

Prime is initially one-shot only. Persistent/restart repository-agent turns remain
on the native backend and a Prime request for such a session fails closed.

## Rollback

No data migration is required. Stop requesting `agent_backend="prime"` (and unset
`LAPLACE_ZETSU_AGENT_BACKEND` if set) to return immediately to the existing native
Qwen repository agent. The native implementation is not deleted by this patch.
