# Zetsu

Zetsu is the authenticated Laplace↔Codex pairing layer. It is served at `/mcp`
by the existing loopback-bound Laplace Operator process. HTTPS terminates at the
normal Laplace ingress; Qwen, CodeV, databases, and control ports remain private.

Zetsu schema version is 1.5 and Skill version is 1.5.0. The tool set is
`search`, `get_evidence`, `project_context`, `experiment_context`, `delegate`,
`agent_task`, `agent_task_status`, `cancel_agent_task`, `rtl_task`, `verify`, and
`get_result`. Normal Laplace capabilities determine which tools each identity sees.

## Configure Codex

Start the normal local development stack with CodeV intentionally disabled:

```bash
laplace zetsu start --nocodev
```

The command starts the owned Qwen → Operator supervisors, explicitly selects the
configured vLLM executable, waits for exact model identities, checks authenticated
Zetsu availability, and leaves the services running in the background. CodeV
absence is intentional in this topology. Private logs and PID ownership records are stored below
`$LAPLACE_STATE_ROOT` or `~/.local/state/laplace`. If a later service fails to
start, only supervisors created by that invocation are stopped. Use
`laplace zetsu stop` for the matching ownership-safe shutdown. `--dry-run` prints
the resolved commands without starting a process.

In each repository that should use Zetsu, run:

```bash
laplace zetsu
```

With no subcommand, Laplace installs or refreshes the managed project Skill and
user-level MCP registration when needed, then always runs both status and the
authenticated retrieval connectivity test. For a loopback deployment, this CLI
command reads the Plus credential directly from the protected local Operator
token file for its own probes; it never prints or writes that value into Codex
configuration.

Start a new Codex session with the same checked configuration and protected
environment-only credential by running:

```bash
laplace zetsu codex
```

This performs the configure, status, and retrieval preflight first, then replaces
itself with Codex and supplies `LAPLACE_ZETSU_TOKEN` only in that child process.
Arguments after `--` are forwarded, for example `laplace zetsu codex -- --version`.
The credential remains absent from the shell, command line, and Codex config.

Create an owner-bound Laplace bearer credential outside Git and export it in the
environment that starts Codex:

```bash
export LAPLACE_ZETSU_TOKEN='<secret value>'
laplace zetsu configure --endpoint https://laplace.example.org/mcp --json
laplace zetsu status --json
laplace zetsu test --json
```

`configure` preserves the intended two-level scope: it merges one managed Zetsu
MCP block into `$CODEX_HOME/config.toml` (or `~/.codex/config.toml`) and installs
the repository-local `.agents/skills/zetsu/SKILL.md`. The MCP registration is
therefore available to Codex sessions, while each project carries the usage
policy that teaches Codex when to use it. The bearer value is referenced through
an environment variable and is never written to either file.

The installed Codex CLI automatically defers MCP tool schemas until tool search
selects the server. All registered capabilities remain available, but unused Zetsu
does not eagerly serialize their full schemas into the model context.

Run `configure` from a new project repository to install or refresh its managed
Skill. Existing unrelated Codex configuration is preserved. An unmarked
user-owned Zetsu table or Skill is rejected instead of overwritten. `remove`
removes only managed Zetsu content.

## Operating policy

Codex keeps direct responsibility for its current checkout, shell, Git, builds,
and simple local tasks. Use the Zetsu surfaces according to the work required:

- `search`, `project_context`, `experiment_context`, then `get_evidence` for
  compact owner-authorized knowledge and selective evidence expansion;
- `delegate` for bounded Qwen reasoning that does not need repository mutation;
- `agent_task` for a coherent bounded Qwen repository task that benefits from
  autonomous inspect/edit/test/retry behavior;
- `rtl_task` for one policy-eligible bounded RTL/SystemVerilog implementation or
  repair through the dedicated CodeV economy route;
- `verify` for deterministic verification evidence exposed by Laplace.

CodeV remains the RTL specialist. `agent_task` uses only Quality or Standard and
must not replace the existing `rtl_task` path.

Before repository-bound work, run `laplace zetsu status --json`. The repository
object must report the canonical registered root, logical ID, owner grant,
granted revision, current HEAD, clean state, and `agent_task_ready: true`.
`repository_not_registered` and `repository_not_authorized` are explicit security
states; Zetsu never silently authorizes a client path.

Inspect owned lifecycle state without querying SQLite:

```bash
laplace zetsu sessions --json
laplace zetsu worktrees --json
```

Clean failed or expired sessions are reconciled before quota denial. Dirty
worktrees remain recoverable and are never automatically deleted. A committed
canonical HEAD advance is synchronized for a new session. Caller-only dirty files
are not copied; when a target is absent from the exact committed session revision,
the deterministic remediation is to checkpoint/commit and delegate again.

## Qwen repository agent

`agent_task` binds a Qwen session to the authenticated owner, logical repository
ID, server-authorized isolated worktree, exact committed base revision, and bounded tool policy.
It can search/read the worktree, request compact owner-authorized retrieval,
edit existing text files, create bounded new text files, and execute only the
verification allowlist. It has no generic shell, network access, `.git` access,
or arbitrary filesystem access.

Related reads and exact anchored edits can be batched into one bounded action.
For an authorized mutation, the caller may request `apply_to_repository`: after
the caller-bound verifier passes following the latest mutation, Laplace hashes
and persists the exact patch, locks the canonical destination, rechecks its bound
revision and clean status, and applies the patch atomically. Destination drift or
any patch/postcondition mismatch fails closed. The eager result contains only
status, changed paths, verifier result, unresolved failures, evidence/checkpoint
references, patch identity, and promotion state. `verify` with `include_patch`
expands the exact persisted handoff only when an anomaly requires it.

Large successful results use durable bounded paging through `get_result`; they are
not execution failures or unbounded context injections.

Mutating tasks cannot finish successfully until deterministic verification has
passed after the latest mutation. Cancellation, command limits, step limits, and
overall wall-clock limits are checked during the iterative loop. A supplied
`session_id` resumes its persistent checkpoint; owner/repository/base-revision or
objective mismatch fails closed.

Long-running Qwen tasks compact around 80% of the configured context window.
Semantic history is summarized while exact execution state remains separate and
persistent: objective, worktree revision/status, changed paths, validation
history/results, unresolved failures, evidence references, step/accounting state,
and the next execution state. Failed or insufficient compaction stops the task
rather than continuing with incomplete state.

Once the bound verifier passes after a mutation, the agent stops without a final
model round. A successful applied handoff with no unresolved failures is
authoritative; Codex should not reread all changed files or rerun the same verifier
unless it observes an anomaly.

## Compact retrieval and provenance

Retrieval is progressive. `search`, `project_context`, and
`experiment_context` return short ranked excerpts with compact evidence and chunk
IDs. Result count and character budgets are bounded; use `get_evidence` only for
selected IDs. Owner/source provenance is retained at each expansion.

The Qwen agent uses the same owner-scoped retrieval boundary. Retrieved corpus
material is compact read-only context and is never mounted as an agent filesystem.
Telemetry distinguishes exact model-reported usage, when supplied by the serving
API, from explicitly approximate evidence/context token estimates. Prompt and
response bodies are not persisted solely for token measurement.

## Codex-token sanity check

The engineering token check is defined by `configs/benchmarks/zetsu_token_tasks.json`
and must remain exactly three single-run pairs: the same frozen task once with Codex
alone and once with Codex+Zetsu. Before either condition, record the task-config
SHA-256, exact prompt SHA-256, immutable base revision, the deterministic Git
commit+tree state SHA-256, Codex model/reasoning setting, and distinct fresh
session/worktree identities. `scripts/benchmark_zetsu_token_efficiency.py
--print-repository-state-sha256 <worktree>` derives the repository-state digest.

Read-only tasks pass only when the exact expected answer is returned and the
worktree remains clean at the frozen base. The implementation task may change only
its predefined files and must pass its frozen commands. Baseline metadata must
report zero Zetsu calls; the Zetsu condition must report at least one. Aggregate
`1 - Codex_tokens_with_Zetsu / Codex_tokens_baseline` only over PASS/PASS pairs
with comparable methodology and exact Codex total usage. Prefer the final cumulative
Codex token record; when the installed schema exposes only per-turn usage, sum each
completed turn once. Unavailable Codex fields stay null. Local Qwen tokens and
approximate Zetsu evidence/context estimates are reported separately and never added
to Codex-credit consumption. This is a three-pair engineering sanity check, not a
statistical benchmark.

The maintained token-efficiency sanity check uses
`configs/benchmarks/zetsu_token_tasks.json` with
`scripts/benchmark_zetsu_token_efficiency.py`. Older benchmark outputs remain
historical diagnostic evidence; deleted legacy task specifications are not runtime
inputs and are never reconstructed or reinterpreted as current configuration.

## Laplace Client boundary

Laplace Client remains the explicit pairing mechanism for remote PCs or user
folders. Zetsu does not infer or silently acquire filesystem or shell access to
the machine running Codex. Repository work is limited to the server-authorized
worktree associated with the agent session; remote-folder access still requires
the normal Laplace Client pair/grant flow documented in [LAPLACE_CLIENT.md](LAPLACE_CLIENT.md).

## Transport and troubleshooting

Enable the normal bearer API on the Operator service and proxy the whole Laplace
origin, including `/mcp`, through HTTPS. Browser sessions are rejected at `/mcp`;
bearer authentication is mandatory and invalid Origin is rejected. Secrets belong
in the protected Operator token file and client process environment.

- `missing_token_env`: export the configured token environment variable in the
  Codex process environment.
- `mcp_connection_failed`: check HTTPS/DNS, Operator service state, and ingress
  streaming/timeouts.
- stale/incompatible managed versions: rerun `laplace zetsu configure`.
- `zetsu_agent_checkpoint_*`: inspect owner/repository/revision/objective binding;
  do not bypass a failed resume check.
- `zetsu_agent_compaction_*`: preserve the checkpoint and diagnose serving/context
  behavior before resuming.
- `repository_not_registered` or `repository_not_authorized`: ask an administrator
  to register/grant the logical repository; never provide an arbitrary client path.
- `repository_state_not_materialized`: commit the caller's intended change and
  delegate a new session; Zetsu never copies dirty files or auto-commits.
- `per_user_worktree_quota`: inspect `laplace zetsu sessions --json`; clean stale
  sessions are reclaimed, while dirty sessions remain protected.
- model readiness false: inspect `/api/v1/readiness`; MCP reachability does not
  certify Qwen or an independently configured RTL worker.

For the local development deployment, `laplace zetsu start --nocodev` starts the
selected Qwen profile and Operator and waits for both to become ready. Thereafter,
bare `laplace zetsu` from any project performs configure-if-needed, status, and
the authenticated retrieval test. The matching `laplace zetsu stop` command stops
only supervisors recorded with matching owned process identities.

The RTL-worker topology, including the Step-C SiliconMind qualification line,
is independently qualified and lifecycle-managed.
`laplace zetsu start --nocodev` intentionally starts the certified P8 Qwen path
without activating the RTL worker; an explicit RTL experiment or later lifecycle
orchestration may activate that independent route without changing P8 identity.

See [QWEN38_PRODUCTION_P8.md](QWEN38_PRODUCTION_P8.md) for the current Qwen3.8
production invariant, artifact provenance, portability rules, and fail-closed
recovery policy.
