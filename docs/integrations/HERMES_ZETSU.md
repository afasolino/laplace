# Hermes Agent + Laplace Zetsu

Step 3.5 decision: **EXTERNALIZE**.

Reviewed snapshots:

- Laplace `feature/laplace-v3`: `0c88eb3b748f36701d5246ab5b13b6b384d5f972`
- NousResearch/hermes-agent: `5cc47c994beb243407bb4c8ba47d2ab421cda9cf`

Hermes owns the commodity product layer: CLI/TUI frontend, session persistence
and search, memory UX, skills, cron, and optional delegation. Laplace remains the
governed repository, authorization, worktree, ACI, mutation-verification, and
execution backend. No Hermes runtime is vendored into Laplace.

## Boundary

The supported repository path is:

```text
Hermes
  -> stdio MCP
  -> laplace-zetsu-mcp
  -> authenticated loopback Zetsu
  -> Laplace authorization / worktree / ACI / deterministic verifier
```

Laplace already exposes the MCP bridge:

```bash
laplace-zetsu-mcp \
  --repo <repo> \
  --state-root <state-root> \
  --endpoint http://127.0.0.1:8765/mcp
```

The bridge obtains its bearer credential from Laplace local state or the
existing token environment. Never copy the bearer value into Hermes
configuration, project files, skill content, command arguments, cron prompts, or
logs.

## Register Zetsu with Hermes

Current Hermes owns its MCP configuration. Register the existing bridge with:

```bash
hermes mcp add zetsu \
  --command laplace-zetsu-mcp \
  --args \
  --repo "$PWD" \
  --state-root "$PWD/.runtime/v2-live-cert" \
  --endpoint http://127.0.0.1:8765/mcp
```

In the reviewed Hermes source, `--args` is `argparse.REMAINDER`; it must be the
last Hermes option. Everything after it belongs to `laplace-zetsu-mcp`.
Do not append Hermes `--env`, `--auth`, `--preset`, or other Hermes options
after `--args`.

The Laplace bridge loads its own local credential, so do not pass bearer
headers, bearer values, or token variables through Hermes MCP configuration.

Inspect the Hermes-owned registration with:

```bash
hermes mcp list
hermes mcp test zetsu
```

Hermes conceptually exposes configured MCP servers as dynamic MCP toolsets. At
the pinned Hermes snapshot, live one-shot validation rejected `mcp-zetsu` as an
unknown `--toolsets` entry but accepted the raw configured server alias `zetsu`.
Runtime tools were named `mcp__zetsu__<tool>`. For this pinned integration, use
`zetsu` in explicit session/per-job allowlists; treat `mcp-zetsu` as descriptive
upstream terminology rather than a certified invocation alias.

## Hard no-bypass repository boundary

A skill is guidance, not a security boundary. The Hermes session itself must be
started with an explicit toolset selection.

Minimal governed repository session:

```bash
hermes chat \
  --toolsets "zetsu,skills" \
  --skills laplace-governed
```

Safe frontend convenience features may be added without granting native
repository I/O:

```bash
hermes chat \
  --toolsets "zetsu,skills,session_search,memory,cronjob" \
  --skills laplace-governed
```

For governed repository work, do **not** enable any of:

```text
terminal
file
code_execution
debugging
coding
hermes-cli
hermes-acp
hermes-api-server
hermes-cron
all
*
```

`coding` is specifically forbidden: current Hermes defines it as a coding
posture/toolset containing native file, terminal, `execute_code`, and delegation
capabilities. Platform-wide toolsets such as `hermes-cli` are also broad
supersets. The explicit `--toolsets` pin is therefore part of this integration
contract.

Do not use Hermes `-w` / `--worktree` or a repository cron `workdir` as an
alternate repository execution path. Repository reads and mutations must remain
Zetsu MCP calls.

## Hermes-specific skill

Laplace already has `.agents/skills/zetsu/SKILL.md`; that is a separate
Codex/Laplace integration surface. Do not bulk-trust the Laplace repository as a
Hermes skill source.

Install only the Hermes-specific template:

```text
integrations/hermes/laplace-governed/SKILL.md
```

For certification, use isolated Hermes state:

```bash
export HERMES_HOME="$PWD/.runtime/v35/hermes-home"
mkdir -p "$HERMES_HOME/skills/laplace-governed"
cp integrations/hermes/laplace-governed/SKILL.md \
  "$HERMES_HOME/skills/laplace-governed/SKILL.md"
hermes skills list
```

Hermes remains responsible for normal skill installation/update lifecycle.
Laplace does not implement another Hermes skill registry.

## Sessions

Session persistence, search, and resume are Hermes-owned frontend behavior:

```bash
hermes sessions list
hermes chat --continue
# or
hermes chat --resume <session-id>
```

Hermes session history is not authoritative Laplace mutation or verification
evidence.

At the pinned Hermes snapshot, top-level one-shot `-z` does not restore
`--resume` / `--continue` conversation history. Use normal `hermes chat` for
resume certification. Live resume also restored Hermes' persisted YOLO approval
mode; approval-mode persistence is frontend state and never replaces the
explicit restricted `--toolsets` boundary required for governed repository work.

## Memory

Hermes built-in memory and optional memory providers remain frontend/user
convenience state. The current provider CLI includes:

```bash
hermes memory status
```

Hermes memory does not replace Step 3.6 hierarchical memory, provenance,
retention, or consolidation policy.


## Project context and Personal Corpus

`project_context` is authenticated Personal Corpus retrieval; it is not a
repository-identity or repository-authorization oracle. An empty/ungrounded
packet can therefore mean that no matching indexed personal source exists.

For the Step 3.5 live gate, the supported Personal Corpus API indexed the current
checkout's `README.md`; `project_context(query="Laplace")` then returned
`grounded=true`, non-empty evidence, and snapshot revision 2. An attempted
second Python-source ingestion returned HTTP 409 after `README.md` had already
been indexed. That partial-index observation is retained as a non-blocking corpus
issue for Step 3.5; it did not affect the governed repository-agent path.

Repository revision synchronization is independent of Personal Corpus
materialization. The verified `agent_task` live gate synchronized the clean
canonical authorization grant to
`0c88eb3b748f36701d5246ab5b13b6b384d5f972`, after which
`revision_sync_required=False`.

## Delegation: separate certification gate

Delegation is **not** included in the default governed toolset above.

Current Hermes subagents inherit the parent's enabled toolsets and cannot grant
themselves a different model-facing toolset. Therefore delegation may be tested
only from an explicitly pinned parent such as:

```bash
hermes chat \
  --toolsets "zetsu,skills,delegation" \
  --skills laplace-governed
```

Before treating delegation as certified for governed repository work, inspect
the child-visible tools and prove that it has no native `file`, `terminal`,
`code_execution`, `coding`, or platform-wide toolset route to the repository.
A child claim is not proof of that boundary.

## Cron

Hermes owns scheduling. A scheduled repository task must set a per-job toolset
restriction:

```text
cronjob(
    action="create",
    name="laplace-repository-check",
    schedule="...",
    enabled_toolsets=["zetsu", "skills"],
    skills=["laplace-governed"],
    prompt="... self-contained governed repository task ...",
)
```

Current agent-facing management uses actions such as:

```text
cronjob(action="list")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

The CLI likewise uses `hermes cron run`, not an obsolete `trigger` command.
Do not set a repository `workdir`, and do not grant native file/shell/code or
broad platform toolsets to a governed cron job.

Cron sessions are fresh. The prompt and attached skill must therefore state the
repository identity, read/write intent, and deterministic verifier requirement
explicitly.

## Zetsu contract from Hermes

With MCP server name `zetsu`, use the Zetsu tools exposed by the running bridge.

Hermes runtime tool names observed in the pinned live gate use the form
`mcp__zetsu__<tool>` (for example `mcp__zetsu__project_context` and
`mcp__zetsu__agent_task`).

The currently certified Laplace contract includes repository context, agent task
submission/status/cancellation, deterministic verification, and persisted
result retrieval.

A mutating task must carry an explicit Laplace-accepted
`verification_argv`. Mutation authority remains transient per turn and is never
inferred merely because `apply_patch` exists in policy. A write is not complete
until Laplace reports the latest mutation verified.

Hermes must not replace or weaken:

- repository authorization;
- canonical worktree ownership and containment;
- path validation;
- bounded ACI;
- mutation/verified epochs;
- deterministic verifier policy;
- promotion rules;
- cancellation/result ownership.

## Step 3.5 live acceptance

Record PASS / FAIL / BLOCKED independently for:

1. Hermes binary/version and current CLI contract;
2. isolated Hermes skill discovery with `hermes skills list`;
3. MCP registration and `hermes mcp test zetsu`;
4. pinned server-alias/tool discovery (`zetsu`, `mcp__zetsu__...`);
5. authorized structural/read context;
6. path-escape rejection;
7. read-only agent task;
8. mutation without accepted verifier rejected;
9. mutation with explicit accepted verifier plus latest-mutation verification;
10. status/result retrieval and cancellation when practical;
11. session list/resume;
12. memory status smoke, without treating Hermes memory as Laplace evidence;
13. delegation boundary separately proving no native repository authority;
14. cron create/run/remove with per-job
    `enabled_toolsets=["zetsu", "skills"]`;
15. credential scan showing no Laplace bearer value in Hermes config/logs;
16. production checkout unchanged.

If Hermes, its provider/model, quota, or another required external service is
unavailable, mark that live item **BLOCKED**, never PASS.


## Step 3.5 live certification result

The mandatory external-compatibility gate passed against the reviewed snapshots.
This PASS is evidence for finalization only; the v3 ladder remains Step 3.5
YELLOW until the final patch is committed, pushed, remote-SHA verified, and the
immutable online commit is inspected.

Live evidence:

- read/context: PASS — `project_context`, `ast_context`, `search`, and
  `get_evidence` returned governed evidence through `mcp__zetsu__...` tools;
- containment: PASS — `../laplace/README.md` was rejected with
  `ast_context_path_escape`;
- mutation without verifier: PASS by expected safety rejection —
  `agent_mutation_requires_verifier`;
- verified mutation: PASS — session `v35-hermes-verified-20260828a` changed only
  `tests/test_zetsu_mcp.py` inside the governed worktree, ran exactly
  `pytest tests/test_zetsu_mcp.py -q`, reached verified mutation epoch 1, did
  not promote to the canonical repository, and persisted result
  `res_ce758d677813d2f0bf0c18e495785fdb`;
- lifecycle/result: PASS — task status, `verify`, and `get_result(result.json)`
  agreed on result identity, changed path, verifier success, and durable hashes;
- cancellation boundary: PASS for terminal-state refusal — cancellation returned
  `cancelled=false` for the already-SUCCEEDED task; queued cancellation was not
  practically exercised because the scheduler had no queued task;
- sessions: PASS — session search located prior work, and normal
  `hermes chat --resume` / `--continue` restored the exact prior chunk ID;
  top-level `-z` resume/continue was a harness mismatch for this Hermes snapshot;
- memory UX: PASS — a temporary frontend-memory marker was added, persisted, and
  removed; this is not Step 3.6 memory certification;
- delegation: PASS — parent was explicitly restricted to
  `zetsu,skills,delegation`; pinned-source inheritance was inspected and the one
  leaf child used only the Zetsu project-context path. Delegation id:
  `deleg_9867603b`;
- cron: PASS for create/list/manual-run/remove — job `24a5d87f1628` stored
  `enabled_toolsets=["zetsu", "skills"]`, no repository `workdir`, and its
  output proved grounded `mcp__zetsu__project_context` access. Automatic gateway
  firing was not exercised because the isolated Hermes gateway was not running;
  manual execution also exposed a non-fatal MCP stdio coroutine warning and did
  not populate `hermes cron runs`, while the output artifact was written;
- credential safety: PASS — an actual-value scan of isolated Hermes state found
  zero Laplace bearer-token matches;
- repository safety: PASS — the development canonical checkout and immutable
  production/control-plane checkout remained clean and unchanged.

The live gate therefore supports the Step 3.5 architectural decision
**EXTERNALIZE**: Hermes remains the frontend/session/skills/memory-UX/cron layer,
while Laplace/Zetsu retains repository authorization, worktrees, ACI, transient
mutation authority, deterministic verification, status/cancellation/result
ownership, and promotion policy.

## Upstream references reviewed for Step 3.5

The reviewed Hermes snapshot covers:

- `hermes_cli/subcommands/mcp.py` — MCP CLI, including remainder-style
  `--args`;
- `toolsets.py` and toolset reference — native, composite, platform, and
  dynamic MCP toolsets;
- `agent/coding_context.py` — coding posture and explicit toolset pin behavior;
- `hermes_cli/main.py` and sessions code — normal chat resume/continue versus one-shot dispatch;
- skills documentation — `~/.hermes/skills` and current skills CLI;
- cron tooling/scheduler — per-job `enabled_toolsets`, manual `run`, output, and execution-ledger behavior;
- delegation documentation/tool — inherited parent tool access.

This is an external interoperability contract. Future Hermes changes require
re-running the live Step 3.5 compatibility gate; they are not silently treated
as certified.
