# Laplace architecture generation 3

Laplace is a local research and engineering workspace. v2 provides a neutral
standalone `LaplaceCore`, an authenticated optional Zetsu/Codex adapter, private
retrieval, structured memory, deterministic verification, and isolated
repository-agent worktrees. It is localhost-only by default; documents,
credentials, model prompts, and repository contents are not uploaded to a cloud
service.

Package SemVer is `2.0.0`; it is intentionally separate from the architecture
generation and the non-A6000 certification campaign. The immutable certified
production/control-plane checkout is `/home/giando/work/laplace` on
`production/v1`; do not modify it while developing or validating v2.

## Runtime model

Laplace Core owns provider-neutral capabilities and can be constructed without
MCP or Zetsu. Zetsu is an authenticated adapter over the same Core services. The
certified serving policy uses Qwen3.8 P7/MTP for Quality/Standard work. CodeV is
an optional, bounded RTL/SystemVerilog Economy specialist; it is intentionally
disabled in the normal development topology:

```bash
cd /home/giando/work/laplace-v2
laplace zetsu start --nocodev --repo /home/giando/work/laplace-v2
laplace zetsu status --repo /home/giando/work/laplace-v2 --json
```

`--nocodev` is an explicit topology choice, not degraded operation. C8 records
the SiliconMind-vs-CodeV experiment as `BLOCKED` because the candidate artifact
and certified full-topology vLLM executable are unavailable locally. No model
promotion is implied by fixture or unit-test results.

## Codex/Zetsu workflow

From a canonical administrator-registered and authorized project:

```bash
cd <registered-project>
laplace zetsu
laplace zetsu codex
```

The bare command configures managed project usage, checks MCP and authenticated
retrieval, and reports repository-bound readiness. It reports
`repository_not_registered` or `repository_not_authorized` instead of claiming
that repository work is ready. One-time administrator registration/grant remains
required because authorization is a security boundary.

Useful read-only inspection commands are:

```bash
laplace zetsu status --json
laplace zetsu sessions --json
laplace zetsu worktrees --json
```

`agent_task` uses a server-authorized isolated worktree, an owner/repository
binding, the current clean canonical commit, and a deterministic verification
allowlist. A clean terminated session is safely reclaimed; a dirty session is
preserved. A committed canonical HEAD advance is synchronized before a new
session. Uncommitted caller-only content is not copied into an agent worktree and
is reported as unmaterialized state when relevant.
The machine-readable error for caller-only dirty content is
`repository_state_not_materialized`.

The normal result is a compact inline response. Large successful results are
durably persisted and exposed through bounded paging; they are not converted into
false execution failures or injected wholesale into Codex context. Explicit
`session_id` resumes the owner-bound session and does not allocate another slot.

## Core capabilities

- **Memory:** owner/project-scoped structured memory with a fail-closed schema
  compatibility contract and deterministic lexical token matching. It is separate
  from the Personal Corpus/RAG index, which retains file/page/section/chunk
  provenance.
- **Rules and context:** authoritative rules are assembled before advisory memory
  and retrieval. The Context Planner bounds and compacts context without changing
  numerical results.
- **Repository intelligence:** RepoMap and symbol/reference indexing are advisory
  under an explicit lightweight parser contract. Exact file reads remain required
  for complex C/C++ and Verilog/SystemVerilog constructs.
- **ACI:** bounded reads support continuation pages; large writes use ordered,
  hashed, same-directory chunks with atomic finalization and cancellation-safe
  abort. No shell or network expansion is provided.
- **Trajectories and hooks:** owner-bound event replay, cooperative hook deadlines,
  fail-closed security pre-hooks, observational post-hooks, and explicit late
  callback diagnostics preserve deterministic lifecycle state.
- **Skills and consolidation:** skills require approval/activation boundaries;
  idle consolidation is shadow-only and does not silently mutate authoritative
  memory or repository state.
- **Logical subagents:** queue admission is FIFO and bounded. GPU reservations
  are atomic, transient scarcity queues, queue deadlines are separate from
  execution budgets, and reservations release on every terminal path.

## Standalone commands

Install the package and development tools locally with `uv` when available:

```bash
uv sync --extra dev
# pip fallback:
python -m pip install -e '.[dev]'
laplace --version
```

The v2 tests and quality gates are deterministic and do not require a live model:

```bash
PYTHONPATH=src pytest -q
ruff check src tests
mypy src
git diff --check
```

The legacy project/research CLI remains available for compatibility, but the v2
control-plane workflow is the Core/Operator/Zetsu path above. Local numerical
work is performed by deterministic Python libraries; models interpret structured
results and never execute commands extracted from documents or model output.

## Persistent terminal agent

`laplace chat` is a thin, loopback-only terminal client of the resident
Operator—not a second Core, scheduler, sandbox, or model process. Its default
mode creates one owner-bound repository-agent session and reuses it across
natural-language turns. Plain Qwen/Core conversation is explicitly selected
with `/mode chat`.

```bash
laplace zetsu start --nocodev
cd <registered-project>
laplace chat --repo-id <logical-repository-id>
```

Agent inspection works without CodeV. Edits require an explicit deterministic
`--verification` argv; `--access confirm` asks before a verifier-backed turn.
The terminal and Operator GUI share the same session, worktree lifecycle,
durable transcript, cancellation endpoint, status evidence, and paged result
artifacts. See [Laplace Chat](docs/LAPLACE_CHAT.md) and the
[feature matrix](docs/FEATURE_MATRIX.md).

## Security and lifecycle boundaries

Only administrator-registered canonical Git roots can receive repository grants.
Clients cannot supply arbitrary filesystem roots. Grants are owner-scoped and
validated with repository identity/device/inode checks. Symlink traversal,
nested repositories, submodules, arbitrary shell/network access, cross-user
session reuse, and dirty-worktree deletion fail closed.

Quota recovery reconciles clean failed/expired sessions before denying capacity,
without stealing genuine active work or deleting dirty worktrees. Use
`laplace zetsu sessions --json` for diagnosis; routine SQLite access is not part
of the user workflow.

## Start, status, and shutdown

```bash
laplace zetsu start --nocodev --repo <development-repository>
laplace zetsu status --repo <development-repository> --json
laplace zetsu sessions --json
laplace zetsu stop
```

Runtime shutdown validates owned process identities and reports survivors. Do not
use `pkill`, broad GPU matching, force removal of worktrees, or commands that
touch the production checkout. For deployment rollback, stop the v2 runtime,
restore the previously certified `production/v1` control-plane artifact through
the deployment procedure, and verify its recorded commit; do not force-push or
rewrite the v2 branch.

## Documentation and evidence

- [User workflow](docs/USER_GUIDE.md)
- [Laplace Chat terminal guide](docs/LAPLACE_CHAT.md)
- [Feature matrix and production evidence](docs/FEATURE_MATRIX.md)
- [Governed self-improvement guide](docs/SELF_IMPROVEMENT.md)
- [Zetsu operations](docs/ZETSU.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Worktree lifecycle](docs/AGENT_WORKTREES.md)
- [Qwen3.8 migration and rollback](docs/QWEN38_PRODUCTION_MIGRATION.md)
- [Release policy](docs/RELEASE_POLICY.md)
- [Corrective roadmap certifications](docs/v2-corrective-roadmap/certifications)

Each certification records exact tests, limitations, security decisions, and
whether production was modified. G0 was already certified and is not repeated by
the corrective pass.
