# Laplace v2 feature matrix

This matrix is the user-facing map of what Laplace v2 actually provides. It
distinguishes implemented contracts from explicitly bounded or unavailable work;
it is not a promise that a local model, GPU, repository grant, or optional
runtime is present on every host.

| Feature | Public surface | Safety and persistence contract | Evidence |
| --- | --- | --- | --- |
| Ordinary conversation | Operator PWA; `laplace chat --mode chat` | Tool-free `LaplaceCore.chat`; owner-scoped conversation history; no repository worktree | `tests/test_operator_api.py`, chat focused suite |
| Persistent repository agent | Operator Agent view; default `laplace chat` | One owner/repo-bound server session and isolated worktree reused through `/messages`; no local second Core/scheduler | `tests/test_operator_agent_conversation.py`, `tests/test_zetsu_agent_checkpoint.py` |
| Bounded edits | Agent UI edit mode; `laplace chat --verification …` | Exact verifier argv is mandatory after mutation; read access refuses verifier-backed turns; user confirmation is per turn in confirm mode | ACI, checkpoint, CLI tests |
| Monitoring and cancellation | `/status`, `/diff`, `/tests`, `/context`, `/cancel`; GUI buttons | Deterministic GET/status/result paths only; zero model calls for monitoring; cancellation is an authenticated Operator mutation | chat CLI/client tests, Operator API tests |
| Large results | Zetsu and Operator result-page route | Inline response is bounded; artifacts are owner/repo/session-bound and byte-paged; server-local paths are not returned | result-store and Operator conversation tests |
| Session resume | `laplace chat --resume last`; GUI worktree Resume | Local terminal metadata is root/repo-bound; Operator transcript validates remote session identity; corrupt state fails closed | chat session and conversation security tests |
| Repository readiness | `laplace zetsu`, `status --json` | Registered canonical root, identity, grant, clean committed HEAD, and readiness reported separately; no implicit authorization | repository lifecycle/hotfix tests |
| Worktree lifecycle | Zetsu sessions/worktrees; Operator worktree view | Owner-only quota, stale clean reclamation, dirty preservation, durable result evidence, no ordinary force delete | lifecycle, sandbox, hotfix tests |
| Retrieval and personal corpus | Operator Knowledge; Core retrieval | Owner-scoped, incremental, provenance-bearing documents; corpus content is read-only to agent and never mounted into a worktree | corpus/authentication tests |
| Memory, rules, context | Core services | Schema compatibility fails closed; rules precede advisory memory/retrieval; bounded compaction | memory/rules/context tests |
| Bounded ACI | Repository agent | Path validation, continuation reads, atomic chunked writes, no generic shell/network surface | ACI and agent tests |
| Scheduler and GPU admission | Tiered serving; repository/logical agents | FIFO bounded queue, atomic reservation accounting, separate queue/execution budgets, release on terminal paths | corrective scheduler tests |
| Qwen serving | Operator routes and Zetsu | Existing certified local serving/routing is preserved; normal Qwen operation does not require CodeV | serving and runtime tests |
| CodeV RTL specialization | Explicit full topology only | Optional specialist. `--nocodev` is normal for chat/agent; no RTL terminal route is claimed without a live server contract | C8 certification / runtime checks |
| Codex integration | `laplace zetsu`, `laplace zetsu codex` | Authenticated MCP adapter over the same Core; it never silently gains arbitrary local or remote filesystem authority | Zetsu tests and `docs/ZETSU.md` |
| Remote browser access | SSH tunnel or HTTPS reverse proxy | Operator and model ports remain loopback; browser auth is cookie/CSRF protected | `docs/REMOTE_ACCESS.md`, auth/API tests |
| Remote local-folder access | `laplace-client` pair/grant/serve | Explicit paired device plus workspace grant; no browser, Zetsu, or agent silently gains this authority | client service/bridge tests and `docs/LAPLACE_CLIENT.md` |

## Supported operating patterns

1. **Local engineering:** start the Operator with `laplace zetsu start --nocodev`,
   enter a registered repository, and use `laplace chat` or the Agent PWA.
2. **Codex-assisted work:** configure the repository with `laplace zetsu`, then
   start `laplace zetsu codex`. The MCP adapter and terminal/GUI use the same
   authorization and worktree controls.
3. **Another machine, server-owned repository:** use an SSH tunnel or HTTPS
   reverse proxy to the Operator. The repository and worktree remain on the
   Operator host.
4. **Another machine, that machine's local files:** pair `laplace-client` and
   grant a specific workspace with the smallest needed operations. This is an
   explicit local-file channel, not an implicit side effect of chat or Zetsu.

## Deliberate limits

- A repository must be explicitly registered and granted; the logical repository
  ID may need `laplace chat --repo-id …` when it differs from the Git directory.
- Uncommitted caller files are never copied to an isolated worktree. Commit a
  checkpoint and start a new delegation instead.
- The normal `--nocodev` topology has no CodeV/RTL specialist and does not claim
  one.
- A persistent worktree has bounded wall time and quota. Clean failures can be
  reclaimed; dirty worktrees remain recoverable until an owner explicitly acts.
- No cloud inference, telemetry, remote document upload, arbitrary shell, or
  arbitrary network authority is added by these surfaces.
