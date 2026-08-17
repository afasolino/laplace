# Zetsu

Zetsu is the authenticated Laplace↔Codex pairing layer. It is served at `/mcp`
by the existing loopback-bound Laplace Operator process; there is no second
public daemon or weaker identity store. HTTPS terminates at the normal Laplace
ingress, and Qwen, CodeV, databases, and control ports remain private.

Zetsu schema version is 1.1, Skill version is 1.1.0, and the server supports MCP
2026-07-28 plus compatibility with 2025-11-25, 2025-06-18, and 2025-03-26.
Available tools are `search`, `get_evidence`, `project_context`,
`experiment_context`, `delegate`, `rtl_task`, and `verify`. Normal Laplace
capabilities determine which tools each identity sees.

## Configure Codex

Create an owner-bound normal Laplace bearer credential outside Git and export
it in the environment that starts Codex:

```bash
export LAPLACE_ZETSU_TOKEN='<secret value>'
laplace zetsu configure --endpoint https://laplace.example.org/mcp --json
laplace zetsu status --json
laplace zetsu test --json
```

`configure` merges one marked Zetsu block into `$CODEX_HOME/config.toml` (or
`~/.codex/config.toml`) and installs `.agents/skills/zetsu/SKILL.md` in the
detected repository. It never stores the credential. It is idempotent and
upgrades owned v1 configuration. It refuses an unmarked user-owned Zetsu table
or Skill instead of overwriting it.

`status` reports configuration versions, native Codex recognition, MCP
reachability and negotiated protocol, tools, server revision, configured model
identities, and independent Laplace/Qwen/CodeV readiness. `test` additionally
performs an authenticated `tools/list` and a real bounded `search`. `remove`
removes only the marked Zetsu block and managed Skill; unrelated Codex settings
remain content-equivalent.

Run `configure` again to repair owned version or endpoint drift. `status` exits
2 for stale, incomplete, unreachable, or incompatible configuration, making it
safe for automation.

## Compact retrieval

Retrieval is progressive. `search`, `project_context`, and
`experiment_context` default to short ranked excerpts with compact evidence and
chunk IDs. Result count is bounded to 20 and `max_chars` is enforced between
512 and 24,000 characters. Use `get_evidence` only for selected IDs. Full
papers, log trees, repository dumps, and experiment trees are never returned by
default. Owner and source provenance is preserved at every expansion.

`delegate` uses Qwen through quality/standard routing for bounded work.
`rtl_task` accepts one bounded module implementation/repair only after the
existing deterministic eligibility policy passes, then uses the CodeV economy
lane. Zetsu exposes no generic shell or filesystem. Codex continues to use its
own repository, shell, Git, builds, and tests directly for ordinary local work.

## Transport and troubleshooting

Enable the normal bearer API on the Operator service and proxy the whole
Laplace origin, including `/mcp`, through HTTPS. Browser sessions are rejected
at `/mcp`; bearer authentication is mandatory, and an invalid Origin is
rejected. Secrets belong in the protected Operator token file and the client
process environment, never in a repository.

- `missing_token_env`: export the configured environment variable in the Codex
  process environment.
- `mcp_connection_failed`: check HTTPS/DNS, Operator service state, and ingress
  streaming/timeouts.
- incompatible local versions: rerun `laplace zetsu configure`.
- model readiness false: inspect `/api/v1/readiness`; Zetsu reachability alone
  does not certify Qwen or CodeV.
