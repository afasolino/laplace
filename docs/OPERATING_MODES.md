# Operating modes

## Standalone Core

Standalone Laplace constructs `LaplaceCore` directly and can use local retrieval,
memory, rules, repository intelligence, trajectories, hooks, skills, bounded ACI,
logical scheduling, and deterministic verification without MCP or Zetsu. A
repository-agent implementation is injected through the neutral Core service
boundary; absent authorization or a bound adapter fails closed.

## Authenticated Operator/Zetsu mode

The Operator provides registered-user authentication, owner-private Personal
Corpus/RAG, canonical repository registrations and grants, isolated worktrees,
model routing, quotas, lifecycle scheduling, and `/mcp`. Zetsu is an optional
Codex adapter over those same Core services. It does not infer access to a client
filesystem, shell, network, or remote folder.

## Runtime topology

Quality and Standard use the certified Qwen3.8 P7/MTP serving route. CodeV is an
optional Economy RTL/SystemVerilog specialist. The ordinary local topology is
explicitly started with:

```bash
laplace zetsu start --nocodev
```

CodeV absence in this mode is intentional. Full Qwen-plus-CodeV startup is
reserved for an explicit RTL experiment or operator action and must be followed
by a return to `--nocodev` when complete.

## Development and production

The v2 development worktree is `/home/giando/work/laplace-v2` on
`feature/laplace-v2`. The stable production checkout is
`/home/giando/work/laplace` on `production/v1` and is read-only during v2 work.
Runtime state, credentials, databases, worktrees, logs, and result artifacts live
outside Git under the configured state root. There is no shared-folder shortcut
between the development and production source trees.
