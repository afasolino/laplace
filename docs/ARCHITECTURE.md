# Laplace architecture generation 3

Laplace architecture generation 3 is a local control plane with a neutral Core and explicit adapters.
`LaplaceCore` composes routing, retrieval, memory, rules, repository intelligence,
trajectories, hooks, skills, consolidation, logical scheduling, and deterministic
verification. It can be constructed without importing or initializing MCP/Zetsu.
The authenticated Zetsu service adapts MCP requests to those Core services and
binds the existing owner-scoped repository-agent implementation through the
neutral `RepositoryAgentService` protocol.

## Serving and topology

The main NVIDIA GPU hosts the certified Qwen3.8 P7/MTP Quality/Standard route.
CodeV is a separate optional Economy route restricted to policy-eligible bounded
RTL/SystemVerilog tasks. `laplace zetsu start --nocodev` starts the normal lighter
topology with CodeV intentionally absent. A full topology is an explicit C8 or
operator action and does not change model promotion policy.

The Operator process owns authentication, bearer-token storage, MCP transport,
model readiness, state roots, lifecycle records, and exact-identity shutdown.
Model serving, Core policy, and MCP presentation remain separate layers.

## Repository-agent isolation

An administrator registers a canonical Git root and grants a logical `repo_id` to
an owner. The agent sandbox resolves only that registered root, validates Git
identity and device/inode binding, and creates a worktree from an exact committed
revision. A clean canonical HEAD advance is synchronized before a new worktree;
uncommitted caller state is never copied. Dirty state is preserved and reported.

The persistent lifecycle manager owns creation, resume, cancellation, terminal
result delivery, safe clean release, stale/expired reconciliation, and quota
accounting. The scheduler queues transient capacity scarcity, atomically reserves
GPU headroom for concurrent admissions, and gives queue wait and execution
budgets separate meanings. `laplace zetsu sessions --json` is the read-only
administrative inspection surface.

## Data and context

Memory is owner/project scoped and deterministic; its schema marker is negotiated
before DDL and incompatible or newer schemas fail closed. Personal Corpus/RAG is
a separate owner-authorized evidence store with file/page/section/chunk
provenance. Rules outrank advisory memory and retrieval. The Context Planner
constructs bounded packets and compacts history without treating summaries as
authoritative state.

RepoMap and symbol/reference results are advisory. The supported repository
intelligence contract is intentionally lightweight for C/C++ and Verilog/SystemVerilog;
complex declarations, preprocessing, elaboration, and parameterized RTL are not
claimed to be fully parsed. Exact file reads are required before mutation.

Bounded ACI provides safe continuation pages for large reads and ordered,
hash-checked chunks for large same-directory writes. Finalization is atomic;
cancellation or drift aborts without copying dirty caller state.

## Lifecycle and evidence

Trajectories append owner-bound events and replay exact state. Skills require
approval and activation. Idle consolidation is shadow-only. Hooks are trusted
host-registered callbacks with cooperative deadlines: Python thread timeouts do
not claim physical termination, late results are discarded and diagnosed, pre
hooks fail closed, and post hooks remain observational.

Logical subagent results are either bounded inline values or durable paged
artifacts. A successful oversized result is still execution success. Completion
cache retention is bounded while durable artifacts remain owner/repository/
session-scoped.

## Production boundary and rollback

Development changes belong in `/home/giando/work/laplace-v2` on
the active review branch. Production checkout locations are deliberately not
part of this architecture contract.
immutable production/control-plane checkout. No test, runtime topology switch,
worktree, or certification step may modify it. Rollback is an explicit deployment
operation to the recorded certified v1 commit; it is never implemented by
force-pushing or rewriting the v2 branch.
