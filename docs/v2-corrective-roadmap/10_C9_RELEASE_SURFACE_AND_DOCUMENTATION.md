# C9 — Align Release Metadata and Documentation with Actual v2

## Goal

Make the repository's public/release surface accurately describe the implemented system.

## Required updates

Inspect and update, where applicable:
- `pyproject.toml`;
- `README.md`;
- `docs/USER_GUIDE.md`;
- `docs/ZETSU.md`;
- deployment/runtime documentation;
- troubleshooting;
- architecture documentation;
- rollback instructions.

Remove obsolete claims that describe the project primarily as the old v0.7/FormalScience/Ollama workspace if those claims no longer represent the current branch.

Document:
- standalone Laplace Core;
- optional Zetsu adapter;
- Qwen3.8 quality/standard routing;
- optional CodeV and `--nocodev`;
- repository-agent isolation and worktree lifecycle;
- queueing/capacity behavior;
- bounded ACI;
- memory vs Personal Corpus/RAG;
- rules;
- repository intelligence;
- trajectories;
- context planner;
- skills/hooks;
- idle consolidation;
- logical subagents;
- large-result paging;
- start/status/test/stop;
- production/development worktree separation;
- rollback.

## Versioning

Choose a v2-compatible package version using the repository's existing versioning policy. Do not invent a release tag unless the project already defines the tagging step.

All CLI examples must be executable against current code.

Run documentation/reference checks sufficient to catch stale command/module names.
