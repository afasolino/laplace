---
name: laplace-governed
description: Use when Hermes must inspect, modify, verify, delegate, or schedule repository work governed by Laplace.
version: 0.3.0
platforms: [linux]
metadata:
  hermes:
    tags: [laplace, zetsu, mcp, governance, verification]
    related_skills: []
---

# Laplace Governed Repository Work

## When to use

Use this skill for repository work that must preserve Laplace authorization,
canonical contained worktrees, bounded ACI, transient mutation authority, and
deterministic post-mutation verification.

This skill is advisory. The hard boundary is the Hermes session toolset.

For the pinned Hermes Step 3.5 snapshot, use the configured MCP server alias
`zetsu` in explicit toolset allowlists. Live one-shot validation rejected the
conceptual `mcp-zetsu` alias; runtime tools were named `mcp__zetsu__<tool>`.

## Required Hermes boundary

For ordinary repository work, the parent session must be explicitly pinned to:

```text
zetsu,skills
```

Frontend convenience may add `session_search`, `memory`, or `cronjob`.

Do not expose these repository-bypass toolsets:

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

Do not use Hermes `-w` / `--worktree` or a repository cron `workdir` as a
substitute for Laplace worktree governance.

## Procedure

1. Use the available `mcp__zetsu__*` structural/read tools for repository context.
2. Use the Zetsu agent-task tool for coherent repository work.
3. For a mutation, provide an explicit deterministic Laplace-accepted
   `verification_argv` before work begins.
4. Use Zetsu task status/cancellation interfaces for lifecycle control.
5. Establish that the latest mutation is verified through Zetsu/Laplace
   evidence.
6. Retrieve the persisted Zetsu result when needed.

## Governance rules

- A Laplace authorization, containment, path, verifier, revision, or promotion
  rejection is final for that operation. Never retry it using Hermes-native
  filesystem/shell/code tools.
- Mutation authority is transient. The presence of `apply_patch` in policy is
  not by itself permission to mutate.
- Never place bearer tokens, authorization headers, or Laplace credentials in
  Hermes prompts, skills, cron definitions, project files, or MCP arguments.
- Hermes sessions and memory are frontend convenience state, not authoritative
  repository evidence.
- Re-specify the restricted toolset on resumed sessions. Hermes may persist
  frontend approval state (including YOLO mode), which does not confer Laplace
  repository authority.
- Delegation is separately gated. If enabled, the parent must remain explicitly
  MCP-restricted and child-visible tools must be inspected before repository
  delegation is certified.
- Scheduled repository tasks must use per-job
  `enabled_toolsets=["zetsu", "skills"]`.
- A child/subagent self-report is not proof of repository mutation,
  verification, or tool-boundary compliance.

## Completion checklist

Repository work is complete only when:

- the intended repository and access mode are authorized;
- any mutation occurred through Zetsu/Laplace governance;
- the deterministic verifier passed after the latest mutation;
- no unresolved failure or pending mutation remains;
- status/result evidence agrees with the claimed outcome;
- no Hermes-native repository bypass tool was used.
