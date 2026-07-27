# User capability and repository isolation

## Basic

Basic is enforced in the backend, not only in the GUI. Basic can call `/api/v1/chat`
and inspect its capability record. Operator endpoints return 403, agent-session
creation returns 403 before Git is invoked, and every chat backend call receives
`tools=()`. Basic cannot supply a repository root, command, path, tool definition, or
model endpoint.

## Plus

An operator first registers a canonical Git toplevel under a repository ID and grants
that ID to a Plus user at a verified commit. The client can submit only the ID. Before
the first model call, the server creates a new detached worktree below its controlled
per-user session root and freezes this binding:

- user and session IDs;
- repository ID and canonical registered root;
- dedicated worktree and base commit;
- grant revision;
- allowlisted tool policy and quotas;
- network disabled;
- allowlisted environment only.

Every later action re-reads the repository grant. Revocation or any revision change
invalidates the session immediately. A session owned by one user cannot be named by
another.

Path validation rejects absolute paths, `..`, symlinks, hard links, foreign filesystems,
bind-mount points found in `/proc/self/mountinfo`, nested `.git` roots, and configured
submodule paths. The only production mutation tool accepts a unified text patch,
rejects binary/symlink/rename/cross-path forms, validates every diff path, runs fixed
`git apply --check`, applies with whitespace errors fatal, and runs `git diff --check`.
No command text emitted by a model is executed.

Clean worktrees can be removed explicitly. Dirty worktrees are preserved and reported;
they are never force-deleted. Audit JSONL records hashes and binding/action metadata,
not credentials or prompt/response bodies.

The HTTP API exposes create, message/run, status, and cancel operations. Status and
cancel revalidate the user and live grant. Final result visibility is scoped to the
owning session. Audit rows include capability, session, mode, repository/sandbox IDs,
requested/effective lane, route, tool/network policies, queue wait, limits,
validation/escalation, and trace ID. Denials are recorded without raw prompts or
protected repository contents.

## Residual isolation limits

This host does not expose an established repository-scoped container or mandatory
access-control sandbox to Laplace. The implementation therefore uses the strongest
available repository controls: server-owned roots, detached worktrees, canonical path
and mount checks, fixed Git argv, fixed CWD, minimal environment, no network tool, and
per-session state. These controls do not claim to be a kernel security boundary
against arbitrary native code. Consequently, Plus does not expose a general shell,
Docker socket, mount operation, arbitrary executable, or host-administration tool.
