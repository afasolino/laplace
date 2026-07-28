# Operating modes

## Desktop/local mode

Desktop mode is for one OS user working in an explicitly selected FormalScience
project. The `laplace` CLI and project GUI operate on project-owned sources,
collections, attachments, conversations and artifacts. A configured local provider
is optional; deterministic fixture configuration is available for tests only.

Desktop mode does not create server users, personal corpora or managed repository
grants. A project is not a server repository registration.

## Server/multi-user mode

Server mode uses registered-email authentication, revision-bound sessions,
independent capabilities, owner-private personal corpora, governed corpora,
registered repositories and isolated Agent worktrees. Server state is outside Git
and clients see logical IDs rather than canonical paths.

The Operator surface administers users, repositories, providers, routes, queues and
governance. Provider invocation and model-process lifecycle remain separate.

## Shared contracts, distinct policy

Both modes share:

- provider-neutral request/response records;
- citation and progress presentation records;
- corpus/source/artifact/provenance vocabulary;
- configuration precedence and diagnostic redaction;
- offline fixture evaluation and release gates.

Mode-specific authentication, storage ownership and administration remain explicit.
There is no third GUI and no automatic state synchronization between modes. Desktop
repository synchronization is a confirmed Git protocol described in
[DESKTOP_REPOSITORY_SYNC.md](DESKTOP_REPOSITORY_SYNC.md), not a shared folder.
