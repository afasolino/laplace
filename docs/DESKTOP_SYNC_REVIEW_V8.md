# v8 desktop sync review

The v8 fixture suite verifies clean, dirty, staged, unstaged, untracked,
divergent-base, same-HEAD/different-branch, detached, symlink, hardlink, submodule,
nested-repository, file-quota, binary, interrupted, replay, invalid-host,
cross-user, patch-export, and audit scenarios.

Classification:

- repository inspection, approval, durable client records, and patch export:
  **complete usable**;
- `FixtureSyncService`: **reference implementation**;
- SSH and HTTPS transports: **partial staged — verification policy only**;
- server-side transfer restart: **partial staged — in-memory reference buffer**;
- force push and automatic merge: **intentionally unsupported**.

Patch apply requires exact approval, SHA-256, size, paths, base HEAD, branch, and
a clean target. Untracked files are shown but excluded. Binary patches, renames,
copies, links, submodules, nested repositories, mount crossings, and arbitrary
folder selection fail closed.

Linux fixtures run locally. Portable logical-path contracts reject Windows
separator injection; native Windows execution is certified only by the remote
matrix. Folder upload is a personal-corpus workflow. Documentation must never
describe arbitrary folders as repositories.

