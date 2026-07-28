# Storage, ownership, and retention

Laplace keeps source code in Git and runtime state outside Git. The configured state root must be private and must not be treated as model memory. NVIDIA VRAM, system RAM, iGPU memory, and Ryzen AI NPU memory are separate pools.

| Data class | Owner and logical location | Indexed | Default lifecycle and access |
| --- | --- | --- | --- |
| Conversation messages | User; external conversation SQLite | No, unless explicitly sent in a request | Persists until conversation deletion; owner-only |
| Drafts | User; active conversation record | No | Replaced on save/submit; owner-only |
| Conversation attachments | User/request scoped when enabled | Not automatically | Separate from the personal corpus; owner-only |
| Personal corpus sources | User; `<state_root>/personal_corpora/owners/<HMAC pseudonym>/` | Only after explicit confirmation | Immediate retrieval removal on delete; 30-day soft delete by default |
| Shared governed corpus | Administrator-governed local corpus | Governed pipeline only | Separate provenance/licence/access policy; no automatic personal promotion |
| Repository files | Repository owner; registered Git root | Not by personal ingestion | Access only through active logical grants |
| Agent worktrees | Requesting user; external worktree root | No | Clean release; dirty/failed retention, 30-day expiry by default |
| Generated artifacts | Requesting user; external artifact content store | No | Owner/policy scoped; SHA-256 verified on read |
| Audit/provenance logs | Operator-controlled external logs | No | Bounded by deployment policy; secret/path redaction |
| Sessions and registry | Local administrator; external auth state | No | Session expiry/revocation; registry retained until controlled replacement |

Personal-corpus Operator inventory is aggregate and pseudonymous. Operator content access is disabled by policy. Canonical repository/worktree paths are Operator-only. Normal users receive logical IDs and clean filenames.

## Backup set

Take a consistent backup with the Operator stopped or through SQLite's backup mechanism. Record the application Git commit and schema versions. Back up the external registry, databases, keys, source content, artifacts, and retained worktrees according to organizational policy. The personal-corpus HMAC key and artifact pseudonym key must remain paired with their databases. Encrypt secret-bearing backups, preserve `0700` directories and `0600` files, and do not copy them into certification bundles.

Models are separately provisioned local artifacts and do not need to be mixed into application-state backups. Certification outputs are evidence, not restorable runtime state.

## Restore and verification

Restore into a new private external state root, keep the service on loopback, validate the user registry, inspect SQLite integrity, then run health/readiness. Verify one owner-scoped conversation, one personal corpus search with file/page/section/chunk citation, one source hash, one repository grant, one retained worktree status, and one artifact integrity check. Only then restore SSH/reverse-proxy access.

## Disk pressure and purge

Ingestion fails closed before crossing the protected free-space threshold. It does not delete user sources to make room. Operators should expire soft-deleted personal sources, exported/discardable worktrees, old logs, and old certification outputs using policy-specific tools. Never delete active SQLite/WAL files or arbitrary owner directories.

Purge must retain the minimal audited lifecycle event required by policy while removing source and derived content. Personal deletion and shared-corpus removal are distinct actions.

## Safe shutdown

1. Stop accepting new browser/Agent work and allow bounded writes to finish.
2. Stop the Operator service.
3. Stop only model processes whose Laplace ownership record matches executable, model, host, and port.
4. Confirm ports 8765, 8102, and 8103 are closed as expected.
5. Confirm unrelated GPU processes remain and Laplace-owned GPU memory is released.

Never use broad `pkill`, GPU-wide process matching, or recursive deletion as a shutdown mechanism.
