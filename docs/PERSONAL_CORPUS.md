# Personal reference corpus

## What it is

**Knowledge / My corpus** is an owner-private, local retrieval store. It is separate from conversation attachments, the shared governed corpus, repositories, and Agent worktrees. A personal upload is never promoted to the shared corpus automatically and is never searchable before the user selects **Index accepted files**.

The browser sends only content that the user selects with the folder picker, drag-and-drop, or controlled ZIP fallback. It cannot reveal or submit arbitrary client filesystem paths.

## Upload and index

1. Sign in with the `personal_corpus` capability and open **Knowledge**.
2. Create or select a corpus.
3. Select a folder, drop files/folders, or choose one ZIP fallback.
4. Select **Preview selected folder**.
5. Review every accepted/rejected row, rejection reason, format support label, size, and secret warning.
6. Remove or cancel the staged upload if the manifest is not expected.
7. Select **Index accepted files** explicitly.
8. Use **Search test** and inspect the file/page/section/chunk citation and snapshot revision.

An interrupted staging session remains in the owner-scoped server registry. On reload, the browser queries that registry without storing an upload identifier locally. Reselecting identical files resumes safely; identical already-staged bytes are idempotent. The browser never persists file content. Cancel removes quarantine content. Indexing and upload finalization use idempotency keys.

## Versioned format policy

The read-only `/api/v1/personal-corpus/policy` endpoint is authoritative.

Accepted documents are `.pdf`, `.docx`, `.md`, `.markdown`, `.txt`, `.json`, `.jsonl`, and `.log`. Accepted engineering references are `.py`, `.pyi`, `.v`, `.vh`, `.sv`, and `.svh`. `Makefile`, `Dockerfile`, and `CMakeLists.txt` are the only supported extensionless basenames. Markdown, ordinary text, logs, and unsupported engineering domains are retrieval-only; indexing does not imply compilation or Agent support.

Laplace rejects old Office `.doc`, macro-enabled `.docm`, executables, libraries, object files, images, audio/video, encrypted or password-protected PDFs, arbitrary/nested archives, ZIP symlink or special entries, macro content, embedded executables, and external DOCX relationships.

Validation combines normalized logical path, extension/basename, client MIME as an untrusted hint, magic bytes, binary detection, and parser result. It rejects traversal, absolute/NUL/reserved names, hidden/cache/temp paths, duplicate paths, Unicode normalization collisions, excessive ZIP ratios, and policy quota violations. PDF and DOCX parsing runs in bounded worker processes with timeout and memory limits. OCR is disabled. JSON and JSONL are parsed strictly. Text/code/log extraction preserves lines, normalizes UTF-8, and records replacement count. Secret-like material produces a warning but is not silently changed.

## Retrieval controls

Chat and Agent expose:

- **No retrieval**
- **My personal corpus**
- **Shared governed corpus**
- **Both permitted corpora**
- **Selected personal corpus**

Response details state the requested selection, whether each source class was actually used, corpus identity, snapshot revision, source title, chunk, and citation. Personal retrieval is owner-authorized at request time. Agent receives personal excerpts as read-only context under a policy that denies copying corpus content into the repository. The personal corpus is not an Agent filesystem tool.

The current tiered Chat/Agent path reports shared-corpus requests but does not yet query the shared governed corpus in that path; `shared.retrieval_used=false` is truthful. Existing governed retrieval and Deep Research remain separate. Research has no personal-corpus selector in this release.

## Storage, isolation, and deletion

Runtime data is under `<state_root>/personal_corpora/`, outside Git. Owner directories use an HMAC pseudonym, never an email address. The SQLite registry uses WAL and a busy timeout; source, staging, key, and database permissions are private. Original and extracted content receive SHA-256 hashes. Chunks use deterministic `line-chunks-v1-3000-300` parameters.

Every corpus, upload, source, chunk, download, and search is scoped to the authenticated stable user ID. Cross-user access returns not found and does not disclose filenames or hashes. Operators receive aggregate pseudonymous inventory only; content access is disabled by policy. Download verifies ownership. Deletion deactivates chunks immediately, then retains source bytes for the default 30-day soft-delete period. Purge is an explicit lifecycle operation.

Default limits are 2,000 files per batch, 64 MiB per file, 512 MiB expanded batch, 512 MiB extracted text, 5 GiB active source bytes per user, ZIP ratio 100:1, and 512 MiB protected free disk. Operators should review the policy endpoint because deployment overrides may differ.

## Backup, restore, and recovery

Stop the owning service or take a SQLite-consistent snapshot. Back up `personal_corpora/registry.sqlite3`, `owners/`, `provenance.jsonl`, and `owner_hmac.key` together to encrypted access-controlled storage. The HMAC key is required to preserve owner-directory identity. Never place the backup in Git.

Restore the complete set into a mode-`0700` external state root, preserve file modes, start on loopback, run health/readiness, list corpora as fixture owners, and verify a search citation and hash before restoring remote access. On crash, quarantined staging is not retrieved; the owner can resume an intact staging session or cancel it. Never delete SQLite WAL/SHM files while the service is running.
