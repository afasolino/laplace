# Artifact provenance and privacy

Every generated artifact receives a sortable 128-bit ULID. The user-visible path and content stay clean; identity, ownership, repository/worktree binding, source-state fingerprint, producing model/tool, parent lineage, SHA-256, visibility, and authorization policy live in the external SQLite registry.

Owner identifiers exposed in provenance are HMAC-based pseudonyms using a separate private 32-byte key. The raw internal user ID is used only for server-side authorization and the private lifecycle audit. Normal exports contain neither.

Create writes content to a mode-`0600` temporary file, fsyncs, atomically renames it, then commits registry metadata. Rename preserves the artifact ID and parent lineage. Content update atomically replaces bytes and updates the hash. Delete removes content and retains a tombstone.

Every read/export recomputes SHA-256 with constant-time comparison. A mismatch emits a denied lifecycle event and fails as `artifact_integrity_failure`. Queries require artifact ID plus authenticated owner and repository binding, so cross-user and cross-repository access returns not found.

Normal users receive a clean name, relative path, creation time, visibility, content hash, and an explicit download link only for their generated result. They cannot enumerate artifact IDs, storage keys, owner pseudonyms, or other users' entries. The compact internal export is an authenticated Operator/admin action used only for an explicitly requested audit/certification bundle.
