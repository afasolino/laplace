"""Reference server/fixture transport for logical-repository synchronization."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .architecture import RepositoryService
from .sync_protocol import SyncError, SyncReceiptV1, validate_patch


@dataclass(frozen=True)
class RegisteredSyncRepository:
    logical_repository_id: str
    canonical_root: Path


class FixtureSyncService:
    """CPU-only resumable transport; it stores patches but never pushes."""

    def __init__(
        self,
        *,
        repositories: RepositoryService,
        registrations: tuple[RegisteredSyncRepository, ...],
        maximum_patch_bytes: int = 64 * 1024 * 1024,
        maximum_files: int = 2_000,
    ) -> None:
        self.repositories = repositories
        self.registrations = {
            item.logical_repository_id: RegisteredSyncRepository(
                item.logical_repository_id,
                item.canonical_root.resolve(),
            )
            for item in registrations
        }
        self.maximum_patch_bytes = maximum_patch_bytes
        self.maximum_files = maximum_files
        self._buffers: dict[str, bytearray] = {}
        self._metadata: dict[str, tuple[str, str, str]] = {}

    def _authorize(self, owner_id: str, logical_repository_id: str) -> None:
        try:
            self.repositories.require_active(owner_id, logical_repository_id)
        except RuntimeError as exc:
            raise SyncError("sync_repository_not_authorized") from exc
        registration = self.registrations.get(logical_repository_id)
        if registration is None:
            raise SyncError("sync_repository_unavailable")
        root = registration.canonical_root
        if root.is_symlink() or not (root / ".git").exists():
            raise SyncError("sync_repository_unavailable")

    def _require_base(self, logical_repository_id: str, base_head: str) -> None:
        registration = self.registrations[logical_repository_id]
        completed = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(registration.canonical_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env={
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "LC_ALL": "C",
            },
        )
        if completed.returncode != 0 or completed.stdout.strip().lower() != base_head:
            raise SyncError("sync_base_conflict")

    def upload_chunk(
        self,
        *,
        owner_id: str,
        logical_repository_id: str,
        operation_id: str,
        base_head: str,
        offset: int,
        content: bytes,
        final: bool,
        expected_sha256: str,
    ) -> SyncReceiptV1:
        self._authorize(owner_id, logical_repository_id)
        self._require_base(logical_repository_id, base_head)
        metadata = (owner_id, logical_repository_id, base_head)
        existing_metadata = self._metadata.setdefault(operation_id, metadata)
        if existing_metadata != metadata:
            raise SyncError("sync_replay_identity_conflict")
        buffer = self._buffers.setdefault(operation_id, bytearray())
        if offset < len(buffer):
            end = offset + len(content)
            if end > len(buffer) or bytes(buffer[offset:end]) != content:
                raise SyncError("sync_resume_offset_conflict")
            replay_hash = (
                hashlib.sha256(buffer).hexdigest() if final and end == len(buffer) else None
            )
            if replay_hash is not None and replay_hash != expected_sha256:
                raise SyncError("sync_patch_hash_mismatch")
            return SyncReceiptV1(
                operation_id=operation_id,
                accepted_offset=end,
                complete=replay_hash is not None,
                patch_sha256=replay_hash,
            )
        if offset != len(buffer):
            raise SyncError("sync_resume_offset_conflict")
        if len(buffer) + len(content) > self.maximum_patch_bytes:
            raise SyncError("sync_patch_size_exceeded")
        buffer.extend(content)
        patch_hash: str | None = None
        if final:
            patch = bytes(buffer)
            validate_patch(
                patch,
                maximum_bytes=self.maximum_patch_bytes,
                maximum_files=self.maximum_files,
            )
            patch_hash = hashlib.sha256(patch).hexdigest()
            if patch_hash != expected_sha256:
                raise SyncError("sync_patch_hash_mismatch")
        return SyncReceiptV1(
            operation_id=operation_id,
            accepted_offset=len(buffer),
            complete=final,
            patch_sha256=patch_hash,
        )

    def download_patch(
        self,
        *,
        owner_id: str,
        logical_repository_id: str,
        operation_id: str,
    ) -> bytes:
        self._authorize(owner_id, logical_repository_id)
        metadata = self._metadata.get(operation_id)
        if metadata is None or metadata[:2] != (owner_id, logical_repository_id):
            raise SyncError("sync_operation_not_found")
        return bytes(self._buffers[operation_id])

    def sanitized_operation(self, owner_id: str, operation_id: str) -> dict[str, object]:
        metadata = self._metadata.get(operation_id)
        if metadata is None or metadata[0] != owner_id:
            raise SyncError("sync_operation_not_found")
        buffer = self._buffers[operation_id]
        return {
            "schema_version": 1,
            "operation_id": operation_id,
            "logical_repository_id": metadata[1],
            "base_head": metadata[2],
            "received_bytes": len(buffer),
            "canonical_path_exposed": False,
            "force_push": False,
        }
