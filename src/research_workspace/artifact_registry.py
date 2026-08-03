"""Atomic artifact content registry with invisible ULID provenance."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence, TypeAlias

from .artifact_provenance import (
    clean_relative_path,
    new_ulid,
    private_hmac_key,
    pseudonymous_owner_id,
)

JsonObject: TypeAlias = dict[str, object]


class ArtifactRegistryError(RuntimeError):
    """Artifact authorization or integrity failed closed."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    pseudonymous_owner_id: str
    owner_user_id: str
    created_at_utc: str
    run_id: str | None
    session_id: str | None
    repo_id: str | None
    worktree_id: str | None
    relative_path: str
    source_state_fingerprint: str
    generator_model_route: str
    producing_tool_call_id: str | None
    parent_artifact_ids: tuple[str, ...]
    content_sha256: str
    visibility: str
    authorization_policy: str
    deleted_at_utc: str | None = None

    def internal(self) -> JsonObject:
        value = asdict(self)
        value["parent_artifact_ids"] = list(self.parent_artifact_ids)
        return value

    def normal(self) -> JsonObject:
        return {
            "name": Path(self.relative_path).name,
            "relative_path": self.relative_path,
            "created_at_utc": self.created_at_utc,
            "visibility": self.visibility,
            "content_sha256": self.content_sha256,
        }


class ArtifactRegistry:
    """Keep clean content separate from authorization and lineage metadata."""

    def __init__(
        self,
        database_path: Path,
        content_root: Path,
        event_log_path: Path,
        pseudonym_key_path: Path,
    ) -> None:
        self.database_path = database_path.resolve()
        self.content_root = content_root.resolve()
        self.event_log_path = event_log_path.resolve()
        for parent in {
            self.database_path.parent,
            self.content_root,
            self.event_log_path.parent,
            pseudonym_key_path.resolve().parent,
        }:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._key = private_hmac_key(pseudonym_key_path)
        self._lock = threading.RLock()
        if not self.event_log_path.exists():
            descriptor = os.open(
                self.event_log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    pseudonymous_owner_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    run_id TEXT,
                    session_id TEXT,
                    repo_id TEXT,
                    worktree_id TEXT,
                    relative_path TEXT NOT NULL,
                    source_state_fingerprint TEXT NOT NULL,
                    generator_model_route TEXT NOT NULL,
                    producing_tool_call_id TEXT,
                    parent_artifact_ids_json TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    authorization_policy TEXT NOT NULL,
                    deleted_at_utc TEXT,
                    UNIQUE(owner_user_id, repo_id, relative_path)
                );
                CREATE INDEX IF NOT EXISTS artifacts_owner_repo
                ON artifacts(owner_user_id, repo_id, deleted_at_utc);
                """
            )
        os.chmod(self.database_path, 0o600)
        os.chmod(self.event_log_path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _owner_root(self, owner_user_id: str, repo_id: str | None) -> Path:
        owner = pseudonymous_owner_id(self._key, owner_user_id)
        repository = repo_id or "personal"
        if not repository.replace("-", "").replace("_", "").isalnum():
            raise ArtifactRegistryError("invalid_repo_id")
        candidate = self.content_root / owner / repository
        if self.content_root != candidate and self.content_root not in candidate.parents:
            raise ArtifactRegistryError("artifact_path_escape")
        with self._lock:
            candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
            root = candidate.resolve(strict=True)
        if self.content_root != root and self.content_root not in root.parents:
            raise ArtifactRegistryError("artifact_path_escape")
        return root

    @staticmethod
    def _record(row: sqlite3.Row) -> ArtifactRecord:
        parents: object = json.loads(str(row["parent_artifact_ids_json"]))
        if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
            raise ArtifactRegistryError("artifact_registry_corrupt")
        return ArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            pseudonymous_owner_id=str(row["pseudonymous_owner_id"]),
            owner_user_id=str(row["owner_user_id"]),
            created_at_utc=str(row["created_at_utc"]),
            run_id=str(row["run_id"]) if row["run_id"] is not None else None,
            session_id=(
                str(row["session_id"]) if row["session_id"] is not None else None
            ),
            repo_id=str(row["repo_id"]) if row["repo_id"] is not None else None,
            worktree_id=(
                str(row["worktree_id"]) if row["worktree_id"] is not None else None
            ),
            relative_path=str(row["relative_path"]),
            source_state_fingerprint=str(row["source_state_fingerprint"]),
            generator_model_route=str(row["generator_model_route"]),
            producing_tool_call_id=(
                str(row["producing_tool_call_id"])
                if row["producing_tool_call_id"] is not None
                else None
            ),
            parent_artifact_ids=tuple(parents),
            content_sha256=str(row["content_sha256"]),
            visibility=str(row["visibility"]),
            authorization_policy=str(row["authorization_policy"]),
            deleted_at_utc=(
                str(row["deleted_at_utc"]) if row["deleted_at_utc"] is not None else None
            ),
        )

    def _event(
        self,
        record: ArtifactRecord,
        *,
        action: str,
        decision: str,
        reason: str,
        capability_tier: str,
        trace_id: str,
    ) -> None:
        value = {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": record.owner_user_id,
            "capability_tier": capability_tier,
            "artifact_id": record.artifact_id,
            "repo_id": record.repo_id,
            "session_id": record.session_id,
            "action": action,
            "decision": decision,
            "reason": reason,
            "content_sha256": record.content_sha256,
            "trace_id": trace_id,
        }
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with self._lock:
            descriptor = os.open(self.event_log_path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def create(
        self,
        *,
        owner_user_id: str,
        content: bytes,
        relative_path: str,
        source_state_fingerprint: str,
        generator_model_route: str,
        capability_tier: str,
        trace_id: str,
        repo_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        worktree_id: str | None = None,
        producing_tool_call_id: str | None = None,
        parent_artifact_ids: Sequence[str] = (),
        visibility: str = "private",
        authorization_policy: str = "owner-and-repository-v1",
    ) -> ArtifactRecord:
        clean = clean_relative_path(relative_path)
        root = self._owner_root(owner_user_id, repo_id)
        target = root.joinpath(*clean.parts)
        if target.exists():
            raise ArtifactRegistryError("artifact_already_exists")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(path.is_symlink() for path in [target.parent, *target.parents] if path.exists()):
            raise ArtifactRegistryError("artifact_symlink_escape")
        digest = hashlib.sha256(content).hexdigest()
        record = ArtifactRecord(
            artifact_id=new_ulid(),
            pseudonymous_owner_id=pseudonymous_owner_id(self._key, owner_user_id),
            owner_user_id=owner_user_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            run_id=run_id,
            session_id=session_id,
            repo_id=repo_id,
            worktree_id=worktree_id,
            relative_path=str(clean),
            source_state_fingerprint=source_state_fingerprint,
            generator_model_route=generator_model_route,
            producing_tool_call_id=producing_tool_call_id,
            parent_artifact_ids=tuple(parent_artifact_ids),
            content_sha256=digest,
            visibility=visibility,
            authorization_policy=authorization_policy,
        )
        temporary = target.with_name(f".{target.name}.{record.artifact_id}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            try:
                with self._lock, self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO artifacts VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
                        )
                        """,
                        (
                            record.artifact_id,
                            record.pseudonymous_owner_id,
                            record.owner_user_id,
                            record.created_at_utc,
                            record.run_id,
                            record.session_id,
                            record.repo_id,
                            record.worktree_id,
                            record.relative_path,
                            record.source_state_fingerprint,
                            record.generator_model_route,
                            record.producing_tool_call_id,
                            json.dumps(list(record.parent_artifact_ids)),
                            record.content_sha256,
                            record.visibility,
                            record.authorization_policy,
                        ),
                    )
            except Exception:
                target.unlink(missing_ok=True)
                raise
        finally:
            temporary.unlink(missing_ok=True)
        self._event(
            record,
            action="create",
            decision="allowed",
            reason="registered_atomically",
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        return record

    def require(
        self,
        artifact_id: str,
        *,
        owner_user_id: str,
        repo_id: str | None,
        include_deleted: bool = False,
    ) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE artifact_id = ? AND owner_user_id = ? AND repo_id IS ?
                """,
                (artifact_id, owner_user_id, repo_id),
            ).fetchone()
        if row is None:
            raise ArtifactRegistryError("artifact_not_found")
        record = self._record(row)
        if record.deleted_at_utc is not None and not include_deleted:
            raise ArtifactRegistryError("artifact_not_found")
        return record

    def _path(self, record: ArtifactRecord) -> Path:
        root = self._owner_root(record.owner_user_id, record.repo_id)
        target = root.joinpath(*clean_relative_path(record.relative_path).parts).resolve()
        if target != root and root not in target.parents:
            raise ArtifactRegistryError("artifact_path_escape")
        return target

    def read(
        self,
        artifact_id: str,
        *,
        owner_user_id: str,
        repo_id: str | None,
        capability_tier: str,
        trace_id: str,
    ) -> bytes:
        record = self.require(
            artifact_id,
            owner_user_id=owner_user_id,
            repo_id=repo_id,
        )
        target = self._path(record)
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise ArtifactRegistryError("artifact_integrity_failure") from exc
        if not hmac_compare(hashlib.sha256(content).hexdigest(), record.content_sha256):
            self._event(
                record,
                action="read",
                decision="denied",
                reason="artifact_integrity_failure",
                capability_tier=capability_tier,
                trace_id=trace_id,
            )
            raise ArtifactRegistryError("artifact_integrity_failure")
        self._event(
            record,
            action="read",
            decision="allowed",
            reason="owner_repository_and_hash_verified",
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        return content

    def rename(
        self,
        artifact_id: str,
        *,
        owner_user_id: str,
        repo_id: str | None,
        new_relative_path: str,
        capability_tier: str,
        trace_id: str,
    ) -> ArtifactRecord:
        record = self.require(artifact_id, owner_user_id=owner_user_id, repo_id=repo_id)
        source = self._path(record)
        clean = clean_relative_path(new_relative_path)
        target = self._owner_root(owner_user_id, repo_id).joinpath(*clean.parts)
        if target.exists():
            raise ArtifactRegistryError("artifact_already_exists")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(source, target)
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "UPDATE artifacts SET relative_path = ? WHERE artifact_id = ?",
                    (str(clean), artifact_id),
                )
        except Exception:
            os.replace(target, source)
            raise
        updated = ArtifactRecord(**{**record.__dict__, "relative_path": str(clean)})
        self._event(
            updated,
            action="rename",
            decision="allowed",
            reason="identity_preserved",
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        return updated

    def update_content(
        self,
        artifact_id: str,
        *,
        owner_user_id: str,
        repo_id: str | None,
        content: bytes,
        capability_tier: str,
        trace_id: str,
    ) -> ArtifactRecord:
        record = self.require(artifact_id, owner_user_id=owner_user_id, repo_id=repo_id)
        target = self._path(record)
        old = self.read(
            artifact_id,
            owner_user_id=owner_user_id,
            repo_id=repo_id,
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        digest = hashlib.sha256(content).hexdigest()
        temporary = target.with_name(f".{target.name}.{new_ulid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            try:
                with self._lock, self._connect() as connection:
                    connection.execute(
                        "UPDATE artifacts SET content_sha256 = ? WHERE artifact_id = ?",
                        (digest, artifact_id),
                    )
            except Exception:
                target.write_bytes(old)
                raise
        finally:
            temporary.unlink(missing_ok=True)
        updated = ArtifactRecord(**{**record.__dict__, "content_sha256": digest})
        self._event(
            updated,
            action="update",
            decision="allowed",
            reason="content_hash_updated",
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        return updated

    def delete(
        self,
        artifact_id: str,
        *,
        owner_user_id: str,
        repo_id: str | None,
        capability_tier: str,
        trace_id: str,
    ) -> ArtifactRecord:
        record = self.require(artifact_id, owner_user_id=owner_user_id, repo_id=repo_id)
        target = self._path(record)
        target.unlink(missing_ok=True)
        deleted_at = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE artifacts SET deleted_at_utc = ? WHERE artifact_id = ?",
                (deleted_at, artifact_id),
            )
        tombstone = ArtifactRecord(**{**record.__dict__, "deleted_at_utc": deleted_at})
        self._event(
            tombstone,
            action="delete",
            decision="allowed",
            reason="content_removed_tombstone_retained",
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        return tombstone

    def export_normal(
        self,
        artifact_id: str,
        *,
        owner_user_id: str,
        repo_id: str | None,
        destination: Path,
        capability_tier: str,
        trace_id: str,
    ) -> Path:
        record = self.require(artifact_id, owner_user_id=owner_user_id, repo_id=repo_id)
        content = self.read(
            artifact_id,
            owner_user_id=owner_user_id,
            repo_id=repo_id,
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / Path(record.relative_path).name
        target.write_bytes(content)
        self._event(
            record,
            action="export",
            decision="allowed",
            reason="normal_export_without_internal_provenance",
            capability_tier=capability_tier,
            trace_id=trace_id,
        )
        return target

    def compact_operator_export(self) -> list[JsonObject]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM artifacts ORDER BY created_at_utc").fetchall()
        return [self._record(row).internal() for row in rows]


def hmac_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)
