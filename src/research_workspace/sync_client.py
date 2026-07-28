"""Desktop Git inspection, confirmed synchronization and resumable records."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from .sync_protocol import (
    ChangeKind,
    FileChangeV1,
    RepositorySnapshotV1,
    SyncDirection,
    SyncError,
    SyncOperationV1,
    SyncPlanV1,
    SyncState,
    SyncTransport,
    clean_logical_path,
    confirmation_for,
    operation_id,
    validate_patch,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _git(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(  # nosec B603 B607
        ["git", "-c", "core.quotepath=false", "-C", str(root), *arguments],
        capture_output=True,
        text=not binary,
        check=False,
        timeout=30,
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "LC_ALL": "C",
        },
    )
    if completed.returncode != 0:
        raise SyncError("git_inspection_failed")
    output = completed.stdout
    if binary:
        if not isinstance(output, bytes):
            raise SyncError("git_output_type_invalid")
    elif not isinstance(output, str):
        raise SyncError("git_output_type_invalid")
    return cast(bytes | str, output)


def _safe_remote(value: str) -> str:
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme in {"https", "ssh"} and parsed.hostname:
        return f"{parsed.scheme}://{parsed.hostname}/<redacted-path>"
    match = value.split(":", 1)
    if len(match) == 2 and "@" in match[0]:
        host = match[0].rsplit("@", 1)[1]
        return f"ssh://{host}/<redacted-path>"
    if value:
        return "configured-remote/<redacted-path>"
    return "unavailable"


class RepositoryInspector:
    def __init__(
        self,
        *,
        maximum_files: int = 2_000,
        maximum_patch_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.maximum_files = maximum_files
        self.maximum_patch_bytes = maximum_patch_bytes

    def _root(self, selected: Path) -> Path:
        if selected.is_symlink():
            raise SyncError("repository_symlink_rejected")
        candidate = selected.resolve()
        top = str(_git(candidate, "rev-parse", "--show-toplevel")).strip()
        resolved = Path(top).resolve()
        if resolved != candidate:
            raise SyncError("repository_selection_not_git_top_level")
        if (candidate / ".git").is_symlink():
            raise SyncError("git_metadata_symlink_rejected")
        submodules = str(_git(candidate, "submodule", "status", "--recursive")).strip()
        if submodules:
            raise SyncError("submodules_not_supported")
        return candidate

    def _changes(self, root: Path) -> tuple[FileChangeV1, ...]:
        raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True)
        assert isinstance(raw, bytes)
        values = raw.decode("utf-8").split("\x00")
        changes: list[FileChangeV1] = []
        for record in values:
            if not record:
                continue
            if len(record) < 4 or record[2] != " ":
                raise SyncError("git_status_malformed")
            code, path_value = record[:2], record[3:]
            if "R" in code or "C" in code:
                raise SyncError("rename_or_copy_requires_manual_commit")
            try:
                logical = clean_logical_path(path_value)
            except ValueError as exc:
                raise SyncError("sync_path_rejected") from exc
            if code == "??":
                kind = ChangeKind.UNTRACKED
                included = False
            elif "D" in code:
                kind = ChangeKind.DELETED
                included = True
            elif "A" in code:
                kind = ChangeKind.ADDED
                included = True
            else:
                kind = ChangeKind.MODIFIED
                included = True
            changes.append(
                FileChangeV1(
                    logical_path=logical,
                    kind=kind,
                    staged=code[0] not in {" ", "?"},
                    included=included,
                )
            )
        if len(changes) > self.maximum_files:
            raise SyncError("sync_file_count_exceeded")
        return tuple(sorted(changes, key=lambda item: item.logical_path))

    def _validate_changed_files(
        self,
        root: Path,
        changes: tuple[FileChangeV1, ...],
    ) -> None:
        root_device = root.stat().st_dev
        for change in changes:
            if not change.included or change.kind is ChangeKind.DELETED:
                continue
            path = root / change.logical_path
            current = path
            while current != root:
                if current.name == ".git" or (current.is_dir() and (current / ".git").exists()):
                    raise SyncError("nested_repository_rejected")
                current = current.parent
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise SyncError("symlink_rejected")
            if not stat.S_ISREG(metadata.st_mode):
                raise SyncError("non_regular_file_rejected")
            if metadata.st_nlink != 1:
                raise SyncError("hardlink_rejected")
            if metadata.st_dev != root_device:
                raise SyncError("mount_boundary_rejected")

    def snapshot(self, selected: Path, *, logical_repository_id: str) -> RepositorySnapshotV1:
        root = self._root(selected)
        changes = self._changes(root)
        self._validate_changed_files(root, changes)
        remote_lines = str(_git(root, "remote", "-v")).splitlines()
        remotes = tuple(
            sorted(
                {
                    _safe_remote(parts[1])
                    for line in remote_lines
                    if len(parts := line.split()) >= 2
                }
            )
        )
        branch = str(_git(root, "branch", "--show-current")).strip() or "DETACHED"
        head = str(_git(root, "rev-parse", "HEAD")).strip().lower()
        return RepositorySnapshotV1(
            logical_repository_id=logical_repository_id,
            branch=branch,
            head=head,
            dirty=bool(changes),
            remotes=remotes,
            changes=changes,
        )

    def plan_upload(self, selected: Path, *, logical_repository_id: str) -> tuple[SyncPlanV1, bytes]:
        root = self._root(selected)
        snapshot = self.snapshot(root, logical_repository_id=logical_repository_id)
        patch_raw = _git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--binary",
            "HEAD",
            "--",
            binary=True,
        )
        assert isinstance(patch_raw, bytes)
        paths = validate_patch(
            patch_raw,
            maximum_bytes=self.maximum_patch_bytes,
            maximum_files=self.maximum_files,
        )
        included = {
            change.logical_path for change in snapshot.changes if change.included
        }
        if set(paths) != included:
            raise SyncError("snapshot_patch_mismatch")
        patch_hash = hashlib.sha256(patch_raw).hexdigest()
        material = "\n".join(
            [logical_repository_id, snapshot.head, snapshot.branch, patch_hash]
        )
        plan = SyncPlanV1(
            plan_id="sync-plan-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24],
            logical_repository_id=logical_repository_id,
            direction=SyncDirection.UPLOAD,
            base_head=snapshot.head,
            branch=snapshot.branch,
            changed_paths=paths,
            patch_sha256=patch_hash,
            patch_size_bytes=len(patch_raw),
        )
        return plan, patch_raw


class SyncOperationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_operations (
                    operation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    patch BLOB NOT NULL,
                    state TEXT NOT NULL,
                    transferred_bytes INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    conflict_reason TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_events (
                    event_id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    logical_repository_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    patch_sha256 TEXT NOT NULL,
                    occurred_at_utc TEXT NOT NULL
                )
                """
            )
        if os.name != "nt":
            os.chmod(self.database_path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def put(self, operation: SyncOperationV1, patch: bytes) -> SyncOperationV1:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT plan_json, patch FROM sync_operations WHERE operation_id=?",
                (operation.operation_id,),
            ).fetchone()
            serialized = operation.plan.model_dump_json()
            if existing is not None and (
                str(existing["plan_json"]) != serialized or bytes(existing["patch"]) != patch
            ):
                raise SyncError("sync_operation_identity_conflict")
            connection.execute(
                """
                INSERT INTO sync_operations(
                    operation_id, owner_id, plan_json, patch, state,
                    transferred_bytes, created_at_utc, updated_at_utc, conflict_reason
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    state=excluded.state,
                    transferred_bytes=excluded.transferred_bytes,
                    updated_at_utc=excluded.updated_at_utc,
                    conflict_reason=excluded.conflict_reason
                """,
                (
                    operation.operation_id,
                    operation.owner_id,
                    serialized,
                    patch,
                    operation.state.value,
                    operation.transferred_bytes,
                    operation.created_at_utc,
                    operation.updated_at_utc,
                    operation.conflict_reason,
                ),
            )
            event_material = (
                f"{operation.operation_id}\n{operation.state.value}\n"
                f"{operation.transferred_bytes}\n{operation.updated_at_utc}"
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO sync_events(
                    event_id, operation_id, owner_id, logical_repository_id,
                    state, patch_sha256, occurred_at_utc
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    "sync-event-" + hashlib.sha256(event_material.encode("utf-8")).hexdigest()[:24],
                    operation.operation_id,
                    operation.owner_id,
                    operation.plan.logical_repository_id,
                    operation.state.value,
                    operation.plan.patch_sha256,
                    operation.updated_at_utc,
                ),
            )
            connection.commit()
        return operation

    def get(self, owner_id: str, operation_id_value: str) -> tuple[SyncOperationV1, bytes]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sync_operations WHERE operation_id=? AND owner_id=?",
                (operation_id_value, owner_id),
            ).fetchone()
        if row is None:
            raise SyncError("sync_operation_not_found")
        plan = SyncPlanV1.model_validate_json(str(row["plan_json"]))
        return (
            SyncOperationV1(
                operation_id=str(row["operation_id"]),
                owner_id=str(row["owner_id"]),
                plan=plan,
                state=SyncState(str(row["state"])),
                transferred_bytes=int(row["transferred_bytes"]),
                created_at_utc=str(row["created_at_utc"]),
                updated_at_utc=str(row["updated_at_utc"]),
                conflict_reason=(
                    str(row["conflict_reason"])
                    if row["conflict_reason"] is not None
                    else None
                ),
            ),
            bytes(row["patch"]),
        )

    def history(self, owner_id: str, operation_id_value: str) -> tuple[dict[str, object], ...]:
        self.get(owner_id, operation_id_value)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, logical_repository_id, state, patch_sha256, occurred_at_utc
                FROM sync_events
                WHERE operation_id=? AND owner_id=?
                ORDER BY occurred_at_utc, event_id
                """,
                (operation_id_value, owner_id),
            ).fetchall()
        return tuple(
            {
                "event_id": str(row["event_id"]),
                "operation_id": operation_id_value,
                "logical_repository_id": str(row["logical_repository_id"]),
                "state": str(row["state"]),
                "patch_sha256": str(row["patch_sha256"]),
                "occurred_at_utc": str(row["occurred_at_utc"]),
            }
            for row in rows
        )


class DesktopSyncClient:
    def __init__(
        self,
        *,
        owner_id: str,
        operations: SyncOperationStore,
        transport: SyncTransport,
        chunk_bytes: int = 64 * 1024,
    ) -> None:
        if not 1_024 <= chunk_bytes <= 4 * 1024 * 1024:
            raise ValueError("sync chunk size must be 1 KiB..4 MiB")
        self.owner_id = owner_id
        self.operations = operations
        self.transport = transport
        self.chunk_bytes = chunk_bytes

    def prepare(self, plan: SyncPlanV1, patch: bytes) -> SyncOperationV1:
        if hashlib.sha256(patch).hexdigest() != plan.patch_sha256:
            raise SyncError("sync_patch_hash_mismatch")
        timestamp = _now()
        operation = SyncOperationV1(
            operation_id=operation_id(plan, self.owner_id),
            owner_id=self.owner_id,
            plan=plan,
            state=SyncState.PLANNED,
            transferred_bytes=0,
            created_at_utc=timestamp,
            updated_at_utc=timestamp,
        )
        return self.operations.put(operation, patch)

    def upload(
        self,
        operation_id_value: str,
        *,
        confirmation: str,
    ) -> SyncOperationV1:
        operation, patch = self.operations.get(self.owner_id, operation_id_value)
        if confirmation != confirmation_for(operation.plan.plan_id):
            raise SyncError("sync_confirmation_required")
        offset = operation.transferred_bytes
        while offset < len(patch):
            end = min(offset + self.chunk_bytes, len(patch))
            receipt = self.transport.upload_chunk(
                owner_id=self.owner_id,
                logical_repository_id=operation.plan.logical_repository_id,
                operation_id=operation.operation_id,
                base_head=operation.plan.base_head,
                offset=offset,
                content=patch[offset:end],
                final=end == len(patch),
                expected_sha256=operation.plan.patch_sha256,
            )
            if receipt.accepted_offset != end:
                raise SyncError("sync_transport_offset_mismatch")
            offset = end
            operation = operation.model_copy(
                update={
                    "state": (
                        SyncState.TRANSFERRED if receipt.complete else SyncState.TRANSFERRING
                    ),
                    "transferred_bytes": offset,
                    "updated_at_utc": _now(),
                }
            )
            self.operations.put(operation, patch)
        return operation

    def export_patch(self, operation_id_value: str, destination: Path) -> Path:
        operation, patch = self.operations.get(self.owner_id, operation_id_value)
        if destination.exists() or destination.is_symlink():
            raise SyncError("patch_export_target_exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(patch)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != operation.plan.patch_sha256:
            raise SyncError("patch_export_integrity_failed")
        return destination


def apply_confirmed_patch(
    selected: Path,
    *,
    plan: SyncPlanV1,
    patch: bytes,
    confirmation: str,
) -> Literal["APPLIED"]:
    root = RepositoryInspector()._root(selected)
    if confirmation != confirmation_for(plan.plan_id):
        raise SyncError("sync_confirmation_required")
    if str(_git(root, "rev-parse", "HEAD")).strip().lower() != plan.base_head:
        raise SyncError("sync_base_conflict")
    paths = validate_patch(patch, maximum_bytes=64 * 1024 * 1024, maximum_files=2_000)
    if paths != plan.changed_paths or hashlib.sha256(patch).hexdigest() != plan.patch_sha256:
        raise SyncError("sync_patch_plan_mismatch")
    completed = subprocess.run(  # nosec B603 B607
        ["git", "-C", str(root), "apply", "--check", "--whitespace=error", "-"],
        input=patch,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise SyncError("sync_patch_conflict")
    applied = subprocess.run(  # nosec B603 B607
        ["git", "-C", str(root), "apply", "--whitespace=error", "-"],
        input=patch,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if applied.returncode != 0:
        raise SyncError("sync_patch_apply_failed")
    return "APPLIED"
