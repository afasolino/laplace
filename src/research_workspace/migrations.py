"""Locked, backed-up forward migrations for explicitly selected state fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
from contextlib import AbstractContextManager, closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

CURRENT_STORE_SCHEMA = 1
STATE_MANIFEST = "state_manifest.json"
MIGRATION_JOURNAL = ".migration-journal.json"
MIGRATION_LOCK = ".laplace-migration.lock"
MAX_STORE_BYTES = 256 * 1024 * 1024


class MigrationError(RuntimeError):
    """Migration failed closed with a sanitized category."""


class StoreKind(StrEnum):
    PROJECT_SQLITE = "project_sqlite"
    GLOBAL_REGISTRY = "global_registry"
    REGISTERED_USERS = "registered_users"
    SESSIONS = "sessions"
    CONVERSATIONS = "conversations"
    REPOSITORY_AUTHORIZATION = "repository_authorization"
    WORKTREES = "worktrees"
    PERSONAL_CORPUS = "personal_corpus"
    ARTIFACT_REGISTRY = "artifact_registry"
    RESEARCH_JOBS = "research_jobs"
    AUDIT = "audit"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class StoreEntry(_Strict):
    store_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    kind: StoreKind
    relative_path: str = Field(min_length=1, max_length=500)
    format: Literal["sqlite", "json", "yaml", "jsonl"]
    schema_version: int = Field(ge=0, le=CURRENT_STORE_SCHEMA)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("kind", mode="before")
    @classmethod
    def kind_from_json(cls, value: object) -> object:
        return StoreKind(value) if isinstance(value, str) else value


class StateManifest(_Strict):
    schema_version: Literal[1] = 1
    state_id: str = Field(pattern=r"^fixture-[a-z0-9-]{1,80}$")
    stores: tuple[StoreEntry, ...] = Field(min_length=1)

    @field_validator("stores", mode="before")
    @classmethod
    def stores_as_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class PreflightResult:
    state_id: str
    status: str
    stores: tuple[dict[str, object], ...]

    def public(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "state_id": self.state_id,
            "status": self.status,
            "stores": list(self.stores),
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_path(value: str) -> PurePosixPath:
    if "\x00" in value or "\\" in value:
        raise MigrationError("invalid_store_path")
    logical = PurePosixPath(value)
    if logical.is_absolute() or not logical.parts or any(
        part in {"", ".", ".."} for part in logical.parts
    ):
        raise MigrationError("invalid_store_path")
    return logical


def _resolve_store(root: Path, entry: StoreEntry) -> Path:
    logical = _logical_path(entry.relative_path)
    candidate = root.joinpath(*logical.parts)
    if candidate.is_symlink():
        raise MigrationError("store_symlink_rejected")
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise MigrationError("store_path_escape")
    return resolved


def _atomic_bytes(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def _load_manifest(root: Path) -> StateManifest:
    path = root / STATE_MANIFEST
    if path.is_symlink():
        raise MigrationError("manifest_symlink_rejected")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        return StateManifest.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise MigrationError("state_manifest_invalid") from exc


def _verify_permissions(root: Path, path: Path) -> None:
    if os.name == "nt":
        return
    if stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise MigrationError("state_root_permissions_unsafe")
    if stat.S_IMODE(path.stat().st_mode) & 0o022:
        raise MigrationError("store_permissions_unsafe")


def _integrity(path: Path, format_name: str) -> None:
    if not path.is_file() or path.stat().st_size > MAX_STORE_BYTES:
        raise MigrationError("store_missing_or_oversize")
    try:
        if format_name == "sqlite":
            with closing(
                sqlite3.connect(
                    f"file:{path}?mode=ro",
                    uri=True,
                    timeout=5,
                )
            ) as connection:
                result = connection.execute("PRAGMA quick_check").fetchone()
            if result is None or result[0] != "ok":
                raise MigrationError("sqlite_integrity_failed")
        elif format_name == "json":
            value: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise MigrationError("json_store_invalid")
        elif format_name == "yaml":
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise MigrationError("yaml_store_invalid")
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MigrationError("jsonl_store_invalid")
    except MigrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError, sqlite3.Error) as exc:
        raise MigrationError("store_integrity_failed") from exc


def preflight(state_root: Path) -> PreflightResult:
    if state_root.is_symlink():
        raise MigrationError("state_root_symlink_rejected")
    root = state_root.resolve()
    if not root.is_dir():
        raise MigrationError("state_root_missing")
    if os.name != "nt" and stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise MigrationError("state_root_permissions_unsafe")
    manifest = _load_manifest(root)
    store_ids: set[str] = set()
    paths: set[Path] = set()
    checks: list[dict[str, object]] = []
    for entry in manifest.stores:
        if entry.store_id in store_ids:
            raise MigrationError("duplicate_store_id")
        store_ids.add(entry.store_id)
        path = _resolve_store(root, entry)
        if path in paths:
            raise MigrationError("duplicate_store_path")
        paths.add(path)
        _verify_permissions(root, path)
        _integrity(path, entry.format)
        if _sha256(path) != entry.sha256:
            raise MigrationError("store_hash_mismatch")
        checks.append(
            {
                "store_id": entry.store_id,
                "kind": entry.kind.value,
                "format": entry.format,
                "from_version": entry.schema_version,
                "to_version": CURRENT_STORE_SCHEMA,
                "action": (
                    "MIGRATE" if entry.schema_version < CURRENT_STORE_SCHEMA else "NO_CHANGE"
                ),
            }
        )
    expected_kinds = set(StoreKind)
    actual_kinds = {entry.kind for entry in manifest.stores}
    if actual_kinds != expected_kinds:
        raise MigrationError("state_store_inventory_incomplete")
    return PreflightResult(manifest.state_id, "PASS", tuple(checks))


class _MigrationLock(AbstractContextManager["_MigrationLock"]):
    def __init__(self, root: Path) -> None:
        self.path = root / MIGRATION_LOCK

    def __enter__(self) -> _MigrationLock:
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise MigrationError("migration_locked") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "pid": os.getpid(), "created_at_utc": _now()}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.path.unlink(missing_ok=True)


def _backup(root: Path, manifest: StateManifest) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root.parent / f"{root.name}.migration-backups" / stamp
    backup.mkdir(parents=True, exist_ok=False, mode=0o700)
    files: list[dict[str, object]] = []
    for relative in [STATE_MANIFEST, *(entry.relative_path for entry in manifest.stores)]:
        logical = _logical_path(relative)
        source = root.joinpath(*logical.parts)
        target = backup.joinpath(*logical.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(source, target)
        if os.name != "nt":
            os.chmod(target, 0o600)
        files.append(
            {
                "relative_path": logical.as_posix(),
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
    _write_json(
        backup / "backup_manifest.json",
        {
            "schema_version": 1,
            "state_id": manifest.state_id,
            "created_at_utc": _now(),
            "files": files,
        },
    )
    return backup


def _restore_backup(root: Path, backup: Path, *, expected_state_id: str) -> None:
    backup_root = (root.parent / f"{root.name}.migration-backups").resolve()
    resolved = backup.resolve()
    if backup_root != resolved and backup_root not in resolved.parents:
        raise MigrationError("backup_path_not_authorized")
    try:
        raw: object = json.loads((resolved / "backup_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError("backup_manifest_invalid") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("state_id") != expected_state_id
        or not isinstance(raw.get("files"), list)
    ):
        raise MigrationError("backup_manifest_invalid")
    for item in raw["files"]:
        if not isinstance(item, dict):
            raise MigrationError("backup_manifest_invalid")
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise MigrationError("backup_manifest_invalid")
        logical = _logical_path(relative)
        source = resolved.joinpath(*logical.parts)
        if source.is_symlink() or _sha256(source) != expected:
            raise MigrationError("backup_integrity_failed")
        target = root.joinpath(*logical.parts)
        _atomic_bytes(target, source.read_bytes())


def _migrate_store(path: Path, entry: StoreEntry) -> None:
    if entry.schema_version == CURRENT_STORE_SCHEMA:
        return
    if entry.schema_version != 0:
        raise MigrationError("no_ordered_migration")
    if entry.format == "sqlite":
        try:
            with closing(sqlite3.connect(path, timeout=15)) as connection:
                connection.execute("PRAGMA busy_timeout=15000")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS laplace_store_schema (
                        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                        schema_version INTEGER NOT NULL,
                        migrated_at_utc TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO laplace_store_schema(singleton, schema_version, migrated_at_utc)
                    VALUES(1, ?, ?)
                    ON CONFLICT(singleton) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        migrated_at_utc=excluded.migrated_at_utc
                    """,
                    (CURRENT_STORE_SCHEMA, _now()),
                )
                connection.execute(f"PRAGMA user_version={CURRENT_STORE_SCHEMA}")
                connection.commit()
        except sqlite3.Error as exc:
            raise MigrationError("sqlite_migration_failed") from exc
        return
    if entry.format == "json":
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise MigrationError("json_store_invalid")
        raw = {**raw, "schema_version": CURRENT_STORE_SCHEMA, "migrated_at_utc": _now()}
        _write_json(path, raw)
        return
    if entry.format == "yaml":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise MigrationError("yaml_store_invalid")
        raw = {**raw, "schema_version": CURRENT_STORE_SCHEMA, "migrated_at_utc": _now()}
        _atomic_bytes(
            path,
            yaml.safe_dump(raw, sort_keys=True, allow_unicode=True).encode("utf-8"),
        )
        return
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    header = {
        "schema_version": CURRENT_STORE_SCHEMA,
        "event": "STORE_SCHEMA_MIGRATED",
        "occurred_at_utc": _now(),
    }
    encoded = "\n".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in [header, *events]
    )
    _atomic_bytes(path, (encoded + "\n").encode("utf-8"))


def _append_audit(root: Path, value: dict[str, object]) -> None:
    path = root / "migration_audit.jsonl"
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def migrate(
    state_root: Path,
    *,
    expected_state_id: str,
    dry_run: bool,
    fail_after: int | None = None,
) -> dict[str, object]:
    root = state_root.resolve()
    initial = preflight(root)
    if initial.state_id != expected_state_id:
        raise MigrationError("state_identity_mismatch")
    if dry_run:
        return {
            "schema_version": 1,
            "status": "DRY_RUN_PASS",
            "state_id": initial.state_id,
            "preflight": initial.public(),
            "backup_created": False,
            "stores_changed": sum(
                1 for item in initial.stores if item["action"] == "MIGRATE"
            ),
        }
    with _MigrationLock(root):
        manifest = _load_manifest(root)
        backup = _backup(root, manifest)
        journal_path = root / MIGRATION_JOURNAL
        journal: dict[str, object] = {
            "schema_version": 1,
            "state_id": manifest.state_id,
            "status": "PREPARED",
            "backup": str(backup.relative_to(root.parent)),
            "started_at_utc": _now(),
        }
        _write_json(journal_path, journal)
        changed = 0
        try:
            for entry in manifest.stores:
                if entry.schema_version == CURRENT_STORE_SCHEMA:
                    continue
                journal["status"] = "APPLYING"
                journal["current_store_id"] = entry.store_id
                _write_json(journal_path, journal)
                _migrate_store(_resolve_store(root, entry), entry)
                changed += 1
                if fail_after is not None and changed >= fail_after:
                    raise MigrationError("simulated_migration_interruption")
            updated_entries: list[StoreEntry] = []
            for entry in manifest.stores:
                path = _resolve_store(root, entry)
                _integrity(path, entry.format)
                updated_entries.append(
                    entry.model_copy(
                        update={
                            "schema_version": CURRENT_STORE_SCHEMA,
                            "sha256": _sha256(path),
                        }
                    )
                )
            updated = manifest.model_copy(update={"stores": tuple(updated_entries)})
            _write_json(root / STATE_MANIFEST, updated.model_dump(mode="json"))
            preflight(root)
            journal.update({"status": "COMPLETE", "completed_at_utc": _now()})
            _write_json(journal_path, journal)
        except BaseException as exc:
            _restore_backup(root, backup, expected_state_id=manifest.state_id)
            journal.update({"status": "ROLLED_BACK", "rolled_back_at_utc": _now()})
            _write_json(journal_path, journal)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise MigrationError("migration_failed_rolled_back") from exc
        _append_audit(
            root,
            {
                "schema_version": 1,
                "event": "STATE_MIGRATION_COMPLETED",
                "state_id": manifest.state_id,
                "stores_changed": changed,
                "occurred_at_utc": _now(),
                "backup_id": backup.name,
            },
        )
        return {
            "schema_version": 1,
            "status": "PASS",
            "state_id": manifest.state_id,
            "stores_changed": changed,
            "backup_id": backup.name,
            "integrity": "PASS",
            "audit_recorded": True,
        }


def recover_interrupted(state_root: Path, *, expected_state_id: str) -> dict[str, object]:
    root = state_root.resolve()
    journal_path = root / MIGRATION_JOURNAL
    try:
        raw: object = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError("migration_journal_invalid") from exc
    if not isinstance(raw, dict) or raw.get("state_id") != expected_state_id:
        raise MigrationError("migration_journal_invalid")
    status = raw.get("status")
    if status in {"COMPLETE", "ROLLED_BACK", "RECOVERED"}:
        return {"schema_version": 1, "status": "NO_RECOVERY_REQUIRED"}
    backup_value = raw.get("backup")
    if not isinstance(backup_value, str):
        raise MigrationError("migration_journal_invalid")
    backup = (root.parent / backup_value).resolve()
    _restore_backup(root, backup, expected_state_id=expected_state_id)
    raw["status"] = "RECOVERED"
    raw["recovered_at_utc"] = _now()
    _write_json(journal_path, raw)
    (root / MIGRATION_LOCK).unlink(missing_ok=True)
    preflight(root)
    _append_audit(
        root,
        {
            "schema_version": 1,
            "event": "STATE_MIGRATION_RECOVERED",
            "state_id": expected_state_id,
            "occurred_at_utc": _now(),
            "backup_id": backup.name,
        },
    )
    return {"schema_version": 1, "status": "RECOVERED", "backup_id": backup.name}


def rollback_to_backup(
    state_root: Path,
    *,
    expected_state_id: str,
    backup_id: str,
) -> dict[str, object]:
    if (
        not backup_id
        or len(backup_id) > 80
        or re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", backup_id) is None
    ):
        raise MigrationError("backup_id_invalid")
    root = state_root.resolve()
    preflight_result = preflight(root)
    if preflight_result.state_id != expected_state_id:
        raise MigrationError("state_identity_mismatch")
    backup = root.parent / f"{root.name}.migration-backups" / backup_id
    with _MigrationLock(root):
        _restore_backup(root, backup, expected_state_id=expected_state_id)
        preflight(root)
        _append_audit(
            root,
            {
                "schema_version": 1,
                "event": "STATE_MIGRATION_ROLLED_BACK",
                "state_id": expected_state_id,
                "occurred_at_utc": _now(),
                "backup_id": backup_id,
            },
        )
    return {"schema_version": 1, "status": "ROLLED_BACK", "backup_id": backup_id}


def create_synthetic_v0_state(root: Path, *, state_id: str = "fixture-v7-old") -> StateManifest:
    """Create an authorized-content-free old-state fixture for migration tests."""

    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name != "nt":
        os.chmod(root, 0o700)
    definitions = (
        ("project", StoreKind.PROJECT_SQLITE, "stores/project.sqlite3", "sqlite"),
        ("registry", StoreKind.GLOBAL_REGISTRY, "stores/registry.json", "json"),
        ("users", StoreKind.REGISTERED_USERS, "stores/users.yaml", "yaml"),
        ("sessions", StoreKind.SESSIONS, "stores/sessions.sqlite3", "sqlite"),
        ("conversations", StoreKind.CONVERSATIONS, "stores/conversations.sqlite3", "sqlite"),
        (
            "repositories",
            StoreKind.REPOSITORY_AUTHORIZATION,
            "stores/repositories.sqlite3",
            "sqlite",
        ),
        ("worktrees", StoreKind.WORKTREES, "stores/worktrees.sqlite3", "sqlite"),
        ("personal_corpus", StoreKind.PERSONAL_CORPUS, "stores/corpus.sqlite3", "sqlite"),
        ("artifacts", StoreKind.ARTIFACT_REGISTRY, "stores/artifacts.sqlite3", "sqlite"),
        ("research", StoreKind.RESEARCH_JOBS, "stores/research.json", "json"),
        ("audit", StoreKind.AUDIT, "stores/audit.jsonl", "jsonl"),
    )
    entries: list[StoreEntry] = []
    for store_id, kind, relative, format_name in definitions:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if format_name == "sqlite":
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE TABLE legacy_records(record_id TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO legacy_records(record_id, value) VALUES(?, ?)",
                    (f"{store_id}-record", "synthetic"),
                )
                connection.commit()
        elif format_name == "json":
            _write_json(path, {"schema_version": 0, "records": [{"id": store_id}]})
        elif format_name == "yaml":
            _atomic_bytes(
                path,
                yaml.safe_dump(
                    {"schema_version": 0, "users": [{"user_id": "fixture-user"}]},
                    sort_keys=True,
                ).encode("utf-8"),
            )
        else:
            _atomic_bytes(
                path,
                (
                    json.dumps(
                        {"schema_version": 0, "event": "SYNTHETIC", "subject": store_id},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        if os.name != "nt":
            os.chmod(path, 0o600)
        entries.append(
            StoreEntry(
                store_id=store_id,
                kind=kind,
                relative_path=relative,
                format=format_name,  # type: ignore[arg-type]
                schema_version=0,
                sha256=_sha256(path),
            )
        )
    manifest = StateManifest(state_id=state_id, stores=tuple(entries))
    _write_json(root / STATE_MANIFEST, manifest.model_dump(mode="json"))
    return manifest
