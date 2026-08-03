"""Fixture-safe storage governance, retention, backup, and recovery contracts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION: Final = 1
PURGE_CONFIRMATION: Final = "PURGE_FIXTURE_STATE"
BACKUP_CONFIRMATION: Final = "EXPORT_FIXTURE_BACKUP"


class GovernanceError(RuntimeError):
    """Raised when a governance invariant is violated."""


class AssetCategory(StrEnum):
    CONVERSATION = "conversation"
    DRAFT = "draft"
    ATTACHMENT = "attachment"
    CORPUS = "corpus"
    EXTRACTED_TEXT = "extracted_text"
    INDEX = "index"
    WORKTREE = "worktree"
    ARTIFACT = "artifact"
    AUDIT_EVENT = "audit_event"
    SESSION = "session"
    BACKUP = "backup"


class AssetStatus(StrEnum):
    ACTIVE = "active"
    TOMBSTONED = "tombstoned"
    PURGED = "purged"


class GovernancePolicy(BaseModel):
    """Versioned quota, pressure, and retention policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    per_user_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    global_bytes: int = Field(default=1024 * 1024 * 1024, ge=1)
    minimum_free_bytes: int = Field(default=64 * 1024 * 1024, ge=0)
    retention_days: dict[AssetCategory, int] = Field(
        default_factory=lambda: {category: 30 for category in AssetCategory}
    )
    export_before_delete: frozenset[AssetCategory] = frozenset(
        {
            AssetCategory.CONVERSATION,
            AssetCategory.DRAFT,
            AssetCategory.CORPUS,
            AssetCategory.ARTIFACT,
        }
    )
    ownership_transfer_allowed: frozenset[AssetCategory] = frozenset(
        {
            AssetCategory.CONVERSATION,
            AssetCategory.DRAFT,
            AssetCategory.ATTACHMENT,
            AssetCategory.CORPUS,
            AssetCategory.EXTRACTED_TEXT,
            AssetCategory.INDEX,
            AssetCategory.WORKTREE,
            AssetCategory.ARTIFACT,
        }
    )

    @field_validator("schema_version")
    @classmethod
    def _schema_is_current(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported governance schema version: {value}")
        return value

    @field_validator("retention_days")
    @classmethod
    def _retention_complete(
        cls, value: dict[AssetCategory, int]
    ) -> dict[AssetCategory, int]:
        if set(value) != set(AssetCategory):
            raise ValueError("retention_days must define every asset category")
        if any(days < 0 for days in value.values()):
            raise ValueError("retention days cannot be negative")
        return value


class AssetRecord(BaseModel):
    """Owner-scoped asset metadata; physical paths remain internal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    asset_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    owner_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    category: AssetCategory
    byte_count: int = Field(ge=0)
    scoped_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", exclude=True)
    created_at: datetime
    retain_until: datetime
    status: AssetStatus = AssetStatus.ACTIVE
    provenance_id: str = Field(min_length=1, max_length=256)
    export_receipt: str | None = Field(default=None, max_length=512)


class StorageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    active_bytes: int = Field(ge=0)
    active_assets: int = Field(ge=0)
    tombstoned_assets: int = Field(ge=0)
    owners: int = Field(ge=0)
    by_category: dict[AssetCategory, int]
    disk_free_bytes: int = Field(ge=0)
    disk_pressure: bool


class PurgeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    owner_id: str
    category: AssetCategory
    byte_count: int = Field(ge=0)
    requires_export: bool
    provenance_id: str


class PurgePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    generated_at: datetime
    dry_run: bool = True
    candidates: tuple[PurgeCandidate, ...]
    reclaimable_bytes: int = Field(ge=0)


class BackupEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_path: str
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("logical_path")
    @classmethod
    def _safe_logical_path(cls, value: str) -> str:
        logical = PurePosixPath(value)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise ValueError("backup logical path is unsafe")
        return value


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    backup_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    created_at: datetime
    encryption_provider: str = Field(min_length=1, max_length=128)
    key_reference: str = Field(min_length=1, max_length=256)
    entries: tuple[BackupEntry, ...]
    provenance_chain: tuple[str, ...]


class EncryptedBackupProvider(Protocol):
    """Encryption boundary; implementations receive a reference, never key material."""

    @property
    def provider_id(self) -> str: ...

    def export(self, source: Path, destination: Path, *, key_reference: str) -> str: ...

    def validate(self, destination: Path, *, key_reference: str) -> bool: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class GovernanceStore:
    """SQLite-backed governance store intended for explicit isolated fixture roots."""

    def __init__(
        self,
        root: Path,
        *,
        policy: GovernancePolicy,
        namespace_secret: bytes,
    ) -> None:
        if not root.is_absolute():
            raise GovernanceError("governance root must be absolute")
        if len(namespace_secret) < 16:
            raise GovernanceError("namespace secret must contain at least 16 bytes")
        self.root = root.resolve()
        self.policy = policy
        self._secret = namespace_secret
        self.data_root = self.root / "governed-data"
        self.database = self.root / "governance.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    owner_id TEXT PRIMARY KEY,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    deletion_requested_at TEXT
                );
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES accounts(owner_id),
                    category TEXT NOT NULL,
                    relative_path TEXT,
                    byte_count INTEGER NOT NULL,
                    scoped_sha256 TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    retain_until TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provenance_id TEXT NOT NULL,
                    export_receipt TEXT
                );
                CREATE INDEX IF NOT EXISTS assets_owner_status
                    ON assets(owner_id, status);
                CREATE TABLE IF NOT EXISTS governance_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    owner_namespace TEXT,
                    asset_id TEXT,
                    details_json TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _owner_namespace(self, owner_id: str) -> str:
        return hmac.new(self._secret, owner_id.encode(), hashlib.sha256).hexdigest()

    def _scoped_digest(self, owner_id: str, content_digest: str) -> str:
        message = f"{owner_id}:{content_digest}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def _asset_path(
        self, owner_id: str, category: AssetCategory, asset_id: str
    ) -> tuple[Path, str]:
        relative = Path(self._owner_namespace(owner_id)) / category.value / f"{asset_id}.bin"
        target = (self.data_root / relative).resolve()
        if not target.is_relative_to(self.data_root):
            raise GovernanceError("asset path escaped governed root")
        return target, relative.as_posix()

    def register_account(self, owner_id: str) -> None:
        if "@" in owner_id:
            raise GovernanceError("raw email addresses are not valid storage owner IDs")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO accounts(owner_id) VALUES (?)", (owner_id,)
            )

    def disable_account(self, owner_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            changed = connection.execute(
                "UPDATE accounts SET disabled = 1 WHERE owner_id = ?", (owner_id,)
            ).rowcount
        if changed != 1:
            raise GovernanceError("unknown account")

    def request_account_deletion(self, owner_id: str, *, now: datetime | None = None) -> None:
        timestamp = (now or _utc_now()).isoformat()
        with closing(self._connect()) as connection, connection:
            changed = connection.execute(
                """
                UPDATE accounts
                SET disabled = 1, deletion_requested_at = ?
                WHERE owner_id = ?
                """,
                (timestamp, owner_id),
            ).rowcount
        if changed != 1:
            raise GovernanceError("unknown account")

    def _active_bytes(self, connection: sqlite3.Connection, owner_id: str | None) -> int:
        query = "SELECT COALESCE(SUM(byte_count), 0) FROM assets WHERE status = ?"
        parameters: tuple[object, ...] = (AssetStatus.ACTIVE.value,)
        if owner_id is not None:
            query += " AND owner_id = ?"
            parameters += (owner_id,)
        return int(connection.execute(query, parameters).fetchone()[0])

    def admit(self, owner_id: str, byte_count: int) -> None:
        if byte_count < 0:
            raise GovernanceError("byte count cannot be negative")
        with closing(self._connect()) as connection, connection:
            account = connection.execute(
                "SELECT disabled FROM accounts WHERE owner_id = ?", (owner_id,)
            ).fetchone()
            if account is None:
                raise GovernanceError("unknown account")
            if bool(account["disabled"]):
                raise GovernanceError("account is disabled")
            if self._active_bytes(connection, owner_id) + byte_count > self.policy.per_user_bytes:
                raise GovernanceError("per-user storage quota exceeded")
            if self._active_bytes(connection, None) + byte_count > self.policy.global_bytes:
                raise GovernanceError("global storage quota exceeded")
        free = shutil.disk_usage(self.root).free
        if free - byte_count < self.policy.minimum_free_bytes:
            raise GovernanceError("disk-pressure admission denied")

    def store_asset(
        self,
        owner_id: str,
        asset_id: str,
        category: AssetCategory,
        payload: bytes,
        *,
        provenance_id: str,
        now: datetime | None = None,
    ) -> AssetRecord:
        created_at = now or _utc_now()
        self.admit(owner_id, len(payload))
        target, relative = self._asset_path(owner_id, category, asset_id)
        content_digest = hashlib.sha256(payload).hexdigest()
        scoped_digest = self._scoped_digest(owner_id, content_digest)
        retain_until = created_at + timedelta(days=self.policy.retention_days[category])
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{asset_id}.", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO assets(
                        asset_id, owner_id, category, relative_path, byte_count,
                        scoped_sha256, content_sha256, created_at, retain_until,
                        status, provenance_id, export_receipt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        asset_id,
                        owner_id,
                        category.value,
                        relative,
                        len(payload),
                        scoped_digest,
                        content_digest,
                        created_at.isoformat(),
                        retain_until.isoformat(),
                        AssetStatus.ACTIVE.value,
                        provenance_id,
                    ),
                )
                os.replace(temporary, target)
                self._event(
                    connection,
                    "asset_stored",
                    owner_id=owner_id,
                    asset_id=asset_id,
                    details={"category": category.value, "byte_count": len(payload)},
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        return self.asset(owner_id, asset_id)

    def _row_to_asset(self, row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            asset_id=str(row["asset_id"]),
            owner_id=str(row["owner_id"]),
            category=AssetCategory(str(row["category"])),
            byte_count=int(row["byte_count"]),
            scoped_sha256=str(row["scoped_sha256"]),
            content_sha256=str(row["content_sha256"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            retain_until=datetime.fromisoformat(str(row["retain_until"])),
            status=AssetStatus(str(row["status"])),
            provenance_id=str(row["provenance_id"]),
            export_receipt=(
                str(row["export_receipt"]) if row["export_receipt"] is not None else None
            ),
        )

    def asset(self, owner_id: str, asset_id: str) -> AssetRecord:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE owner_id = ? AND asset_id = ?",
                (owner_id, asset_id),
            ).fetchone()
        if row is None:
            raise GovernanceError("asset not found")
        return self._row_to_asset(row)

    def summary(self) -> StorageSummary:
        with closing(self._connect()) as connection, connection:
            active_bytes = self._active_bytes(connection, None)
            active_assets = int(
                connection.execute(
                    "SELECT COUNT(*) FROM assets WHERE status = ?",
                    (AssetStatus.ACTIVE.value,),
                ).fetchone()[0]
            )
            tombstones = int(
                connection.execute(
                    "SELECT COUNT(*) FROM assets WHERE status = ?",
                    (AssetStatus.TOMBSTONED.value,),
                ).fetchone()[0]
            )
            owners = int(connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])
            category_rows = connection.execute(
                """
                SELECT category, COALESCE(SUM(byte_count), 0)
                FROM assets WHERE status = ? GROUP BY category
                """,
                (AssetStatus.ACTIVE.value,),
            ).fetchall()
        by_category = {category: 0 for category in AssetCategory}
        for category, byte_count in category_rows:
            by_category[AssetCategory(str(category))] = int(byte_count)
        free = shutil.disk_usage(self.root).free
        return StorageSummary(
            active_bytes=active_bytes,
            active_assets=active_assets,
            tombstoned_assets=tombstones,
            owners=owners,
            by_category=by_category,
            disk_free_bytes=free,
            disk_pressure=free < self.policy.minimum_free_bytes,
        )

    def plan_purge(self, *, now: datetime | None = None) -> PurgePlan:
        generated_at = now or _utc_now()
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT * FROM assets
                WHERE status = ? AND retain_until <= ?
                ORDER BY owner_id, asset_id
                """,
                (AssetStatus.ACTIVE.value, generated_at.isoformat()),
            ).fetchall()
        candidates = tuple(
            PurgeCandidate(
                asset_id=str(row["asset_id"]),
                owner_id=str(row["owner_id"]),
                category=AssetCategory(str(row["category"])),
                byte_count=int(row["byte_count"]),
                requires_export=AssetCategory(str(row["category"]))
                in self.policy.export_before_delete,
                provenance_id=str(row["provenance_id"]),
            )
            for row in rows
        )
        return PurgePlan(
            generated_at=generated_at,
            candidates=candidates,
            reclaimable_bytes=sum(candidate.byte_count for candidate in candidates),
        )

    def tombstone(
        self,
        plan: PurgePlan,
        *,
        confirmation: str,
        export_receipts: dict[str, str] | None = None,
    ) -> int:
        if confirmation != PURGE_CONFIRMATION:
            raise GovernanceError("exact purge confirmation is required")
        receipts = export_receipts or {}
        changed = 0
        with self._transaction() as connection:
            for candidate in plan.candidates:
                receipt = receipts.get(candidate.asset_id)
                if candidate.requires_export and not receipt:
                    raise GovernanceError(
                        f"export-before-delete receipt missing for {candidate.asset_id}"
                    )
                changed += connection.execute(
                    """
                    UPDATE assets
                    SET status = ?, export_receipt = ?
                    WHERE asset_id = ? AND owner_id = ? AND status = ?
                    """,
                    (
                        AssetStatus.TOMBSTONED.value,
                        receipt,
                        candidate.asset_id,
                        candidate.owner_id,
                        AssetStatus.ACTIVE.value,
                    ),
                ).rowcount
                self._event(
                    connection,
                    "asset_tombstoned",
                    owner_id=candidate.owner_id,
                    asset_id=candidate.asset_id,
                    details={
                        "provenance_id": candidate.provenance_id,
                        "export_receipt": receipt,
                    },
                )
        return changed

    def purge_tombstones(self, *, confirmation: str) -> int:
        if confirmation != PURGE_CONFIRMATION:
            raise GovernanceError("exact purge confirmation is required")
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM assets WHERE status = ? ORDER BY asset_id",
                (AssetStatus.TOMBSTONED.value,),
            ).fetchall()
            for row in rows:
                relative = row["relative_path"]
                if relative:
                    target = (self.data_root / str(relative)).resolve()
                    if not target.is_relative_to(self.data_root):
                        raise GovernanceError("tombstone path escaped governed root")
                    target.unlink(missing_ok=True)
                connection.execute(
                    """
                    UPDATE assets SET status = ?, relative_path = NULL
                    WHERE asset_id = ?
                    """,
                    (AssetStatus.PURGED.value, str(row["asset_id"])),
                )
                self._event(
                    connection,
                    "asset_purged",
                    owner_id=str(row["owner_id"]),
                    asset_id=str(row["asset_id"]),
                    details={"provenance_id": str(row["provenance_id"])},
                )
        return len(rows)

    def transfer_ownership(
        self,
        source_owner: str,
        target_owner: str,
        asset_ids: Sequence[str],
    ) -> int:
        changed = 0
        with self._transaction() as connection:
            target = connection.execute(
                "SELECT disabled FROM accounts WHERE owner_id = ?", (target_owner,)
            ).fetchone()
            if target is None or bool(target["disabled"]):
                raise GovernanceError("target account is missing or disabled")
            for asset_id in asset_ids:
                row = connection.execute(
                    """
                    SELECT * FROM assets
                    WHERE owner_id = ? AND asset_id = ? AND status = ?
                    """,
                    (source_owner, asset_id, AssetStatus.ACTIVE.value),
                ).fetchone()
                if row is None:
                    raise GovernanceError("transfer source asset not found")
                category = AssetCategory(str(row["category"]))
                if category not in self.policy.ownership_transfer_allowed:
                    raise GovernanceError(f"ownership transfer forbidden for {category.value}")
                old_path = (self.data_root / str(row["relative_path"])).resolve()
                new_path, new_relative = self._asset_path(target_owner, category, asset_id)
                if new_path.exists():
                    raise GovernanceError("transfer target already exists")
                new_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                new_scoped_digest = self._scoped_digest(
                    target_owner, str(row["content_sha256"])
                )
                os.replace(old_path, new_path)
                connection.execute(
                    """
                    UPDATE assets
                    SET owner_id = ?, relative_path = ?, scoped_sha256 = ?
                    WHERE asset_id = ?
                    """,
                    (target_owner, new_relative, new_scoped_digest, asset_id),
                )
                self._event(
                    connection,
                    "ownership_transferred",
                    owner_id=target_owner,
                    asset_id=asset_id,
                    details={
                        "from_owner_namespace": self._owner_namespace(source_owner),
                        "provenance_id": str(row["provenance_id"]),
                    },
                )
                changed += 1
        return changed

    def create_backup_manifest(
        self,
        backup_id: str,
        *,
        provider: EncryptedBackupProvider,
        key_reference: str,
        destination: Path,
        confirmation: str,
    ) -> BackupManifest:
        if confirmation != BACKUP_CONFIRMATION:
            raise GovernanceError("exact backup confirmation is required")
        if not key_reference or any(secret in key_reference.lower() for secret in ("key=", "token=")):
            raise GovernanceError("use an external key reference, not embedded key material")
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM assets WHERE status = ? ORDER BY owner_id, asset_id",
                (AssetStatus.ACTIVE.value,),
            ).fetchall()
        entries: list[BackupEntry] = []
        provenance: list[str] = []
        for row in rows:
            source = (self.data_root / str(row["relative_path"])).resolve()
            if not source.is_relative_to(self.data_root) or not source.is_file():
                raise GovernanceError("backup source is missing or unsafe")
            logical = (
                Path(self._owner_namespace(str(row["owner_id"])))
                / str(row["category"])
                / f"{row['asset_id']}.bin"
            ).as_posix()
            entries.append(
                BackupEntry(
                    logical_path=logical,
                    byte_count=int(row["byte_count"]),
                    sha256=_sha256(source),
                )
            )
            provenance.append(str(row["provenance_id"]))
        receipt = provider.export(self.data_root, destination, key_reference=key_reference)
        if not receipt or not provider.validate(destination, key_reference=key_reference):
            raise GovernanceError("encrypted backup provider validation failed")
        return BackupManifest(
            backup_id=backup_id,
            created_at=_utc_now(),
            encryption_provider=provider.provider_id,
            key_reference=key_reference,
            entries=tuple(entries),
            provenance_chain=tuple(provenance),
        )

    @staticmethod
    def verify_backup_manifest(manifest: BackupManifest, restored_root: Path) -> None:
        root = restored_root.resolve()
        seen: set[str] = set()
        for entry in manifest.entries:
            if entry.logical_path in seen:
                raise GovernanceError("duplicate backup entry")
            seen.add(entry.logical_path)
            target = (root / entry.logical_path).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise GovernanceError(f"backup entry missing or unsafe: {entry.logical_path}")
            if target.stat().st_size != entry.byte_count or _sha256(target) != entry.sha256:
                raise GovernanceError(f"backup entry integrity failure: {entry.logical_path}")

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        *,
        owner_id: str | None,
        asset_id: str | None,
        details: dict[str, object],
    ) -> None:
        connection.execute(
            """
            INSERT INTO governance_events(
                occurred_at, event_type, owner_namespace, asset_id, details_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                _utc_now().isoformat(),
                event_type,
                self._owner_namespace(owner_id) if owner_id else None,
                asset_id,
                _json(details),
            ),
        )
