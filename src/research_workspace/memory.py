"""Persistent owner/project-scoped episodic and semantic memory.

Memory is intentionally separate from Personal Corpus.  This backend stores
explicitly admitted learned state in SQLite, keeps immutable mutation history,
and uses deterministic lexical matching until a separately certified local
semantic index is available.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
MemoryKind = Literal["episodic", "semantic"]
MemoryState = Literal["ACTIVE", "SUPERSEDED", "DELETED"]
MemoryWriteMode = Literal["explicit_user", "deterministic_event", "approved_model_proposal"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MEMORY_ID = re.compile(r"^mem_[a-f0-9]{32}$")
_MAX_CONTENT_CHARS = 32_000
_MAX_METADATA_BYTES = 16_384
_SCHEMA_VERSION = 1


class MemoryError(RuntimeError):
    """Base error for fail-closed memory operations."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class MemoryValidationError(MemoryError):
    """A caller supplied malformed or inadmissible memory input."""


class MemoryNotFoundError(MemoryError):
    """A memory is absent from the requested owner/project scope."""


class MemoryCorruptionError(MemoryError):
    """Durable memory or provenance failed strict decoding."""


class MemoryKindValue(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise MemoryValidationError(f"invalid_{label}")
    return value


def _validate_text(value: str, *, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise MemoryValidationError(f"invalid_{label}")
    return value.strip()


def _json_object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MemoryValidationError(f"invalid_{label}")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MemoryValidationError(f"invalid_{label}") from exc
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise MemoryValidationError(f"{label}_too_large")
    return cast(JsonObject, value)


@dataclass(frozen=True)
class MemoryProvenance:
    """Required provenance for every accepted memory write."""

    source_kind: Literal["user_explicit", "deterministic_event", "model_proposal"]
    source_id: str
    actor_id: str
    approved: bool
    created_at_utc: str = field(default_factory=_now)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, str) or self.source_kind not in {
            "user_explicit",
            "deterministic_event",
            "model_proposal",
        }:
            raise MemoryValidationError("invalid_provenance_source_kind")
        _validate_identifier(self.source_id, label="provenance_source_id")
        _validate_identifier(self.actor_id, label="provenance_actor_id")
        if not isinstance(self.approved, bool):
            raise MemoryValidationError("invalid_provenance_approval")
        _validate_text(self.created_at_utc, label="provenance_timestamp", maximum=64)
        _json_object(dict(self.metadata), label="provenance_metadata")
        if self.source_kind == "model_proposal" and not self.approved:
            raise MemoryValidationError("model_proposal_requires_approval")

    def to_json(self) -> JsonObject:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "actor_id": self.actor_id,
            "approved": self.approved,
            "created_at_utc": self.created_at_utc,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, value: object) -> "MemoryProvenance":
        raw = _json_object(value, label="provenance")
        required = {
            "source_kind",
            "source_id",
            "actor_id",
            "approved",
            "created_at_utc",
            "metadata",
        }
        if set(raw) != required:
            raise MemoryCorruptionError("memory_provenance_schema_invalid")
        source_kind = raw["source_kind"]
        if source_kind not in {"user_explicit", "deterministic_event", "model_proposal"}:
            raise MemoryCorruptionError("memory_provenance_source_invalid")
        if not isinstance(raw["source_id"], str) or not isinstance(raw["actor_id"], str):
            raise MemoryCorruptionError("memory_provenance_identity_invalid")
        if not isinstance(raw["approved"], bool) or not isinstance(raw["created_at_utc"], str):
            raise MemoryCorruptionError("memory_provenance_fields_invalid")
        metadata = raw["metadata"]
        if not isinstance(metadata, dict):
            raise MemoryCorruptionError("memory_provenance_metadata_invalid")
        try:
            return cls(
                source_kind=source_kind,
                source_id=raw["source_id"],
                actor_id=raw["actor_id"],
                approved=raw["approved"],
                created_at_utc=raw["created_at_utc"],
                metadata=metadata,
            )
        except MemoryError as exc:
            raise MemoryCorruptionError("memory_provenance_invalid", {"category": exc.category}) from exc


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    owner_id: str
    project_id: str
    kind: MemoryKind
    content: str
    state: MemoryState
    version: int
    created_at_utc: str
    updated_at_utc: str
    provenance: MemoryProvenance
    content_sha256: str
    contradiction_key: str | None = None
    supersedes_memory_id: str | None = None

    def __post_init__(self) -> None:
        if _MEMORY_ID.fullmatch(self.memory_id) is None:
            raise MemoryValidationError("invalid_memory_id")
        _validate_identifier(self.owner_id, label="owner_id")
        _validate_identifier(self.project_id, label="project_id")
        if not isinstance(self.kind, str) or self.kind not in {"episodic", "semantic"}:
            raise MemoryValidationError("invalid_memory_kind")
        _validate_text(self.content, label="memory_content", maximum=_MAX_CONTENT_CHARS)
        if not isinstance(self.state, str) or self.state not in {
            "ACTIVE",
            "SUPERSEDED",
            "DELETED",
        }:
            raise MemoryValidationError("invalid_memory_state")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise MemoryValidationError("invalid_memory_version")
        for value, label in (
            (self.created_at_utc, "created_timestamp"),
            (self.updated_at_utc, "updated_timestamp"),
        ):
            _validate_text(value, label=label, maximum=64)
        if not re.fullmatch(r"[a-f0-9]{64}", self.content_sha256):
            raise MemoryValidationError("invalid_memory_hash")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_sha256:
            raise MemoryCorruptionError("memory_content_hash_mismatch")
        if self.contradiction_key is not None:
            _validate_text(self.contradiction_key, label="contradiction_key", maximum=256)
        if self.supersedes_memory_id is not None and _MEMORY_ID.fullmatch(self.supersedes_memory_id) is None:
            raise MemoryValidationError("invalid_superseded_memory_id")

    def to_json(self) -> JsonObject:
        value = asdict(self)
        value["provenance"] = self.provenance.to_json()
        return value


class MemoryBackend(Protocol):
    """Replaceable persistence/search boundary for learned memory."""

    def add(self, record: MemoryRecord) -> MemoryRecord: ...

    def get(self, owner_id: str, project_id: str, memory_id: str) -> MemoryRecord: ...

    def search(
        self,
        owner_id: str,
        project_id: str,
        query: str,
        *,
        kind: MemoryKind | None = None,
        limit: int = 20,
    ) -> tuple[MemoryRecord, ...]: ...

    def supersede(self, old: MemoryRecord, replacement: MemoryRecord) -> MemoryRecord: ...

    def delete(self, record: MemoryRecord) -> MemoryRecord: ...

    def history(self, owner_id: str, project_id: str, memory_id: str) -> tuple[JsonObject, ...]: ...

    def health(self) -> JsonObject: ...


def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
    provenance_raw: object
    try:
        provenance_raw = json.loads(str(row["provenance_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MemoryCorruptionError("memory_provenance_json_corrupt") from exc
    try:
        source_kind = cast(MemoryKind, str(row["kind"]))
        return MemoryRecord(
            memory_id=str(row["memory_id"]),
            owner_id=str(row["owner_id"]),
            project_id=str(row["project_id"]),
            kind=source_kind,
            content=str(row["content"]),
            state=cast(MemoryState, str(row["state"])),
            version=int(row["version"]),
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
            provenance=MemoryProvenance.from_json(provenance_raw),
            content_sha256=str(row["content_sha256"]),
            contradiction_key=(
                str(row["contradiction_key"]) if row["contradiction_key"] is not None else None
            ),
            supersedes_memory_id=(
                str(row["supersedes_memory_id"])
                if row["supersedes_memory_id"] is not None
                else None
            ),
        )
    except (MemoryError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, MemoryCorruptionError):
            raise
        raise MemoryCorruptionError("memory_record_corrupt") from exc


class SQLiteMemoryBackend:
    """Private SQLite backend with atomic writes and deterministic lexical search."""

    def __init__(self, path: Path, *, clock: Callable[[], str] = _now) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.RLock()
        self._initialize()
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_schema (
                        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                        schema_version INTEGER NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        memory_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('episodic','semantic')),
                        content TEXT NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('ACTIVE','SUPERSEDED','DELETED')),
                        version INTEGER NOT NULL CHECK(version >= 1),
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL,
                        provenance_json TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        contradiction_key TEXT,
                        supersedes_memory_id TEXT
                    );
                    CREATE INDEX IF NOT EXISTS memory_active_scope
                        ON memory_entries(owner_id, project_id, state, kind);
                    CREATE INDEX IF NOT EXISTS memory_contradictions
                        ON memory_entries(owner_id, project_id, contradiction_key, state);
                    CREATE TABLE IF NOT EXISTS memory_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at_utc TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS memory_events_lookup
                        ON memory_events(owner_id, project_id, memory_id, event_id);
                    """,
                )
                connection.execute(
                    """
                    INSERT INTO memory_schema(singleton, schema_version, updated_at_utc)
                    VALUES (1, ?, ?)
                    ON CONFLICT(singleton) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (_SCHEMA_VERSION, self.clock()),
                )
        except (sqlite3.DatabaseError, OSError) as exc:
            raise MemoryCorruptionError("memory_database_unavailable") from exc

    @staticmethod
    def _scope(owner_id: str, project_id: str) -> tuple[str, str]:
        return (
            _validate_identifier(owner_id, label="owner_id"),
            _validate_identifier(project_id, label="project_id"),
        )

    @staticmethod
    def _require_memory_id(memory_id: str) -> str:
        if not isinstance(memory_id, str) or _MEMORY_ID.fullmatch(memory_id) is None:
            raise MemoryValidationError("invalid_memory_id")
        return memory_id

    @staticmethod
    def _event_payload(record: MemoryRecord, *, old_state: str | None = None) -> str:
        value: JsonObject = {
            "record": record.to_json(),
            "old_state": old_state,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def _fetch(
        self,
        connection: sqlite3.Connection,
        owner_id: str,
        project_id: str,
        memory_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT * FROM memory_entries
            WHERE owner_id=? AND project_id=? AND memory_id=?
            """,
            (owner_id, project_id, memory_id),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError("memory_not_found")
        return cast(sqlite3.Row, row)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if record.contradiction_key is not None:
                        conflict = connection.execute(
                            """
                            SELECT memory_id FROM memory_entries
                            WHERE owner_id=? AND project_id=? AND state='ACTIVE'
                              AND contradiction_key=?
                            LIMIT 1
                            """,
                            (record.owner_id, record.project_id, record.contradiction_key),
                        ).fetchone()
                        if conflict is not None:
                            raise MemoryValidationError("contradiction_requires_supersession")
                    connection.execute(
                        """
                        INSERT INTO memory_entries(
                            memory_id, owner_id, project_id, kind, content, state, version,
                            created_at_utc, updated_at_utc, provenance_json, content_sha256,
                            contradiction_key, supersedes_memory_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.memory_id,
                            record.owner_id,
                            record.project_id,
                            record.kind,
                            record.content,
                            record.state,
                            record.version,
                            record.created_at_utc,
                            record.updated_at_utc,
                            json.dumps(record.provenance.to_json(), sort_keys=True, separators=(",", ":")),
                            record.content_sha256,
                            record.contradiction_key,
                            record.supersedes_memory_id,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_events(
                            memory_id, owner_id, project_id, event_type, occurred_at_utc, payload_json
                        ) VALUES (?, ?, ?, 'CREATED', ?, ?)
                        """,
                        (
                            record.memory_id,
                            record.owner_id,
                            record.project_id,
                            self.clock(),
                            self._event_payload(record),
                        ),
                    )
            except MemoryError:
                raise
            except sqlite3.IntegrityError as exc:
                raise MemoryValidationError("memory_id_exists") from exc
            except sqlite3.DatabaseError as exc:
                raise MemoryError("memory_write_failed") from exc
        return record

    def get(self, owner_id: str, project_id: str, memory_id: str) -> MemoryRecord:
        owner_id, project_id = self._scope(owner_id, project_id)
        memory_id = self._require_memory_id(memory_id)
        with self._connect() as connection:
            return _record_from_row(self._fetch(connection, owner_id, project_id, memory_id))

    @staticmethod
    def _tokens(query: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(re.findall(r"[\w-]{2,}", query.casefold())))

    def search(
        self,
        owner_id: str,
        project_id: str,
        query: str,
        *,
        kind: MemoryKind | None = None,
        limit: int = 20,
    ) -> tuple[MemoryRecord, ...]:
        owner_id, project_id = self._scope(owner_id, project_id)
        query = _validate_text(query, label="memory_query", maximum=4_000)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise MemoryValidationError("invalid_memory_limit")
        if kind is not None and kind not in {"episodic", "semantic"}:
            raise MemoryValidationError("invalid_memory_kind")
        tokens = self._tokens(query)
        if not tokens:
            raise MemoryValidationError("memory_query_has_no_terms")
        clauses = ["owner_id=?", "project_id=?", "state='ACTIVE'"]
        parameters: list[object] = [owner_id, project_id]
        if kind is not None:
            clauses.append("kind=?")
            parameters.append(kind)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memory_entries WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at_utc DESC, memory_id ASC",
                parameters,
            ).fetchall()
        records = tuple(_record_from_row(row) for row in rows)

        def score(record: MemoryRecord) -> tuple[int, str, str]:
            text = record.content.casefold()
            matched = sum(token in text for token in tokens)
            return matched, record.updated_at_utc, record.memory_id

        ranked = sorted(
            (record for record in records if score(record)[0] > 0),
            key=score,
            reverse=True,
        )
        return tuple(ranked[:limit])

    def supersede(self, old: MemoryRecord, replacement: MemoryRecord) -> MemoryRecord:
        if old.owner_id != replacement.owner_id or old.project_id != replacement.project_id:
            raise MemoryValidationError("memory_scope_mismatch")
        if old.state != "ACTIVE":
            raise MemoryValidationError("memory_not_active")
        if replacement.supersedes_memory_id != old.memory_id:
            raise MemoryValidationError("supersession_link_missing")
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    current = _record_from_row(
                        self._fetch(connection, old.owner_id, old.project_id, old.memory_id)
                    )
                    if current.state != "ACTIVE":
                        raise MemoryValidationError("memory_not_active")
                    if replacement.contradiction_key is not None:
                        conflict = connection.execute(
                            """
                            SELECT memory_id FROM memory_entries
                            WHERE owner_id=? AND project_id=? AND state='ACTIVE'
                              AND contradiction_key=? AND memory_id != ?
                            LIMIT 1
                            """,
                            (
                                replacement.owner_id,
                                replacement.project_id,
                                replacement.contradiction_key,
                                old.memory_id,
                            ),
                        ).fetchone()
                        if conflict is not None:
                            raise MemoryValidationError("contradiction_requires_supersession")
                    connection.execute(
                        "UPDATE memory_entries SET state='SUPERSEDED', updated_at_utc=? WHERE memory_id=?",
                        (self.clock(), old.memory_id),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_entries(
                            memory_id, owner_id, project_id, kind, content, state, version,
                            created_at_utc, updated_at_utc, provenance_json, content_sha256,
                            contradiction_key, supersedes_memory_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            replacement.memory_id,
                            replacement.owner_id,
                            replacement.project_id,
                            replacement.kind,
                            replacement.content,
                            replacement.state,
                            replacement.version,
                            replacement.created_at_utc,
                            replacement.updated_at_utc,
                            json.dumps(replacement.provenance.to_json(), sort_keys=True, separators=(",", ":")),
                            replacement.content_sha256,
                            replacement.contradiction_key,
                            replacement.supersedes_memory_id,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_events(
                            memory_id, owner_id, project_id, event_type, occurred_at_utc, payload_json
                        ) VALUES (?, ?, ?, 'SUPERSEDED', ?, ?)
                        """,
                        (
                            replacement.memory_id,
                            replacement.owner_id,
                            replacement.project_id,
                            self.clock(),
                            json.dumps(
                                {
                                    "old_memory_id": old.memory_id,
                                    "old_state": "ACTIVE",
                                    "replacement": replacement.to_json(),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_events(
                            memory_id, owner_id, project_id, event_type, occurred_at_utc, payload_json
                        ) VALUES (?, ?, ?, 'SUPERSEDED', ?, ?)
                        """,
                        (
                            old.memory_id,
                            old.owner_id,
                            old.project_id,
                            self.clock(),
                            json.dumps(
                                {
                                    "replacement_memory_id": replacement.memory_id,
                                    "old_state": "ACTIVE",
                                    "replacement": replacement.to_json(),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ),
                    )
            except MemoryError:
                raise
            except sqlite3.IntegrityError as exc:
                raise MemoryValidationError("memory_id_exists") from exc
            except sqlite3.DatabaseError as exc:
                raise MemoryError("memory_supersession_failed") from exc
        return replacement

    def delete(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    current = _record_from_row(
                        self._fetch(connection, record.owner_id, record.project_id, record.memory_id)
                    )
                    if current.state == "DELETED":
                        return current
                    connection.execute(
                        "UPDATE memory_entries SET state='DELETED', updated_at_utc=? WHERE memory_id=?",
                        (self.clock(), record.memory_id),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_events(
                            memory_id, owner_id, project_id, event_type, occurred_at_utc, payload_json
                        ) VALUES (?, ?, ?, 'DELETED', ?, ?)
                        """,
                        (
                            record.memory_id,
                            record.owner_id,
                            record.project_id,
                            self.clock(),
                            self._event_payload(current, old_state=current.state),
                        ),
                    )
            except MemoryError:
                raise
            except sqlite3.DatabaseError as exc:
                raise MemoryError("memory_delete_failed") from exc
        return self.get(record.owner_id, record.project_id, record.memory_id)

    def history(self, owner_id: str, project_id: str, memory_id: str) -> tuple[JsonObject, ...]:
        owner_id, project_id = self._scope(owner_id, project_id)
        memory_id = self._require_memory_id(memory_id)
        with self._connect() as connection:
            self._fetch(connection, owner_id, project_id, memory_id)
            rows = connection.execute(
                """
                SELECT event_id, event_type, occurred_at_utc, payload_json
                FROM memory_events
                WHERE owner_id=? AND project_id=? AND memory_id=?
                ORDER BY event_id ASC
                """,
                (owner_id, project_id, memory_id),
            ).fetchall()
        events: list[JsonObject] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MemoryCorruptionError("memory_event_json_corrupt") from exc
            if not isinstance(payload, dict):
                raise MemoryCorruptionError("memory_event_payload_invalid")
            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "event_type": str(row["event_type"]),
                    "occurred_at_utc": str(row["occurred_at_utc"]),
                    "payload": payload,
                }
            )
        return tuple(events)

    def health(self) -> JsonObject:
        try:
            with self._connect() as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM memory_entries"
                ).fetchone()
                for entry in connection.execute("SELECT * FROM memory_entries").fetchall():
                    _record_from_row(entry)
                for event in connection.execute("SELECT payload_json FROM memory_events").fetchall():
                    payload = json.loads(str(event["payload_json"]))
                    if not isinstance(payload, dict):
                        raise MemoryCorruptionError("memory_event_payload_invalid")
            integrity_value = str(integrity[0]) if integrity is not None else "unknown"
            if integrity_value != "ok":
                raise MemoryCorruptionError("memory_sqlite_integrity_failed")
            return {
                "status": "READY",
                "schema_version": _SCHEMA_VERSION,
                "path": str(self.path),
                "entries": int(row["count"]) if row is not None else 0,
                "backend": "sqlite_local_lexical_v1",
            }
        except MemoryError:
            raise
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MemoryCorruptionError("memory_health_failed") from exc


class MemoryService:
    """Admission and scope facade over a replaceable memory backend."""

    def __init__(self, backend: MemoryBackend) -> None:
        self.backend = backend

    @staticmethod
    def _write_mode(mode: str) -> MemoryWriteMode:
        if not isinstance(mode, str) or mode not in {
            "explicit_user",
            "deterministic_event",
            "approved_model_proposal",
        }:
            raise MemoryValidationError("memory_write_requires_explicit_intent")
        return cast(MemoryWriteMode, mode)

    @staticmethod
    def _provenance(
        provenance: MemoryProvenance | Mapping[str, object],
        *,
        mode: MemoryWriteMode,
    ) -> MemoryProvenance:
        value = (
            provenance
            if isinstance(provenance, MemoryProvenance)
            else MemoryProvenance.from_json(provenance)
        )
        expected = {
            "explicit_user": "user_explicit",
            "deterministic_event": "deterministic_event",
            "approved_model_proposal": "model_proposal",
        }[mode]
        if value.source_kind != expected:
            raise MemoryValidationError("memory_provenance_intent_mismatch")
        if mode == "approved_model_proposal" and not value.approved:
            raise MemoryValidationError("model_proposal_requires_approval")
        return value

    @staticmethod
    def _record(
        *,
        owner_id: str,
        project_id: str,
        kind: MemoryKind,
        content: str,
        provenance: MemoryProvenance,
        contradiction_key: str | None,
        supersedes_memory_id: str | None = None,
        version: int = 1,
        clock: Callable[[], str] = _now,
    ) -> MemoryRecord:
        content = _validate_text(content, label="memory_content", maximum=_MAX_CONTENT_CHARS)
        now = clock()
        return MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex}",
            owner_id=_validate_identifier(owner_id, label="owner_id"),
            project_id=_validate_identifier(project_id, label="project_id"),
            kind=kind,
            content=content,
            state="ACTIVE",
            version=version,
            created_at_utc=now,
            updated_at_utc=now,
            provenance=provenance,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            contradiction_key=contradiction_key,
            supersedes_memory_id=supersedes_memory_id,
        )

    def add(
        self,
        *,
        owner_id: str,
        project_id: str,
        kind: MemoryKind,
        content: str,
        provenance: MemoryProvenance | Mapping[str, object],
        write_mode: str,
        contradiction_key: str | None = None,
    ) -> MemoryRecord:
        mode = self._write_mode(write_mode)
        accepted_provenance = self._provenance(provenance, mode=mode)
        if contradiction_key is not None:
            contradiction_key = _validate_text(
                contradiction_key, label="contradiction_key", maximum=256
            )
        record = self._record(
            owner_id=owner_id,
            project_id=project_id,
            kind=kind,
            content=content,
            provenance=accepted_provenance,
            contradiction_key=contradiction_key,
        )
        return self.backend.add(record)

    def get(self, *, owner_id: str, project_id: str, memory_id: str) -> MemoryRecord:
        return self.backend.get(owner_id, project_id, memory_id)

    def search(
        self,
        *,
        owner_id: str,
        project_id: str,
        query: str,
        kind: MemoryKind | None = None,
        limit: int = 20,
    ) -> tuple[MemoryRecord, ...]:
        return self.backend.search(owner_id, project_id, query, kind=kind, limit=limit)

    def update(
        self,
        *,
        owner_id: str,
        project_id: str,
        memory_id: str,
        content: str,
        provenance: MemoryProvenance | Mapping[str, object],
        write_mode: str,
    ) -> MemoryRecord:
        current = self.get(owner_id=owner_id, project_id=project_id, memory_id=memory_id)
        if current.state != "ACTIVE":
            raise MemoryValidationError("memory_not_active")
        mode = self._write_mode(write_mode)
        accepted_provenance = self._provenance(provenance, mode=mode)
        replacement = self._record(
            owner_id=owner_id,
            project_id=project_id,
            kind=current.kind,
            content=content,
            provenance=accepted_provenance,
            contradiction_key=current.contradiction_key,
            supersedes_memory_id=current.memory_id,
            version=current.version + 1,
        )
        return self.backend.supersede(current, replacement)

    def supersede(
        self,
        *,
        owner_id: str,
        project_id: str,
        memory_id: str,
        content: str,
        provenance: MemoryProvenance | Mapping[str, object],
        write_mode: str,
    ) -> MemoryRecord:
        return self.update(
            owner_id=owner_id,
            project_id=project_id,
            memory_id=memory_id,
            content=content,
            provenance=provenance,
            write_mode=write_mode,
        )

    def delete(self, *, owner_id: str, project_id: str, memory_id: str) -> MemoryRecord:
        current = self.get(owner_id=owner_id, project_id=project_id, memory_id=memory_id)
        return self.backend.delete(current)

    def history(self, *, owner_id: str, project_id: str, memory_id: str) -> tuple[JsonObject, ...]:
        return self.backend.history(owner_id, project_id, memory_id)

    def health(self) -> JsonObject:
        return self.backend.health()
