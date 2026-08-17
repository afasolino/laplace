"""Persistent owner/device-isolated queue for outbound Laplace Client connections."""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Mapping, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
ClientAction = Literal["list", "read", "search", "write", "git", "run"]
_ACTIONS = {"list", "read", "search", "write", "git", "run"}
_TERMINAL = {"COMPLETE", "FAILED", "CANCELLED"}


class ClientServiceError(RuntimeError):
    """A client device or operation failed an ownership/state check."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ClientServiceError(f"invalid_{label}")
    return value


class ClientDeviceStore:
    """SQLite authority for pair/reconnect, operation lifecycle, and cancellation."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS client_devices (
                    device_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    paired_at_utc TEXT NOT NULL,
                    last_seen_at_utc TEXT NOT NULL,
                    revoked_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS client_devices_owner
                    ON client_devices(owner_user_id, revoked_at_utc);
                CREATE TABLE IF NOT EXISTS client_operations (
                    operation_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result_json TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    claimed_at_utc TEXT,
                    completed_at_utc TEXT,
                    FOREIGN KEY(device_id) REFERENCES client_devices(device_id)
                );
                CREATE INDEX IF NOT EXISTS client_operations_device_state
                    ON client_operations(device_id, state, created_at_utc);
                """
            )
        if os.name != "nt":
            os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @staticmethod
    def _device(row: sqlite3.Row) -> JsonObject:
        return {
            "device_id": str(row["device_id"]),
            "name": str(row["name"]),
            "capabilities": json.loads(str(row["capabilities_json"])),
            "paired_at_utc": str(row["paired_at_utc"]),
            "last_seen_at_utc": str(row["last_seen_at_utc"]),
            "revoked": row["revoked_at_utc"] is not None,
        }

    @staticmethod
    def _operation(row: sqlite3.Row) -> JsonObject:
        result = json.loads(str(row["result_json"])) if row["result_json"] is not None else None
        return {
            "operation_id": str(row["operation_id"]),
            "device_id": str(row["device_id"]),
            "workspace_id": str(row["workspace_id"]),
            "action": str(row["action"]),
            "arguments": json.loads(str(row["arguments_json"])),
            "state": str(row["state"]),
            "result": result,
            "cancellation_requested": str(row["state"]) == "CANCEL_REQUESTED",
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
            "completed_at_utc": row["completed_at_utc"],
        }

    def pair(
        self,
        owner_user_id: str,
        *,
        name: str,
        capabilities: Mapping[str, object],
        device_id: str | None = None,
    ) -> JsonObject:
        owner = _identifier(owner_user_id, label="owner_user_id")
        normalized_name = name.strip()
        if not normalized_name or len(normalized_name) > 160:
            raise ClientServiceError("invalid_device_name")
        encoded = json.dumps(dict(capabilities), sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 32_000:
            raise ClientServiceError("client_capabilities_too_large")
        now = _now()
        with self._lock, self._connect() as connection:
            if device_id is None:
                selected = "dev_" + secrets.token_hex(16)
                connection.execute(
                    """
                    INSERT INTO client_devices (
                        device_id, owner_user_id, name, capabilities_json,
                        paired_at_utc, last_seen_at_utc, revoked_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (selected, owner, normalized_name, encoded, now, now),
                )
            else:
                selected = device_id
                row = connection.execute(
                    "SELECT owner_user_id, revoked_at_utc FROM client_devices WHERE device_id=?",
                    (selected,),
                ).fetchone()
                if row is None or str(row["owner_user_id"]) != owner:
                    raise ClientServiceError("client_device_not_found")
                if row["revoked_at_utc"] is not None:
                    raise ClientServiceError("client_device_revoked")
                connection.execute(
                    """
                    UPDATE client_devices SET name=?, capabilities_json=?, last_seen_at_utc=?
                    WHERE device_id=? AND owner_user_id=?
                    """,
                    (normalized_name, encoded, now, selected, owner),
                )
            row = connection.execute(
                "SELECT * FROM client_devices WHERE device_id=?", (selected,)
            ).fetchone()
        assert row is not None
        return self._device(row)

    def list_devices(self, owner_user_id: str) -> list[JsonObject]:
        owner = _identifier(owner_user_id, label="owner_user_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM client_devices WHERE owner_user_id=?
                ORDER BY paired_at_utc, device_id
                """,
                (owner,),
            ).fetchall()
        return [self._device(row) for row in rows]

    def _require_device(
        self, connection: sqlite3.Connection, owner_user_id: str, device_id: str
    ) -> sqlite3.Row:
        owner = _identifier(owner_user_id, label="owner_user_id")
        if not re.fullmatch(r"dev_[a-f0-9]{32}", device_id):
            raise ClientServiceError("client_device_not_found")
        row = connection.execute(
            "SELECT * FROM client_devices WHERE device_id=? AND owner_user_id=?",
            (device_id, owner),
        ).fetchone()
        if row is None:
            raise ClientServiceError("client_device_not_found")
        if row["revoked_at_utc"] is not None:
            raise ClientServiceError("client_device_revoked")
        return cast(sqlite3.Row, row)

    def revoke(self, owner_user_id: str, device_id: str) -> JsonObject:
        with self._lock, self._connect() as connection:
            self._require_device(connection, owner_user_id, device_id)
            now = _now()
            connection.execute(
                "UPDATE client_devices SET revoked_at_utc=?, last_seen_at_utc=? WHERE device_id=?",
                (now, now, device_id),
            )
            connection.execute(
                """
                UPDATE client_operations SET state='CANCELLED', updated_at_utc=?, completed_at_utc=?
                WHERE device_id=? AND state IN ('PENDING', 'CLAIMED', 'CANCEL_REQUESTED')
                """,
                (now, now, device_id),
            )
        return {"status": "REVOKED", "device_id": device_id}

    def heartbeat(
        self,
        owner_user_id: str,
        device_id: str,
        capabilities: Mapping[str, object],
    ) -> JsonObject:
        encoded = json.dumps(dict(capabilities), sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 32_000:
            raise ClientServiceError("client_capabilities_too_large")
        with self._lock, self._connect() as connection:
            self._require_device(connection, owner_user_id, device_id)
            connection.execute(
                """
                UPDATE client_devices SET capabilities_json=?, last_seen_at_utc=?
                WHERE device_id=?
                """,
                (encoded, _now(), device_id),
            )
        return {"status": "CONNECTED", "device_id": device_id}

    def enqueue(
        self,
        owner_user_id: str,
        device_id: str,
        *,
        workspace_id: str,
        action: str,
        arguments: Mapping[str, object],
    ) -> JsonObject:
        if action not in _ACTIONS:
            raise ClientServiceError("client_action_not_supported")
        if not re.fullmatch(r"ws-[a-f0-9]{24}", workspace_id):
            raise ClientServiceError("client_workspace_id_invalid")
        encoded = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 2_100_000:
            raise ClientServiceError("client_operation_too_large")
        operation_id = "cop_" + secrets.token_hex(16)
        now = _now()
        with self._lock, self._connect() as connection:
            self._require_device(connection, owner_user_id, device_id)
            active = connection.execute(
                """
                SELECT COUNT(*) AS count FROM client_operations
                WHERE device_id=? AND state NOT IN ('COMPLETE', 'FAILED', 'CANCELLED')
                """,
                (device_id,),
            ).fetchone()
            if active is not None and int(active["count"]) >= 100:
                raise ClientServiceError("client_operation_capacity")
            connection.execute(
                """
                INSERT INTO client_operations (
                    operation_id, owner_user_id, device_id, workspace_id, action,
                    arguments_json, state, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    operation_id,
                    owner_user_id,
                    device_id,
                    workspace_id,
                    action,
                    encoded,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM client_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        assert row is not None
        return self._operation(row)

    def claim(self, owner_user_id: str, device_id: str) -> JsonObject | None:
        stale_before = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_device(connection, owner_user_id, device_id)
            now = _now()
            connection.execute(
                """
                UPDATE client_operations SET state='PENDING', claimed_at_utc=NULL,
                    updated_at_utc=?
                WHERE device_id=? AND state='CLAIMED' AND claimed_at_utc < ?
                """,
                (now, device_id, stale_before),
            )
            row = connection.execute(
                """
                SELECT * FROM client_operations
                WHERE device_id=? AND owner_user_id=? AND state='PENDING'
                ORDER BY created_at_utc, operation_id LIMIT 1
                """,
                (device_id, owner_user_id),
            ).fetchone()
            if row is None:
                connection.execute(
                    "UPDATE client_devices SET last_seen_at_utc=? WHERE device_id=?",
                    (now, device_id),
                )
                return None
            connection.execute(
                """
                UPDATE client_operations SET state='CLAIMED', claimed_at_utc=?, updated_at_utc=?
                WHERE operation_id=? AND state='PENDING'
                """,
                (now, now, str(row["operation_id"])),
            )
            claimed = connection.execute(
                "SELECT * FROM client_operations WHERE operation_id=?",
                (str(row["operation_id"]),),
            ).fetchone()
            connection.execute(
                "UPDATE client_devices SET last_seen_at_utc=? WHERE device_id=?",
                (now, device_id),
            )
        assert claimed is not None
        return self._operation(claimed)

    def get_operation(self, owner_user_id: str, operation_id: str) -> JsonObject:
        if not re.fullmatch(r"cop_[a-f0-9]{32}", operation_id):
            raise ClientServiceError("client_operation_not_found")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM client_operations
                WHERE operation_id=? AND owner_user_id=?
                """,
                (operation_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise ClientServiceError("client_operation_not_found")
        return self._operation(row)

    def complete(
        self,
        owner_user_id: str,
        device_id: str,
        operation_id: str,
        *,
        result: Mapping[str, object],
        failed: bool,
    ) -> JsonObject:
        encoded = json.dumps(dict(result), sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 2_100_000:
            raise ClientServiceError("client_result_too_large")
        with self._lock, self._connect() as connection:
            self._require_device(connection, owner_user_id, device_id)
            row = connection.execute(
                """
                SELECT * FROM client_operations
                WHERE operation_id=? AND owner_user_id=? AND device_id=?
                """,
                (operation_id, owner_user_id, device_id),
            ).fetchone()
            if row is None:
                raise ClientServiceError("client_operation_not_found")
            state = str(row["state"])
            if state in _TERMINAL:
                return self._operation(row)
            final = "CANCELLED" if state == "CANCEL_REQUESTED" else ("FAILED" if failed else "COMPLETE")
            now = _now()
            connection.execute(
                """
                UPDATE client_operations SET state=?, result_json=?, updated_at_utc=?,
                    completed_at_utc=? WHERE operation_id=?
                """,
                (final, encoded, now, now, operation_id),
            )
            updated = connection.execute(
                "SELECT * FROM client_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        assert updated is not None
        return self._operation(updated)

    def cancel(self, owner_user_id: str, operation_id: str) -> JsonObject:
        current = self.get_operation(owner_user_id, operation_id)
        state = str(current["state"])
        if state in _TERMINAL:
            return current
        final = "CANCELLED" if state == "PENDING" else "CANCEL_REQUESTED"
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE client_operations SET state=?, updated_at_utc=?,
                    completed_at_utc=CASE WHEN ?='CANCELLED' THEN ? ELSE completed_at_utc END
                WHERE operation_id=? AND owner_user_id=?
                """,
                (final, now, final, now, operation_id, owner_user_id),
            )
        return self.get_operation(owner_user_id, operation_id)
