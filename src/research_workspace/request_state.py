"""Truthful high-level request state tracking for bounded polling."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]

REQUEST_STATES = frozenset(
    {
        "VALIDATING",
        "QUEUED",
        "ADMITTED",
        "PREPARING_CONTEXT",
        "RETRIEVING",
        "GENERATING",
        "VALIDATING_OUTPUT",
        "ESCALATING",
        "COMPLETE",
        "CANCELLED",
        "TIMED_OUT",
        "FAILED",
    }
)
TERMINAL_STATES = frozenset({"COMPLETE", "CANCELLED", "TIMED_OUT", "FAILED"})


class RequestStateError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class RequestStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_states (
                    request_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    started_monotonic REAL NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    queue_position INTEGER,
                    requested_lane TEXT,
                    effective_lane TEXT,
                    model_name TEXT,
                    retrieval_json TEXT NOT NULL,
                    operation TEXT,
                    cancelled INTEGER NOT NULL DEFAULT 0 CHECK(cancelled IN (0,1))
                )
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def create(
        self,
        request_id: str,
        *,
        owner_user_id: str,
        request_type: str,
        trace_id: str,
        requested_lane: str | None = None,
    ) -> JsonObject:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,159}", request_id):
            raise RequestStateError("invalid_request_id")
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO request_states (
                        request_id, owner_user_id, request_type, state, trace_id,
                        started_monotonic, created_at_utc, updated_at_utc,
                        requested_lane, retrieval_json
                    ) VALUES (?, ?, ?, 'VALIDATING', ?, ?, ?, ?, ?, '{}')
                    """,
                    (
                        request_id,
                        owner_user_id,
                        request_type,
                        trace_id,
                        time.monotonic(),
                        now,
                        now,
                        requested_lane,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise RequestStateError("request_already_active") from exc
        return self.get(owner_user_id, request_id)

    def transition(
        self,
        request_id: str,
        *,
        owner_user_id: str,
        state: str,
        queue_position: int | None = None,
        effective_lane: str | None = None,
        model_name: str | None = None,
        retrieval: JsonObject | None = None,
        operation: str | None = None,
    ) -> JsonObject:
        if state not in REQUEST_STATES:
            raise RequestStateError("invalid_request_state")
        current = self.get(owner_user_id, request_id)
        if str(current["state"]) in TERMINAL_STATES:
            return current
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE request_states
                SET state=?, updated_at_utc=?, queue_position=?,
                    effective_lane=COALESCE(?, effective_lane),
                    model_name=COALESCE(?, model_name),
                    retrieval_json=COALESCE(?, retrieval_json),
                    operation=COALESCE(?, operation)
                WHERE request_id=? AND owner_user_id=?
                """,
                (
                    state,
                    datetime.now(UTC).isoformat(),
                    queue_position,
                    effective_lane,
                    model_name,
                    (
                        json.dumps(retrieval, sort_keys=True, separators=(",", ":"))
                        if retrieval is not None
                        else None
                    ),
                    operation,
                    request_id,
                    owner_user_id,
                ),
            )
        return self.get(owner_user_id, request_id)

    def get(self, owner_user_id: str, request_id: str) -> JsonObject:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM request_states
                WHERE request_id=? AND owner_user_id=?
                """,
                (request_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise RequestStateError("request_not_found")
        return {
            "request_id": str(row["request_id"]),
            "request_type": str(row["request_type"]),
            "state": str(row["state"]),
            "trace_id": str(row["trace_id"]),
            "elapsed_seconds": round(
                max(0.0, time.monotonic() - float(row["started_monotonic"])), 3
            ),
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
            "queue_position": row["queue_position"],
            "requested_lane": row["requested_lane"],
            "effective_lane": row["effective_lane"],
            "model_name": row["model_name"],
            "retrieval": json.loads(str(row["retrieval_json"])),
            "operation": row["operation"],
            "cancel_requested": bool(row["cancelled"]),
        }

    def cancel(self, owner_user_id: str, request_id: str) -> JsonObject:
        current = self.get(owner_user_id, request_id)
        if str(current["state"]) in TERMINAL_STATES:
            return current
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE request_states
                SET state='CANCELLED', cancelled=1, updated_at_utc=?
                WHERE request_id=? AND owner_user_id=?
                """,
                (datetime.now(UTC).isoformat(), request_id, owner_user_id),
            )
        return self.get(owner_user_id, request_id)
