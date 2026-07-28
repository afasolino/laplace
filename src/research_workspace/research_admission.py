"""Owner isolation and conservative admission for Deep Research jobs."""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]


class ResearchAdmissionError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class ResearchAdmissionStore:
    """SQLite WAL ownership and queue state for research workflows."""

    def __init__(self, path: Path, *, global_active: int = 2, per_user_active: int = 1) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.global_active = global_active
        self.per_user_active = per_user_active
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_admission (
                    research_job_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    queue_reason TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    cancelled_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS research_owner_state
                ON research_admission(owner_user_id, state);
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def create(self, owner_user_id: str, research_job_id: str) -> JsonObject:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            global_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM research_admission WHERE state IN ('ADMITTED', 'RUNNING')"
                ).fetchone()[0]
            )
            user_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM research_admission
                    WHERE owner_user_id = ? AND state IN ('ADMITTED', 'RUNNING')
                    """,
                    (owner_user_id,),
                ).fetchone()[0]
            )
            if global_count >= self.global_active:
                state, reason = "QUEUED", "global_deep_research_limit"
            elif user_count >= self.per_user_active:
                state, reason = "QUEUED", "per_user_deep_research_limit"
            else:
                state, reason = "ADMITTED", None
            connection.execute(
                """
                INSERT INTO research_admission (
                    research_job_id, owner_user_id, state, queue_reason,
                    created_at_utc, updated_at_utc, cancelled_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (research_job_id, owner_user_id, state, reason, now, now),
            )
        return self.status(owner_user_id, research_job_id)

    def _row(self, owner_user_id: str, research_job_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM research_admission
                WHERE research_job_id = ? AND owner_user_id = ?
                """,
                (research_job_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise ResearchAdmissionError("research_job_not_found")
        return cast(sqlite3.Row, row)

    def status(self, owner_user_id: str, research_job_id: str) -> JsonObject:
        row = self._row(owner_user_id, research_job_id)
        with self._connect() as connection:
            queued = connection.execute(
                """
                SELECT research_job_id FROM research_admission
                WHERE state = 'QUEUED'
                ORDER BY created_at_utc
                """
            ).fetchall()
        position = next(
            (
                index
                for index, current in enumerate(queued, start=1)
                if str(current["research_job_id"]) == research_job_id
            ),
            0,
        )
        return {
            "research_job_id": research_job_id,
            "state": str(row["state"]),
            "queue_reason": (
                str(row["queue_reason"]) if row["queue_reason"] is not None else None
            ),
            "queue_position": position,
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
        }

    def begin(self, owner_user_id: str, research_job_id: str) -> JsonObject:
        current = self.status(owner_user_id, research_job_id)
        if current["state"] == "QUEUED":
            raise ResearchAdmissionError("capacity_guardrail")
        if current["state"] == "CANCELLED":
            raise ResearchAdmissionError("research_job_cancelled")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE research_admission SET state = 'RUNNING', updated_at_utc = ?
                WHERE research_job_id = ? AND owner_user_id = ?
                """,
                (now, research_job_id, owner_user_id),
            )
        return self.status(owner_user_id, research_job_id)

    def finish(self, owner_user_id: str, research_job_id: str, *, failed: bool = False) -> None:
        self._row(owner_user_id, research_job_id)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE research_admission SET state = ?, updated_at_utc = ?
                WHERE research_job_id = ? AND owner_user_id = ?
                """,
                ("FAILED" if failed else "COMPLETE", now, research_job_id, owner_user_id),
            )
            self._promote(connection, now)

    def cancel(self, owner_user_id: str, research_job_id: str) -> JsonObject:
        self._row(owner_user_id, research_job_id)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE research_admission
                SET state = 'CANCELLED', updated_at_utc = ?, cancelled_at_utc = ?
                WHERE research_job_id = ? AND owner_user_id = ?
                """,
                (now, now, research_job_id, owner_user_id),
            )
            self._promote(connection, now)
        return self.status(owner_user_id, research_job_id)

    def _promote(self, connection: sqlite3.Connection, now: str) -> None:
        active_rows = connection.execute(
            """
            SELECT owner_user_id FROM research_admission
            WHERE state IN ('ADMITTED', 'RUNNING')
            """
        ).fetchall()
        active_users = {str(row["owner_user_id"]) for row in active_rows}
        capacity = self.global_active - len(active_rows)
        if capacity <= 0:
            return
        queued = connection.execute(
            """
            SELECT research_job_id, owner_user_id FROM research_admission
            WHERE state = 'QUEUED' ORDER BY created_at_utc
            """
        ).fetchall()
        for row in queued:
            owner = str(row["owner_user_id"])
            if owner in active_users:
                continue
            connection.execute(
                """
                UPDATE research_admission
                SET state = 'ADMITTED', queue_reason = NULL, updated_at_utc = ?
                WHERE research_job_id = ?
                """,
                (now, str(row["research_job_id"])),
            )
            active_users.add(owner)
            capacity -= 1
            if capacity <= 0:
                break
