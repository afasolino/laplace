"""Public fixture for an idempotent SQLite transition task."""

from __future__ import annotations

import sqlite3


def record_transition(connection: sqlite3.Connection, key: str, value: str) -> bool:
    """Record one immutable transition and return whether it was newly inserted."""
    if not key or not value:
        raise ValueError("key and value must be non-empty")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS transitions (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    # Intentional seeded defect: duplicate calls fail and partial writes are not rolled back.
    connection.execute("INSERT INTO transitions(key, value) VALUES (?, ?)", (key, value))
    return True
