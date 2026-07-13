"""Public fixture for an idempotent SQLite state transition."""

from __future__ import annotations

import sqlite3


def record_state(connection: sqlite3.Connection, name: str, value: str) -> bool:
    """Record state and report whether this call wrote a row."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS state_entries (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute("INSERT INTO state_entries(name, value) VALUES (?, ?)", (name, value))
    return True
