from __future__ import annotations

import sqlite3
from collections.abc import Iterable


def record_batch(connection: sqlite3.Connection, values: Iterable[tuple[str, str]]) -> int:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS batch_entries(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    inserted = 0
    with connection:
        for key, value in values:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO batch_entries(key, value) VALUES(?, ?)", (key, value)
            )
            inserted += cursor.rowcount
    return inserted
