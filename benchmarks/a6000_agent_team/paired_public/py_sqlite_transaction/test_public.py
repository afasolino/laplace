from __future__ import annotations

import sqlite3

from store import record_transition


def test_first_transition_is_recorded() -> None:
    with sqlite3.connect(":memory:") as connection:
        assert record_transition(connection, "sample", "created") is True
        assert connection.execute("SELECT value FROM transitions WHERE key='sample'").fetchone() == (
            "created",
        )
