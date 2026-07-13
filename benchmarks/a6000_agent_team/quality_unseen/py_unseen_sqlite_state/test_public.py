import sqlite3

from state import record_state


def test_first_state_entry_is_recorded() -> None:
    with sqlite3.connect(":memory:") as connection:
        assert record_state(connection, "job", "created") is True
        assert connection.execute("SELECT value FROM state_entries").fetchone() == ("created",)
