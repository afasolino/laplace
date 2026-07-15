import sqlite3

from batch_store import record_batch


def test_basic_batch_is_idempotent() -> None:
    with sqlite3.connect(":memory:") as connection:
        assert record_batch(connection, [("a", "1"), ("b", "2")]) == 2
        assert record_batch(connection, [("a", "1")]) == 0
