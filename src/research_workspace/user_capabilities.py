"""Persistent, server-authoritative user capability assignments."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class CapabilityTier(StrEnum):
    BASIC = "basic"
    PLUS = "plus"
    OPERATOR = "operator"


class UserCapabilityError(RuntimeError):
    """A user capability check failed closed."""


@dataclass(frozen=True)
class UserCapability:
    user_id: str
    tier: CapabilityTier
    enabled: bool
    revision: int
    updated_at_utc: str


def _valid_identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {label}")
    return value


class UserCapabilityStore:
    """SQLite store used as the only authority for Basic/Plus/Operator rights."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_capabilities (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
                    revision INTEGER NOT NULL,
                    updated_at_utc TEXT NOT NULL
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

    def set_user(
        self,
        user_id: str,
        tier: CapabilityTier,
        *,
        enabled: bool = True,
    ) -> UserCapability:
        normalized = _valid_identifier(user_id, label="user_id")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT revision FROM user_capabilities WHERE user_id = ?",
                (normalized,),
            ).fetchone()
            revision = int(row["revision"]) + 1 if row is not None else 1
            connection.execute(
                """
                INSERT INTO user_capabilities (
                    user_id, tier, enabled, revision, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    tier=excluded.tier,
                    enabled=excluded.enabled,
                    revision=excluded.revision,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (normalized, tier.value, int(enabled), revision, now),
            )
        return UserCapability(normalized, tier, enabled, revision, now)

    def get(self, user_id: str) -> UserCapability:
        normalized = _valid_identifier(user_id, label="user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, tier, enabled, revision, updated_at_utc
                FROM user_capabilities WHERE user_id = ?
                """,
                (normalized,),
            ).fetchone()
        if row is None:
            raise UserCapabilityError("unknown_user")
        return UserCapability(
            user_id=str(row["user_id"]),
            tier=CapabilityTier(str(row["tier"])),
            enabled=bool(row["enabled"]),
            revision=int(row["revision"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )

    def require(self, user_id: str, allowed: frozenset[CapabilityTier]) -> UserCapability:
        capability = self.get(user_id)
        if not capability.enabled:
            raise UserCapabilityError("user_disabled")
        if capability.tier not in allowed:
            raise UserCapabilityError("capability_denied")
        return capability
