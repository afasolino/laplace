"""Persistent, server-authoritative independent capability assignments.

The legacy tier remains a migration/default profile. Authorization decisions must use
the named capability set returned by :meth:`UserCapabilityStore.require_capability`.
"""

from __future__ import annotations

import os
import json
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


class Capability(StrEnum):
    CHAT = "chat"
    AGENT = "agent"
    RESEARCH = "research"
    OPERATOR = "operator"
    ADMIN = "admin"
    PERSONAL_CORPUS = "personal_corpus"
    SHARED_CORPUS_INGEST = "shared_corpus_ingest"
    REPOSITORY_ADMIN = "repository_admin"
    MODEL_ADMIN = "model_admin"


_DEFAULT_CAPABILITIES: dict[CapabilityTier, frozenset[Capability]] = {
    CapabilityTier.BASIC: frozenset({Capability.CHAT}),
    CapabilityTier.PLUS: frozenset(
        {Capability.CHAT, Capability.AGENT, Capability.PERSONAL_CORPUS}
    ),
    # This reproduces the pre-v6 Operator surface. Agent and personal corpus can
    # then be granted independently without removing administrative access.
    CapabilityTier.OPERATOR: frozenset(
        {
            Capability.CHAT,
            Capability.RESEARCH,
            Capability.OPERATOR,
            Capability.ADMIN,
            Capability.SHARED_CORPUS_INGEST,
            Capability.REPOSITORY_ADMIN,
            Capability.MODEL_ADMIN,
        }
    ),
}


def default_capabilities(tier: CapabilityTier) -> frozenset[Capability]:
    """Return the secure legacy-profile defaults for a tier."""

    return _DEFAULT_CAPABILITIES[tier]


class UserCapabilityError(RuntimeError):
    """A user capability check failed closed."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class UserCapability:
    user_id: str
    tier: CapabilityTier
    enabled: bool
    revision: int
    updated_at_utc: str
    capabilities: frozenset[Capability] = frozenset()

    def has(self, capability: Capability) -> bool:
        return self.enabled and capability in self.capabilities

    def public(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "tier": self.tier.value,
            "enabled": self.enabled,
            "revision": self.revision,
            "updated_at_utc": self.updated_at_utc,
            "capabilities": sorted(item.value for item in self.capabilities),
        }


def _valid_identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {label}")
    return value


class UserCapabilityStore:
    """SQLite authority for independent rights with an idempotent v1-to-v2 migration."""

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
                    updated_at_utc TEXT NOT NULL,
                    capabilities_json TEXT
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(user_capabilities)"
                ).fetchall()
            }
            if "capabilities_json" not in columns:
                connection.execute(
                    "ALTER TABLE user_capabilities ADD COLUMN capabilities_json TEXT"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL,
                    migrated_at_utc TEXT NOT NULL
                )
                """
            )
            rows = connection.execute(
                """
                SELECT user_id, tier FROM user_capabilities
                WHERE capabilities_json IS NULL
                """
            ).fetchall()
            for row in rows:
                tier = CapabilityTier(str(row["tier"]))
                connection.execute(
                    """
                    UPDATE user_capabilities SET capabilities_json = ?
                    WHERE user_id = ?
                    """,
                    (
                        _encode_capabilities(default_capabilities(tier)),
                        str(row["user_id"]),
                    ),
                )
            connection.execute(
                """
                INSERT INTO capability_schema (
                    singleton, schema_version, migrated_at_utc
                ) VALUES (1, 2, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    migrated_at_utc=excluded.migrated_at_utc
                """,
                (datetime.now(UTC).isoformat(),),
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
        capabilities: frozenset[Capability] | None = None,
    ) -> UserCapability:
        normalized = _valid_identifier(user_id, label="user_id")
        effective = capabilities if capabilities is not None else default_capabilities(tier)
        _validate_capabilities(effective)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT tier, enabled, revision, capabilities_json
                FROM user_capabilities WHERE user_id = ?
                """,
                (normalized,),
            ).fetchone()
            changed = (
                row is None
                or str(row["tier"]) != tier.value
                or bool(row["enabled"]) is not enabled
                or _decode_capabilities(row["capabilities_json"], tier) != effective
            )
            revision = (
                int(row["revision"]) + 1
                if row is not None and changed
                else (int(row["revision"]) if row is not None else 1)
            )
            connection.execute(
                """
                INSERT INTO user_capabilities (
                    user_id, tier, enabled, revision, updated_at_utc,
                    capabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    tier=excluded.tier,
                    enabled=excluded.enabled,
                    revision=excluded.revision,
                    updated_at_utc=excluded.updated_at_utc,
                    capabilities_json=excluded.capabilities_json
                """,
                (
                    normalized,
                    tier.value,
                    int(enabled),
                    revision,
                    now,
                    _encode_capabilities(effective),
                ),
            )
        return UserCapability(normalized, tier, enabled, revision, now, effective)

    def set_capabilities(
        self,
        user_id: str,
        capabilities: frozenset[Capability],
        *,
        enabled: bool | None = None,
    ) -> UserCapability:
        """Replace a user's independent capability set and increment its revision."""

        current = self.get(user_id)
        return self.set_user(
            current.user_id,
            current.tier,
            enabled=current.enabled if enabled is None else enabled,
            capabilities=capabilities,
        )

    def get(self, user_id: str) -> UserCapability:
        normalized = _valid_identifier(user_id, label="user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, tier, enabled, revision, updated_at_utc,
                       capabilities_json
                FROM user_capabilities WHERE user_id = ?
                """,
                (normalized,),
            ).fetchone()
        if row is None:
            raise UserCapabilityError("unknown_user")
        tier = CapabilityTier(str(row["tier"]))
        return UserCapability(
            user_id=str(row["user_id"]),
            tier=tier,
            enabled=bool(row["enabled"]),
            revision=int(row["revision"]),
            updated_at_utc=str(row["updated_at_utc"]),
            capabilities=_decode_capabilities(row["capabilities_json"], tier),
        )

    def require(self, user_id: str, allowed: frozenset[CapabilityTier]) -> UserCapability:
        capability = self.get(user_id)
        if not capability.enabled:
            raise UserCapabilityError("user_disabled")
        if capability.tier not in allowed:
            raise UserCapabilityError("capability_denied")
        return capability

    def require_capability(
        self, user_id: str, required: Capability
    ) -> UserCapability:
        assignment = self.get(user_id)
        if not assignment.enabled:
            raise UserCapabilityError("user_disabled")
        if required not in assignment.capabilities:
            raise UserCapabilityError("capability_denied")
        return assignment

    def has(self, user_id: str, required: Capability) -> bool:
        try:
            return self.require_capability(user_id, required).enabled
        except (UserCapabilityError, ValueError):
            return False

    def list_users(self) -> list[UserCapability]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id FROM user_capabilities ORDER BY user_id"
            ).fetchall()
        return [self.get(str(row["user_id"])) for row in rows]


def _validate_capabilities(values: frozenset[Capability]) -> None:
    if len(values) > len(Capability) or any(not isinstance(item, Capability) for item in values):
        raise ValueError("invalid capabilities")


def _encode_capabilities(values: frozenset[Capability]) -> str:
    _validate_capabilities(values)
    return json.dumps(
        sorted(item.value for item in values),
        separators=(",", ":"),
    )


def _decode_capabilities(
    raw: object, tier: CapabilityTier
) -> frozenset[Capability]:
    if raw is None:
        return default_capabilities(tier)
    try:
        value: object = json.loads(str(raw))
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError
        parsed = frozenset(Capability(item) for item in value)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise UserCapabilityError("malformed_capability_assignment") from exc
    _validate_capabilities(parsed)
    return parsed
