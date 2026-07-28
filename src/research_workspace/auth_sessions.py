"""Opaque browser sessions, credential verification, and authentication audit."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import TypeAlias

from .auth_registry import (
    AuthRegistryError,
    RegisteredUser,
    RegisteredUserRegistry,
    hash_secret,
    normalize_email,
    validate_password,
    verify_secret,
)
from .user_capabilities import CapabilityTier

JsonObject: TypeAlias = dict[str, object]


class AuthSessionError(RuntimeError):
    """Authentication failed without exposing account existence."""

    def __init__(
        self,
        category: str = "authentication_failed",
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class SessionRecord:
    session_hash: str
    user_id: str
    email: str
    display_name: str
    role: str
    capability_tier: CapabilityTier
    default_lane: str
    registry_revision: str
    issued_at: float
    last_activity: float
    idle_expires_at: float
    absolute_expires_at: float

    def public_account(self) -> JsonObject:
        return {
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "capability_tier": self.capability_tier.value,
            "default_lane": self.default_lane,
        }


@dataclass(frozen=True)
class NewSession:
    identifier: str
    csrf_token: str
    record: SessionRecord


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthAuditLog:
    """Private append-only authentication events with no credential material."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        if not self.path.exists():
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        os.chmod(self.path, 0o600)
        self._lock = threading.Lock()

    def append(
        self,
        action: str,
        *,
        outcome: str,
        user_id: str | None = None,
        normalized_email: str | None = None,
        reason: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        value: JsonObject = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "action": action,
            "outcome": outcome,
            "user_id": user_id,
            "email_sha256": (
                hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()
                if normalized_email is not None
                else None
            ),
            "reason": reason,
            "trace_id": trace_id,
        }
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with self._lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class LoginRateLimiter:
    """Per-IP and per-normalized-email exponential login backoff."""

    def __init__(self, *, free_failures: int = 4, maximum_backoff: float = 300.0) -> None:
        self.free_failures = free_failures
        self.maximum_backoff = maximum_backoff
        self._lock = threading.Lock()
        self._failures: dict[str, tuple[int, float]] = {}

    def _keys(self, ip: str, email: str) -> tuple[str, str]:
        return f"ip:{ip}", f"email:{hashlib.sha256(email.encode('utf-8')).hexdigest()}"

    def require_allowed(self, ip: str, email: str) -> None:
        now = time.monotonic()
        with self._lock:
            blocked_until = max(
                (self._failures.get(key, (0, 0.0))[1] for key in self._keys(ip, email)),
                default=0.0,
            )
        if blocked_until > now:
            raise AuthSessionError(
                "authentication_rate_limited",
                retry_after_seconds=max(0.1, blocked_until - now),
            )

    def failure(self, ip: str, email: str) -> float:
        now = time.monotonic()
        backoff = 0.0
        with self._lock:
            for key in self._keys(ip, email):
                count, _ = self._failures.get(key, (0, 0.0))
                count += 1
                key_backoff = (
                    min(self.maximum_backoff, float(2 ** (count - self.free_failures)))
                    if count > self.free_failures
                    else 0.0
                )
                backoff = max(backoff, key_backoff)
                self._failures[key] = (count, now + key_backoff)
        return backoff

    def success(self, ip: str, email: str) -> None:
        with self._lock:
            for key in self._keys(ip, email):
                self._failures.pop(key, None)


class SessionStore:
    """SQLite WAL session store containing only hashes of browser credentials."""

    def __init__(
        self,
        path: Path,
        *,
        idle_timeout_seconds: int = 30 * 60,
        absolute_timeout_seconds: int = 12 * 60 * 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if idle_timeout_seconds < 60 or absolute_timeout_seconds < idle_timeout_seconds:
            raise ValueError("invalid session timeout")
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self.idle_timeout_seconds = idle_timeout_seconds
        self.absolute_timeout_seconds = absolute_timeout_seconds
        self._clock = clock
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_hash TEXT PRIMARY KEY,
                    csrf_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    capability_tier TEXT NOT NULL,
                    default_lane TEXT NOT NULL,
                    registry_revision TEXT NOT NULL,
                    issued_at REAL NOT NULL,
                    last_activity REAL NOT NULL,
                    idle_expires_at REAL NOT NULL,
                    absolute_expires_at REAL NOT NULL,
                    revoked_at REAL
                );
                CREATE INDEX IF NOT EXISTS sessions_user_active
                ON sessions(user_id, revoked_at);
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create(self, user: RegisteredUser, registry_revision: str) -> NewSession:
        identifier = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        session_hash = _hash_token(identifier)
        now = float(self._clock())
        record = SessionRecord(
            session_hash=session_hash,
            user_id=user.user_id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            capability_tier=user.capability_tier,
            default_lane=user.default_lane,
            registry_revision=registry_revision,
            issued_at=now,
            last_activity=now,
            idle_expires_at=now + self.idle_timeout_seconds,
            absolute_expires_at=now + self.absolute_timeout_seconds,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_hash, csrf_hash, user_id, email, display_name, role,
                    capability_tier, default_lane, registry_revision, issued_at,
                    last_activity, idle_expires_at, absolute_expires_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record.session_hash,
                    _hash_token(csrf_token),
                    record.user_id,
                    record.email,
                    record.display_name,
                    record.role,
                    record.capability_tier.value,
                    record.default_lane,
                    record.registry_revision,
                    record.issued_at,
                    record.last_activity,
                    record.idle_expires_at,
                    record.absolute_expires_at,
                ),
            )
        return NewSession(identifier=identifier, csrf_token=csrf_token, record=record)

    @staticmethod
    def _record(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_hash=str(row["session_hash"]),
            user_id=str(row["user_id"]),
            email=str(row["email"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            capability_tier=CapabilityTier(str(row["capability_tier"])),
            default_lane=str(row["default_lane"]),
            registry_revision=str(row["registry_revision"]),
            issued_at=float(row["issued_at"]),
            last_activity=float(row["last_activity"]),
            idle_expires_at=float(row["idle_expires_at"]),
            absolute_expires_at=float(row["absolute_expires_at"]),
        )

    def resolve(self, identifier: str, registry: RegisteredUserRegistry) -> SessionRecord:
        digest = _hash_token(identifier)
        now = float(self._clock())
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_hash = ? AND revoked_at IS NULL",
                (digest,),
            ).fetchone()
            if row is None:
                raise AuthSessionError("authentication_required")
            record = self._record(row)
            if now >= record.idle_expires_at or now >= record.absolute_expires_at:
                connection.execute(
                    "UPDATE sessions SET revoked_at = ? WHERE session_hash = ?",
                    (now, digest),
                )
                raise AuthSessionError("session_expired")
            user = registry.by_user_id(record.user_id)
            if (
                user is None
                or not user.enabled
                or user.normalized_email != normalize_email(record.email)
                or user.role != record.role
                or user.capability_tier is not record.capability_tier
                or user.default_lane != record.default_lane
                or registry.user_revision(user) != record.registry_revision
            ):
                connection.execute(
                    "UPDATE sessions SET revoked_at = ? WHERE session_hash = ?",
                    (now, digest),
                )
                raise AuthSessionError("session_revoked")
            idle_expires = min(
                now + self.idle_timeout_seconds,
                record.absolute_expires_at,
            )
            connection.execute(
                """
                UPDATE sessions SET last_activity = ?, idle_expires_at = ?
                WHERE session_hash = ?
                """,
                (now, idle_expires, digest),
            )
        return replace(record, last_activity=now, idle_expires_at=idle_expires)

    def rotate_csrf(self, identifier: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET csrf_hash = ?
                WHERE session_hash = ? AND revoked_at IS NULL
                """,
                (_hash_token(token), _hash_token(identifier)),
            )
            if cursor.rowcount != 1:
                raise AuthSessionError("authentication_required")
        return token

    def validate_csrf(self, identifier: str, supplied: str | None) -> None:
        if supplied is None:
            raise AuthSessionError("csrf_validation_failed")
        digest = _hash_token(identifier)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT csrf_hash FROM sessions WHERE session_hash = ? AND revoked_at IS NULL",
                (digest,),
            ).fetchone()
        if row is None or not hmac.compare_digest(
            str(row["csrf_hash"]),
            _hash_token(supplied),
        ):
            raise AuthSessionError("csrf_validation_failed")

    def revoke(self, identifier: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET revoked_at = ? WHERE session_hash = ?",
                (float(self._clock()), _hash_token(identifier)),
            )

    def revoke_user(self, user_id: str) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (float(self._clock()), user_id),
            )
            return cursor.rowcount

    def revoke_all(self) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET revoked_at = ? WHERE revoked_at IS NULL",
                (float(self._clock()),),
            )
            return cursor.rowcount

    def active_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM sessions WHERE revoked_at IS NULL"
            ).fetchone()
        return int(row["count"]) if row is not None else 0


class RegisteredEmailAuth:
    """High-level activation, login, session rotation, and revocation service."""

    def __init__(
        self,
        registry: RegisteredUserRegistry,
        sessions: SessionStore,
        audit: AuthAuditLog,
        *,
        rate_limiter: LoginRateLimiter | None = None,
    ) -> None:
        self.registry = registry
        self.sessions = sessions
        self.audit = audit
        self.rate_limiter = rate_limiter or LoginRateLimiter()
        self._dummy_hash = hash_secret(secrets.token_urlsafe(32))

    def login(
        self,
        email: str,
        password: str,
        *,
        client_ip: str,
        trace_id: str,
    ) -> NewSession:
        try:
            normalized = normalize_email(email)
        except AuthRegistryError:
            normalized = hashlib.sha256(email.encode("utf-8", errors="replace")).hexdigest()
        self.rate_limiter.require_allowed(client_ip, normalized)
        user = self.registry.by_email(email)
        encoded = user.password_hash if user is not None else self._dummy_hash
        matched = verify_secret(encoded, password)
        if user is None or not user.enabled or user.must_change_password or not matched:
            backoff = self.rate_limiter.failure(client_ip, normalized)
            self.audit.append(
                "LOGIN",
                outcome="DENIED",
                user_id=user.user_id if user is not None else None,
                normalized_email=normalized,
                reason="generic_authentication_failure",
                trace_id=trace_id,
            )
            raise AuthSessionError(
                "authentication_failed",
                retry_after_seconds=backoff if backoff > 0 else None,
            )
        self.rate_limiter.success(client_ip, normalized)
        session = self.sessions.create(user, self.registry.user_revision(user))
        self.audit.append(
            "LOGIN",
            outcome="SUCCESS",
            user_id=user.user_id,
            normalized_email=normalized,
            trace_id=trace_id,
        )
        return session

    def activate(
        self,
        email: str,
        activation_code: str,
        new_password: str,
        *,
        client_ip: str,
        trace_id: str,
    ) -> NewSession:
        normalized = normalize_email(email)
        self.rate_limiter.require_allowed(client_ip, normalized)
        user = self.registry.by_email(email)
        encoded = user.password_hash if user is not None else self._dummy_hash
        matched = verify_secret(encoded, activation_code)
        if user is None or not user.enabled or not user.must_change_password or not matched:
            backoff = self.rate_limiter.failure(client_ip, normalized)
            self.audit.append(
                "ACTIVATION",
                outcome="DENIED",
                user_id=user.user_id if user is not None else None,
                normalized_email=normalized,
                reason="generic_authentication_failure",
                trace_id=trace_id,
            )
            raise AuthSessionError(
                "authentication_failed",
                retry_after_seconds=backoff if backoff > 0 else None,
            )
        validate_password(new_password)
        new_hash = hash_secret(new_password, password_policy=True)
        self.registry.update_user(
            user.user_id,
            password_hash=new_hash,
            must_change_password=False,
        )
        self.sessions.revoke_user(user.user_id)
        current = self.registry.require_user(user.user_id)
        session = self.sessions.create(current, self.registry.user_revision(current))
        self.rate_limiter.success(client_ip, normalized)
        self.audit.append(
            "ACTIVATION",
            outcome="SUCCESS",
            user_id=user.user_id,
            normalized_email=normalized,
            trace_id=trace_id,
        )
        return session

    def change_password(
        self,
        record: SessionRecord,
        current_password: str,
        new_password: str,
        *,
        trace_id: str,
    ) -> NewSession:
        user = self.registry.require_user(record.user_id)
        if user.must_change_password or not verify_secret(user.password_hash, current_password):
            self.audit.append(
                "PASSWORD_CHANGE",
                outcome="DENIED",
                user_id=user.user_id,
                reason="current_password_invalid",
                trace_id=trace_id,
            )
            raise AuthSessionError("authentication_failed")
        validate_password(new_password)
        self.registry.update_user(
            user.user_id,
            password_hash=hash_secret(new_password, password_policy=True),
            must_change_password=False,
        )
        self.sessions.revoke_user(user.user_id)
        current = self.registry.require_user(user.user_id)
        session = self.sessions.create(current, self.registry.user_revision(current))
        self.audit.append(
            "PASSWORD_CHANGE",
            outcome="SUCCESS",
            user_id=user.user_id,
            trace_id=trace_id,
        )
        return session
