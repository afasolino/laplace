"""Strict external registered-email authentication registry."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import yaml
from argon2 import PasswordHasher, extract_parameters
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type

from .user_capabilities import CapabilityTier

JsonObject: TypeAlias = dict[str, object]

_SCHEMA_VERSION = 1
_ROLES = frozenset({"user", "operator", "auditor", "admin"})
_LANES = frozenset({"quality", "standard", "economy"})
_USER_FIELDS = frozenset(
    {
        "email",
        "user_id",
        "display_name",
        "enabled",
        "capability_tier",
        "role",
        "default_lane",
        "authorized_repo_ids",
        "password_hash",
        "must_change_password",
    }
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EMAIL = re.compile(r"[^@\s]{1,254}@[^@\s]{1,253}")

# OWASP's current minimum Argon2id profile: 19 MiB, t=2, p=1.
PASSWORD_HASHER = PasswordHasher(
    time_cost=2,
    memory_cost=19 * 1024,
    parallelism=1,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)


class AuthRegistryError(RuntimeError):
    """The external registry could not be accepted safely."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise AuthRegistryError("duplicate_registry_key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def normalize_email(value: str) -> str:
    """Normalize only the login comparison form, preserving stored display text."""

    normalized = value.strip().lower()
    if len(normalized) > 320 or not _EMAIL.fullmatch(normalized):
        raise AuthRegistryError("invalid_email")
    return normalized


def validate_password(value: str) -> None:
    """Accept password-manager strings, including 64+ characters, without truncation."""

    if len(value) < 12:
        raise AuthRegistryError("password_too_short")
    if len(value) > 1024 or len(value.encode("utf-8")) > 4096:
        raise AuthRegistryError("password_too_long")
    if "\x00" in value:
        raise AuthRegistryError("password_contains_nul")


def hash_secret(value: str, *, password_policy: bool = False) -> str:
    if password_policy:
        validate_password(value)
    elif not value or len(value) > 1024:
        raise AuthRegistryError("invalid_activation_code")
    return PASSWORD_HASHER.hash(value)


def verify_secret(encoded: str, supplied: str) -> bool:
    try:
        return bool(PASSWORD_HASHER.verify(encoded, supplied))
    except (InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def _valid_argon2id(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("$argon2id$"):
        raise AuthRegistryError("malformed_password_hash")
    try:
        parameters = extract_parameters(value)
    except (InvalidHashError, TypeError, ValueError) as exc:
        raise AuthRegistryError("malformed_password_hash") from exc
    if parameters.type is not Type.ID:
        raise AuthRegistryError("malformed_password_hash")
    if (
        parameters.memory_cost < 19 * 1024
        or parameters.time_cost < 2
        or parameters.parallelism < 1
    ):
        raise AuthRegistryError("weak_password_hash")
    return value


@dataclass(frozen=True)
class RegisteredUser:
    email: str
    user_id: str
    display_name: str
    enabled: bool
    capability_tier: CapabilityTier
    role: str
    default_lane: str
    authorized_repo_ids: tuple[str, ...]
    password_hash: str
    must_change_password: bool

    @property
    def normalized_email(self) -> str:
        return normalize_email(self.email)

    def public(self, *, include_email: bool = True) -> JsonObject:
        value: JsonObject = {
            "display_name": self.display_name,
            "enabled": self.enabled,
            "capability_tier": self.capability_tier.value,
            "role": self.role,
            "default_lane": self.default_lane,
            "authorized_repo_ids": list(self.authorized_repo_ids),
            "must_change_password": self.must_change_password,
        }
        if include_email:
            value["email"] = self.email
        return value

    def registry_value(self) -> JsonObject:
        value = asdict(self)
        value["capability_tier"] = self.capability_tier.value
        value["authorized_repo_ids"] = list(self.authorized_repo_ids)
        return value


@dataclass(frozen=True)
class RegistrySnapshot:
    users_by_email: dict[str, RegisteredUser]
    users_by_id: dict[str, RegisteredUser]
    revision: str
    raw_sha256: str


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise AuthRegistryError("invalid_registry_field", {"field": field})
    return value


def _parse_user(value: object) -> RegisteredUser:
    if not isinstance(value, dict) or set(value) != _USER_FIELDS:
        unknown = sorted(str(item) for item in set(value or {}) - _USER_FIELDS) if isinstance(value, dict) else []
        raise AuthRegistryError("invalid_user_schema", {"unknown_fields": unknown})
    try:
        email = str(value["email"])
        normalize_email(email)
        user_id = str(value["user_id"])
        if not _IDENTIFIER.fullmatch(user_id):
            raise AuthRegistryError("invalid_user_id")
        display_name = str(value["display_name"]).strip()
        if not display_name or len(display_name) > 160:
            raise AuthRegistryError("invalid_display_name")
        capability = CapabilityTier(str(value["capability_tier"]))
        role = str(value["role"])
        lane = str(value["default_lane"])
        if role not in _ROLES:
            raise AuthRegistryError("invalid_role")
        if lane not in _LANES:
            raise AuthRegistryError("invalid_default_lane")
        repositories = value["authorized_repo_ids"]
        if not isinstance(repositories, list) or any(
            not isinstance(item, str) or not _IDENTIFIER.fullmatch(item)
            for item in repositories
        ):
            raise AuthRegistryError("invalid_authorized_repo_ids")
        if len(set(repositories)) != len(repositories):
            raise AuthRegistryError("duplicate_authorized_repo_id")
        return RegisteredUser(
            email=email,
            user_id=user_id,
            display_name=display_name,
            enabled=_strict_bool(value["enabled"], field="enabled"),
            capability_tier=capability,
            role=role,
            default_lane=lane,
            authorized_repo_ids=tuple(repositories),
            password_hash=_valid_argon2id(value["password_hash"]),
            must_change_password=_strict_bool(
                value["must_change_password"], field="must_change_password"
            ),
        )
    except KeyError as exc:
        raise AuthRegistryError("invalid_user_schema") from exc


def parse_registry(raw_bytes: bytes) -> RegistrySnapshot:
    if len(raw_bytes) > 4_000_000:
        raise AuthRegistryError("registry_too_large")
    try:
        raw: object = yaml.load(raw_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AuthRegistryError("malformed_registry") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "users"}:
        raise AuthRegistryError("invalid_registry_schema")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != _SCHEMA_VERSION:
        raise AuthRegistryError("unsupported_registry_schema")
    values = raw["users"]
    if not isinstance(values, list):
        raise AuthRegistryError("invalid_registry_users")
    by_email: dict[str, RegisteredUser] = {}
    by_id: dict[str, RegisteredUser] = {}
    for item in values:
        user = _parse_user(item)
        email = user.normalized_email
        if email in by_email:
            raise AuthRegistryError("duplicate_normalized_email")
        if user.user_id in by_id:
            raise AuthRegistryError("duplicate_user_id")
        by_email[email] = user
        by_id[user.user_id] = user
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return RegistrySnapshot(
        users_by_email=by_email,
        users_by_id=by_id,
        revision=digest[:24],
        raw_sha256=digest,
    )


def require_private_registry_permissions(path: Path) -> None:
    try:
        parent_mode = path.parent.stat().st_mode & 0o777
        file_mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise AuthRegistryError("registry_unavailable", {"error_type": type(exc).__name__}) from exc
    if parent_mode & 0o077:
        raise AuthRegistryError("registry_parent_permissions", {"required": "0700"})
    if file_mode & 0o077 or file_mode & 0o600 != 0o600:
        raise AuthRegistryError("registry_file_permissions", {"required": "0600"})


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def write_registry(path: Path, users: list[RegisteredUser]) -> RegistrySnapshot:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "users": [user.registry_value() for user in sorted(users, key=lambda item: item.normalized_email)],
    }
    raw = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")
    snapshot = parse_registry(raw)
    _atomic_write(path, raw)
    return snapshot


class RegisteredUserRegistry:
    """Atomically reloadable registry that retains its last valid snapshot."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.RLock()
        self._snapshot: RegistrySnapshot | None = None
        self.reload()

    @property
    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            if self._snapshot is None:
                raise AuthRegistryError("registry_not_loaded")
            return self._snapshot

    def reload(self) -> RegistrySnapshot:
        require_private_registry_permissions(self.path)
        raw = self.path.read_bytes()
        parsed = parse_registry(raw)
        with self._lock:
            self._snapshot = parsed
        return parsed

    def try_reload(self) -> tuple[bool, str | None, RegistrySnapshot]:
        try:
            value = self.reload()
            return True, None, value
        except AuthRegistryError as exc:
            return False, exc.category, self.snapshot

    def by_email(self, email: str) -> RegisteredUser | None:
        try:
            normalized = normalize_email(email)
        except AuthRegistryError:
            return None
        return self.snapshot.users_by_email.get(normalized)

    def by_user_id(self, user_id: str) -> RegisteredUser | None:
        return self.snapshot.users_by_id.get(user_id)

    def require_user(self, user_id: str) -> RegisteredUser:
        user = self.by_user_id(user_id)
        if user is None:
            raise AuthRegistryError("unknown_user")
        return user

    @staticmethod
    def user_revision(user: RegisteredUser) -> str:
        """Hash only access-relevant account state so unrelated users stay signed in."""

        encoded = json.dumps(
            user.registry_value(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def mutate(
        self,
        transform: Callable[[dict[str, RegisteredUser]], dict[str, RegisteredUser]],
    ) -> RegistrySnapshot:
        with self._lock:
            current = dict(self.snapshot.users_by_id)
            updated = transform(current)
            snapshot = write_registry(self.path, list(updated.values()))
            self._snapshot = snapshot
            return snapshot

    def upsert(self, user: RegisteredUser) -> RegistrySnapshot:
        def transform(values: dict[str, RegisteredUser]) -> dict[str, RegisteredUser]:
            for current in values.values():
                if (
                    current.normalized_email == user.normalized_email
                    and current.user_id != user.user_id
                ):
                    raise AuthRegistryError("duplicate_normalized_email")
            values[user.user_id] = user
            return values

        return self.mutate(transform)

    def update_user(self, user_id: str, **changes: object) -> RegistrySnapshot:
        def transform(values: dict[str, RegisteredUser]) -> dict[str, RegisteredUser]:
            current = values.get(user_id)
            if current is None:
                raise AuthRegistryError("unknown_user")
            normalized_changes = dict(changes)
            repositories = normalized_changes.get("authorized_repo_ids")
            if isinstance(repositories, tuple):
                normalized_changes["authorized_repo_ids"] = list(repositories)
            values[user_id] = _parse_user(
                {**current.registry_value(), **normalized_changes}
            )
            return values

        return self.mutate(transform)
