"""Invisible artifact identity and privacy-preserving provenance helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path, PurePosixPath

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(*, timestamp_ms: int | None = None) -> str:
    """Return a monotonic-sortable 128-bit ULID without a third-party runtime."""

    current = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    if current < 0 or current >= 2**48:
        raise ValueError("ULID timestamp is outside the 48-bit range")
    value = (current << 80) | int.from_bytes(secrets.token_bytes(10), "big")
    encoded = ["0"] * 26
    for index in range(25, -1, -1):
        encoded[index] = _CROCKFORD[value & 31]
        value >>= 5
    return "".join(encoded)


def private_hmac_key(path: Path) -> bytes:
    """Load or atomically create the external owner-pseudonym key."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(resolved.parent, 0o700)
    if not resolved.exists():
        temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(secrets.token_bytes(32))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, resolved)
        finally:
            temporary.unlink(missing_ok=True)
    if os.name != "nt" and resolved.stat().st_mode & 0o077:
        raise PermissionError("artifact pseudonym key must have mode 0600")
    value = resolved.read_bytes()
    if len(value) != 32:
        raise ValueError("artifact pseudonym key has an invalid length")
    return value


def pseudonymous_owner_id(key: bytes, internal_user_id: str) -> str:
    digest = hmac.new(key, internal_user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:32]


def clean_relative_path(value: str) -> PurePosixPath:
    """Validate a clean user-visible relative artifact path."""

    if "\x00" in value or "\\" in value:
        raise ValueError("invalid artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ValueError("invalid artifact path")
    if any(
        not part
        or part.startswith(".laplace")
        or any(ord(character) < 32 for character in part)
        or any(character in {'"', "'", ";"} for character in part)
        for part in path.parts
    ):
        raise ValueError("invalid artifact path")
    return path
