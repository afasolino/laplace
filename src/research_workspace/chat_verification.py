"""Session verifier persistence for ``laplace chat``.

The remote repository agent remains authoritative for verifier admission and
execution. The local store remembers the latest caller-selected verifier for
resume convenience; verifier replacement is allowed during development and
remote candidate-assurance state remains authoritative for verification claims.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SCHEMA_VERSION = 1
_MAX_ARGS = 64
_MAX_ARG_CHARS = 4096
_MAX_TOTAL_CHARS = 32768


class ChatVerificationError(RuntimeError):
    pass


def _normalize(argv: Sequence[str] | None) -> tuple[str, ...] | None:
    if argv is None:
        return None
    if isinstance(argv, (str, bytes)) or not 1 <= len(argv) <= _MAX_ARGS:
        raise ChatVerificationError("verification_argv_invalid")
    values: list[str] = []
    total = 0
    for item in argv:
        if not isinstance(item, str) or not item or "\x00" in item or len(item) > _MAX_ARG_CHARS:
            raise ChatVerificationError("verification_argv_invalid")
        total += len(item)
        if total > _MAX_TOTAL_CHARS:
            raise ChatVerificationError("verification_argv_invalid")
        values.append(item)
    return tuple(values)


class ChatVerificationStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def _path(self, session_id: str) -> Path:
        if _ID_RE.fullmatch(session_id) is None:
            raise ChatVerificationError("verification_session_id_invalid")
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> tuple[str, ...] | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChatVerificationError("verification_contract_corrupt") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
            raise ChatVerificationError("verification_contract_corrupt")
        if raw.get("session_id") != session_id:
            raise ChatVerificationError("verification_contract_corrupt")
        argv = raw.get("argv")
        if not isinstance(argv, list):
            raise ChatVerificationError("verification_contract_corrupt")
        try:
            return _normalize(argv)
        except ChatVerificationError as exc:
            raise ChatVerificationError("verification_contract_corrupt") from exc

    def save(self, session_id: str, argv: Sequence[str]) -> tuple[str, ...]:
        normalized = _normalize(argv)
        assert normalized is not None
        path = self._path(session_id)
        payload = json.dumps(
            {"schema_version": _SCHEMA_VERSION, "session_id": session_id, "argv": list(normalized)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        except (OSError, ValueError):
            try:
                temporary.unlink(missing_ok=True)
            finally:
                raise
        return normalized


def resolve_verification(
    store: ChatVerificationStore,
    session_id: str,
    supplied: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Resolve resume semantics without weakening the remote fail-closed check."""

    explicit = _normalize(supplied)
    persisted = store.load(session_id)
    if persisted is None:
        if explicit is not None:
            store.save(session_id, explicit)
        return explicit
    if explicit is None:
        return persisted
    if explicit != persisted:
        store.save(session_id, explicit)
    return explicit
