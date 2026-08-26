"""Small, durable terminal-session state for ``laplace chat``.

Remote engineering-task state remains authoritative in the resident Operator.
This store contains only terminal preferences, chat transcript, and the remote
agent-session identifier.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_SCHEMA_VERSION = 4
_READABLE_SCHEMA_VERSIONS = frozenset({2, 3, _SCHEMA_VERSION})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_MAX_MESSAGES = 4096
_MAX_MESSAGE_CHARS = 128_000


class ChatSessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    mode: str = "chat"


@dataclass(frozen=True)
class ChatSession:
    session_id: str
    repo_id: str
    repository_root: str
    lane: str
    domain: str
    interaction_mode: str
    access_mode: str
    remote_agent_session_id: str | None
    active_turn_id: str | None
    messages: tuple[StoredMessage, ...]
    created_at_unix: float
    updated_at_unix: float


def _validate_id(value: str, label: str) -> str:
    if _ID_RE.fullmatch(value) is None:
        raise ChatSessionError(f"invalid_{label}")
    return value


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        if not handle.read(1):
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(), msvcrt.LK_LOCK, 1  # type: ignore[attr-defined]
        )
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(), msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
        )
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ChatSessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(self.root, 0o700)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_validate_id(session_id, 'session_id')}.json"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        lock = self.root / ".lock"
        with lock.open("a+b") as handle:
            _lock_file(handle)
            try:
                yield
            finally:
                _unlock_file(handle)

    @staticmethod
    def _parse(payload: object) -> ChatSession:
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") not in _READABLE_SCHEMA_VERSIONS
        ):
            raise ChatSessionError("chat_session_schema_incompatible")
        try:
            raw_messages = payload.get("messages", [])
            if not isinstance(raw_messages, list) or len(raw_messages) > _MAX_MESSAGES:
                raise ChatSessionError("chat_session_messages_invalid")
            messages: list[StoredMessage] = []
            for raw in raw_messages:
                if not isinstance(raw, Mapping):
                    raise ChatSessionError("chat_session_message_invalid")
                role = str(raw.get("role", ""))
                content = str(raw.get("content", ""))
                mode = str(raw.get("mode", "chat"))
                if (
                    role not in {"user", "assistant", "system"}
                    or mode not in {"agent", "chat"}
                    or len(content) > _MAX_MESSAGE_CHARS
                ):
                    raise ChatSessionError("chat_session_message_invalid")
                messages.append(StoredMessage(role=role, content=content, mode=mode))
            session = ChatSession(
                session_id=str(payload["session_id"]),
                repo_id=str(payload["repo_id"]),
                repository_root=str(payload["repository_root"]),
                lane=str(payload["lane"]),
                domain=str(payload["domain"]),
                interaction_mode=str(payload["interaction_mode"]),
                access_mode=str(payload["access_mode"]),
                remote_agent_session_id=(
                    str(payload["remote_agent_session_id"])
                    if payload.get("remote_agent_session_id")
                    else None
                ),
                active_turn_id=(
                    str(payload["active_turn_id"])
                    if payload.get("active_turn_id")
                    else None
                ),
                messages=tuple(messages),
                created_at_unix=float(payload["created_at_unix"]),
                updated_at_unix=float(payload["updated_at_unix"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChatSessionError("chat_session_invalid") from exc
        _validate_id(session.session_id, "session_id")
        _validate_id(session.repo_id, "repo_id")
        if session.remote_agent_session_id:
            _validate_id(session.remote_agent_session_id, "remote_agent_session_id")
        if session.active_turn_id:
            _validate_id(session.active_turn_id, "active_turn_id")
        if session.lane not in {"quality", "standard", "economy"}:
            raise ChatSessionError("chat_session_lane_invalid")
        if _DOMAIN_RE.fullmatch(session.domain) is None:
            raise ChatSessionError("chat_session_domain_invalid")
        if session.interaction_mode not in {"agent", "chat"}:
            raise ChatSessionError("chat_session_interaction_mode_invalid")
        if session.access_mode not in {"read", "confirm", "write"}:
            raise ChatSessionError("chat_session_access_mode_invalid")
        if (
            not math.isfinite(session.created_at_unix)
            or not math.isfinite(session.updated_at_unix)
            or session.created_at_unix <= 0
            or session.updated_at_unix < session.created_at_unix
        ):
            raise ChatSessionError("chat_session_timestamp_invalid")
        return session

    @staticmethod
    def _serialize(session: ChatSession) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_id": session.session_id,
            "repo_id": session.repo_id,
            "repository_root": session.repository_root,
            "lane": session.lane,
            "domain": session.domain,
            "interaction_mode": session.interaction_mode,
            "access_mode": session.access_mode,
            "remote_agent_session_id": session.remote_agent_session_id,
            "active_turn_id": session.active_turn_id,
            "messages": [asdict(message) for message in session.messages],
            "created_at_unix": session.created_at_unix,
            "updated_at_unix": session.updated_at_unix,
        }

    def create(
        self,
        *,
        repo_id: str,
        repository_root: str,
        lane: str,
        domain: str,
        interaction_mode: str,
        access_mode: str,
    ) -> ChatSession:
        _validate_id(repo_id, "repo_id")
        if lane not in {"quality", "standard", "economy"}:
            raise ChatSessionError("invalid_lane")
        if interaction_mode not in {"agent", "chat"}:
            raise ChatSessionError("invalid_interaction_mode")
        if access_mode not in {"read", "confirm", "write"}:
            raise ChatSessionError("invalid_access_mode")
        if _DOMAIN_RE.fullmatch(domain) is None:
            raise ChatSessionError("invalid_domain")
        now = time.time()
        session = ChatSession(
            session_id=f"chat-{uuid.uuid4().hex}",
            repo_id=repo_id,
            repository_root=str(Path(repository_root).resolve()),
            lane=lane,
            domain=domain,
            interaction_mode=interaction_mode,
            access_mode=access_mode,
            remote_agent_session_id=None,
            active_turn_id=None,
            messages=(),
            created_at_unix=now,
            updated_at_unix=now,
        )
        with self._lock():
            _atomic_write(self._path(session.session_id), self._serialize(session))
        return session

    def load(
        self,
        session_id: str,
        *,
        repo_id: str,
        repository_root: str,
    ) -> ChatSession:
        path = self._path(session_id)
        with self._lock():
            try:
                if not path.exists():
                    raise ChatSessionError("chat_session_not_found")
                if path.is_symlink() or not path.is_file():
                    raise ChatSessionError("chat_session_not_regular")
                if os.name != "nt" and path.stat().st_mode & 0o077:
                    raise ChatSessionError("chat_session_permissions_too_open")
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ChatSessionError("chat_session_not_found") from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise ChatSessionError("chat_session_corrupt") from exc
        session = self._parse(payload)
        if session.repo_id != repo_id:
            raise ChatSessionError("chat_session_repository_mismatch")
        if str(Path(session.repository_root).resolve()) != str(Path(repository_root).resolve()):
            raise ChatSessionError("chat_session_repository_root_mismatch")
        return session

    def save(self, session: ChatSession) -> ChatSession:
        updated = ChatSession(
            **{**session.__dict__, "updated_at_unix": time.time()}
        )
        payload = self._serialize(updated)
        validated = self._parse(payload)
        with self._lock():
            _atomic_write(self._path(validated.session_id), payload)
        return validated

    def append_message(
        self,
        session: ChatSession,
        *,
        role: str,
        content: str,
        mode: str = "chat",
    ) -> ChatSession:
        if role not in {"user", "assistant", "system"}:
            raise ChatSessionError("invalid_message_role")
        if mode not in {"agent", "chat"}:
            raise ChatSessionError("invalid_message_mode")
        if len(content) > _MAX_MESSAGE_CHARS:
            raise ChatSessionError("message_too_large_for_terminal_history")
        messages = (*session.messages, StoredMessage(role=role, content=content, mode=mode))
        if len(messages) > _MAX_MESSAGES:
            raise ChatSessionError("chat_history_requires_core_compaction")
        return self.save(ChatSession(**{**session.__dict__, "messages": messages}))

    def compact(self, session: ChatSession, *, keep: int = 80) -> ChatSession:
        """Bound local display history without asking a model to summarize it."""

        if not 4 <= keep <= 512:
            raise ChatSessionError("invalid_compaction_window")
        if len(session.messages) <= keep:
            return session
        removed = len(session.messages) - keep
        marker = StoredMessage(
            role="system",
            content=f"[{removed} earlier local transcript messages compacted]",
            mode=session.interaction_mode,
        )
        messages = (marker, *session.messages[-keep:])
        return self.save(ChatSession(**{**session.__dict__, "messages": messages}))

    def update(self, session: ChatSession, **changes: object) -> ChatSession:
        allowed = {
            "lane",
            "domain",
            "interaction_mode",
            "access_mode",
            "remote_agent_session_id",
            "active_turn_id",
            "messages",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ChatSessionError(f"unsupported_session_update:{','.join(sorted(unknown))}")
        return self.save(ChatSession(**{**session.__dict__, **changes}))

    def last(
        self,
        *,
        repo_id: str,
        repository_root: str,
    ) -> ChatSession | None:
        candidates: list[ChatSession] = []
        with self._lock():
            paths = list(self.root.glob("chat-*.json"))
        for path in paths:
            try:
                if path.is_symlink() or not path.is_file():
                    raise ChatSessionError("chat_session_not_regular")
                if os.name != "nt" and path.stat().st_mode & 0o077:
                    raise ChatSessionError("chat_session_permissions_too_open")
                payload = json.loads(path.read_text(encoding="utf-8"))
                session = self._parse(payload)
            except (OSError, json.JSONDecodeError, ChatSessionError) as exc:
                raise ChatSessionError("chat_session_store_contains_corrupt_state") from exc
            if (
                session.repo_id == repo_id
                and str(Path(session.repository_root).resolve())
                == str(Path(repository_root).resolve())
            ):
                candidates.append(session)
        return max(candidates, key=lambda item: item.updated_at_unix) if candidates else None
