"""Owner-isolated server-side chat conversation persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]


class ConversationError(RuntimeError):
    """Conversation access failed closed."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    owner_user_id: str
    title: str
    archived: bool
    draft: str
    created_at_utc: str
    updated_at_utc: str

    def public(self) -> JsonObject:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "archived": self.archived,
            "draft": self.draft,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


class ConversationStore:
    """SQLite WAL store with owner predicates on every data operation."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    archived INTEGER NOT NULL CHECK(archived IN (0, 1)),
                    draft TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    deleted_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS conversations_owner
                ON conversations(owner_user_id, updated_at_utc);
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    UNIQUE(conversation_id, ordinal),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );
                CREATE TABLE IF NOT EXISTS agent_conversations (
                    agent_session_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL UNIQUE,
                    owner_user_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );
                CREATE INDEX IF NOT EXISTS agent_conversations_owner
                ON agent_conversations(owner_user_id, created_at_utc);
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

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(
            conversation_id=str(row["conversation_id"]),
            owner_user_id=str(row["owner_user_id"]),
            title=str(row["title"]),
            archived=bool(row["archived"]),
            draft=str(row["draft"]),
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )

    def create(self, owner_user_id: str, *, title: str = "New conversation") -> Conversation:
        cleaned = title.strip()[:160] or "New conversation"
        identifier = f"conv-{uuid.uuid4().hex}"
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, owner_user_id, title, archived, draft,
                    created_at_utc, updated_at_utc, deleted_at_utc
                ) VALUES (?, ?, ?, 0, '', ?, ?, NULL)
                """,
                (identifier, owner_user_id, cleaned, now, now),
            )
        return Conversation(identifier, owner_user_id, cleaned, False, "", now, now)

    def require(self, owner_user_id: str, conversation_id: str) -> Conversation:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM conversations
                WHERE conversation_id = ? AND owner_user_id = ? AND deleted_at_utc IS NULL
                """,
                (conversation_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise ConversationError("conversation_not_found")
        return self._conversation(row)

    def list(self, owner_user_id: str, *, include_archived: bool = True) -> list[JsonObject]:
        where = (
            "owner_user_id = ? AND deleted_at_utc IS NULL"
            if include_archived
            else "owner_user_id = ? AND deleted_at_utc IS NULL AND archived = 0"
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM conversations WHERE {where} ORDER BY updated_at_utc DESC",  # nosec B608 - fixed clauses
                (owner_user_id,),
            ).fetchall()
        return [self._conversation(row).public() for row in rows]

    def update(
        self,
        owner_user_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        archived: bool | None = None,
        draft: str | None = None,
    ) -> Conversation:
        current = self.require(owner_user_id, conversation_id)
        new_title = (title.strip()[:160] or current.title) if title is not None else current.title
        new_archived = archived if archived is not None else current.archived
        new_draft = draft[:100_000] if draft is not None else current.draft
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, archived = ?, draft = ?, updated_at_utc = ?
                WHERE conversation_id = ? AND owner_user_id = ? AND deleted_at_utc IS NULL
                """,
                (
                    new_title,
                    int(new_archived),
                    new_draft,
                    now,
                    conversation_id,
                    owner_user_id,
                ),
            )
        return Conversation(
            conversation_id,
            owner_user_id,
            new_title,
            new_archived,
            new_draft,
            current.created_at_utc,
            now,
        )

    def delete(self, owner_user_id: str, conversation_id: str) -> None:
        self.require(owner_user_id, conversation_id)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations SET deleted_at_utc = ?, updated_at_utc = ?, draft = ''
                WHERE conversation_id = ? AND owner_user_id = ?
                """,
                (now, now, conversation_id, owner_user_id),
            )

    def append_message(
        self,
        owner_user_id: str,
        conversation_id: str,
        *,
        role: str,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> JsonObject:
        if role not in {"user", "assistant"} or not content or len(content) > 500_000:
            raise ConversationError("invalid_conversation_message")
        self.require(owner_user_id, conversation_id)
        now = datetime.now(UTC).isoformat()
        message_id = f"msg-{uuid.uuid4().hex}"
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(ordinal), -1) + 1 AS ordinal
                FROM conversation_messages
                WHERE conversation_id = ? AND owner_user_id = ?
                """,
                (conversation_id, owner_user_id),
            ).fetchone()
            ordinal = int(row["ordinal"]) if row is not None else 0
            connection.execute(
                """
                INSERT INTO conversation_messages (
                    message_id, conversation_id, owner_user_id, ordinal, role,
                    content, metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    owner_user_id,
                    ordinal,
                    role,
                    content,
                    json.dumps(dict(metadata or {}), sort_keys=True),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE conversations SET updated_at_utc = ?, draft = ''
                WHERE conversation_id = ? AND owner_user_id = ?
                """,
                (now, conversation_id, owner_user_id),
            )
        return {
            "message_id": message_id,
            "ordinal": ordinal,
            "role": role,
            "content": content,
            "metadata": dict(metadata or {}),
            "created_at_utc": now,
        }

    def get_with_messages(self, owner_user_id: str, conversation_id: str) -> JsonObject:
        conversation = self.require(owner_user_id, conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, ordinal, role, content, metadata_json, created_at_utc
                FROM conversation_messages
                WHERE conversation_id = ? AND owner_user_id = ?
                ORDER BY ordinal
                """,
                (conversation_id, owner_user_id),
            ).fetchall()
        return {
            **conversation.public(),
            "messages": [
                {
                    "message_id": str(row["message_id"]),
                    "ordinal": int(row["ordinal"]),
                    "role": str(row["role"]),
                    "content": str(row["content"]),
                    "metadata": json.loads(str(row["metadata_json"])),
                    "created_at_utc": str(row["created_at_utc"]),
                }
                for row in rows
            ],
        }

    def bind_agent_session(
        self,
        owner_user_id: str,
        agent_session_id: str,
        *,
        repo_id: str,
        title: str,
    ) -> JsonObject:
        """Idempotently bind one owner/repository agent session to durable messages."""

        if not owner_user_id or not agent_session_id or not repo_id:
            raise ConversationError("invalid_agent_conversation_binding")
        cleaned_title = title.strip()[:160] or "New agent session"
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT conversation_id, repo_id, created_at_utc
                FROM agent_conversations
                WHERE agent_session_id = ? AND owner_user_id = ?
                """,
                (agent_session_id, owner_user_id),
            ).fetchone()
            if existing is not None:
                if str(existing["repo_id"]) != repo_id:
                    raise ConversationError("agent_conversation_repository_mismatch")
                return {
                    "agent_session_id": agent_session_id,
                    "conversation_id": str(existing["conversation_id"]),
                    "repo_id": repo_id,
                    "created_at_utc": str(existing["created_at_utc"]),
                }
            collision = connection.execute(
                "SELECT 1 FROM agent_conversations WHERE agent_session_id = ?",
                (agent_session_id,),
            ).fetchone()
            if collision is not None:
                raise ConversationError("agent_conversation_not_found")
            conversation_id = f"conv-{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, owner_user_id, title, archived, draft,
                    created_at_utc, updated_at_utc, deleted_at_utc
                ) VALUES (?, ?, ?, 0, '', ?, ?, NULL)
                """,
                (conversation_id, owner_user_id, cleaned_title, now, now),
            )
            connection.execute(
                """
                INSERT INTO agent_conversations (
                    agent_session_id, conversation_id, owner_user_id, repo_id,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (agent_session_id, conversation_id, owner_user_id, repo_id, now),
            )
        return {
            "agent_session_id": agent_session_id,
            "conversation_id": conversation_id,
            "repo_id": repo_id,
            "created_at_utc": now,
        }

    def _require_agent_binding(
        self, owner_user_id: str, agent_session_id: str
    ) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT agent_session_id, conversation_id, repo_id, created_at_utc
                FROM agent_conversations
                WHERE agent_session_id = ? AND owner_user_id = ?
                """,
                (agent_session_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise ConversationError("agent_conversation_not_found")
        return cast(sqlite3.Row, row)

    def append_agent_message(
        self,
        owner_user_id: str,
        agent_session_id: str,
        *,
        role: str,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> JsonObject:
        binding = self._require_agent_binding(owner_user_id, agent_session_id)
        return self.append_message(
            owner_user_id,
            str(binding["conversation_id"]),
            role=role,
            content=content,
            metadata=metadata,
        )

    def get_agent_conversation(
        self,
        owner_user_id: str,
        agent_session_id: str,
        *,
        limit: int = 200,
    ) -> JsonObject:
        """Return a bounded newest-first window, reordered for display."""

        if not 1 <= limit <= 200:
            raise ConversationError("invalid_agent_conversation_limit")
        binding = self._require_agent_binding(owner_user_id, agent_session_id)
        conversation_id = str(binding["conversation_id"])
        with self._connect() as connection:
            total_row = connection.execute(
                """
                SELECT COUNT(*) AS total FROM conversation_messages
                WHERE conversation_id = ? AND owner_user_id = ?
                """,
                (conversation_id, owner_user_id),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT message_id, ordinal, role, content, metadata_json, created_at_utc
                FROM conversation_messages
                WHERE conversation_id = ? AND owner_user_id = ?
                ORDER BY ordinal DESC LIMIT ?
                """,
                (conversation_id, owner_user_id, limit),
            ).fetchall()
        total = int(total_row["total"]) if total_row is not None else 0
        return {
            "agent_session_id": agent_session_id,
            "conversation_id": conversation_id,
            "repo_id": str(binding["repo_id"]),
            "created_at_utc": str(binding["created_at_utc"]),
            "total_messages": total,
            "truncated": total > len(rows),
            "messages": [
                {
                    "message_id": str(row["message_id"]),
                    "ordinal": int(row["ordinal"]),
                    "role": str(row["role"]),
                    "content": str(row["content"]),
                    "metadata": json.loads(str(row["metadata_json"])),
                    "created_at_utc": str(row["created_at_utc"]),
                }
                for row in reversed(rows)
            ],
        }

    def agent_turn(
        self,
        owner_user_id: str,
        agent_session_id: str,
        turn_id: str,
    ) -> JsonObject | None:
        """Find one durable submitted turn without exposing another owner's data."""

        if not turn_id or len(turn_id) > 160:
            raise ConversationError("invalid_agent_turn_id")
        binding = self._require_agent_binding(owner_user_id, agent_session_id)
        conversation_id = str(binding["conversation_id"])
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, ordinal, role, metadata_json, created_at_utc
                FROM conversation_messages
                WHERE conversation_id=? AND owner_user_id=?
                ORDER BY ordinal
                """,
                (conversation_id, owner_user_id),
            ).fetchall()
        submitted: JsonObject | None = None
        completed: JsonObject | None = None
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"]))
            except json.JSONDecodeError as exc:
                raise ConversationError("agent_conversation_message_invalid") from exc
            if not isinstance(metadata, dict) or metadata.get("turn_id") != turn_id:
                continue
            message = {
                "message_id": str(row["message_id"]),
                "ordinal": int(row["ordinal"]),
                "role": str(row["role"]),
                "created_at_utc": str(row["created_at_utc"]),
                "metadata": metadata,
            }
            if message["role"] == "user" and submitted is None:
                submitted = message
            elif message["role"] == "assistant":
                completed = message
        if submitted is None:
            return None
        return {
            "turn_id": turn_id,
            "submitted_message": submitted,
            "completed_message": completed,
            "status": "COMPLETED" if completed is not None else "SUBMITTED",
        }
