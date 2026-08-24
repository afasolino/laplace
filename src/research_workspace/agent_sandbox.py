"""Plus agent sessions bound to dedicated, server-created Git worktrees."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess  # nosec B404
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, TypeAlias

from .repository_authorization import (
    RepositoryAuthorizationError,
    RepositoryAuthorizationStore,
    validate_workspace_path,
)

JsonObject: TypeAlias = dict[str, object]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class AgentSandboxError(RuntimeError):
    """An agent session could not be confined to its authorized worktree."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class AgentToolPolicy:
    policy_id: str
    allowed_tools: tuple[str, ...]
    network_enabled: bool = False
    max_commands: int = 100
    max_wall_seconds: int = 1_800

    def __post_init__(self) -> None:
        if self.network_enabled:
            raise ValueError("local Plus agent policies cannot enable network")
        if not self.allowed_tools or self.max_commands < 1 or self.max_wall_seconds < 1:
            raise ValueError("invalid agent tool policy")


@dataclass(frozen=True)
class AgentSessionBinding:
    session_id: str
    user_id: str
    repo_id: str
    canonical_repository_root: str
    worktree_root: str
    base_revision: str
    grant_revision: int
    tool_policy: AgentToolPolicy
    environment: Mapping[str, str]
    created_at_utc: str

    def to_json(self) -> JsonObject:
        value = asdict(self)
        value["tool_policy"] = asdict(self.tool_policy)
        value["environment"] = dict(self.environment)
        return value


def _session_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("invalid session_id")
    return value


class AgentSandboxManager:
    """Persistent, quota-bounded worktree lifecycle without client-supplied roots."""

    def __init__(
        self,
        sandbox_root: Path,
        authorizations: RepositoryAuthorizationStore,
        *,
        runner: CommandRunner = subprocess.run,
        environment_allowlist: frozenset[str] = frozenset(
            {"LANG", "LC_ALL", "PATH", "PYTHONUTF8", "TZ"}
        ),
        per_user_quota: int = 8,
        global_quota: int = 64,
        retention_days: int = 30,
    ) -> None:
        self.sandbox_root = sandbox_root.resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.sandbox_root, 0o700)
        self.authorizations = authorizations
        self._runner = runner
        self.environment_allowlist = environment_allowlist
        if per_user_quota < 1 or global_quota < per_user_quota or retention_days < 1:
            raise ValueError("invalid worktree quotas")
        self.per_user_quota = per_user_quota
        self.global_quota = global_quota
        self.retention_days = retention_days
        self.database = self.sandbox_root / "worktree_sessions.sqlite3"
        self._lock = threading.RLock()
        self._sessions: dict[str, AgentSessionBinding] = {}
        self._initialize()
        self._recover()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS worktree_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version INTEGER NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worktree_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    canonical_repository_root TEXT NOT NULL,
                    worktree_root TEXT NOT NULL UNIQUE,
                    base_revision TEXT NOT NULL,
                    grant_revision INTEGER NOT NULL,
                    tool_policy_json TEXT NOT NULL,
                    environment_json TEXT NOT NULL,
                    task_title TEXT NOT NULL,
                    instruction_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    lane TEXT,
                    sanitized_model_name TEXT,
                    command_count INTEGER NOT NULL DEFAULT 0,
                    changed_paths_json TEXT NOT NULL DEFAULT '[]',
                    diff_hash TEXT,
                    verification_summary TEXT,
                    export_state TEXT NOT NULL DEFAULT 'NONE',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    retention_expires_at_utc TEXT,
                    idempotency_key TEXT,
                    UNIQUE(user_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS worktree_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT INTO worktree_schema (singleton, schema_version, updated_at_utc)
                VALUES (1, 1, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (datetime.now(UTC).isoformat(),),
            )
        os.chmod(self.database, 0o600)

    def _recover(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worktree_sessions
                WHERE state IN (
                    'ACTIVE', 'RUNNING', 'DIRTY', 'FAILED', 'CANCELLED_DIRTY',
                    'STALE_GRANT'
                )
                """
            ).fetchall()
        for row in rows:
            root = Path(str(row["worktree_root"]))
            if not root.is_dir():
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE worktree_sessions
                        SET state='UNAVAILABLE', updated_at_utc=?
                        WHERE session_id=?
                        """,
                        (datetime.now(UTC).isoformat(), str(row["session_id"])),
                    )
                continue
            try:
                policy_value: object = json.loads(str(row["tool_policy_json"]))
                environment_value: object = json.loads(str(row["environment_json"]))
                if not isinstance(policy_value, dict) or not isinstance(environment_value, dict):
                    raise ValueError
                policy = AgentToolPolicy(
                    policy_id=str(policy_value["policy_id"]),
                    allowed_tools=tuple(str(item) for item in policy_value["allowed_tools"]),
                    network_enabled=bool(policy_value["network_enabled"]),
                    max_commands=int(policy_value["max_commands"]),
                    max_wall_seconds=int(policy_value["max_wall_seconds"]),
                )
                binding = AgentSessionBinding(
                    session_id=str(row["session_id"]),
                    user_id=str(row["user_id"]),
                    repo_id=str(row["repo_id"]),
                    canonical_repository_root=str(row["canonical_repository_root"]),
                    worktree_root=str(row["worktree_root"]),
                    base_revision=str(row["base_revision"]),
                    grant_revision=int(row["grant_revision"]),
                    tool_policy=policy,
                    environment={str(key): str(value) for key, value in environment_value.items()},
                    created_at_utc=str(row["created_at_utc"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self._sessions[binding.session_id] = binding

    def create(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        tool_policy: AgentToolPolicy,
        environment: Mapping[str, str] | None = None,
        task_title: str = "New Agent task",
        instruction_digest: str = "",
        lane: str | None = None,
        sanitized_model_name: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentSessionBinding:
        identifier = _session_identifier(session_id)
        title = task_title.strip()
        if not title or len(title) > 200:
            raise AgentSandboxError("invalid_task_title")
        if instruction_digest and not re.fullmatch(r"[a-f0-9]{64}", instruction_digest):
            raise AgentSandboxError("invalid_instruction_digest")
        if idempotency_key is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}", idempotency_key
        ):
            raise AgentSandboxError("invalid_idempotency_key")
        supplied_environment = dict(environment or {})
        forbidden = sorted(set(supplied_environment) - self.environment_allowlist)
        if forbidden:
            raise AgentSandboxError("environment_not_allowed", {"variables": forbidden})
        with self._lock:
            if idempotency_key is not None:
                with self._connect() as connection:
                    existing = connection.execute(
                        """
                        SELECT session_id FROM worktree_sessions
                        WHERE user_id=? AND idempotency_key=?
                        """,
                        (user_id, idempotency_key),
                    ).fetchone()
                if existing is not None:
                    return self.require_active(str(existing["session_id"]), user_id=user_id)
            if identifier in self._sessions:
                raise AgentSandboxError("session_exists", {"session_id": identifier})
            self._require_quota(user_id)
        grant = self.authorizations.require_grant(user_id, repo_id)
        target = self.sandbox_root / user_id / identifier
        if target.exists():
            raise AgentSandboxError("worktree_exists", {"path": str(target)})
        target.parent.mkdir(parents=True, exist_ok=True)
        result = self._runner(
            [
                "git",
                "-C",
                str(grant.repository.canonical_root),
                "worktree",
                "add",
                "--detach",
                str(target),
                grant.base_revision,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            raise AgentSandboxError(
                "worktree_creation_failed",
                {"returncode": result.returncode, "stderr": result.stderr[-2_000:]},
            )
        head = self._runner(
            ["git", "-C", str(target), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if (
            head.returncode != 0
            or head.stdout.strip()
            and head.stdout.strip() != grant.base_revision
        ):
            self._runner(
                [
                    "git",
                    "-C",
                    str(grant.repository.canonical_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(target),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            raise AgentSandboxError("base_revision_race")
        binding = AgentSessionBinding(
            session_id=identifier,
            user_id=user_id,
            repo_id=repo_id,
            canonical_repository_root=str(grant.repository.canonical_root),
            worktree_root=str(target.resolve(strict=True)),
            base_revision=grant.base_revision,
            grant_revision=grant.revision,
            tool_policy=tool_policy,
            environment=supplied_environment,
            created_at_utc=datetime.now(UTC).isoformat(),
        )
        now = binding.created_at_utc
        retention_expires = (
            datetime.fromisoformat(now) + timedelta(days=self.retention_days)
        ).isoformat()
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO worktree_sessions (
                        session_id, user_id, repo_id, canonical_repository_root,
                        worktree_root, base_revision, grant_revision,
                        tool_policy_json, environment_json, task_title,
                        instruction_digest, state, lane, sanitized_model_name,
                        created_at_utc, updated_at_utc, retention_expires_at_utc,
                        idempotency_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        user_id,
                        repo_id,
                        binding.canonical_repository_root,
                        binding.worktree_root,
                        binding.base_revision,
                        binding.grant_revision,
                        json.dumps(asdict(tool_policy), sort_keys=True),
                        json.dumps(supplied_environment, sort_keys=True),
                        title,
                        instruction_digest,
                        lane,
                        sanitized_model_name,
                        now,
                        now,
                        retention_expires,
                        idempotency_key,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise AgentSandboxError("session_exists") from exc
            self._event(connection, binding, "CREATED", "ACTIVE", {})
            self._sessions[identifier] = binding
        return binding

    def _require_quota(self, user_id: str) -> None:
        active_states = (
            "ACTIVE",
            "RUNNING",
            "DIRTY",
            "FAILED",
            "CANCELLED_DIRTY",
            "STALE_GRANT",
        )
        with self._connect() as connection:
            global_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM worktree_sessions
                    WHERE state IN (?, ?, ?, ?, ?, ?)
                    """,
                    active_states,
                ).fetchone()["count"]
            )
            user_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM worktree_sessions
                    WHERE user_id=? AND state IN (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, *active_states),
                ).fetchone()["count"]
            )
        if user_count >= self.per_user_quota:
            raise AgentSandboxError(
                "per_user_worktree_quota",
                {"limit": self.per_user_quota, "active": user_count},
            )
        if global_count >= self.global_quota:
            raise AgentSandboxError(
                "global_worktree_quota",
                {"limit": self.global_quota, "active": global_count},
            )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        binding: AgentSessionBinding,
        event: str,
        state: str,
        details: JsonObject,
    ) -> None:
        connection.execute(
            """
            INSERT INTO worktree_events (
                session_id, user_id, event, state, timestamp_utc, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                binding.session_id,
                binding.user_id,
                event,
                state,
                datetime.now(UTC).isoformat(),
                json.dumps(details, sort_keys=True, separators=(",", ":")),
            ),
        )

    def require_active(self, session_id: str, *, user_id: str) -> AgentSessionBinding:
        identifier = _session_identifier(session_id)
        binding = self._sessions.get(identifier)
        if binding is None or binding.user_id != user_id:
            raise AgentSandboxError("unknown_agent_session")
        try:
            current = self.authorizations.require_grant(binding.user_id, binding.repo_id)
        except RepositoryAuthorizationError as exc:
            self._set_state(binding, "STALE_GRANT", "GRANT_REVOKED", exc.evidence)
            raise AgentSandboxError("repository_authorization_revoked", exc.evidence) from exc
        if current.revision != binding.grant_revision:
            self._set_state(
                binding,
                "STALE_GRANT",
                "GRANT_CHANGED",
                {
                    "session_revision": binding.grant_revision,
                    "current_revision": current.revision,
                },
            )
            raise AgentSandboxError(
                "repository_authorization_changed",
                {
                    "session_revision": binding.grant_revision,
                    "current_revision": current.revision,
                },
            )
        root = Path(binding.worktree_root)
        if not root.is_dir():
            self._set_state(binding, "UNAVAILABLE", "WORKTREE_UNAVAILABLE", {})
            raise AgentSandboxError("worktree_unavailable")
        return binding

    def _set_state(
        self,
        binding: AgentSessionBinding,
        state: str,
        event: str,
        details: JsonObject,
        *,
        completed: bool = False,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state=?, updated_at_utc=?,
                    completed_at_utc=CASE WHEN ? THEN ? ELSE completed_at_utc END
                WHERE session_id=? AND user_id=?
                """,
                (
                    state,
                    now,
                    int(completed),
                    now,
                    binding.session_id,
                    binding.user_id,
                ),
            )
            self._event(connection, binding, event, state, details)

    def validate_path(self, session_id: str, *, user_id: str, relative_path: str) -> Path:
        binding = self.require_active(session_id, user_id=user_id)
        try:
            return validate_workspace_path(Path(binding.worktree_root), relative_path)
        except RepositoryAuthorizationError as exc:
            raise AgentSandboxError(exc.category, exc.evidence) from exc

    def close_if_clean(self, session_id: str, *, user_id: str) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        root = Path(binding.worktree_root)
        status = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0 or status.stdout.strip():
            changed_paths, diff_hash = self._diff_summary(binding)
            self._update_diff(binding, changed_paths, diff_hash)
            self._set_state(
                binding,
                "DIRTY",
                "CLOSE_PRESERVED_DIRTY",
                {"changed_paths": list(changed_paths), "diff_hash": diff_hash},
            )
            return {
                "status": "PRESERVED_DIRTY_WORKTREE",
                "session_id": session_id,
                "changed_paths": list(changed_paths),
                "diff_hash": diff_hash,
            }
        remove = self._runner(
            [
                "git",
                "-C",
                binding.canonical_repository_root,
                "worktree",
                "remove",
                str(root),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if remove.returncode != 0:
            raise AgentSandboxError(
                "clean_worktree_release_failed",
                {"returncode": remove.returncode, "stderr": remove.stderr[-2_000:]},
            )
        self._set_state(binding, "CLOSED_CLEAN", "CLOSED_CLEAN", {}, completed=True)
        self._sessions.pop(session_id, None)
        return {"status": "RELEASED_CLEAN_WORKTREE", "session_id": session_id}

    def status(self, session_id: str, *, user_id: str) -> JsonObject:
        """Revalidate a binding and report its owned worktree state."""

        binding = self.require_active(session_id, user_id=user_id)
        root = Path(binding.worktree_root)
        status = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0:
            raise AgentSandboxError(
                "worktree_status_failed",
                {"returncode": status.returncode, "stderr": status.stderr[-2_000:]},
            )
        dirty = bool(status.stdout.strip())
        changed_paths, diff_hash = self._diff_summary(binding)
        self._update_diff(binding, changed_paths, diff_hash)
        state = "DIRTY" if dirty else "ACTIVE"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM worktree_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is not None and str(row["state"]) in {"RUNNING", "FAILED", "CANCELLED_DIRTY"}:
            state = str(row["state"])
        return {
            "status": "DIRTY" if dirty else "ACTIVE_CLEAN",
            "lifecycle_state": state,
            "session_id": session_id,
            "repo_id": binding.repo_id,
            "base_revision": binding.base_revision,
            "grant_revision": binding.grant_revision,
            "changed_paths": list(changed_paths),
            "diff_hash": diff_hash,
        }

    def _diff_summary(self, binding: AgentSessionBinding) -> tuple[tuple[str, ...], str | None]:
        root = Path(binding.worktree_root)
        names = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        paths: list[str] = []
        if names.returncode == 0:
            for line in names.stdout.splitlines():
                path = line[3:] if len(line) > 3 else ""
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                if path:
                    paths.append(path)
        diff = self._runner(
            ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        untracked = self._runner(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        material = diff.stdout
        if untracked.returncode == 0:
            for path in sorted(untracked.stdout.splitlines()):
                candidate = root / path
                try:
                    content = candidate.read_bytes()
                except OSError:
                    continue
                material += f"\nuntracked:{path}:{hashlib.sha256(content).hexdigest()}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest() if material else None
        return tuple(sorted(set(paths))), digest

    def _update_diff(
        self,
        binding: AgentSessionBinding,
        changed_paths: tuple[str, ...],
        diff_hash: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET changed_paths_json=?, diff_hash=?, updated_at_utc=?
                WHERE session_id=? AND user_id=?
                """,
                (
                    json.dumps(list(changed_paths), separators=(",", ":")),
                    diff_hash,
                    datetime.now(UTC).isoformat(),
                    binding.session_id,
                    binding.user_id,
                ),
            )

    def list_mine(self, user_id: str) -> list[JsonObject]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worktree_sessions
                WHERE user_id=? ORDER BY created_at_utc DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._public_record(row, operator=False) for row in rows]

    def operator_inventory(self) -> list[JsonObject]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worktree_sessions ORDER BY created_at_utc DESC
                """
            ).fetchall()
        return [self._public_record(row, operator=True) for row in rows]

    def inspect(self, session_id: str, *, user_id: str, operator: bool = False) -> JsonObject:
        identifier = _session_identifier(session_id)
        with self._connect() as connection:
            if operator:
                row = connection.execute(
                    "SELECT * FROM worktree_sessions WHERE session_id=?",
                    (identifier,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM worktree_sessions
                    WHERE session_id=? AND user_id=?
                    """,
                    (identifier, user_id),
                ).fetchone()
        if row is None:
            raise AgentSandboxError("unknown_agent_session")
        return self._public_record(row, operator=operator)

    @staticmethod
    def _public_record(row: sqlite3.Row, *, operator: bool) -> JsonObject:
        value: JsonObject = {
            "session_id": str(row["session_id"]),
            "user_id": str(row["user_id"]) if operator else "self",
            "repo_id": str(row["repo_id"]),
            "base_revision": str(row["base_revision"]),
            "grant_revision": int(row["grant_revision"]),
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
            "completed_at_utc": row["completed_at_utc"],
            "task_title": str(row["task_title"]),
            "instruction_digest": str(row["instruction_digest"]),
            "state": str(row["state"]),
            "lane": row["lane"],
            "model_name": row["sanitized_model_name"],
            "tool_policy": json.loads(str(row["tool_policy_json"])),
            "command_count": int(row["command_count"]),
            "changed_paths": json.loads(str(row["changed_paths_json"])),
            "diff_hash": row["diff_hash"],
            "verification_summary": row["verification_summary"],
            "export_state": str(row["export_state"]),
            "retention_expires_at_utc": row["retention_expires_at_utc"],
            "network_enabled": False,
        }
        if operator:
            value["canonical_repository_root"] = str(row["canonical_repository_root"])
            value["worktree_root"] = str(row["worktree_root"])
        return value

    def start_task(
        self,
        session_id: str,
        *,
        user_id: str,
        lane: str,
        sanitized_model_name: str,
        instruction_digest: str | None = None,
    ) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        if instruction_digest is not None and not re.fullmatch(r"[a-f0-9]{64}", instruction_digest):
            raise AgentSandboxError("invalid_instruction_digest")
        self._set_state(
            binding,
            "RUNNING",
            "TASK_STARTED",
            {"lane": lane, "model_name": sanitized_model_name},
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions SET lane=?, sanitized_model_name=?
                    , instruction_digest=COALESCE(?, instruction_digest)
                WHERE session_id=? AND user_id=?
                """,
                (
                    lane,
                    sanitized_model_name,
                    instruction_digest,
                    session_id,
                    user_id,
                ),
            )
        return self.inspect(session_id, user_id=user_id)

    def record_result(
        self,
        session_id: str,
        *,
        user_id: str,
        command_count: int,
        verification_summary: str,
        failed: bool = False,
    ) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        changed_paths, diff_hash = self._diff_summary(binding)
        state = "FAILED" if failed else ("DIRTY" if changed_paths else "ACTIVE")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state=?, command_count=?, changed_paths_json=?, diff_hash=?,
                    verification_summary=?, updated_at_utc=?
                WHERE session_id=? AND user_id=?
                """,
                (
                    state,
                    command_count,
                    json.dumps(list(changed_paths), separators=(",", ":")),
                    diff_hash,
                    verification_summary[:2_000],
                    datetime.now(UTC).isoformat(),
                    session_id,
                    user_id,
                ),
            )
            self._event(
                connection,
                binding,
                "TASK_FAILED" if failed else "TASK_COMPLETED",
                state,
                {"changed_paths": list(changed_paths), "diff_hash": diff_hash},
            )
        return self.inspect(session_id, user_id=user_id)

    def resume(self, session_id: str, *, user_id: str) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        self._set_state(binding, "ACTIVE", "RESUMED", {})
        return self.inspect(session_id, user_id=user_id)

    def cancel(self, session_id: str, *, user_id: str) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        result = self.close_if_clean(session_id, user_id=user_id)
        if result["status"] == "PRESERVED_DIRTY_WORKTREE":
            self._set_state(binding, "CANCELLED_DIRTY", "CANCELLED_DIRTY", {})
        return result

    def discard(
        self,
        session_id: str,
        *,
        user_id: str,
        confirmation: str,
        operator: bool = False,
    ) -> JsonObject:
        if confirmation != f"discard:{session_id}":
            raise AgentSandboxError("discard_confirmation_required")
        record = self.inspect(session_id, user_id=user_id, operator=operator)
        owner = str(record["user_id"]) if operator else user_id
        if operator:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM worktree_sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
            if row is None:
                raise AgentSandboxError("unknown_agent_session")
            root = str(row["worktree_root"])
            repository = str(row["canonical_repository_root"])
        else:
            binding = self._sessions.get(session_id)
            if binding is None or binding.user_id != user_id:
                raise AgentSandboxError("unknown_agent_session")
            root = binding.worktree_root
            repository = binding.canonical_repository_root
        result = self._runner(
            ["git", "-C", repository, "worktree", "remove", "--force", root],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            raise AgentSandboxError(
                "worktree_discard_failed",
                {"returncode": result.returncode, "stderr": result.stderr[-2_000:]},
            )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state='DISCARDED', completed_at_utc=?, updated_at_utc=?
                WHERE session_id=?
                """,
                (datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat(), session_id),
            )
            binding = self._sessions.get(session_id)
            if binding is not None:
                self._event(connection, binding, "DISCARDED", "DISCARDED", {})
        self._sessions.pop(session_id, None)
        return {
            "status": "DISCARDED",
            "session_id": session_id,
            "owner": owner if operator else "self",
        }

    def request_export(
        self, session_id: str, *, user_id: str, promotion: bool = False
    ) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        state = "PROMOTION_REQUESTED" if promotion else "EXPORT_REQUESTED"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions SET export_state=?, updated_at_utc=?
                WHERE session_id=? AND user_id=?
                """,
                (state, datetime.now(UTC).isoformat(), session_id, user_id),
            )
            self._event(connection, binding, state, state, {})
        return {"status": state, "session_id": session_id}

    def patch(self, session_id: str, *, user_id: str) -> bytes:
        binding = self.require_active(session_id, user_id=user_id)
        result = self._runner(
            [
                "git",
                "-C",
                binding.worktree_root,
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise AgentSandboxError("patch_export_failed")
        patch = result.stdout
        untracked = self._runner(
            [
                "git",
                "-C",
                binding.worktree_root,
                "ls-files",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if untracked.returncode != 0:
            raise AgentSandboxError("patch_export_failed")
        for relative in sorted(untracked.stdout.splitlines()):
            validate_workspace_path(Path(binding.worktree_root), relative)
            addition = self._runner(
                [
                    "git",
                    "-C",
                    binding.worktree_root,
                    "diff",
                    "--no-index",
                    "--binary",
                    "--",
                    "/dev/null",
                    relative,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if addition.returncode not in {0, 1}:
                raise AgentSandboxError("patch_export_failed")
            patch += addition.stdout
        return patch.encode("utf-8")

    def history(self, session_id: str, *, user_id: str) -> list[JsonObject]:
        self.inspect(session_id, user_id=user_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event, state, timestamp_utc, details_json
                FROM worktree_events
                WHERE session_id=? AND user_id=? ORDER BY sequence
                """,
                (session_id, user_id),
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "event": str(row["event"]),
                "state": str(row["state"]),
                "timestamp_utc": str(row["timestamp_utc"]),
                "details": json.loads(str(row["details_json"])),
            }
            for row in rows
        ]

    @staticmethod
    def fixed_environment(binding: AgentSessionBinding) -> dict[str, str]:
        operator_bin = str(Path(sys.executable).parent)
        base = {
            "PATH": f"{operator_bin}:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
        }
        base.update(binding.environment)
        base["LAPLACE_AGENT_SESSION"] = binding.session_id
        base["LAPLACE_REPOSITORY_ID"] = binding.repo_id
        return base
