"""Plus agent sessions bound to dedicated, server-created Git worktrees."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess  # nosec B404
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, TypeAlias, cast

from .repository_authorization import (
    RepositoryAuthorizationError,
    RepositoryAuthorizationStore,
    validate_workspace_path,
)
from .task_labels import derive_task_label

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


_ACTIVE_QUOTA_STATES = (
    "ACTIVE",
    "RUNNING",
    "DIRTY",
    "FAILED",
    "INTERRUPTED_RESUMABLE",
    "STALE_DIRTY",
)
_GC_TERMINAL_STATES = (
    "SUCCEEDED",
    "FAILED_TERMINAL",
    "SUCCEEDED_LEGACY",
    "FAILED_TERMINAL_LEGACY",
    "CANCELLED_DIRTY",
    "STALE_GRANT",
    "CLEANUP_PENDING",
)


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
        recover_cleanup: bool = True,
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
        self.recover_cleanup = recover_cleanup
        self.database = self.sandbox_root / "worktree_sessions.sqlite3"
        self.ownership_root = self.sandbox_root / ".ownership"
        self.ownership_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.ownership_root, 0o700)
        self._lock = threading.RLock()
        self._sessions: dict[str, AgentSessionBinding] = {}
        self._live_executors: set[str] = set()
        self._capacity_notifiers: list[Callable[[], None]] = []
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
                    current_task_label TEXT,
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
                    ownership_token TEXT,
                    cleanup_eligible INTEGER NOT NULL DEFAULT 0 CHECK(cleanup_eligible IN (0, 1)),
                    result_id TEXT,
                    physical_state TEXT NOT NULL DEFAULT 'PRESENT',
                    executor_pid INTEGER,
                    executor_start_ticks INTEGER,
                    executor_boot_id TEXT,
                    wall_deadline_utc TEXT,
                    last_heartbeat_utc TEXT,
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
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(worktree_sessions)")
            }
            migrations = {
                "ownership_token": "ALTER TABLE worktree_sessions ADD COLUMN ownership_token TEXT",
                "cleanup_eligible": (
                    "ALTER TABLE worktree_sessions ADD COLUMN cleanup_eligible INTEGER "
                    "NOT NULL DEFAULT 0 CHECK(cleanup_eligible IN (0, 1))"
                ),
                "result_id": "ALTER TABLE worktree_sessions ADD COLUMN result_id TEXT",
                "physical_state": (
                    "ALTER TABLE worktree_sessions ADD COLUMN physical_state TEXT "
                    "NOT NULL DEFAULT 'PRESENT'"
                ),
                "executor_pid": (
                    "ALTER TABLE worktree_sessions ADD COLUMN executor_pid INTEGER"
                ),
                "executor_start_ticks": (
                    "ALTER TABLE worktree_sessions ADD COLUMN executor_start_ticks INTEGER"
                ),
                "executor_boot_id": (
                    "ALTER TABLE worktree_sessions ADD COLUMN executor_boot_id TEXT"
                ),
                "wall_deadline_utc": (
                    "ALTER TABLE worktree_sessions ADD COLUMN wall_deadline_utc TEXT"
                ),
                "last_heartbeat_utc": (
                    "ALTER TABLE worktree_sessions ADD COLUMN last_heartbeat_utc TEXT"
                ),
                "current_task_label": (
                    "ALTER TABLE worktree_sessions ADD COLUMN current_task_label TEXT"
                ),
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
            # v1 Zetsu rows recorded terminal events but left their lifecycle
            # state ACTIVE/FAILED. Reclassify only unmistakable Zetsu sessions;
            # evidence and cleanliness are validated later before any release.
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state='SUCCEEDED_LEGACY', completed_at_utc=COALESCE(
                        completed_at_utc,
                        (SELECT MAX(timestamp_utc) FROM worktree_events e
                         WHERE e.session_id=worktree_sessions.session_id
                           AND e.user_id=worktree_sessions.user_id
                           AND e.event='TASK_COMPLETED')
                    ), updated_at_utc=?
                WHERE ownership_token IS NULL
                  AND task_title='Zetsu Qwen delegated task'
                  AND session_id LIKE 'zetsu-%'
                  AND physical_state='PRESENT'
                  AND state='ACTIVE'
                  AND EXISTS (
                      SELECT 1 FROM worktree_events e
                      WHERE e.session_id=worktree_sessions.session_id
                        AND e.user_id=worktree_sessions.user_id
                        AND e.event='TASK_COMPLETED'
                  )
                """,
                (datetime.now(UTC).isoformat(),),
            )
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state='FAILED_TERMINAL_LEGACY', completed_at_utc=COALESCE(
                        completed_at_utc,
                        (SELECT MAX(timestamp_utc) FROM worktree_events e
                         WHERE e.session_id=worktree_sessions.session_id
                           AND e.user_id=worktree_sessions.user_id
                           AND e.event='TASK_FAILED')
                    ), updated_at_utc=?
                WHERE ownership_token IS NULL
                  AND task_title='Zetsu Qwen delegated task'
                  AND session_id LIKE 'zetsu-%'
                  AND physical_state='PRESENT'
                  AND state='FAILED'
                  AND EXISTS (
                      SELECT 1 FROM worktree_events e
                      WHERE e.session_id=worktree_sessions.session_id
                        AND e.user_id=worktree_sessions.user_id
                        AND e.event='TASK_FAILED'
                  )
                """,
                (datetime.now(UTC).isoformat(),),
            )
            connection.execute(
                """
                INSERT INTO worktree_schema (singleton, schema_version, updated_at_utc)
                VALUES (1, 2, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (datetime.now(UTC).isoformat(),),
            )
        os.chmod(self.database, 0o600)

    def _recover(self) -> None:
        # No task thread survives an operator-process restart. Reconcile its durable
        # state before reconstructing the in-memory active-session map.
        self.reconcile(dry_run=False, limit=self.global_quota)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worktree_sessions
                WHERE state IN (
                    'ACTIVE', 'RUNNING', 'DIRTY', 'FAILED', 'INTERRUPTED_RESUMABLE',
                    'STALE_DIRTY'
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
        # Recover a crash between durable terminal persistence and physical release.
        if self.recover_cleanup:
            self.collect_garbage(dry_run=False, limit=self.global_quota)

    @staticmethod
    def _process_start_ticks(pid: int) -> int | None:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            closing = raw.rfind(")")
            if closing < 0:
                return None
            fields = raw[closing + 2 :].split()
            return int(fields[19])
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _boot_identity() -> str | None:
        try:
            value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def _ownership_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.ownership_root / f"{digest}.json"

    def register_capacity_notifier(self, notifier: Callable[[], None]) -> None:
        with self._lock:
            self._capacity_notifiers.append(notifier)

    def _notify_capacity_change(self) -> None:
        for notifier in tuple(self._capacity_notifiers):
            notifier()

    def _write_ownership(self, binding: AgentSessionBinding, token: str) -> None:
        path = self._ownership_path(binding.session_id)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        value = {
            "schema_version": 1,
            "session_id": binding.session_id,
            "user_id": binding.user_id,
            "repo_id": binding.repo_id,
            "canonical_repository_root": binding.canonical_repository_root,
            "worktree_root": binding.worktree_root,
            "base_revision": binding.base_revision,
            "ownership_token": token,
        }
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

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
        # Serialize quota reconciliation through durable registration. Without
        # this reservation boundary, concurrent creators could both observe the
        # final free slot before either inserts its worktree record.
        with self._lock:
            return self._create_locked(
                user_id=user_id,
                repo_id=repo_id,
                session_id=session_id,
                tool_policy=tool_policy,
                environment=environment,
                task_title=task_title,
                instruction_digest=instruction_digest,
                lane=lane,
                sanitized_model_name=sanitized_model_name,
                idempotency_key=idempotency_key,
            )

    def _create_locked(
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
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", user_id):
            raise AgentSandboxError("invalid_user_id")
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
            prepared = self.authorizations.prepare_new_session(user_id, repo_id)
            grant = prepared.grant
            self._require_quota(user_id)
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
        if head.returncode != 0 or head.stdout.strip() != grant.base_revision:
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
        try:
            self.authorizations.assert_revision(grant)
        except RepositoryAuthorizationError as exc:
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
            raise AgentSandboxError(exc.category, exc.evidence) from exc
        binding = AgentSessionBinding(
            session_id=identifier,
            user_id=user_id,
            repo_id=grant.repository.repo_id,
            canonical_repository_root=str(grant.repository.canonical_root),
            worktree_root=str(target.resolve(strict=True)),
            base_revision=grant.base_revision,
            grant_revision=grant.revision,
            tool_policy=tool_policy,
            environment=supplied_environment,
            created_at_utc=datetime.now(UTC).isoformat(),
        )
        ownership_token = os.urandom(32).hex()
        try:
            self._write_ownership(binding, ownership_token)
        except OSError as exc:
            self._runner(
                [
                    "git",
                    "-C",
                    binding.canonical_repository_root,
                    "worktree",
                    "remove",
                    "--force",
                    binding.worktree_root,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            raise AgentSandboxError("worktree_ownership_persistence_failed") from exc
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
                        idempotency_key, ownership_token
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?)
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
                        ownership_token,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                self._ownership_path(identifier).unlink(missing_ok=True)
                self._runner(
                    [
                        "git",
                        "-C",
                        binding.canonical_repository_root,
                        "worktree",
                        "remove",
                        "--force",
                        binding.worktree_root,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                raise AgentSandboxError("session_exists") from exc
            self._event(
                connection,
                binding,
                "CREATED",
                "ACTIVE",
                {"revision_sync": prepared.revision_sync},
            )
            self._sessions[identifier] = binding
        return binding

    def _require_quota(self, user_id: str) -> None:
        # GC performs reconciliation first, then usage is recomputed.
        self.collect_garbage(dry_run=False, user_id=user_id, limit=self.per_user_quota)
        placeholders = ",".join("?" for _ in _ACTIVE_QUOTA_STATES)
        with self._connect() as connection:
            global_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) AS count FROM worktree_sessions
                    WHERE physical_state='PRESENT' AND state IN ({placeholders})
                    """,
                    _ACTIVE_QUOTA_STATES,
                ).fetchone()["count"]
            )
            user_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) AS count FROM worktree_sessions
                    WHERE user_id=? AND physical_state='PRESENT'
                      AND state IN ({placeholders})
                    """,
                    (user_id, *_ACTIVE_QUOTA_STATES),
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
    ) -> int:
        cursor = connection.execute(
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
        if cursor.lastrowid is None:
            raise AgentSandboxError("agent_event_sequence_unavailable")
        return int(cursor.lastrowid)

    def record_progress(
        self,
        session_id: str,
        *,
        user_id: str,
        event: str,
        details: JsonObject,
    ) -> JsonObject:
        """Append bounded, owner-scoped execution progress without changing state.

        This is deliberately an internal lifecycle boundary, not a general event
        logger. Callers provide only deterministic event names and compact public
        metadata; the durable worktree event sequence is the reconnect cursor.
        """

        if event not in {
            "TURN_SUBMITTED",
            "TURN_STARTED",
            "TASK_STARTED",
            "REPOSITORY_READ_STARTED",
            "REPOSITORY_SEARCH_STARTED",
            "RETRIEVAL_STARTED",
            "TOOL_STARTED",
            "VERIFICATION_STARTED",
            "VERIFICATION_COMPLETED",
            "QUANTUM_CONTINUING",
            "TASK_YIELDED_RESUMABLE",
            "TASK_COMPLETED",
            "TASK_FAILED",
            "TASK_CANCELLED",
            "TURN_COMPLETED",
            "TURN_FAILED",
            "TURN_CANCELLED",
            "TURN_YIELDED_RESUMABLE",
        }:
            raise AgentSandboxError("agent_progress_event_invalid")
        try:
            encoded = json.dumps(details, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise AgentSandboxError("agent_progress_details_invalid") from exc
        if len(encoded.encode("utf-8")) > 4_096:
            raise AgentSandboxError("agent_progress_details_invalid")
        binding = self.require_active(session_id, user_id=user_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM worktree_sessions WHERE session_id=? AND user_id=?",
                (binding.session_id, binding.user_id),
            ).fetchone()
            if row is None:
                raise AgentSandboxError("unknown_agent_session")
            sequence = self._event(
                connection,
                binding,
                event,
                str(row["state"]),
                dict(details),
            )
        return {
            "sequence": sequence,
            "event": event,
            "state": str(row["state"]),
        }

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
        with self._connect() as connection:
            connection.execute(
                "UPDATE worktree_sessions SET physical_state='REMOVED' WHERE session_id=?",
                (session_id,),
            )
        self._sessions.pop(session_id, None)
        self._notify_capacity_change()
        return {"status": "RELEASED_CLEAN_WORKTREE", "session_id": session_id}

    def status(self, session_id: str, *, user_id: str) -> JsonObject:
        """Revalidate a binding and report its owned worktree state."""

        identifier = _session_identifier(session_id)
        with self._connect() as connection:
            historical = connection.execute(
                """
                SELECT repo_id, base_revision, grant_revision, state, changed_paths_json,
                       diff_hash, verification_summary, physical_state, result_id,
                       current_task_label
                FROM worktree_sessions WHERE session_id=? AND user_id=?
                """,
                (identifier, user_id),
            ).fetchone()
        if historical is None:
            raise AgentSandboxError("unknown_agent_session")
        if str(historical["physical_state"]) == "REMOVED":
            return {
                "status": "TERMINAL_RELEASED",
                "lifecycle_state": str(historical["state"]),
                "physical_state": "REMOVED",
                "session_id": identifier,
                "repo_id": str(historical["repo_id"]),
                "base_revision": str(historical["base_revision"]),
                "grant_revision": int(historical["grant_revision"]),
                "changed_paths": json.loads(str(historical["changed_paths_json"])),
                "diff_hash": historical["diff_hash"],
                "verification_summary": historical["verification_summary"],
                "result_id": historical["result_id"],
                "task_label": (
                    str(historical["current_task_label"])
                    if historical["current_task_label"]
                    else None
                ),
            }
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
        if row is not None and str(row["state"]) in {
            "RUNNING",
            "FAILED",
            "CANCELLED_DIRTY",
            "INTERRUPTED_RESUMABLE",
            "STALE_DIRTY",
            "SUCCEEDED",
        }:
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
            "verification_summary": historical["verification_summary"],
            "result_id": historical["result_id"],
            "task_label": (
                str(historical["current_task_label"])
                if historical["current_task_label"]
                else None
            ),
        }

    def has_session(self, session_id: str, *, user_id: str) -> bool:
        identifier = _session_identifier(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM worktree_sessions WHERE session_id=? AND user_id=?",
                (identifier, user_id),
            ).fetchone()
        return row is not None

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
        changed_paths = json.loads(str(row["changed_paths_json"]))
        clean = isinstance(changed_paths, list) and not changed_paths and row["diff_hash"] is None
        state = str(row["state"])
        physical_present = str(row["physical_state"]) == "PRESENT"
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
            "task_label": (
                str(row["current_task_label"]) if row["current_task_label"] else None
            ),
            "instruction_digest": str(row["instruction_digest"]),
            "state": state,
            "lane": row["lane"],
            "model_name": row["sanitized_model_name"],
            "tool_policy": json.loads(str(row["tool_policy_json"])),
            "command_count": int(row["command_count"]),
            "changed_paths": changed_paths,
            "diff_hash": row["diff_hash"],
            "verification_summary": row["verification_summary"],
            "export_state": str(row["export_state"]),
            "retention_expires_at_utc": row["retention_expires_at_utc"],
            "result_id": row["result_id"],
            "cleanup_eligible": bool(row["cleanup_eligible"]),
            "physical_state": str(row["physical_state"]),
            "worktree_clean": clean,
            "counts_against_quota": physical_present and state in _ACTIVE_QUOTA_STATES,
            "safely_releasable": clean
            and (
                (
                    state in _GC_TERMINAL_STATES
                    and (bool(row["cleanup_eligible"]) or state.endswith("_LEGACY"))
                )
                or state == "FAILED"
            ),
            "network_enabled": False,
        }
        if operator:
            value["canonical_repository_root"] = str(row["canonical_repository_root"])
            value["worktree_root"] = str(row["worktree_root"])
        else:
            value["worktree_path"] = str(row["worktree_root"])
        return value

    def start_task(
        self,
        session_id: str,
        *,
        user_id: str,
        lane: str,
        sanitized_model_name: str,
        instruction_digest: str | None = None,
        remaining_wall_seconds: float | None = None,
        task_label: str | None = None,
    ) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        normalized_task_label = (
            derive_task_label(task_label) if task_label is not None else None
        )
        if instruction_digest is not None and not re.fullmatch(r"[a-f0-9]{64}", instruction_digest):
            raise AgentSandboxError("invalid_instruction_digest")
        if remaining_wall_seconds is not None and not (
            0 < remaining_wall_seconds <= binding.tool_policy.max_wall_seconds
        ):
            raise AgentSandboxError("remaining_wall_budget_invalid")
        now = datetime.now(UTC)
        deadline = now + timedelta(
            seconds=(remaining_wall_seconds or binding.tool_policy.max_wall_seconds)
        )
        start_ticks = self._process_start_ticks(os.getpid())
        boot_id = self._boot_identity()
        if start_ticks is None or boot_id is None:
            raise AgentSandboxError("executor_identity_unavailable")
        start_details: JsonObject = {
            "lane": lane,
            "model_name": sanitized_model_name,
        }
        if normalized_task_label is not None:
            start_details["task_label"] = normalized_task_label
        self._set_state(binding, "RUNNING", "TASK_STARTED", start_details)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions SET lane=?, sanitized_model_name=?,
                    current_task_label=?,
                    instruction_digest=COALESCE(?, instruction_digest), executor_pid=?,
                    executor_start_ticks=?, executor_boot_id=?, wall_deadline_utc=?,
                    last_heartbeat_utc=?
                WHERE session_id=? AND user_id=?
                """,
                (
                    lane,
                    sanitized_model_name,
                    normalized_task_label,
                    instruction_digest,
                    os.getpid(),
                    start_ticks,
                    boot_id,
                    deadline.isoformat(),
                    now.isoformat(),
                    session_id,
                    user_id,
                ),
            )
        self._live_executors.add(session_id)
        return self.inspect(session_id, user_id=user_id)

    def heartbeat_task(self, session_id: str, *, user_id: str) -> None:
        """Refresh a live task lease without changing its deterministic wall deadline."""

        binding = self.require_active(session_id, user_id=user_id)
        if binding.session_id not in self._live_executors:
            raise AgentSandboxError("executor_not_live")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions SET last_heartbeat_utc=?, updated_at_utc=?
                WHERE session_id=? AND user_id=? AND state='RUNNING'
                """,
                (
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                    session_id,
                    user_id,
                ),
            )

    def reconcile_executor_exit(self, session_id: str, *, user_id: str) -> JsonObject:
        """Drop the in-process lease, then classify exact durable resume state."""

        identifier = _session_identifier(session_id)
        self._live_executors.discard(identifier)
        return self.reconcile(
            dry_run=False,
            user_id=user_id,
            session_id=identifier,
            limit=1,
        )

    def record_result(
        self,
        session_id: str,
        *,
        user_id: str,
        command_count: int,
        verification_summary: str,
        failed: bool = False,
        terminal: bool = False,
        resumable: bool = False,
        result_id: str | None = None,
    ) -> JsonObject:
        if result_id is not None and re.fullmatch(r"res_[a-f0-9]{32}", result_id) is None:
            raise AgentSandboxError("result_id_invalid")
        if resumable and (failed or terminal):
            raise AgentSandboxError("result_state_invalid")
        binding = self.require_active(session_id, user_id=user_id)
        changed_paths, diff_hash = self._diff_summary(binding)
        if resumable:
            state = "INTERRUPTED_RESUMABLE"
        elif terminal:
            state = "FAILED_TERMINAL" if failed else "SUCCEEDED"
        else:
            state = "FAILED" if failed else ("DIRTY" if changed_paths else "ACTIVE")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state=?, command_count=?, changed_paths_json=?, diff_hash=?,
                    verification_summary=?, result_id=COALESCE(?, result_id), updated_at_utc=?,
                    completed_at_utc=CASE WHEN ? THEN ? ELSE completed_at_utc END,
                    executor_pid=NULL, executor_start_ticks=NULL,
                    executor_boot_id=NULL, last_heartbeat_utc=NULL
                WHERE session_id=? AND user_id=?
                """,
                (
                    state,
                    command_count,
                    json.dumps(list(changed_paths), separators=(",", ":")),
                    diff_hash,
                    verification_summary[:2_000],
                    result_id,
                    now,
                    int(terminal),
                    now,
                    session_id,
                    user_id,
                ),
            )
            event = (
                "TASK_YIELDED_RESUMABLE"
                if resumable
                else ("TASK_FAILED" if failed else "TASK_COMPLETED")
            )
            current = connection.execute(
                "SELECT current_task_label FROM worktree_sessions WHERE session_id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
            details: JsonObject = {
                "changed_paths": list(changed_paths),
                "diff_hash": diff_hash,
                "result_id": result_id,
            }
            if current is not None and current["current_task_label"]:
                details["task_label"] = str(current["current_task_label"])
            self._event(connection, binding, event, state, details)
        self._live_executors.discard(session_id)
        return self.inspect(session_id, user_id=user_id)

    def record_interrupted(self, session_id: str, *, user_id: str, reason: str) -> JsonObject:
        """Preserve a genuinely resumable interruption and its physical worktree."""

        binding = self.require_active(session_id, user_id=user_id)
        self._set_state(
            binding,
            "INTERRUPTED_RESUMABLE",
            "TASK_INTERRUPTED_RESUMABLE",
            {"reason": reason[:500]},
        )
        self._live_executors.discard(session_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions SET executor_pid=NULL,
                    executor_start_ticks=NULL, executor_boot_id=NULL,
                    last_heartbeat_utc=NULL, updated_at_utc=?
                WHERE session_id=? AND user_id=?
                """,
                (datetime.now(UTC).isoformat(), session_id, user_id),
            )
        return self.inspect(session_id, user_id=user_id)

    def authorize_terminal_cleanup(
        self,
        session_id: str,
        *,
        user_id: str,
        result_id: str,
    ) -> JsonObject:
        """Authorize release only after an owner-bound durable result exists."""

        identifier = _session_identifier(session_id)
        if re.fullmatch(r"res_[a-f0-9]{32}", result_id) is None:
            raise AgentSandboxError("terminal_result_id_invalid")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM worktree_sessions WHERE session_id=? AND user_id=?",
                (identifier, user_id),
            ).fetchone()
            if row is None:
                raise AgentSandboxError("unknown_agent_session")
            state = str(row["state"])
            if state not in _GC_TERMINAL_STATES:
                raise AgentSandboxError("terminal_cleanup_state_invalid", {"state": state})
            connection.execute(
                """
                UPDATE worktree_sessions
                SET cleanup_eligible=1, result_id=?, updated_at_utc=?
                WHERE session_id=? AND user_id=?
                """,
                (result_id, datetime.now(UTC).isoformat(), identifier, user_id),
            )
        report = self.collect_garbage(
            dry_run=False,
            user_id=user_id,
            session_id=identifier,
            limit=1,
        )
        items = report.get("items")
        if isinstance(items, list) and items:
            return cast(JsonObject, items[0])
        with self._connect() as connection:
            released = connection.execute(
                """
                SELECT physical_state, result_id FROM worktree_sessions
                WHERE session_id=? AND user_id=?
                """,
                (identifier, user_id),
            ).fetchone()
        if (
            released is not None
            and str(released["physical_state"]) == "REMOVED"
            and released["result_id"] == result_id
        ):
            return {
                "action": "ALREADY_RELEASED",
                "reason": "idempotent_terminal_cleanup",
                "session_id": identifier,
                "result_id": result_id,
            }
        raise AgentSandboxError("terminal_cleanup_not_examined")

    def resume(self, session_id: str, *, user_id: str) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        self._set_state(binding, "ACTIVE", "RESUMED", {})
        return self.inspect(session_id, user_id=user_id)

    def cancel(self, session_id: str, *, user_id: str) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        if binding.session_id in self._live_executors:
            # A running coordinator owns this worktree.  Cancellation is
            # cooperative and must be observed by the coordinator before the
            # normal clean-worktree lifecycle may reclaim it.
            return {
                "status": "CANCELLATION_REQUESTED",
                "session_id": binding.session_id,
            }
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
                SET state='DISCARDED', physical_state='REMOVED', cleanup_eligible=0,
                    completed_at_utc=?, updated_at_utc=?
                WHERE session_id=?
                """,
                (datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat(), session_id),
            )
            binding = self._sessions.get(session_id)
            if binding is not None:
                self._event(connection, binding, "DISCARDED", "DISCARDED", {})
        self._sessions.pop(session_id, None)
        self._notify_capacity_change()
        return {
            "status": "DISCARDED",
            "session_id": session_id,
            "owner": owner if operator else "self",
        }

    @staticmethod
    def _binding_from_row(row: sqlite3.Row) -> AgentSessionBinding:
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
            return AgentSessionBinding(
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
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentSandboxError("worktree_ownership_record_invalid") from exc

    def _marker_matches(self, row: sqlite3.Row, binding: AgentSessionBinding) -> bool:
        token = row["ownership_token"]
        marker = self._ownership_path(binding.session_id)
        if isinstance(token, str) and token:
            try:
                details = marker.lstat()
                raw: object = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            expected = {
                "schema_version": 1,
                "session_id": binding.session_id,
                "user_id": binding.user_id,
                "repo_id": binding.repo_id,
                "canonical_repository_root": binding.canonical_repository_root,
                "worktree_root": binding.worktree_root,
                "base_revision": binding.base_revision,
                "ownership_token": token,
            }
            return (
                stat.S_ISREG(details.st_mode)
                and not marker.is_symlink()
                and marker.resolve() == marker
                and raw == expected
            )
        # Legacy records are accepted only with the original append-only CREATED
        # event. A forged row alone can never authorize deletion.
        with self._connect() as connection:
            created = connection.execute(
                """
                SELECT COUNT(*) AS count FROM worktree_events
                WHERE session_id=? AND user_id=? AND event='CREATED' AND state='ACTIVE'
                """,
                (binding.session_id, binding.user_id),
            ).fetchone()
        return created is not None and int(created["count"]) == 1

    def _durable_result_matches(self, row: sqlite3.Row, binding: AgentSessionBinding) -> bool:
        result_id = row["result_id"]
        if not isinstance(result_id, str) or re.fullmatch(r"res_[a-f0-9]{32}", result_id) is None:
            return False
        directory = self.sandbox_root / "zetsu_agent_results" / result_id
        manifest = directory / "manifest.json"
        try:
            if (
                directory.is_symlink()
                or directory.resolve() != directory
                or manifest.is_symlink()
                or manifest.resolve() != manifest
            ):
                return False
            raw: object = json.loads(manifest.read_text(encoding="utf-8"))
            changed = json.loads(str(row["changed_paths_json"]))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(raw, dict):
            return False
        artifacts = raw.get("artifacts")
        identity_matches = (
            raw.get("result_id") == result_id
            and raw.get("owner_id_sha256")
            == hashlib.sha256(binding.user_id.encode("utf-8")).hexdigest()
            and raw.get("repo_id") == binding.repo_id
            and raw.get("session_id") == binding.session_id
            and isinstance(artifacts, dict)
            and "result.json" in artifacts
            and (not changed or "handoff.patch" in artifacts)
        )
        if not identity_matches or not isinstance(artifacts, dict):
            return False
        for name, record in artifacts.items():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", name) is None
                or not isinstance(record, dict)
            ):
                return False
            artifact = directory / name
            try:
                details = artifact.lstat()
                if (
                    stat.S_ISLNK(details.st_mode)
                    or not stat.S_ISREG(details.st_mode)
                    or artifact.resolve() != artifact
                    or details.st_dev != record.get("device")
                    or details.st_ino != record.get("inode")
                    or details.st_size != record.get("bytes")
                ):
                    return False
                digest = hashlib.sha256()
                with artifact.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != record.get("sha256"):
                    return False
            except OSError:
                return False
        return True

    def _executor_is_live(self, row: sqlite3.Row, binding: AgentSessionBinding) -> bool:
        """Require both a live in-process lease and a non-reused OS process identity."""

        if binding.session_id not in self._live_executors:
            return False
        pid = row["executor_pid"]
        ticks = row["executor_start_ticks"]
        boot_id = row["executor_boot_id"]
        return (
            isinstance(pid, int)
            and isinstance(ticks, int)
            and isinstance(boot_id, str)
            and pid == os.getpid()
            and ticks == self._process_start_ticks(pid)
            and boot_id == self._boot_identity()
        )

    @staticmethod
    def _row_deadline(row: sqlite3.Row, binding: AgentSessionBinding) -> datetime | None:
        raw = row["wall_deadline_utc"]
        try:
            if isinstance(raw, str) and raw:
                deadline = datetime.fromisoformat(raw)
            else:
                deadline = datetime.fromisoformat(binding.created_at_utc) + timedelta(
                    seconds=binding.tool_policy.max_wall_seconds
                )
        except ValueError:
            return None
        if deadline.tzinfo is None:
            return None
        return deadline.astimezone(UTC)

    @staticmethod
    def _changed_from_status(status: str) -> list[str]:
        changed: list[str] = []
        for line in status.splitlines():
            if len(line) < 4:
                continue
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            changed.append(path.strip('"'))
        return sorted(dict.fromkeys(changed))

    def _valid_resume_checkpoint(
        self,
        row: sqlite3.Row,
        binding: AgentSessionBinding,
        *,
        head: str,
        status_text: str,
        now: datetime,
    ) -> Path | None:
        deadline = self._row_deadline(row, binding)
        if deadline is None or now >= deadline:
            return None
        checkpoint = (
            self.sandbox_root
            / "zetsu_agent_checkpoints"
            / f"{hashlib.sha256(binding.session_id.encode('utf-8')).hexdigest()}.json"
        )
        try:
            details = checkpoint.lstat()
            raw: object = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or checkpoint.resolve() != checkpoint
            or not isinstance(raw, dict)
        ):
            return None
        consumed = raw.get("consumed_wall_seconds")
        expected_changed = self._changed_from_status(status_text)
        return (
            checkpoint
            if raw.get("session_id") == binding.session_id
            and raw.get("user_id_sha256")
            == hashlib.sha256(binding.user_id.encode("utf-8")).hexdigest()
            and raw.get("repo_id") == binding.repo_id
            and raw.get("base_revision") == binding.base_revision
            and raw.get("worktree_head") == head
            and raw.get("worktree_status_sha256")
            == hashlib.sha256(status_text.encode("utf-8")).hexdigest()
            and raw.get("changed_paths") == expected_changed
            and isinstance(consumed, (int, float))
            and not isinstance(consumed, bool)
            and 0 <= float(consumed) < binding.tool_policy.max_wall_seconds
            else None
        )

    def _persist_reconciled_terminal(
        self,
        row: sqlite3.Row,
        binding: AgentSessionBinding,
        *,
        reason: str,
        checkpoint: Path | None = None,
    ) -> sqlite3.Row:
        """Persist an exact clean-worktree diagnostic before authorizing release."""

        from .result_store import ResultStore

        deadline = self._row_deadline(row, binding)
        diagnostic: JsonObject = {
            "schema_version": 1,
            "status": "FAILED",
            "failure_category": reason,
            "session_id": binding.session_id,
            "repo_id": binding.repo_id,
            "prior_lifecycle_state": str(row["state"]),
            "worktree_clean": True,
            "executor_pid": row["executor_pid"],
            "wall_deadline_utc": deadline.isoformat() if deadline is not None else None,
            "checkpoint_preserved": checkpoint is not None,
        }
        artifacts: dict[str, Path | bytes | str] = {
            "result.json": json.dumps(
                diagnostic, indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n"
        }
        if checkpoint is not None:
            artifacts["checkpoint.json"] = checkpoint
        delivery = ResultStore(
            self.sandbox_root / "zetsu_agent_results"
        ).persist(
            user_id=binding.user_id,
            repo_id=binding.repo_id,
            session_id=binding.session_id,
            status="FAILED",
            summary=reason,
            artifacts=artifacts,
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state='FAILED_TERMINAL', cleanup_eligible=1, result_id=?,
                    changed_paths_json='[]', diff_hash=NULL,
                    verification_summary=?, completed_at_utc=COALESCE(completed_at_utc, ?),
                    updated_at_utc=?, executor_pid=NULL, executor_start_ticks=NULL,
                    executor_boot_id=NULL, last_heartbeat_utc=NULL
                WHERE session_id=? AND user_id=?
                """,
                (
                    delivery["result_id"],
                    f"FAILED:{reason}"[:2_000],
                    now,
                    now,
                    binding.session_id,
                    binding.user_id,
                ),
            )
            self._event(
                connection,
                binding,
                "STALE_TASK_RECONCILED",
                "FAILED_TERMINAL",
                {"reason": reason, "result_id": delivery["result_id"]},
            )
            migrated = connection.execute(
                "SELECT * FROM worktree_sessions WHERE session_id=? AND user_id=?",
                (binding.session_id, binding.user_id),
            ).fetchone()
        self._live_executors.discard(binding.session_id)
        if migrated is None:
            raise AgentSandboxError("stale_task_reconciliation_failed")
        return cast(sqlite3.Row, migrated)

    def _reconcile_item(self, row: sqlite3.Row, *, dry_run: bool) -> JsonObject:
        binding = self._binding_from_row(row)
        state = str(row["state"])
        item: JsonObject = {
            "session_id": binding.session_id,
            "user_id": binding.user_id,
            "repo_id": binding.repo_id,
            "prior_state": state,
            "worktree_root": binding.worktree_root,
        }
        if state == "CANCELLED_DIRTY":
            return {**item, "action": "PROTECTED", "reason": "cancelled_dirty_preserved"}
        if state == "STALE_GRANT":
            return {**item, "action": "PROTECTED", "reason": "stale_grant_preserved"}
        if not self._marker_matches(row, binding):
            return {**item, "action": "PROTECTED", "reason": "ownership_proof_invalid"}
        registered, registration_reason = self._registered_worktree(binding)
        if not registered:
            return {**item, "action": "PROTECTED", "reason": registration_reason}
        root = Path(binding.worktree_root)
        if not root.exists():
            if dry_run:
                return {**item, "action": "WOULD_MARK_UNAVAILABLE", "reason": registration_reason}
            now_text = datetime.now(UTC).isoformat()
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE worktree_sessions SET state='UNAVAILABLE', physical_state='REMOVED',
                        cleanup_eligible=0, completed_at_utc=COALESCE(completed_at_utc, ?),
                        updated_at_utc=? WHERE session_id=? AND user_id=?
                    """,
                    (now_text, now_text, binding.session_id, binding.user_id),
                )
            self._sessions.pop(binding.session_id, None)
            self._live_executors.discard(binding.session_id)
            return {**item, "action": "MARKED_UNAVAILABLE", "reason": registration_reason}
        head_result = self._runner(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        status_result = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if head_result.returncode != 0 or status_result.returncode != 0:
            return {**item, "action": "PROTECTED", "reason": "worktree_state_unavailable"}
        dirty = bool(status_result.stdout.strip())
        now = datetime.now(UTC)
        deadline = self._row_deadline(row, binding)
        expired = deadline is None or now >= deadline
        live = self._executor_is_live(row, binding)
        checkpoint = self._valid_resume_checkpoint(
            row,
            binding,
            head=head_result.stdout.strip(),
            status_text=status_result.stdout,
            now=now,
        )
        item.update({"dirty": dirty, "deadline_expired": expired, "executor_live": live})

        if state == "RUNNING" and live:
            return {**item, "action": "RETAINED", "reason": "executor_live"}
        if state == "ACTIVE" and not expired:
            return {**item, "action": "RETAINED", "reason": "admission_deadline_live"}
        if state == "INTERRUPTED_RESUMABLE" and checkpoint is not None:
            return {**item, "action": "RETAINED", "reason": "valid_resumable_checkpoint"}
        if state == "RUNNING" and checkpoint is not None:
            if not dry_run:
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE worktree_sessions SET state='INTERRUPTED_RESUMABLE',
                            executor_pid=NULL, executor_start_ticks=NULL,
                            executor_boot_id=NULL, last_heartbeat_utc=NULL, updated_at_utc=?
                        WHERE session_id=? AND user_id=? AND state='RUNNING'
                        """,
                        (datetime.now(UTC).isoformat(), binding.session_id, binding.user_id),
                    )
            return {
                **item,
                "action": "WOULD_MARK_RESUMABLE" if dry_run else "MARKED_RESUMABLE",
                "reason": "executor_gone_valid_checkpoint",
            }

        reason = (
            "failed_non_resumable"
            if state == "FAILED"
            else "wall_deadline_expired"
            if expired
            else "executor_gone_without_valid_checkpoint"
        )
        if dirty:
            if not dry_run:
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE worktree_sessions SET state='STALE_DIRTY',
                            verification_summary=?, executor_pid=NULL,
                            executor_start_ticks=NULL, executor_boot_id=NULL,
                            last_heartbeat_utc=NULL, updated_at_utc=?
                        WHERE session_id=? AND user_id=?
                        """,
                        (
                            f"FAILED:{reason}"[:2_000],
                            datetime.now(UTC).isoformat(),
                            binding.session_id,
                            binding.user_id,
                        ),
                    )
            return {**item, "action": "PROTECTED", "reason": f"{reason}_dirty"}
        if dry_run:
            return {**item, "action": "WOULD_TERMINATE_AND_RELEASE", "reason": reason}
        terminal = self._persist_reconciled_terminal(
            row, binding, reason=reason, checkpoint=checkpoint
        )
        return self._gc_item(terminal, dry_run=False)

    def reconcile(
        self,
        *,
        dry_run: bool,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 64,
    ) -> JsonObject:
        """Boundedly reconcile durable task state with executor and worktree reality."""

        if not 1 <= limit <= 256:
            raise AgentSandboxError("worktree_reconcile_limit_invalid")
        clauses = [
            "physical_state='PRESENT'",
            "state IN ('ACTIVE','RUNNING','FAILED','INTERRUPTED_RESUMABLE','STALE_DIRTY',"
            "'CANCELLED_DIRTY','STALE_GRANT')",
        ]
        values: list[object] = []
        if user_id is not None:
            clauses.append("user_id=?")
            values.append(user_id)
        if session_id is not None:
            clauses.append("session_id=?")
            values.append(_session_identifier(session_id))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM worktree_sessions WHERE {' AND '.join(clauses)}
                ORDER BY created_at_utc ASC LIMIT ?
                """,
                (*values, limit),
            ).fetchall()
            items = [self._reconcile_item(row, dry_run=dry_run) for row in rows]
        return {
            "status": "DRY_RUN" if dry_run else "COMPLETE",
            "examined": len(items),
            "released": sum(item.get("action") in {"RELEASED", "RECOVERED_RELEASE"} for item in items),
            "resumable": sum(
                item.get("action") in {"RETAINED", "MARKED_RESUMABLE"}
                and item.get("reason") in {
                    "valid_resumable_checkpoint",
                    "executor_gone_valid_checkpoint",
                }
                for item in items
            ),
            "protected": sum(item.get("action") == "PROTECTED" for item in items),
            "items": items,
        }

    def capacity_snapshot(self, *, user_id: str | None = None) -> JsonObject:
        placeholders = ",".join("?" for _ in _ACTIVE_QUOTA_STATES)
        with self._connect() as connection:
            global_live = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) AS count FROM worktree_sessions
                    WHERE physical_state='PRESENT' AND state IN ({placeholders})
                    """,
                    _ACTIVE_QUOTA_STATES,
                ).fetchone()["count"]
            )
            user_live = (
                int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*) AS count FROM worktree_sessions
                        WHERE user_id=? AND physical_state='PRESENT'
                          AND state IN ({placeholders})
                        """,
                        (user_id, *_ACTIVE_QUOTA_STATES),
                    ).fetchone()["count"]
                )
                if user_id is not None
                else None
            )
        return {
            "live_resumable_worktrees": global_live,
            "global_worktree_safety_limit": self.global_quota,
            "user_live_resumable_worktrees": user_live,
            "per_user_worktree_safety_limit": self.per_user_quota,
            "available": global_live < self.global_quota
            and (user_live is None or user_live < self.per_user_quota),
        }

    def next_reconciliation_delay(self, *, user_id: str) -> float | None:
        """Return the exact next durable wall deadline, never a polling interval."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worktree_sessions
                WHERE user_id=? AND physical_state='PRESENT'
                  AND state IN ('ACTIVE','RUNNING','INTERRUPTED_RESUMABLE')
                """,
                (user_id,),
            ).fetchall()
        now = datetime.now(UTC)
        delays: list[float] = []
        for row in rows:
            try:
                binding = self._binding_from_row(row)
            except AgentSandboxError:
                continue
            deadline = self._row_deadline(row, binding)
            if deadline is not None:
                delays.append(max(0.0, (deadline - now).total_seconds()))
        return min(delays) if delays else None

    def _legacy_checkpoint(self, row: sqlite3.Row, binding: AgentSessionBinding) -> Path | None:
        checkpoint = (
            self.sandbox_root
            / "zetsu_agent_checkpoints"
            / f"{hashlib.sha256(binding.session_id.encode('utf-8')).hexdigest()}.json"
        )
        try:
            details = checkpoint.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISREG(details.st_mode)
                or checkpoint.resolve() != checkpoint
            ):
                return None
            raw: object = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        if (
            raw.get("session_id") != binding.session_id
            or raw.get("user_id_sha256")
            != hashlib.sha256(binding.user_id.encode("utf-8")).hexdigest()
            or raw.get("repo_id") != binding.repo_id
            or raw.get("base_revision") != binding.base_revision
        ):
            return None
        changed = raw.get("changed_paths")
        if not isinstance(changed, list) or changed != json.loads(str(row["changed_paths_json"])):
            return None
        return checkpoint

    def _migrate_legacy_result(
        self,
        row: sqlite3.Row,
        binding: AgentSessionBinding,
        checkpoint: Path,
    ) -> sqlite3.Row:
        from .result_store import ResultStore

        state = str(row["state"])
        status = "SUCCESS" if state == "SUCCEEDED_LEGACY" else "FAILED"
        historical = {
            "schema_version": 1,
            "migration": "legacy_zetsu_terminal_v1",
            "status": status,
            "session_id": binding.session_id,
            "repo_id": binding.repo_id,
            "verification_summary": row["verification_summary"],
            "changed_paths": json.loads(str(row["changed_paths_json"])),
            "diff_hash": row["diff_hash"],
            "checkpoint_preserved": True,
        }
        delivery = ResultStore(
            self.sandbox_root / "zetsu_agent_results"
        ).persist(
            user_id=binding.user_id,
            repo_id=binding.repo_id,
            session_id=binding.session_id,
            status=status,
            summary=str(row["verification_summary"] or status),
            artifacts={
                "result.json": json.dumps(
                    historical, indent=2, sort_keys=True, ensure_ascii=False
                )
                + "\n",
                "checkpoint.json": checkpoint,
            },
        )
        migrated_state = "SUCCEEDED" if status == "SUCCESS" else "FAILED_TERMINAL"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state=?, cleanup_eligible=1, result_id=?, updated_at_utc=?
                WHERE session_id=? AND user_id=? AND state=?
                """,
                (
                    migrated_state,
                    delivery["result_id"],
                    datetime.now(UTC).isoformat(),
                    binding.session_id,
                    binding.user_id,
                    state,
                ),
            )
            migrated = connection.execute(
                "SELECT * FROM worktree_sessions WHERE session_id=? AND user_id=?",
                (binding.session_id, binding.user_id),
            ).fetchone()
        if migrated is None:
            raise AgentSandboxError("legacy_terminal_migration_failed")
        return cast(sqlite3.Row, migrated)

    def _registered_worktree(self, binding: AgentSessionBinding) -> tuple[bool, str]:
        root = Path(binding.worktree_root)
        expected = self.sandbox_root / binding.user_id / binding.session_id
        if root != expected or not root.is_absolute():
            return False, "worktree_path_not_owned_layout"
        try:
            repository = self.authorizations.repository(binding.repo_id)
        except RepositoryAuthorizationError:
            return False, "repository_identity_unavailable"
        if str(repository.canonical_root) != binding.canonical_repository_root:
            return False, "foreign_repository"
        listed = self._runner(
            [
                "git",
                "-C",
                binding.canonical_repository_root,
                "worktree",
                "list",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if listed.returncode != 0:
            return False, "git_worktree_inventory_failed"
        paths = [
            Path(line.removeprefix("worktree ")).resolve()
            for line in listed.stdout.splitlines()
            if line.startswith("worktree ")
        ]
        if not root.exists():
            return (root not in paths, "already_absent" if root not in paths else "missing_registered")
        try:
            details = root.lstat()
            resolved = root.resolve(strict=True)
        except OSError:
            return False, "worktree_unavailable"
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode) or resolved != root:
            return False, "worktree_symlink_or_realpath_mismatch"
        if paths.count(root) != 1:
            return False, "worktree_not_uniquely_registered"
        top = self._runner(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        common = self._runner(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        repository_common = self._runner(
            [
                "git",
                "-C",
                binding.canonical_repository_root,
                "rev-parse",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
            return False, "worktree_git_root_mismatch"
        if common.returncode != 0 or repository_common.returncode != 0:
            return False, "worktree_git_common_dir_unavailable"
        observed_common = (root / common.stdout.strip()).resolve()
        expected_common = (
            Path(binding.canonical_repository_root) / repository_common.stdout.strip()
        ).resolve()
        if observed_common != expected_common:
            return False, "worktree_foreign_git_common_dir"
        return True, "owned_registered"

    def _gc_item(self, row: sqlite3.Row, *, dry_run: bool) -> JsonObject:
        binding = self._binding_from_row(row)
        state = str(row["state"])
        legacy = state in {"SUCCEEDED_LEGACY", "FAILED_TERMINAL_LEGACY"}
        item: JsonObject = {
            "session_id": binding.session_id,
            "user_id": binding.user_id,
            "repo_id": binding.repo_id,
            "state": state,
            "physical_state": str(row["physical_state"]),
            "result_id": row["result_id"],
            "worktree_root": binding.worktree_root,
        }
        if str(row["physical_state"]) != "PRESENT":
            return {**item, "action": "NONE", "reason": "already_released"}
        if state not in _GC_TERMINAL_STATES or (
            not legacy and not bool(row["cleanup_eligible"])
        ):
            return {**item, "action": "PROTECTED", "reason": "not_cleanup_eligible_terminal"}
        if not self._marker_matches(row, binding):
            return {**item, "action": "PROTECTED", "reason": "ownership_proof_invalid"}
        registered, reason = self._registered_worktree(binding)
        if not registered:
            return {**item, "action": "PROTECTED", "reason": reason}
        root = Path(binding.worktree_root)
        if legacy and not root.exists():
            return {**item, "action": "PROTECTED", "reason": "legacy_worktree_missing"}
        if not legacy and not self._durable_result_matches(row, binding):
            return {**item, "action": "PROTECTED", "reason": "durable_result_invalid"}
        if not root.exists():
            if not dry_run:
                self._mark_released(binding, prior_state=state, recovered=True)
            return {
                **item,
                "action": "WOULD_RECOVER_RELEASE" if dry_run else "RECOVERED_RELEASE",
                "reason": reason,
            }
        status = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0:
            return {**item, "action": "PROTECTED", "reason": "worktree_status_failed"}
        dirty = bool(status.stdout.strip())
        item["dirty"] = dirty
        if legacy:
            if dirty:
                return {**item, "action": "PROTECTED", "reason": "legacy_worktree_dirty"}
            checkpoint = self._legacy_checkpoint(row, binding)
            if checkpoint is None:
                return {**item, "action": "PROTECTED", "reason": "legacy_checkpoint_invalid"}
            if dry_run:
                return {
                    **item,
                    "action": "WOULD_MIGRATE_AND_RELEASE",
                    "reason": "verified_legacy_terminal_owned",
                }
            migrated = self._migrate_legacy_result(row, binding, checkpoint)
            return self._gc_item(migrated, dry_run=False)
        if dry_run:
            return {**item, "action": "WOULD_RELEASE", "reason": "durable_terminal_owned"}
        removed = self._runner(
            [
                "git",
                "-C",
                binding.canonical_repository_root,
                "worktree",
                "remove",
                *( ["--force"] if dirty else [] ),
                binding.worktree_root,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if removed.returncode != 0:
            return {
                **item,
                "action": "FAILED",
                "reason": "git_worktree_remove_failed",
                "returncode": removed.returncode,
                "stderr_tail": removed.stderr[-2_000:],
            }
        self._mark_released(binding, prior_state=state, recovered=False)
        return {**item, "action": "RELEASED", "reason": "durable_terminal_owned"}

    def _mark_released(
        self, binding: AgentSessionBinding, *, prior_state: str, recovered: bool
    ) -> None:
        released_state = (
            "RELEASED_SUCCESS" if prior_state == "SUCCEEDED" else "RELEASED_TERMINAL"
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worktree_sessions
                SET state=?, physical_state='REMOVED', cleanup_eligible=0,
                    updated_at_utc=?, completed_at_utc=COALESCE(completed_at_utc, ?)
                WHERE session_id=? AND user_id=?
                """,
                (released_state, now, now, binding.session_id, binding.user_id),
            )
            self._event(
                connection,
                binding,
                "TERMINAL_WORKTREE_RELEASE_RECOVERED" if recovered else "TERMINAL_WORKTREE_RELEASED",
                released_state,
                {"result_preserved": True},
            )
        self._sessions.pop(binding.session_id, None)
        self._notify_capacity_change()

    def collect_garbage(
        self,
        *,
        dry_run: bool,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 64,
    ) -> JsonObject:
        """Inspect or release a bounded set of provably owned terminal worktrees."""

        if not 1 <= limit <= 256:
            raise AgentSandboxError("worktree_gc_limit_invalid")
        reconciliation = self.reconcile(
            dry_run=dry_run,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )
        clauses = [
            "physical_state='PRESENT'",
            "(cleanup_eligible=1 OR state IN ('SUCCEEDED_LEGACY', "
            "'FAILED_TERMINAL_LEGACY'))",
        ]
        values: list[object] = []
        if user_id is not None:
            clauses.append("user_id=?")
            values.append(user_id)
        if session_id is not None:
            clauses.append("session_id=?")
            values.append(_session_identifier(session_id))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM worktree_sessions
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at_utc ASC LIMIT ?
                """,
                (*values, limit),
            ).fetchall()
            gc_items = [self._gc_item(row, dry_run=dry_run) for row in rows]
        reconciled_items = reconciliation.get("items")
        items = (
            [cast(JsonObject, item) for item in reconciled_items if isinstance(item, dict)]
            if isinstance(reconciled_items, list)
            else []
        )
        items.extend(gc_items)
        return {
            "status": "DRY_RUN" if dry_run else "COMPLETE",
            "examined": len(items),
            "released": sum(item.get("action") in {"RELEASED", "RECOVERED_RELEASE"} for item in items),
            "protected": sum(item.get("action") == "PROTECTED" for item in items),
            "items": items,
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
        # A clean terminal worktree can be released before a reconnecting CLI
        # fetches its final event.  Keep the durable event feed readable for the
        # original owner without reviving the released worktree.
        self.status(session_id, user_id=user_id)
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

    def events(
        self,
        session_id: str,
        *,
        user_id: str,
        after_sequence: int = 0,
        limit: int = 200,
    ) -> list[JsonObject]:
        """Return a bounded ordered owner-only worktree event page."""

        if not 0 <= after_sequence <= 2**63 - 1 or not 1 <= limit <= 200:
            raise AgentSandboxError("agent_event_cursor_invalid")
        self.inspect(session_id, user_id=user_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event, state, timestamp_utc, details_json
                FROM worktree_events
                WHERE session_id=? AND user_id=? AND sequence>?
                ORDER BY sequence LIMIT ?
                """,
                (session_id, user_id, after_sequence, limit),
            ).fetchall()
        records: list[JsonObject] = []
        for row in rows:
            try:
                details = json.loads(str(row["details_json"]))
            except json.JSONDecodeError as exc:
                raise AgentSandboxError("agent_event_details_invalid") from exc
            if not isinstance(details, dict):
                raise AgentSandboxError("agent_event_details_invalid")
            records.append(
                {
                    "sequence": int(row["sequence"]),
                    "event": str(row["event"]),
                    "state": str(row["state"]),
                    "timestamp_utc": str(row["timestamp_utc"]),
                    "details": details,
                }
            )
        return records

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
