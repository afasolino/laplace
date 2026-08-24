"""Persistent, bounded admission for whole Zetsu agent tasks.

The scheduler deliberately precedes worktree allocation and model invocation. Its
condition variable is the live wake-up mechanism; SQLite is durable state and
restart/audit evidence, not a polling transport.
"""

from __future__ import annotations

import re
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .agent_sandbox import AgentSandboxManager

JsonObject: TypeAlias = dict[str, object]


class AgentSchedulerError(RuntimeError):
    """A bounded task admission request could not proceed."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class AgentCapacityPolicy:
    topology: str
    execution_slots: int
    queue_capacity: int
    certification: str

    def __post_init__(self) -> None:
        if self.topology not in {"full", "nocodev"}:
            raise ValueError("invalid agent runtime topology")
        if not 1 <= self.execution_slots <= 16 or not 1 <= self.queue_capacity <= 256:
            raise ValueError("invalid agent capacity policy")


# These values are intentionally conservative until the certification artifact is
# produced on the target GPU. The nocodev value may only be raised together with
# recorded deterministic stress evidence.
CERTIFIED_AGENT_CAPACITIES: dict[str, AgentCapacityPolicy] = {
    "full": AgentCapacityPolicy(
        topology="full",
        execution_slots=1,
        queue_capacity=16,
        certification="conservative_8gb_baseline_a6000_49140mib_not_transferable",
    ),
    "nocodev": AgentCapacityPolicy(
        topology="nocodev",
        execution_slots=1,
        queue_capacity=16,
        certification="conservative_8gb_baseline_a6000_49140mib_not_transferable",
    ),
}


@dataclass(frozen=True)
class AgentAdmission:
    ticket_id: str
    session_id: str
    queue_position_at_arrival: int
    queue_wait_seconds: float


def capacity_policy(*, codev_enabled: bool) -> AgentCapacityPolicy:
    return CERTIFIED_AGENT_CAPACITIES["full" if codev_enabled else "nocodev"]


class AgentTaskScheduler:
    """FIFO whole-task scheduler with bounded queue/deadline/cancellation."""

    def __init__(
        self,
        database: Path,
        sandboxes: AgentSandboxManager,
        policy: AgentCapacityPolicy,
    ) -> None:
        self.sandboxes = sandboxes
        candidate = database.absolute()
        expected = sandboxes.sandbox_root / "zetsu_agent_scheduler.sqlite3"
        try:
            if candidate != expected or candidate.parent.resolve() != sandboxes.sandbox_root:
                raise AgentSchedulerError("agent_scheduler_path_invalid")
            if candidate.is_symlink() or candidate.exists():
                details = candidate.lstat()
                if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                    raise AgentSchedulerError("agent_scheduler_path_invalid")
        except OSError as exc:
            raise AgentSchedulerError("agent_scheduler_path_invalid") from exc
        self.database = candidate
        self.policy = policy
        self._condition = threading.Condition(threading.RLock())
        self._initialize()
        self._recover()
        self.sandboxes.register_capacity_notifier(self.notify_capacity_changed)

    def notify_capacity_changed(self) -> None:
        with self._condition:
            self._condition.notify_all()

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
                CREATE TABLE IF NOT EXISTS agent_admissions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    instruction_digest TEXT NOT NULL,
                    topology TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    deadline_utc TEXT NOT NULL,
                    admitted_at_utc TEXT,
                    completed_at_utc TEXT,
                    result_id TEXT,
                    failure_category TEXT
                );
                CREATE INDEX IF NOT EXISTS agent_admissions_state_sequence
                    ON agent_admissions(state, sequence);
                CREATE INDEX IF NOT EXISTS agent_admissions_owner_session
                    ON agent_admissions(user_id, session_id, sequence DESC);
                """
            )
        self.database.chmod(0o600)

    def _recover(self) -> None:
        """Never execute orphaned queued payloads or duplicate admitted work."""

        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            queued = connection.execute(
                "SELECT ticket_id FROM agent_admissions WHERE state='QUEUED'"
            ).fetchall()
            for row in queued:
                connection.execute(
                    """
                    UPDATE agent_admissions SET state='CANCELLED', completed_at_utc=?,
                        updated_at_utc=?, failure_category='scheduler_restart'
                    WHERE ticket_id=? AND state='QUEUED'
                    """,
                    (now, now, str(row["ticket_id"])),
                )
            running = connection.execute(
                "SELECT ticket_id, session_id, user_id FROM agent_admissions WHERE state='RUNNING'"
            ).fetchall()
            for row in running:
                recovered_state = "FAILED"
                reason = "scheduler_restart_non_resumable"
                try:
                    status = self.sandboxes.status(
                        str(row["session_id"]), user_id=str(row["user_id"])
                    )
                    if status.get("lifecycle_state") == "INTERRUPTED_RESUMABLE":
                        recovered_state = "RESUMABLE"
                        reason = "scheduler_restart_resumable"
                except Exception:
                    pass
                connection.execute(
                    """
                    UPDATE agent_admissions SET state=?, completed_at_utc=?,
                        updated_at_utc=?, failure_category=?
                    WHERE ticket_id=? AND state='RUNNING'
                    """,
                    (recovered_state, now, now, reason, str(row["ticket_id"])),
                )

    @staticmethod
    def _validate_identity(user_id: str, repo_id: str, session_id: str) -> None:
        identity = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
        if not identity.fullmatch(user_id) or not identity.fullmatch(session_id):
            raise AgentSchedulerError("agent_queue_identity_invalid")
        if not repo_id or len(repo_id) > 128 or "\x00" in repo_id:
            raise AgentSchedulerError("agent_queue_repository_invalid")

    def _head_ticket(self, connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            "SELECT ticket_id FROM agent_admissions WHERE state='QUEUED' ORDER BY sequence LIMIT 1"
        ).fetchone()
        return str(row["ticket_id"]) if row is not None else None

    @staticmethod
    def _running_count(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM agent_admissions WHERE state='RUNNING'"
        ).fetchone()
        return int(row["count"]) if row is not None else 0

    def wait_for_admission(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        instruction_digest: str,
        wait_timeout_seconds: float,
    ) -> AgentAdmission:
        self._validate_identity(user_id, repo_id, session_id)
        if not re.fullmatch(r"[a-f0-9]{64}", instruction_digest):
            raise AgentSchedulerError("agent_queue_instruction_digest_invalid")
        if not 0.01 <= wait_timeout_seconds <= 3_600:
            raise AgentSchedulerError("agent_queue_wait_timeout_invalid")

        # Reconciliation is bounded and precedes the allocation-free queue decision.
        self.sandboxes.reconcile(
            dry_run=False,
            user_id=user_id,
            limit=self.sandboxes.per_user_quota,
        )
        queued_at = time.monotonic()
        created = datetime.now(UTC)
        deadline = created + timedelta(seconds=wait_timeout_seconds)
        ticket_id = f"agentq-{uuid.uuid4().hex}"
        with self._condition:
            with self._connect() as connection:
                duplicate = connection.execute(
                    """
                    SELECT state FROM agent_admissions
                    WHERE user_id=? AND session_id=? AND state IN ('QUEUED','RUNNING')
                    ORDER BY sequence DESC LIMIT 1
                    """,
                    (user_id, session_id),
                ).fetchone()
                if duplicate is not None:
                    raise AgentSchedulerError(
                        "agent_task_duplicate_pending", {"state": str(duplicate["state"])}
                    )
                waiting = int(
                    connection.execute(
                        "SELECT COUNT(*) AS count FROM agent_admissions WHERE state='QUEUED'"
                    ).fetchone()["count"]
                )
                if waiting >= self.policy.queue_capacity:
                    raise AgentSchedulerError(
                        "agent_queue_full", {"limit": self.policy.queue_capacity}
                    )
                connection.execute(
                    """
                    INSERT INTO agent_admissions (
                        ticket_id, session_id, user_id, repo_id, instruction_digest,
                        topology, state, created_at_utc, updated_at_utc, deadline_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, 'QUEUED', ?, ?, ?)
                    """,
                    (
                        ticket_id,
                        session_id,
                        user_id,
                        repo_id,
                        instruction_digest,
                        self.policy.topology,
                        created.isoformat(),
                        created.isoformat(),
                        deadline.isoformat(),
                    ),
                )
                position = waiting + 1

        while True:
            # Never hold the scheduler condition while entering worktree code:
            # terminal release invokes a capacity notifier and would otherwise
            # create a condition/worktree lock inversion.
            self.sandboxes.reconcile(
                dry_run=False,
                user_id=user_id,
                limit=self.sandboxes.per_user_quota,
            )
            capacity = self.sandboxes.capacity_snapshot(user_id=user_id)
            reconciliation_delay = self.sandboxes.next_reconciliation_delay(
                user_id=user_id
            )
            with self._condition:
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT state, failure_category FROM agent_admissions WHERE ticket_id=?",
                        (ticket_id,),
                    ).fetchone()
                    if row is None:
                        raise AgentSchedulerError("agent_queue_ticket_lost")
                    state = str(row["state"])
                    if state == "CANCELLED":
                        category = str(row["failure_category"] or "agent_queue_cancelled")
                        raise AgentSchedulerError(category)
                    remaining = deadline.timestamp() - datetime.now(UTC).timestamp()
                    if remaining <= 0:
                        now = datetime.now(UTC).isoformat()
                        connection.execute(
                            """
                            UPDATE agent_admissions SET state='CANCELLED', completed_at_utc=?,
                                updated_at_utc=?, failure_category='agent_queue_wait_timeout'
                            WHERE ticket_id=? AND state='QUEUED'
                            """,
                            (now, now, ticket_id),
                        )
                        self._condition.notify_all()
                        raise AgentSchedulerError("agent_queue_wait_timeout")
                    can_admit = (
                        self._head_ticket(connection) == ticket_id
                        and self._running_count(connection) < self.policy.execution_slots
                        and capacity.get("available") is True
                    )
                    if can_admit:
                        now = datetime.now(UTC).isoformat()
                        updated = connection.execute(
                            """
                            UPDATE agent_admissions SET state='RUNNING', admitted_at_utc=?,
                                updated_at_utc=? WHERE ticket_id=? AND state='QUEUED'
                            """,
                            (now, now, ticket_id),
                        )
                        if updated.rowcount != 1:
                            continue
                        self._condition.notify_all()
                        return AgentAdmission(
                            ticket_id=ticket_id,
                            session_id=session_id,
                            queue_position_at_arrival=position,
                            queue_wait_seconds=max(0.0, time.monotonic() - queued_at),
                        )
                wake_after = (
                    min(remaining, max(0.01, reconciliation_delay))
                    if reconciliation_delay is not None
                    else remaining
                )
                self._condition.wait(timeout=wake_after)

    def finish(
        self,
        admission: AgentAdmission,
        *,
        state: str,
        result_id: str | None = None,
        failure_category: str | None = None,
    ) -> None:
        if state not in {"RESUMABLE", "SUCCEEDED", "FAILED", "CANCELLED"}:
            raise AgentSchedulerError("agent_scheduler_terminal_state_invalid")
        now = datetime.now(UTC).isoformat()
        with self._condition, self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_admissions SET state=?, result_id=?, failure_category=?,
                    completed_at_utc=?, updated_at_utc=?
                WHERE ticket_id=? AND state='RUNNING'
                """,
                (
                    state,
                    result_id,
                    failure_category,
                    now,
                    now,
                    admission.ticket_id,
                ),
            )
            self._condition.notify_all()

    def cancel(self, *, user_id: str, session_id: str) -> JsonObject:
        self._validate_identity(user_id, "repository", session_id)
        now = datetime.now(UTC).isoformat()
        with self._condition, self._connect() as connection:
            row = connection.execute(
                """
                SELECT ticket_id, state FROM agent_admissions
                WHERE user_id=? AND session_id=? ORDER BY sequence DESC LIMIT 1
                """,
                (user_id, session_id),
            ).fetchone()
            if row is None:
                raise AgentSchedulerError("agent_queue_task_not_found")
            state = str(row["state"])
            if state != "QUEUED":
                return {"status": state, "cancelled": False, "session_id": session_id}
            connection.execute(
                """
                UPDATE agent_admissions SET state='CANCELLED', completed_at_utc=?,
                    updated_at_utc=?, failure_category='agent_queue_cancelled'
                WHERE ticket_id=? AND state='QUEUED'
                """,
                (now, now, str(row["ticket_id"])),
            )
            self._condition.notify_all()
        return {"status": "CANCELLED", "cancelled": True, "session_id": session_id}

    def task_status(self, *, user_id: str, session_id: str) -> JsonObject:
        self._validate_identity(user_id, "repository", session_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_admissions WHERE user_id=? AND session_id=?
                ORDER BY sequence DESC LIMIT 1
                """,
                (user_id, session_id),
            ).fetchone()
            if row is None:
                raise AgentSchedulerError("agent_queue_task_not_found")
            position: int | None = None
            if str(row["state"]) == "QUEUED":
                position = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS count FROM agent_admissions
                        WHERE state='QUEUED' AND sequence<=?
                        """,
                        (int(row["sequence"]),),
                    ).fetchone()["count"]
                )
        return {
            "session_id": session_id,
            "repo_id": str(row["repo_id"]),
            "state": str(row["state"]),
            "queue_position": position,
            "topology": str(row["topology"]),
            "created_at_utc": str(row["created_at_utc"]),
            "admitted_at_utc": row["admitted_at_utc"],
            "completed_at_utc": row["completed_at_utc"],
            "result_id": row["result_id"],
            "failure_category": row["failure_category"],
        }

    def snapshot(self, *, user_id: str | None = None) -> JsonObject:
        with self._connect() as connection:
            counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM agent_admissions GROUP BY state"
                )
            }
        worktrees = self.sandboxes.capacity_snapshot(user_id=user_id)
        return {
            "runtime_topology": self.policy.topology,
            "agent_execution_slot_limit": self.policy.execution_slots,
            "running_agent_tasks": counts.get("RUNNING", 0),
            "queued_agent_tasks": counts.get("QUEUED", 0),
            "pending_queue_capacity": self.policy.queue_capacity,
            "capacity_certification": self.policy.certification,
            **worktrees,
        }
