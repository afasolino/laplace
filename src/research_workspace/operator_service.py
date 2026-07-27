"""Typed local Operator Plane service shared by CLI and HTTP adapters."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, TypeAlias

from .execution_records import (
    RunIdentity,
    RunIdentityStore,
    canonical_json_bytes,
    canonical_sha256,
)
from .model_servers import ModelServerController
from .notifications import LocalNotificationAdapter

JsonObject: TypeAlias = dict[str, object]
RunExecutor = Callable[[RunIdentity, JsonObject, Path], JsonObject]

_ROLES = frozenset({"read", "operate", "approve", "admin"})
_OPERATORS = frozenset({"operate", "admin"})
_APPROVERS = frozenset({"approve", "admin"})
_APPROVAL_ACTIONS = frozenset(
    {
        "START_GPU_RUN",
        "START_MODEL_SERVERS",
        "STOP_MODEL_SERVERS",
        "PROMOTE_RESEARCH_SOURCE",
        "REVIEWER_OVERRIDE",
        "PUBLISH_BUNDLE",
    }
)
_RUN_STATES = frozenset({"PREPARED", "QUEUED", "RUNNING", "COMPLETE", "FAILED"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OperatorServiceError(RuntimeError):
    """A typed Operator Plane action failed validation or authorization."""

    def __init__(self, category: str, evidence: JsonObject) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_role(role: str, allowed: frozenset[str]) -> None:
    if role not in _ROLES:
        raise OperatorServiceError(
            "authorization_failure", {"reason": "unknown role"}
        )
    if role not in allowed:
        raise OperatorServiceError(
            "authorization_failure",
            {"reason": "role is not permitted for this action", "role": role},
        )


def _row_object(row: sqlite3.Row) -> JsonObject:
    return {key: row[key] for key in row.keys()}


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    configuration_sha256: str
    state: str
    project_path: str

    def to_json(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "configuration_sha256": self.configuration_sha256,
            "state": self.state,
            "project_path": self.project_path,
        }


class OperatorService:
    """Local metadata, approvals, and immutable run activation."""

    def __init__(
        self,
        repository_root: Path,
        state_root: Path,
        *,
        model_servers: ModelServerController | None = None,
        notifier: LocalNotificationAdapter | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.state_root = state_root.resolve()
        self.database = self.state_root / "operator.sqlite3"
        self.output_root = self.state_root / "execution"
        self.model_servers = model_servers or ModelServerController(
            self.repository_root,
            self.state_root / "model_servers",
        )
        self.notifier = notifier or LocalNotificationAdapter()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    configuration_sha256 TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    gpu_required INTEGER NOT NULL,
                    project_path TEXT NOT NULL UNIQUE,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    terminal_result_json TEXT
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    requested_by_role TEXT NOT NULL,
                    decided_by_role TEXT,
                    created_at_utc TEXT NOT NULL,
                    decided_at_utc TEXT
                );
                CREATE TABLE IF NOT EXISTS operator_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    timestamp_utc TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def _event(
        self,
        connection: sqlite3.Connection,
        *,
        actor_role: str,
        action: str,
        entity_type: str,
        entity_id: str,
        payload: Mapping[str, object],
    ) -> None:
        serialized = canonical_json_bytes(dict(payload)).decode("utf-8")
        digest = canonical_sha256(dict(payload))
        connection.execute(
            """
            INSERT INTO operator_events (
                event_id, timestamp_utc, actor_role, action, entity_type,
                entity_id, payload_sha256, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"evt-{uuid.uuid4().hex}",
                _now(),
                actor_role,
                action,
                entity_type,
                entity_id,
                digest,
                serialized,
            ),
        )

    @staticmethod
    def _validate_run_configuration(configuration: Mapping[str, object]) -> JsonObject:
        allowed = {
            "task_id",
            "arm_id",
            "model_route",
            "corpus_snapshot_sha256",
            "skills_lock_sha256",
            "smoke_profile",
            "request_sha256",
            "gpu_required",
        }
        unexpected = sorted(set(configuration).difference(allowed))
        if unexpected:
            raise OperatorServiceError(
                "invalid_run_configuration",
                {"reason": "unexpected fields", "fields": unexpected},
            )
        required_text = (
            "task_id",
            "arm_id",
            "model_route",
            "smoke_profile",
        )
        result: JsonObject = {}
        for key in required_text:
            value = configuration.get(key)
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 160
                or not re.fullmatch(r"[A-Za-z0-9._:/+-]+", value)
            ):
                raise OperatorServiceError(
                    "invalid_run_configuration", {"reason": f"invalid {key}"}
                )
            result[key] = value
        for key in (
            "corpus_snapshot_sha256",
            "skills_lock_sha256",
            "request_sha256",
        ):
            value = configuration.get(key)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise OperatorServiceError(
                    "invalid_run_configuration", {"reason": f"invalid {key}"}
                )
            result[key] = value
        gpu_required = configuration.get("gpu_required")
        if not isinstance(gpu_required, bool):
            raise OperatorServiceError(
                "invalid_run_configuration", {"reason": "gpu_required must be boolean"}
            )
        result["gpu_required"] = gpu_required
        return result

    def prepare_run(
        self,
        configuration: Mapping[str, object],
        *,
        actor_role: str,
        run_id: str | None = None,
    ) -> JsonObject:
        _require_role(actor_role, _OPERATORS)
        frozen = self._validate_run_configuration(configuration)
        selected_run_id = run_id or (
            f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
        )
        identity = RunIdentity(
            run_id=selected_run_id,
            task_id=str(frozen["task_id"]),
            arm_id=str(frozen["arm_id"]),
            configuration_sha256=canonical_sha256(frozen),
            request_sha256=str(frozen["request_sha256"]),
        )
        project_path = RunIdentityStore.project_path(self.output_root, selected_run_id)
        now = _now()
        prepared = PreparedRun(
            run_id=selected_run_id,
            configuration_sha256=identity.configuration_sha256,
            state="PREPARED",
            project_path=str(project_path),
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (selected_run_id,)
                ).fetchone()
                if existing is not None:
                    if (
                        existing["configuration_sha256"]
                        == identity.configuration_sha256
                        and existing["request_sha256"] == identity.request_sha256
                    ):
                        result = _row_object(existing)
                        result["status"] = "IDEMPOTENT_EXISTING_RUN"
                        return result
                    raise OperatorServiceError(
                        "run_identity_conflict",
                        {
                            "run_id": selected_run_id,
                            "existing_configuration_sha256": existing[
                                "configuration_sha256"
                            ],
                            "requested_configuration_sha256": (
                                identity.configuration_sha256
                            ),
                        },
                    )
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, task_id, arm_id, configuration_sha256,
                        request_sha256, configuration_json, state, gpu_required,
                        project_path, created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity.run_id,
                        identity.task_id,
                        identity.arm_id,
                        identity.configuration_sha256,
                        identity.request_sha256,
                        canonical_json_bytes(frozen).decode("utf-8"),
                        "PREPARED",
                        int(bool(frozen["gpu_required"])),
                        str(project_path),
                        now,
                        now,
                    ),
                )
                self._event(
                    connection,
                    actor_role=actor_role,
                    action="RUN_PREPARED",
                    entity_type="run",
                    entity_id=selected_run_id,
                    payload=prepared.to_json(),
                )
        except sqlite3.IntegrityError as exc:
            raise OperatorServiceError(
                "run_identity_conflict",
                {"run_id": selected_run_id, "reason": "project path is already assigned"},
            ) from exc
        return {"status": "PREPARED", **prepared.to_json()}

    def request_approval(
        self,
        action: str,
        entity_id: str,
        payload: Mapping[str, object],
        *,
        actor_role: str,
    ) -> JsonObject:
        _require_role(actor_role, _OPERATORS)
        if action not in _APPROVAL_ACTIONS:
            raise OperatorServiceError(
                "invalid_approval_action", {"action": action}
            )
        payload_object = dict(payload)
        payload_sha256 = canonical_sha256(payload_object)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM approvals
                WHERE action = ? AND entity_id = ? AND payload_sha256 = ?
                  AND state IN ('PENDING', 'APPROVED')
                ORDER BY created_at_utc DESC LIMIT 1
                """,
                (action, entity_id, payload_sha256),
            ).fetchone()
            if existing is not None:
                result = _row_object(existing)
                result["status"] = "IDEMPOTENT_EXISTING_APPROVAL"
                return result
            approval_id = f"approval-{uuid.uuid4().hex}"
            now = _now()
            connection.execute(
                """
                INSERT INTO approvals (
                    approval_id, action, entity_id, payload_sha256, payload_json,
                    state, requested_by_role, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    approval_id,
                    action,
                    entity_id,
                    payload_sha256,
                    canonical_json_bytes(payload_object).decode("utf-8"),
                    actor_role,
                    now,
                ),
            )
            self._event(
                connection,
                actor_role=actor_role,
                action="APPROVAL_REQUESTED",
                entity_type="approval",
                entity_id=approval_id,
                payload={
                    "action": action,
                    "entity_id": entity_id,
                    "payload_sha256": payload_sha256,
                },
            )
        notification = self.notifier.send(
            "APPROVAL_REQUIRED",
            {"approval_id": approval_id, "status": "PENDING"},
        )
        return {
            "status": "PENDING",
            "approval_id": approval_id,
            "action": action,
            "entity_id": entity_id,
            "payload_sha256": payload_sha256,
            "notification": notification,
        }

    def decide_approval(
        self,
        approval_id: str,
        *,
        approve: bool,
        actor_role: str,
    ) -> JsonObject:
        _require_role(actor_role, _APPROVERS)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise OperatorServiceError(
                    "approval_not_found", {"approval_id": approval_id}
                )
            decision = "APPROVED" if approve else "REJECTED"
            if row["state"] != "PENDING":
                if row["state"] == decision:
                    result = _row_object(row)
                    result["status"] = "IDEMPOTENT_EXISTING_DECISION"
                    return result
                raise OperatorServiceError(
                    "approval_state_conflict",
                    {
                        "approval_id": approval_id,
                        "existing_state": row["state"],
                        "requested_state": decision,
                    },
                )
            now = _now()
            connection.execute(
                """
                UPDATE approvals
                SET state = ?, decided_by_role = ?, decided_at_utc = ?
                WHERE approval_id = ?
                """,
                (decision, actor_role, now, approval_id),
            )
            self._event(
                connection,
                actor_role=actor_role,
                action=f"APPROVAL_{decision}",
                entity_type="approval",
                entity_id=approval_id,
                payload={"decision": decision},
            )
        return {"status": decision, "approval_id": approval_id}

    def _approved(
        self,
        *,
        approval_id: str | None,
        action: str,
        entity_id: str,
    ) -> bool:
        if approval_id is None:
            return False
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state FROM approvals
                WHERE approval_id = ? AND action = ? AND entity_id = ?
                """,
                (approval_id, action, entity_id),
            ).fetchone()
        return row is not None and row["state"] == "APPROVED"

    def approval_is_valid(
        self,
        approval_id: str,
        action: str,
        entity_id: str,
        *,
        actor_role: str,
    ) -> bool:
        """Read-only approval check used by other typed local services."""

        _require_role(actor_role, _ROLES)
        return self._approved(
            approval_id=approval_id,
            action=action,
            entity_id=entity_id,
        )

    def start_run(
        self,
        run_id: str,
        *,
        approval_id: str | None,
        actor_role: str,
        executor: RunExecutor | None = None,
    ) -> JsonObject:
        _require_role(actor_role, _OPERATORS)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise OperatorServiceError("run_not_found", {"run_id": run_id})
        if row["state"] in {"QUEUED", "RUNNING", "COMPLETE", "FAILED"}:
            result = _row_object(row)
            result["status"] = "IDEMPOTENT_EXISTING_STATE"
            return result
        if row["gpu_required"] and not self._approved(
            approval_id=approval_id,
            action="START_GPU_RUN",
            entity_id=run_id,
        ):
            raise OperatorServiceError(
                "approval_required",
                {"run_id": run_id, "required_action": "START_GPU_RUN"},
            )
        configuration_raw: object = json.loads(row["configuration_json"])
        if not isinstance(configuration_raw, dict):
            raise OperatorServiceError(
                "operator_state_corrupt", {"run_id": run_id}
            )
        identity = RunIdentity(
            run_id=run_id,
            task_id=row["task_id"],
            arm_id=row["arm_id"],
            configuration_sha256=row["configuration_sha256"],
            request_sha256=row["request_sha256"],
        )
        project_path = Path(row["project_path"])
        initialization = RunIdentityStore(project_path).initialize(identity)
        target_state = "RUNNING" if executor is not None else "QUEUED"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE runs SET state = ?, updated_at_utc = ? WHERE run_id = ?",
                (target_state, _now(), run_id),
            )
            self._event(
                connection,
                actor_role=actor_role,
                action=f"RUN_{target_state}",
                entity_type="run",
                entity_id=run_id,
                payload={"identity_initialization": initialization["status"]},
            )
        if executor is None:
            return {
                "status": "QUEUED",
                "run_id": run_id,
                "identity_initialization": initialization,
            }
        try:
            terminal = executor(identity, dict(configuration_raw), project_path)
        except Exception as exc:
            failure = {
                "status": "FAILED",
                "failure_category": "run_executor_failure",
                "error_type": type(exc).__name__,
            }
            self.finalize_run(
                run_id, failure, state="FAILED", actor_role="admin"
            )
            raise
        return self.finalize_run(
            run_id, terminal, state="COMPLETE", actor_role="admin"
        )

    def finalize_run(
        self,
        run_id: str,
        result: Mapping[str, object],
        *,
        state: str,
        actor_role: str,
    ) -> JsonObject:
        _require_role(actor_role, _OPERATORS)
        if state not in {"COMPLETE", "FAILED"}:
            raise OperatorServiceError(
                "invalid_run_state", {"state": state}
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise OperatorServiceError("run_not_found", {"run_id": run_id})
            prior_result = row["terminal_result_json"]
            serialized = canonical_json_bytes(dict(result)).decode("utf-8")
            if row["state"] in {"COMPLETE", "FAILED"}:
                if row["state"] == state and prior_result == serialized:
                    return {
                        "status": "IDEMPOTENT_TERMINAL",
                        "run_id": run_id,
                        "result": dict(result),
                    }
                raise OperatorServiceError(
                    "terminal_result_conflict", {"run_id": run_id}
                )
            identity = RunIdentity(
                run_id=run_id,
                task_id=row["task_id"],
                arm_id=row["arm_id"],
                configuration_sha256=row["configuration_sha256"],
                request_sha256=row["request_sha256"],
            )
            RunIdentityStore(Path(row["project_path"])).write_terminal(
                identity, dict(result)
            )
            connection.execute(
                """
                UPDATE runs
                SET state = ?, terminal_result_json = ?, updated_at_utc = ?
                WHERE run_id = ?
                """,
                (state, serialized, _now(), run_id),
            )
            self._event(
                connection,
                actor_role=actor_role,
                action=f"RUN_{state}",
                entity_type="run",
                entity_id=run_id,
                payload={"result_sha256": canonical_sha256(dict(result))},
            )
        notification_event = "RUN_COMPLETE" if state == "COMPLETE" else "TERMINAL_FAILURE"
        notification = self.notifier.send(
            notification_event, {"run_id": run_id, "status": state}
        )
        return {
            "status": state,
            "run_id": run_id,
            "result": dict(result),
            "notification": notification,
        }

    def model_server_action(
        self,
        action: str,
        *,
        approval_id: str | None,
        actor_role: str,
    ) -> JsonObject:
        if action == "status":
            _require_role(actor_role, _ROLES)
            return self.model_servers.status()
        _require_role(actor_role, _OPERATORS)
        approval_action = {
            "start": "START_MODEL_SERVERS",
            "stop": "STOP_MODEL_SERVERS",
        }.get(action)
        if approval_action is None:
            raise OperatorServiceError(
                "invalid_model_server_action", {"action": action}
            )
        if not self._approved(
            approval_id=approval_id,
            action=approval_action,
            entity_id="phase3",
        ):
            raise OperatorServiceError(
                "approval_required", {"required_action": approval_action}
            )
        if action == "start":
            result = self.model_servers.start()
        else:
            result = self.model_servers.release_owned()
        with self._connect() as connection:
            self._event(
                connection,
                actor_role=actor_role,
                action=f"MODEL_SERVERS_{action.upper()}",
                entity_type="model_servers",
                entity_id="phase3",
                payload={"status": result.get("status")},
            )
        if action == "stop" and result.get("status") == "RELEASED_LAPLACE_OWNED_SERVERS":
            self.notifier.send("GPU_RELEASED", {"status": str(result["status"])})
        return result

    def get_run(self, run_id: str, *, actor_role: str) -> JsonObject:
        _require_role(actor_role, _ROLES)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise OperatorServiceError("run_not_found", {"run_id": run_id})
        result = _row_object(row)
        result.pop("configuration_json", None)
        result.pop("terminal_result_json", None)
        return result

    def events(
        self,
        *,
        actor_role: str,
        after_sequence: int = 0,
        limit: int = 200,
    ) -> list[JsonObject]:
        _require_role(actor_role, _ROLES)
        if after_sequence < 0 or limit < 1 or limit > 500:
            raise OperatorServiceError(
                "invalid_event_cursor",
                {"after_sequence": after_sequence, "limit": limit},
            )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_id, timestamp_utc, actor_role, action,
                       entity_type, entity_id, payload_sha256
                FROM operator_events
                WHERE sequence > ?
                ORDER BY sequence ASC LIMIT ?
                """,
                (after_sequence, limit),
            ).fetchall()
        return [_row_object(row) for row in rows]

    def approvals(
        self,
        *,
        actor_role: str,
        state: str | None = None,
    ) -> list[JsonObject]:
        _require_role(actor_role, _ROLES)
        if state is not None and state not in {"PENDING", "APPROVED", "REJECTED"}:
            raise OperatorServiceError(
                "invalid_approval_state", {"state": state}
            )
        with self._connect() as connection:
            if state is None:
                rows = connection.execute(
                    """
                    SELECT approval_id, action, entity_id, payload_sha256, state,
                           requested_by_role, decided_by_role, created_at_utc,
                           decided_at_utc
                    FROM approvals ORDER BY created_at_utc DESC LIMIT 200
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT approval_id, action, entity_id, payload_sha256, state,
                           requested_by_role, decided_by_role, created_at_utc,
                           decided_at_utc
                    FROM approvals WHERE state = ?
                    ORDER BY created_at_utc DESC LIMIT 200
                    """,
                    (state,),
                ).fetchall()
        return [_row_object(row) for row in rows]

    def record_action(
        self,
        *,
        actor_role: str,
        action: str,
        entity_type: str,
        entity_id: str,
        payload: Mapping[str, object],
    ) -> JsonObject:
        """Append an action performed by another typed local service."""

        _require_role(actor_role, _OPERATORS)
        if not re.fullmatch(r"[A-Z0-9_]{1,80}", action):
            raise OperatorServiceError(
                "invalid_operator_action", {"action": action}
            )
        if not re.fullmatch(r"[a-z_]{1,40}", entity_type):
            raise OperatorServiceError(
                "invalid_operator_action", {"entity_type": entity_type}
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._event(
                connection,
                actor_role=actor_role,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                payload=payload,
            )
            sequence = int(
                connection.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]
            )
        return {"status": "RECORDED", "event_sequence": sequence}

    def summary(self, *, actor_role: str) -> JsonObject:
        _require_role(actor_role, _ROLES)
        with self._connect() as connection:
            run_counts = {
                row["state"]: row["count"]
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM runs GROUP BY state"
                )
            }
            approvals = {
                row["state"]: row["count"]
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM approvals GROUP BY state"
                )
            }
            recent = [
                _row_object(row)
                for row in connection.execute(
                    """
                    SELECT run_id, task_id, arm_id, configuration_sha256, state,
                           gpu_required, created_at_utc, updated_at_utc
                    FROM runs ORDER BY created_at_utc DESC LIMIT 20
                    """
                )
            ]
            event_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM operator_events"
                ).fetchone()[0]
            )
        return {
            "schema_version": 1,
            "status": "OK",
            "local_only": True,
            "run_counts": run_counts,
            "approval_counts": approvals,
            "recent_runs": recent,
            "event_count": event_count,
        }
