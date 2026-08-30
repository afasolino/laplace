"""Durable run identity, append-only events, and local observability."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence, TypeAlias

if os.name == "nt":
    import msvcrt
else:
    import fcntl

JsonObject: TypeAlias = dict[str, object]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
_FORBIDDEN_TRACE_KEYS = frozenset(
    {"prompt", "response", "source", "source_code", "held_out", "secret", "token"}
)


class ExecutionRecordError(RuntimeError):
    """A durable execution record is unsafe or inconsistent."""


class RunIdentityConflict(ExecutionRecordError):
    """A project path already belongs to an incompatible run identity."""

    def __init__(self, evidence: JsonObject) -> None:
        super().__init__("run_identity_conflict: existing project identity is incompatible")
        self.evidence = evidence


class StageWorkflowInterrupted(ExecutionRecordError):
    """A deterministic fixture interruption occurred after a durable stage."""


def _lock_file(handle: BinaryIO) -> None:
    """Acquire an exclusive advisory lock using the host-native primitive."""

    if os.name == "nt":
        handle.seek(0)
        if not handle.read(1):
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_LOCK,  # type: ignore[attr-defined]
            1,
        )
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
            1,
        )
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# Explicit package-internal interfaces shared by durable append-only stores.
lock_file = _lock_file
unlock_file = _unlock_file


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    task_id: str
    arm_id: str
    configuration_sha256: str
    request_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("run_id", self.run_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ExecutionRecordError(f"Unsafe {label}")
        for label, value in (
            ("configuration_sha256", self.configuration_sha256),
            ("request_sha256", self.request_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ExecutionRecordError(f"{label} must be SHA-256")

    def to_json(self) -> JsonObject:
        return asdict(self)


class RunIdentityStore:
    """Create, resume, or return a terminal result for one project identity."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.identity_path = self.project_root / "run_identity.json"
        self.terminal_path = self.project_root / "result_compact.json"
        self.lock_path = self.project_root.parent / f".{self.project_root.name}.identity.lock"

    @staticmethod
    def project_path(output_root: Path, run_id: str) -> Path:
        if not _IDENTIFIER.fullmatch(run_id):
            raise ExecutionRecordError("Unsafe run_id")
        return output_root.resolve() / "runs" / run_id

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            _lock_file(handle)
            try:
                yield
            finally:
                _unlock_file(handle)

    def initialize(self, identity: RunIdentity) -> JsonObject:
        with self._locked():
            if self.identity_path.is_file():
                try:
                    existing_raw: object = json.loads(
                        self.identity_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise ExecutionRecordError("Persisted run identity is malformed") from exc
                if not isinstance(existing_raw, dict):
                    raise ExecutionRecordError("Persisted run identity is not an object")
                existing = dict(existing_raw)
                requested = identity.to_json()
                if existing != requested:
                    raise RunIdentityConflict(
                        {
                            "status": "run_identity_conflict",
                            "project_root": str(self.project_root),
                            "existing_identity": existing,
                            "requested_identity": requested,
                            "identity_path": str(self.identity_path),
                        }
                    )
                if self.terminal_path.is_file():
                    terminal_raw: object = json.loads(
                        self.terminal_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(terminal_raw, dict):
                        raise ExecutionRecordError("Terminal result is malformed")
                    return {
                        "status": "IDEMPOTENT_TERMINAL",
                        "identity": existing,
                        "result": terminal_raw,
                        "model_calls_executed": 0,
                        "eda_runs_executed": 0,
                        "events_appended": 0,
                    }
                return {"status": "RESUME", "identity": existing}
            self.project_root.mkdir(parents=True, exist_ok=False)
            _atomic_json(self.identity_path, identity.to_json())
            return {"status": "CREATED", "identity": identity.to_json()}

    def write_terminal(self, identity: RunIdentity, result: JsonObject) -> None:
        with self._locked():
            if not self.identity_path.is_file():
                raise ExecutionRecordError("Cannot finalize a run without identity")
            existing = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if existing != identity.to_json():
                raise RunIdentityConflict(
                    {
                        "status": "run_identity_conflict",
                        "project_root": str(self.project_root),
                        "existing_identity": existing,
                        "requested_identity": identity.to_json(),
                    }
                )
            if self.terminal_path.is_file():
                prior = json.loads(self.terminal_path.read_text(encoding="utf-8"))
                if prior != result:
                    raise ExecutionRecordError("Terminal result is immutable")
                return
            _atomic_json(self.terminal_path, result)


class AppendOnlyEventLog:
    """Process-safe JSONL event stream with deterministic duplicate suppression."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> None:
        for value in (run_id, task_id, arm_id):
            if not _IDENTIFIER.fullmatch(value):
                raise ExecutionRecordError("Unsafe event-stream identity")
        self.path = path.resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.run_id = run_id
        self.task_id = task_id
        self.arm_id = arm_id

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            _lock_file(handle)
            try:
                yield
            finally:
                _unlock_file(handle)

    def _recover_truncated_final_line(self) -> str | None:
        if not self.path.is_file():
            return None
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return None
        boundary = data.rfind(b"\n") + 1
        fragment = data[boundary:]
        digest = hashlib.sha256(fragment).hexdigest()
        recovery = self.path.with_name(f"{self.path.name}.truncated.{digest[:16]}.bin")
        if not recovery.exists():
            recovery.write_bytes(fragment)
        with self.path.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        return digest

    def _read_unlocked(self) -> list[JsonObject]:
        if not self.path.is_file():
            return []
        events: list[JsonObject] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue
            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExecutionRecordError(
                    f"Invalid event JSON at line {line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise ExecutionRecordError(f"Event line {line_number} is not an object")
            events.append(dict(raw))
        return events

    def read(self) -> list[JsonObject]:
        with self._locked():
            self._recover_truncated_final_line()
            return self._read_unlocked()

    def append(
        self,
        *,
        attempt: int,
        event_type: str,
        from_state: str | None,
        to_state: str | None,
        source_state_fingerprint: str | None,
        payload: JsonObject,
    ) -> JsonObject:
        if attempt < 0:
            raise ExecutionRecordError("Event attempt cannot be negative")
        if not _IDENTIFIER.fullmatch(event_type):
            raise ExecutionRecordError("Unsafe event type")
        if source_state_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", source_state_fingerprint
        ):
            raise ExecutionRecordError("Event source-state fingerprint must be SHA-256")
        payload_sha256 = canonical_sha256(payload)
        event_id = canonical_sha256(
            {
                "run_id": self.run_id,
                "task_id": self.task_id,
                "arm_id": self.arm_id,
                "attempt": attempt,
                "event_type": event_type,
                "from_state": from_state,
                "to_state": to_state,
                "source_state_fingerprint": source_state_fingerprint,
                "payload_sha256": payload_sha256,
            }
        )
        with self._locked():
            recovered_fragment = self._recover_truncated_final_line()
            events = self._read_unlocked()
            for existing_event in events:
                if existing_event.get("event_id") == event_id:
                    return {**existing_event, "deduplicated": True}
            event: JsonObject = {
                "event_id": event_id,
                "event_sequence": len(events) + 1,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "arm_id": self.arm_id,
                "attempt": attempt,
                "event_type": event_type,
                "from_state": from_state,
                "to_state": to_state,
                "source_state_fingerprint": source_state_fingerprint,
                "payload_sha256": payload_sha256,
                "timestamp": utc_now(),
                "payload": payload,
            }
            if recovered_fragment is not None:
                event["recovered_truncated_fragment_sha256"] = recovered_fragment
            encoded = canonical_json_bytes(event) + b"\n"
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            return event


class ResumableStageWorkflow:
    """Run ordered deterministic stages once and resume from durable outputs."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
        stages: Sequence[str],
    ) -> None:
        if not stages or len(set(stages)) != len(stages):
            raise ExecutionRecordError("Workflow stages must be non-empty and unique")
        if any(not _IDENTIFIER.fullmatch(stage) for stage in stages):
            raise ExecutionRecordError("Workflow stage name is unsafe")
        self.root = root.resolve()
        self.stages = tuple(stages)
        self.outputs = self.root / "stage_outputs"
        self.projection_path = self.root / "projection.json"
        self.events = AppendOnlyEventLog(
            self.root / "events.jsonl",
            run_id=run_id,
            task_id=task_id,
            arm_id=arm_id,
        )

    @staticmethod
    def _read_object(path: Path) -> JsonObject:
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionRecordError(f"Stage output is malformed: {path}") from exc
        if not isinstance(value, dict):
            raise ExecutionRecordError(f"Stage output is not an object: {path}")
        return dict(value)

    def run(
        self,
        handlers: Mapping[str, Callable[[JsonObject], JsonObject]],
        *,
        interrupt_after: str | None = None,
    ) -> JsonObject:
        if set(handlers) != set(self.stages):
            raise ExecutionRecordError("Stage handlers do not match the frozen workflow")
        if interrupt_after is not None and interrupt_after not in self.stages:
            raise ExecutionRecordError("Interruption stage is not in the workflow")
        projection: JsonObject = {}
        for index, stage in enumerate(self.stages):
            output_path = self.outputs / f"{index:02d}_{stage}.json"
            if output_path.is_file():
                output = self._read_object(output_path)
            else:
                output = handlers[stage](dict(projection))
                if not isinstance(output, dict):
                    raise ExecutionRecordError(f"Stage {stage} did not return an object")
                _atomic_json(output_path, output)
            prior_stage = self.stages[index - 1] if index else None
            self.events.append(
                attempt=0,
                event_type="stage_completed",
                from_state=prior_stage,
                to_state=stage,
                source_state_fingerprint=None,
                payload={"stage": stage, "output_sha256": canonical_sha256(output)},
            )
            projection[stage] = output
            if interrupt_after == stage:
                raise StageWorkflowInterrupted(f"interrupted_after:{stage}")
        _atomic_json(self.projection_path, projection)
        return projection


@dataclass
class _Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    started_at: str
    started_monotonic: float
    attributes: JsonObject


class LocalTraceRecorder:
    """Small OpenTelemetry-compatible local JSONL exporter with a no-op mode."""

    def __init__(
        self,
        path: Path,
        *,
        trace_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.path = path.resolve()
        self.trace_id = trace_id or uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", self.trace_id):
            raise ExecutionRecordError("trace_id must be 32 lowercase hexadecimal characters")
        self.enabled = enabled
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.metrics_path = self.path.with_name("metrics.json")
        self._metrics: dict[str, int] = {}

    @staticmethod
    def _bounded_attributes(attributes: Mapping[str, object]) -> JsonObject:
        if len(attributes) > 32:
            raise ExecutionRecordError("A trace span may have at most 32 attributes")
        bounded: JsonObject = {}
        for key, value in attributes.items():
            normalized = key.lower()
            if any(fragment in normalized for fragment in _FORBIDDEN_TRACE_KEYS):
                raise ExecutionRecordError(f"Sensitive trace attribute is forbidden: {key}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key):
                raise ExecutionRecordError(f"Unsafe trace attribute name: {key}")
            if isinstance(value, str):
                bounded[key] = value[:256]
            elif isinstance(value, (bool, int, float)) or value is None:
                bounded[key] = value
            else:
                bounded[key] = canonical_sha256(value)
        return bounded

    def _append(self, record: JsonObject) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            _lock_file(lock)
            try:
                with self.path.open("ab") as handle:
                    handle.write(canonical_json_bytes(record) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                _unlock_file(lock)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        parent_span_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[_Span]:
        if not _IDENTIFIER.fullmatch(name):
            raise ExecutionRecordError("Unsafe span name")
        span = _Span(
            name=name,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
            attributes=self._bounded_attributes(attributes or {}),
        )
        status = "OK"
        error_type: str | None = None
        try:
            yield span
        except Exception as exc:
            status = "ERROR"
            error_type = exc.__class__.__name__
            raise
        finally:
            record: JsonObject = {
                "schema_version": 1,
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "start_time": span.started_at,
                "end_time": utc_now(),
                "duration_seconds": round(time.monotonic() - span.started_monotonic, 9),
                "status": status,
                "attributes": span.attributes,
            }
            if error_type is not None:
                record["error_type"] = error_type
            self._append(record)
            self._metrics[name] = self._metrics.get(name, 0) + 1
            if self.enabled:
                _atomic_json(
                    self.metrics_path,
                    {
                        "schema_version": 1,
                        "trace_id": self.trace_id,
                        "span_counts": dict(sorted(self._metrics.items())),
                    },
                )
