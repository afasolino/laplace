"""Typed, local lifecycle hooks for Laplace.

This module deliberately does not discover files, import plugins, invoke
subprocesses, or interpret model output.  A hook is an in-process Python
callback registered by the host, with a bounded timeout and a declared
failure policy.  Security-sensitive pre hooks fail closed; post hooks are
observability-only and their failures are recorded without changing the
operation result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
HookCallback: TypeAlias = Callable[["HookContext"], "HookResult | None"]


class HookStage(str, Enum):
    TASK_START = "TASK_START"
    TASK_RESUME = "TASK_RESUME"
    TASK_CANCEL = "TASK_CANCEL"
    PRE_RETRIEVAL = "PRE_RETRIEVAL"
    POST_RETRIEVAL = "POST_RETRIEVAL"
    PRE_MEMORY_WRITE = "PRE_MEMORY_WRITE"
    POST_MEMORY_WRITE = "POST_MEMORY_WRITE"
    PRE_MUTATION = "PRE_MUTATION"
    POST_MUTATION = "POST_MUTATION"
    PRE_VERIFY = "PRE_VERIFY"
    POST_VERIFY = "POST_VERIFY"
    TASK_COMPLETE = "TASK_COMPLETE"
    TASK_FAILURE = "TASK_FAILURE"
    IDLE_START = "IDLE_START"
    IDLE_END = "IDLE_END"


class HookFailurePolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"
    OBSERVE = "observe"


class HookError(RuntimeError):
    """Base class for fail-closed hook errors."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class HookBlockedError(HookError):
    """A hook explicitly blocked a security-sensitive operation."""


class HookSecurityError(HookError):
    """A security-sensitive hook failed, timed out, or was cancelled."""


class HookCancelledError(HookError):
    """A cancellable hook dispatch observed task cancellation."""


@dataclass(frozen=True)
class HookEvent:
    """Owner-bound event delivered to a typed callback."""

    stage: HookStage
    event_id: str
    idempotency_key: str
    owner_id: str
    project_id: str
    session_id: str
    task_id: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class HookResult:
    """The only callback result accepted by the hook engine."""

    cancel: bool = False
    context_modification: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class HookContext:
    """Read-only callback view with a shared cancellation signal."""

    stage: HookStage
    event_id: str
    idempotency_key: str
    owner_id: str
    project_id: str
    session_id: str
    task_id: str
    payload: Mapping[str, object]
    cancel_event: threading.Event


@dataclass(frozen=True)
class HookFailure:
    hook_name: str
    category: str
    message: str

    def to_json(self) -> JsonObject:
        return {
            "hook_name": self.hook_name,
            "category": self.category,
            "message": self.message,
        }


@dataclass(frozen=True)
class HookReport:
    """Stable result of one idempotent event dispatch."""

    stage: HookStage
    event_id: str
    idempotency_key: str
    replayed: bool
    continued: bool
    blocked: bool
    cancelled: bool
    executed: tuple[str, ...]
    failures: tuple[HookFailure, ...]
    context_modifications: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "stage": self.stage.value,
            "event_id": self.event_id,
            "idempotency_key": self.idempotency_key,
            "replayed": self.replayed,
            "continued": self.continued,
            "blocked": self.blocked,
            "cancelled": self.cancelled,
            "executed": list(self.executed),
            "failures": [failure.to_json() for failure in self.failures],
            "context_modifications": list(self.context_modifications),
        }


@dataclass(frozen=True)
class _HookRegistration:
    name: str
    stage: HookStage
    callback: HookCallback
    owner_id: str | None
    project_id: str | None
    priority: int
    timeout_seconds: float
    enabled: bool
    security_critical: bool
    failure_policy: HookFailurePolicy
    ordinal: int


_PRE_STAGES = frozenset(
    {
        HookStage.PRE_RETRIEVAL,
        HookStage.PRE_MEMORY_WRITE,
        HookStage.PRE_MUTATION,
        HookStage.PRE_VERIFY,
    }
)
_POST_STAGES = frozenset(
    {HookStage.POST_RETRIEVAL, HookStage.POST_MEMORY_WRITE, HookStage.POST_MUTATION, HookStage.POST_VERIFY}
)
_STAGES = frozenset(HookStage)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _validate_payload(value: object, *, depth: int = 0) -> None:
    if depth > 8:
        raise HookError("hook_payload_too_deep")
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise HookError("hook_payload_nonfinite")
        if isinstance(value, str) and len(value) > 16_384:
            raise HookError("hook_payload_string_too_large")
        return
    if isinstance(value, Mapping):
        if len(value) > 256 or not all(isinstance(key, str) and key for key in value):
            raise HookError("hook_payload_mapping_invalid")
        for key, item in value.items():
            if len(key) > 256:
                raise HookError("hook_payload_key_too_large")
            _validate_payload(item, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 256:
            raise HookError("hook_payload_sequence_too_large")
        for item in value:
            _validate_payload(item, depth=depth + 1)
        return
    raise HookError("hook_payload_type_invalid")


class HookService:
    """Deterministic, persistent, typed lifecycle hook dispatcher."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, *, default_timeout_seconds: float = 1.0) -> None:
        self.root = root.resolve()
        self.state_path = self.root / "events.json"
        if not 0.001 <= default_timeout_seconds <= 30.0:
            raise HookError("hook_timeout_out_of_range")
        self.default_timeout_seconds = default_timeout_seconds
        self._registrations: dict[str, _HookRegistration] = {}
        self._next_ordinal = 0
        self._events: dict[str, JsonObject] = {}
        self._revision = 0
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw: object = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HookError("hook_state_unreadable") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "revision", "events"}:
            raise HookError("hook_state_invalid")
        if raw["schema_version"] != self.SCHEMA_VERSION or not isinstance(raw["revision"], int):
            raise HookError("hook_state_schema_unsupported")
        events = raw["events"]
        if not isinstance(events, dict):
            raise HookError("hook_state_events_invalid")
        for key, value in events.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise HookError("hook_state_event_invalid")
            self._validate_stored_event(value)
        self._events = {str(key): cast(JsonObject, value) for key, value in events.items()}
        self._revision = raw["revision"]

    @staticmethod
    def _validate_stored_event(value: Mapping[str, object]) -> None:
        expected = {
            "stage",
            "event_id",
            "idempotency_key",
            "owner_id",
            "project_id",
            "session_id",
            "task_id",
            "payload_sha256",
            "replayed",
            "continued",
            "blocked",
            "cancelled",
            "executed",
            "failures",
            "context_modifications",
        }
        if set(value) != expected:
            raise HookError("hook_state_event_keys_invalid")
        if value["stage"] not in {stage.value for stage in HookStage}:
            raise HookError("hook_state_event_stage_invalid")
        for key in ("event_id", "idempotency_key", "owner_id", "project_id", "session_id", "task_id"):
            if not isinstance(value[key], str) or not value[key]:
                raise HookError("hook_state_event_identity_invalid")
        if not isinstance(value["payload_sha256"], str) or len(value["payload_sha256"]) != 64:
            raise HookError("hook_state_event_hash_invalid")
        for key in ("replayed", "continued", "blocked", "cancelled"):
            if not isinstance(value[key], bool):
                raise HookError("hook_state_event_flag_invalid")
        if not isinstance(value["executed"], list) or not isinstance(value["failures"], list):
            raise HookError("hook_state_event_result_invalid")
        if not all(isinstance(item, str) for item in cast(list[object], value["executed"])):
            raise HookError("hook_state_event_executed_invalid")
        if not all(isinstance(item, dict) for item in cast(list[object], value["failures"])):
            raise HookError("hook_state_event_failures_invalid")
        if not isinstance(value["context_modifications"], list) or not all(
            isinstance(item, str) for item in cast(list[object], value["context_modifications"])
        ):
            raise HookError("hook_state_event_context_invalid")

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload: JsonObject = {
            "schema_version": self.SCHEMA_VERSION,
            "revision": self._revision + 1,
            "events": {key: self._events[key] for key in sorted(self._events)},
        }
        temporary = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8"))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        except OSError as exc:
            raise HookError("hook_state_persist_failed") from exc
        self._revision += 1

    @staticmethod
    def _scope_matches(registration: _HookRegistration, event: HookEvent) -> bool:
        return (
            registration.owner_id is None
            or (
                registration.owner_id == event.owner_id
                and (registration.project_id is None or registration.project_id == event.project_id)
            )
        )

    def register(
        self,
        name: str,
        stage: HookStage,
        callback: HookCallback,
        *,
        owner_id: str | None = None,
        project_id: str | None = None,
        priority: int = 0,
        timeout_seconds: float | None = None,
        enabled: bool = True,
        security_critical: bool | None = None,
        failure_policy: HookFailurePolicy | None = None,
    ) -> None:
        """Register one typed callback; no file or command hook is accepted."""

        if not _SAFE_NAME.fullmatch(name) or name in self._registrations:
            raise HookError("hook_name_invalid_or_duplicate")
        if stage not in _STAGES or not callable(callback):
            raise HookError("hook_registration_invalid")
        if owner_id is not None and (not isinstance(owner_id, str) or not owner_id):
            raise HookError("hook_owner_invalid")
        if project_id is not None and (not isinstance(project_id, str) or not project_id):
            raise HookError("hook_project_invalid")
        if project_id is not None and owner_id is None:
            raise HookError("project_hook_requires_owner")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < -10_000 or priority > 10_000:
            raise HookError("hook_priority_invalid")
        timeout = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.001 <= timeout <= 30.0:
            raise HookError("hook_timeout_out_of_range")
        if not isinstance(enabled, bool):
            raise HookError("hook_enabled_invalid")
        if security_critical is not None and not isinstance(security_critical, bool):
            raise HookError("hook_security_flag_invalid")
        if failure_policy is not None and not isinstance(failure_policy, HookFailurePolicy):
            raise HookError("hook_failure_policy_invalid")
        critical = stage in _PRE_STAGES if security_critical is None else security_critical
        policy = (
            HookFailurePolicy.FAIL_CLOSED if critical else HookFailurePolicy.OBSERVE
        ) if failure_policy is None else failure_policy
        if stage in _POST_STAGES and (critical or policy is not HookFailurePolicy.OBSERVE):
            raise HookError("post_hook_must_be_observability_only")
        if critical and stage in _PRE_STAGES and policy is not HookFailurePolicy.FAIL_CLOSED:
            raise HookError("security_pre_hook_must_fail_closed")
        self._registrations[name] = _HookRegistration(
            name=name,
            stage=stage,
            callback=callback,
            owner_id=owner_id,
            project_id=project_id,
            priority=priority,
            timeout_seconds=timeout,
            enabled=enabled,
            security_critical=critical,
            failure_policy=policy,
            ordinal=self._next_ordinal,
        )
        self._next_ordinal += 1

    def set_enabled(self, name: str, enabled: bool) -> None:
        registration = self._registrations.get(name)
        if registration is None:
            raise HookError("hook_not_found")
        self._registrations[name] = replace(registration, enabled=enabled)

    def registrations(self, *, stage: HookStage | None = None) -> tuple[str, ...]:
        values = (
            registration
            for registration in self._registrations.values()
            if stage is None or registration.stage is stage
        )
        return tuple(
            registration.name
            for registration in sorted(values, key=lambda item: (item.priority, item.ordinal, item.name))
        )

    @staticmethod
    def _event_hash(event: HookEvent) -> str:
        return hashlib.sha256(_canonical(dict(event.payload))).hexdigest()

    @staticmethod
    def _validate_event(event: HookEvent) -> None:
        if not isinstance(event.stage, HookStage):
            raise HookError("hook_event_stage_invalid")
        for value in (
            event.event_id,
            event.idempotency_key,
            event.owner_id,
            event.project_id,
            event.session_id,
            event.task_id,
        ):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise HookError("hook_event_identity_invalid")
        _validate_payload(event.payload)
        if len(_canonical(dict(event.payload))) > 256_000:
            raise HookError("hook_payload_too_large")

    @staticmethod
    def _report_from_stored(value: Mapping[str, object], *, replayed: bool) -> HookReport:
        failures_raw = cast(list[object], value["failures"])
        failures: list[HookFailure] = []
        for raw in failures_raw:
            if not isinstance(raw, dict):
                raise HookError("hook_state_failures_invalid")
            if set(raw) != {"hook_name", "category", "message"} or not all(
                isinstance(raw[key], str) for key in ("hook_name", "category", "message")
            ):
                raise HookError("hook_state_failure_shape_invalid")
            failures.append(
                HookFailure(
                    hook_name=cast(str, raw["hook_name"]),
                    category=cast(str, raw["category"]),
                    message=cast(str, raw["message"]),
                )
            )
        return HookReport(
            stage=HookStage(cast(str, value["stage"])),
            event_id=cast(str, value["event_id"]),
            idempotency_key=cast(str, value["idempotency_key"]),
            replayed=replayed,
            continued=cast(bool, value["continued"]),
            blocked=cast(bool, value["blocked"]),
            cancelled=cast(bool, value["cancelled"]),
            executed=tuple(cast(list[str], value["executed"])),
            failures=tuple(failures),
            context_modifications=tuple(cast(list[str], value["context_modifications"])),
        )

    def _store_report(self, event: HookEvent, report: HookReport) -> None:
        self._events[event.idempotency_key] = {
            "stage": report.stage.value,
            "event_id": report.event_id,
            "idempotency_key": report.idempotency_key,
            "owner_id": event.owner_id,
            "project_id": event.project_id,
            "session_id": event.session_id,
            "task_id": event.task_id,
            "payload_sha256": self._event_hash(event),
            "replayed": False,
            "continued": report.continued,
            "blocked": report.blocked,
            "cancelled": report.cancelled,
            "executed": list(report.executed),
            "failures": [failure.to_json() for failure in report.failures],
            "context_modifications": list(report.context_modifications),
        }
        self._persist()

    def _run_callback(
        self, registration: _HookRegistration, context: HookContext
    ) -> HookResult:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laplace-hook")
        future: Future[HookResult | None] = executor.submit(registration.callback, context)
        try:
            value = future.result(timeout=registration.timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise HookError("hook_timeout", {"hook": registration.name}) from exc
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise HookError(
                "hook_exception", {"hook": registration.name, "exception": type(exc).__name__}
            ) from exc
        else:
            executor.shutdown(wait=True, cancel_futures=True)
        if value is None:
            return HookResult()
        if not isinstance(value, HookResult):
            raise HookError("hook_result_invalid", {"hook": registration.name})
        if len(value.context_modification) > 16_384 or len(value.error_message) > 4_096:
            raise HookError("hook_result_too_large", {"hook": registration.name})
        return value

    def dispatch(self, event: HookEvent, *, cancel_event: threading.Event | None = None) -> HookReport:
        """Run matching hooks once, in stable order, and persist the result."""

        self._validate_event(event)
        signal = cancel_event or threading.Event()
        with self._lock:
            existing = self._events.get(event.idempotency_key)
            if existing is not None:
                if existing.get("payload_sha256") != self._event_hash(event) or existing.get("stage") != event.stage.value:
                    raise HookError("hook_idempotency_conflict")
                if any(
                    existing.get(key) != value
                    for key, value in (
                        ("owner_id", event.owner_id),
                        ("project_id", event.project_id),
                        ("session_id", event.session_id),
                        ("task_id", event.task_id),
                        ("event_id", event.event_id),
                    )
                ):
                    raise HookError("hook_owner_or_project_denied")
                replayed = self._report_from_stored(existing, replayed=True)
                if not replayed.continued:
                    if replayed.blocked:
                        raise HookBlockedError("hook_blocked", {"replayed": True})
                    if replayed.cancelled:
                        raise HookCancelledError("hook_cancelled", {"replayed": True})
                    raise HookSecurityError("hook_replayed_failure", {"replayed": True})
                return replayed
            registrations = [
                registration
                for registration in self._registrations.values()
                if registration.enabled
                and registration.stage is event.stage
                and self._scope_matches(registration, event)
            ]
            registrations.sort(key=lambda item: (item.priority, item.ordinal, item.name))

        executed: list[str] = []
        failures: list[HookFailure] = []
        context_modifications: list[str] = []
        blocked = False
        cancelled = False
        for registration in registrations:
            if signal.is_set() and event.stage is not HookStage.TASK_CANCEL:
                cancelled = True
                failure = HookFailure(registration.name, "hook_cancelled", "task cancellation observed")
                failures.append(failure)
                if registration.security_critical:
                    report = HookReport(
                        event.stage, event.event_id, event.idempotency_key, False, False, False, True,
                        tuple(executed), tuple(failures), tuple(context_modifications)
                    )
                    with self._lock:
                        self._store_report(event, report)
                    raise HookCancelledError("hook_cancelled", failure.to_json())
                continue
            context = HookContext(
                stage=event.stage,
                event_id=event.event_id,
                idempotency_key=event.idempotency_key,
                owner_id=event.owner_id,
                project_id=event.project_id,
                session_id=event.session_id,
                task_id=event.task_id,
                payload=MappingProxyType(dict(event.payload)),
                cancel_event=signal,
            )
            try:
                result = self._run_callback(registration, context)
            except HookError as exc:
                failure = HookFailure(registration.name, exc.category, str(exc))
                failures.append(failure)
                if registration.failure_policy is HookFailurePolicy.FAIL_CLOSED:
                    report = HookReport(
                        event.stage, event.event_id, event.idempotency_key, False, False, False, cancelled,
                        tuple(executed), tuple(failures), tuple(context_modifications)
                    )
                    with self._lock:
                        self._store_report(event, report)
                    raise HookSecurityError(exc.category, failure.to_json()) from exc
                continue
            executed.append(registration.name)
            if result.context_modification:
                context_modifications.append(result.context_modification)
            if result.error_message:
                failures.append(HookFailure(registration.name, "hook_reported_error", result.error_message))
            if result.cancel and event.stage not in _POST_STAGES:
                blocked = True
                failure = HookFailure(
                    registration.name,
                    "hook_blocked",
                    result.error_message or "hook blocked operation",
                )
                if not any(item == failure for item in failures):
                    failures.append(failure)
                report = HookReport(
                    event.stage, event.event_id, event.idempotency_key, False, False, True, cancelled,
                    tuple(executed), tuple(failures), tuple(context_modifications)
                )
                with self._lock:
                    self._store_report(event, report)
                raise HookBlockedError("hook_blocked", failure.to_json())

        report = HookReport(
            event.stage,
            event.event_id,
            event.idempotency_key,
            False,
            not blocked and not cancelled,
            blocked,
            cancelled,
            tuple(executed),
            tuple(failures),
            tuple(context_modifications),
        )
        with self._lock:
            self._store_report(event, report)
        return report

    def emit(
        self,
        stage: HookStage,
        *,
        owner_id: str,
        project_id: str,
        session_id: str,
        task_id: str,
        idempotency_key: str,
        payload: Mapping[str, object] | None = None,
        event_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> HookReport:
        """Convenience adapter for internal Core lifecycle callers."""

        event = HookEvent(
            stage=stage,
            event_id=event_id or idempotency_key,
            idempotency_key=idempotency_key,
            owner_id=owner_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            payload=payload or {},
        )
        return self.dispatch(event, cancel_event=cancel_event)

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


LifecycleHookService = HookService
LifecycleHookStage = HookStage
