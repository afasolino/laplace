"""Bounded logical subagents and fail-closed single-GPU scheduling.

This is a host-side orchestration boundary, not a second agent runtime.  A
logical child receives a typed task and a cancellation signal; only the caller
supplies the executor that can perform model work.  The default execution mode
is serial and the default GPU concurrency limit is one.  Real concurrent
serving is opt-in and cannot be enabled by a task or model response.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Literal, TypeAlias, cast

from .zetsu_results import ZetsuResultError, ZetsuResultStore

JsonObject: TypeAlias = dict[str, object]
GpuStatus = Literal["AVAILABLE", "UNAVAILABLE", "UNCERTAIN"]

_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}"
_ALLOWED_CAPABILITIES = frozenset({
    "read_repository",
    "read_corpus",
    "deterministic_verify",
    "propose_change",
})


class SubagentExecutionMode(str, Enum):
    SERIAL = "serial"
    QUEUED_LOGICAL = "queued_logical"
    REAL_CONCURRENT = "real_concurrent"


class LogicalSubagentError(RuntimeError):
    """A subagent request or scheduler operation failed closed."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class SubagentOutOfMemoryError(LogicalSubagentError):
    """An executor reported a recoverable GPU memory failure."""


SubagentExecutor: TypeAlias = Callable[
    ["LogicalSubagentTask", threading.Event], Mapping[str, object]
]


@dataclass(frozen=True)
class GpuAvailability:
    status: GpuStatus
    total_mib: int
    used_mib: int
    free_mib: int
    compute_pids: tuple[int, ...] = ()
    source: str = "injected-observation"
    captured_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"AVAILABLE", "UNAVAILABLE", "UNCERTAIN"}:
            raise LogicalSubagentError("gpu_status_invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (
            self.total_mib, self.used_mib, self.free_mib
        )):
            raise LogicalSubagentError("gpu_memory_observation_invalid")
        if self.used_mib + self.free_mib > self.total_mib:
            raise LogicalSubagentError("gpu_memory_observation_inconsistent")
        if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in self.compute_pids):
            raise LogicalSubagentError("gpu_process_identity_invalid")
        if not isinstance(self.source, str) or not self.source or len(self.source) > 128:
            raise LogicalSubagentError("gpu_observation_source_invalid")
        if not isinstance(self.captured_at_utc, str) or len(self.captured_at_utc) > 64:
            raise LogicalSubagentError("gpu_observation_timestamp_invalid")

    @classmethod
    def unavailable(cls, reason: str = "no_gpu_probe") -> "GpuAvailability":
        if not reason or len(reason) > 128:
            raise LogicalSubagentError("gpu_unavailable_reason_invalid")
        return cls(
            status="UNCERTAIN",
            total_mib=0,
            used_mib=0,
            free_mib=0,
            source=reason,
            captured_at_utc=datetime.now(UTC).isoformat(),
        )

    def to_json(self) -> JsonObject:
        return {
            "status": self.status,
            "total_mib": self.total_mib,
            "used_mib": self.used_mib,
            "free_mib": self.free_mib,
            "compute_pids": list(self.compute_pids),
            "source": self.source,
            "captured_at_utc": self.captured_at_utc,
        }


GpuProbe = Callable[[], GpuAvailability]


@dataclass(frozen=True)
class SubagentSchedulerPolicy:
    execution_mode: SubagentExecutionMode = SubagentExecutionMode.SERIAL
    max_logical_children: int = 3
    queue_capacity: int = 16
    max_gpu_concurrency: int = 1
    minimum_free_mib: int = 4_096
    max_spawn_depth: int = 1
    task_timeout_seconds: float = 600.0
    allow_real_concurrency: bool = False
    completed_cache_size: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, SubagentExecutionMode):
            raise LogicalSubagentError("subagent_execution_mode_invalid")
        for name in (
            "max_logical_children",
            "queue_capacity",
            "max_gpu_concurrency",
            "minimum_free_mib",
            "max_spawn_depth",
            "completed_cache_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LogicalSubagentError("subagent_policy_invalid", {"field": name})
        if self.max_logical_children > self.queue_capacity:
            raise LogicalSubagentError("subagent_batch_exceeds_queue_capacity")
        if (
            isinstance(self.task_timeout_seconds, bool)
            or not isinstance(self.task_timeout_seconds, (int, float))
            or not 0.01 <= self.task_timeout_seconds <= 86_400
        ):
            raise LogicalSubagentError("subagent_timeout_invalid")
        if self.execution_mode is SubagentExecutionMode.REAL_CONCURRENT:
            if not self.allow_real_concurrency or self.max_gpu_concurrency < 2:
                raise LogicalSubagentError("real_concurrency_not_certified")
        elif self.max_gpu_concurrency != 1:
            raise LogicalSubagentError("single_gpu_policy_requires_one_default_slot")

    @property
    def active_limit(self) -> int:
        if self.execution_mode is SubagentExecutionMode.REAL_CONCURRENT:
            return self.max_gpu_concurrency
        return 1

    def to_json(self) -> JsonObject:
        return {
            "execution_mode": self.execution_mode.value,
            "max_logical_children": self.max_logical_children,
            "queue_capacity": self.queue_capacity,
            "max_gpu_concurrency": self.max_gpu_concurrency,
            "minimum_free_mib": self.minimum_free_mib,
            "max_spawn_depth": self.max_spawn_depth,
            "task_timeout_seconds": float(self.task_timeout_seconds),
            "allow_real_concurrency": self.allow_real_concurrency,
            "completed_cache_size": self.completed_cache_size,
        }


@dataclass(frozen=True)
class LogicalSubagentTask:
    task_id: str
    parent_task_id: str
    owner_id: str
    project_id: str
    goal: str
    context_digest: str
    required_free_mib: int = 4_096
    depth: int = 1
    capabilities: tuple[str, ...] = ("read_repository", "deterministic_verify")
    model_lane: str = "standard"

    def __post_init__(self) -> None:
        import re

        for value, label in (
            (self.task_id, "task_id"),
            (self.parent_task_id, "parent_task_id"),
            (self.owner_id, "owner_id"),
            (self.project_id, "project_id"),
        ):
            if re.fullmatch(_IDENTIFIER, value) is None:
                raise LogicalSubagentError("subagent_identity_invalid", {"field": label})
        if not isinstance(self.goal, str) or not self.goal or len(self.goal) > 8_192 or "\x00" in self.goal:
            raise LogicalSubagentError("subagent_goal_invalid")
        if not isinstance(self.context_digest, str) or len(self.context_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.context_digest
        ):
            raise LogicalSubagentError("subagent_context_digest_invalid")
        if isinstance(self.required_free_mib, bool) or not isinstance(self.required_free_mib, int) or self.required_free_mib < 0:
            raise LogicalSubagentError("subagent_vram_reservation_invalid")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1:
            raise LogicalSubagentError("subagent_depth_invalid")
        if not self.capabilities or len(self.capabilities) > 8 or any(
            capability not in _ALLOWED_CAPABILITIES for capability in self.capabilities
        ):
            raise LogicalSubagentError("subagent_capability_invalid")
        if self.model_lane not in {"quality", "standard", "economy"}:
            raise LogicalSubagentError("subagent_model_lane_invalid")

    @property
    def task_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "task_id": self.task_id,
                    "parent_task_id": self.parent_task_id,
                    "owner_id": self.owner_id,
                    "project_id": self.project_id,
                    "goal": self.goal,
                    "context_digest": self.context_digest,
                    "required_free_mib": self.required_free_mib,
                    "depth": self.depth,
                    "capabilities": list(self.capabilities),
                    "model_lane": self.model_lane,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_json(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "goal": self.goal,
            "context_digest": self.context_digest,
            "required_free_mib": self.required_free_mib,
            "depth": self.depth,
            "capabilities": list(self.capabilities),
            "model_lane": self.model_lane,
        }


SubagentOutcomeState = Literal["SUCCEEDED", "FAILED", "CANCELLED", "GPU_BLOCKED", "RECOVERABLE"]


@dataclass(frozen=True)
class LogicalSubagentOutcome:
    task_id: str
    owner_id: str
    state: SubagentOutcomeState
    result: JsonObject | None
    failure_category: str | None
    queue_position: int
    queue_wait_seconds: float
    elapsed_seconds: float
    gpu_observation: GpuAvailability
    result_id: str | None = None
    result_artifacts: JsonObject | None = None
    delivery_status: Literal["INLINE", "PAGED", "FAILED", "UNAVAILABLE"] = "INLINE"
    delivery_error: str | None = None

    def to_json(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "owner_id": self.owner_id,
            "state": self.state,
            "result": self.result,
            "failure_category": self.failure_category,
            "queue_position": self.queue_position,
            "queue_wait_seconds": self.queue_wait_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "gpu_observation": self.gpu_observation.to_json(),
            "result_id": self.result_id,
            "result_artifacts": self.result_artifacts,
            "delivery_status": self.delivery_status,
            "delivery_error": self.delivery_error,
        }


@dataclass
class _QueuedTask:
    task: LogicalSubagentTask
    executor: SubagentExecutor
    queue_position: int
    submitted_at: float
    cancellation: threading.Event
    done: threading.Event
    outcome: LogicalSubagentOutcome | None = None


def _validate_result(value: Mapping[str, object]) -> JsonObject:
    if not isinstance(value, Mapping):
        raise LogicalSubagentError("subagent_result_invalid")
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise LogicalSubagentError("subagent_result_invalid") from exc
    return dict(value)


class GpuAwareSubagentScheduler:
    """FIFO logical-child scheduler with conservative GPU admission.

    The queue is deliberately in-process because durable whole-agent admission
    already belongs to :class:`AgentTaskScheduler`.  A scheduler restart never
    replays an opaque callable: callers receive no hidden execution or retry.
    """

    def __init__(
        self,
        *,
        policy: SubagentSchedulerPolicy | None = None,
        gpu_probe: GpuProbe | None = None,
        result_store: ZetsuResultStore | None = None,
    ) -> None:
        self.policy = policy or SubagentSchedulerPolicy()
        self.gpu_probe = gpu_probe or GpuAvailability.unavailable
        self._condition = threading.Condition(threading.RLock())
        self._queue: list[_QueuedTask] = []
        self._active: dict[str, _QueuedTask] = {}
        self._completed: OrderedDict[str, LogicalSubagentOutcome] = OrderedDict()
        self.result_store = result_store
        self._closed = False
        self._next_position = 0
        self._worker = threading.Thread(target=self._dispatch, name="laplace-subagent-scheduler", daemon=True)
        self._worker.start()

    def _probe(self) -> GpuAvailability:
        try:
            value = self.gpu_probe()
        except Exception as exc:
            raise LogicalSubagentError("gpu_probe_failed", {"exception": type(exc).__name__}) from exc
        if not isinstance(value, GpuAvailability):
            raise LogicalSubagentError("gpu_probe_result_invalid")
        return value

    def _make_outcome(
        self,
        queued: _QueuedTask,
        *,
        state: SubagentOutcomeState,
        result: JsonObject | None,
        failure_category: str | None,
        queue_wait_seconds: float,
        elapsed_seconds: float,
        gpu: GpuAvailability,
        result_id: str | None = None,
        result_artifacts: JsonObject | None = None,
        delivery_status: Literal["INLINE", "PAGED", "FAILED", "UNAVAILABLE"] = "INLINE",
        delivery_error: str | None = None,
    ) -> LogicalSubagentOutcome:
        return LogicalSubagentOutcome(
            task_id=queued.task.task_id,
            owner_id=queued.task.owner_id,
            state=state,
            result=result,
            failure_category=failure_category,
            queue_position=queued.queue_position,
            queue_wait_seconds=max(0.0, queue_wait_seconds),
            elapsed_seconds=max(0.0, elapsed_seconds),
            gpu_observation=gpu,
            result_id=result_id,
            result_artifacts=result_artifacts,
            delivery_status=delivery_status,
            delivery_error=delivery_error,
        )

    def _remember_completed_locked(
        self, task_id: str, outcome: LogicalSubagentOutcome
    ) -> None:
        self._completed[task_id] = outcome
        self._completed.move_to_end(task_id)
        while len(self._completed) > self.policy.completed_cache_size:
            self._completed.popitem(last=False)

    def _deliver_result(
        self, task: LogicalSubagentTask, result: JsonObject
    ) -> tuple[JsonObject | None, str | None, JsonObject | None, Literal["INLINE", "PAGED", "FAILED", "UNAVAILABLE"], str | None]:
        """Persist the exact result and return a bounded presentation envelope."""

        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        encoded_bytes = encoded.encode("utf-8")
        if self.result_store is None:
            if len(encoded_bytes) <= 128_000:
                return result, None, None, "INLINE", None
            return (
                {
                    "execution_status": "SUCCEEDED",
                    "delivery_status": "UNAVAILABLE",
                },
                None,
                None,
                "UNAVAILABLE",
                "logical_result_store_unconfigured",
            )
        try:
            delivery = self.result_store.persist(
                user_id=task.owner_id,
                repo_id=task.project_id,
                session_id=task.task_id,
                status="SUCCEEDED",
                summary=f"SUCCEEDED:{task.task_id}",
                artifacts={"result.json": encoded_bytes},
            )
        except ZetsuResultError as exc:
            if len(encoded_bytes) <= 128_000:
                return result, None, None, "FAILED", exc.category
            return (
                {
                    "execution_status": "SUCCEEDED",
                    "delivery_status": "FAILED",
                },
                None,
                None,
                "FAILED",
                exc.category,
            )
        result_id = str(delivery["result_id"])
        artifacts = cast(JsonObject, delivery["artifacts"])
        if len(encoded_bytes) <= 128_000:
            return result, result_id, artifacts, "INLINE", None
        return (
            {
                "execution_status": "SUCCEEDED",
                "delivery_status": "PAGED",
                "result_id": result_id,
                "result_artifact": "result.json",
                "result_bytes": len(encoded_bytes),
                "result_sha256": hashlib.sha256(encoded_bytes).hexdigest(),
            },
            result_id,
            artifacts,
            "PAGED",
            None,
        )

    def _finish(self, queued: _QueuedTask, outcome: LogicalSubagentOutcome) -> None:
        with self._condition:
            self._active.pop(queued.task.task_id, None)
            queued.outcome = outcome
            self._remember_completed_locked(queued.task.task_id, outcome)
            queued.done.set()
            self._condition.notify_all()

    def _execute(self, queued: _QueuedTask, gpu: GpuAvailability) -> None:
        started = time.monotonic()
        wait_seconds = started - queued.submitted_at
        if queued.cancellation.is_set():
            self._finish(
                queued,
                self._make_outcome(
                    queued,
                    state="CANCELLED",
                    result=None,
                    failure_category="subagent_cancelled_before_start",
                    queue_wait_seconds=wait_seconds,
                    elapsed_seconds=0.0,
                    gpu=gpu,
                ),
            )
            return
        accepted: JsonObject | None = None
        result_id: str | None = None
        result_artifacts: JsonObject | None = None
        delivery_status: Literal["INLINE", "PAGED", "FAILED", "UNAVAILABLE"] = "UNAVAILABLE"
        delivery_error: str | None = None
        try:
            result = _validate_result(queued.executor(queued.task, queued.cancellation))
            if queued.cancellation.is_set():
                state: SubagentOutcomeState = "CANCELLED"
                category: str | None = "subagent_cancelled"
            else:
                state = "SUCCEEDED"
                category = None
                accepted, result_id, result_artifacts, delivery_status, delivery_error = self._deliver_result(
                    queued.task, result
                )
        except SubagentOutOfMemoryError as exc:
            state = "RECOVERABLE"
            category = exc.category
        except LogicalSubagentError as exc:
            state = "FAILED"
            category = exc.category
        except Exception as exc:
            state = "FAILED"
            category = f"subagent_executor_{type(exc).__name__}"
        self._finish(
            queued,
            self._make_outcome(
                queued,
                state=state,
                result=accepted,
                failure_category=category,
                queue_wait_seconds=wait_seconds,
                elapsed_seconds=time.monotonic() - started,
                gpu=gpu,
                result_id=result_id,
                result_artifacts=result_artifacts,
                delivery_status=delivery_status,
                delivery_error=delivery_error,
            ),
        )

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                while not self._closed and not self._queue:
                    self._condition.wait()
                if self._closed and not self._queue and not self._active:
                    return
                if not self._queue or len(self._active) >= self.policy.active_limit:
                    self._condition.wait(timeout=0.05)
                    continue
                queued = self._queue.pop(0)
                if queued.cancellation.is_set():
                    gpu = GpuAvailability.unavailable("cancelled-before-gpu-probe")
                    outcome = self._make_outcome(
                        queued,
                        state="CANCELLED",
                        result=None,
                        failure_category="subagent_cancelled_before_admission",
                        queue_wait_seconds=time.monotonic() - queued.submitted_at,
                        elapsed_seconds=0.0,
                        gpu=gpu,
                    )
                    queued.outcome = outcome
                    self._remember_completed_locked(queued.task.task_id, outcome)
                    queued.done.set()
                    continue
            try:
                gpu = self._probe()
            except LogicalSubagentError as exc:
                gpu = GpuAvailability.unavailable(exc.category)
            if gpu.status != "AVAILABLE" or gpu.free_mib < max(self.policy.minimum_free_mib, queued.task.required_free_mib):
                outcome = self._make_outcome(
                    queued,
                    state="GPU_BLOCKED",
                    result=None,
                    failure_category=("gpu_uncertain" if gpu.status != "AVAILABLE" else "gpu_headroom_insufficient"),
                    queue_wait_seconds=time.monotonic() - queued.submitted_at,
                    elapsed_seconds=0.0,
                    gpu=gpu,
                )
                with self._condition:
                    queued.outcome = outcome
                    self._remember_completed_locked(queued.task.task_id, outcome)
                    queued.done.set()
                    self._condition.notify_all()
                continue
            with self._condition:
                if self._closed:
                    queued.cancellation.set()
                if queued.cancellation.is_set():
                    outcome = self._make_outcome(
                        queued,
                        state="CANCELLED",
                        result=None,
                        failure_category="subagent_cancelled_before_start",
                        queue_wait_seconds=time.monotonic() - queued.submitted_at,
                        elapsed_seconds=0.0,
                        gpu=gpu,
                    )
                    queued.outcome = outcome
                    self._remember_completed_locked(queued.task.task_id, outcome)
                    queued.done.set()
                    self._condition.notify_all()
                    continue
                self._active[queued.task.task_id] = queued
                threading.Thread(
                    target=self._execute,
                    args=(queued, gpu),
                    name=f"laplace-subagent-{queued.task.task_id}",
                    daemon=True,
                ).start()

    def run_batch(
        self,
        tasks: Sequence[LogicalSubagentTask],
        executor: SubagentExecutor,
    ) -> tuple[LogicalSubagentOutcome, ...]:
        """Queue one bounded logical batch and return outcomes in task order."""

        if not tasks:
            return ()
        if len(tasks) > self.policy.max_logical_children:
            raise LogicalSubagentError("subagent_batch_too_large")
        if not callable(executor):
            raise LogicalSubagentError("subagent_executor_invalid")
        task_list = tuple(tasks)
        identifiers = [task.task_id for task in task_list]
        if len(set(identifiers)) != len(identifiers):
            raise LogicalSubagentError("subagent_task_id_duplicate")
        first_owner = task_list[0].owner_id
        first_project = task_list[0].project_id
        if any(task.owner_id != first_owner or task.project_id != first_project for task in task_list):
            raise LogicalSubagentError("subagent_batch_scope_mismatch")
        if any(task.depth > self.policy.max_spawn_depth for task in task_list):
            raise LogicalSubagentError("subagent_spawn_depth_exceeded")
        entries: list[_QueuedTask] = []
        with self._condition:
            if self._closed:
                raise LogicalSubagentError("subagent_scheduler_closed")
            if len(self._queue) + len(self._active) + len(task_list) > self.policy.queue_capacity:
                raise LogicalSubagentError("subagent_queue_full")
            for task in task_list:
                if task.task_id in self._completed or task.task_id in self._active or any(
                    item.task.task_id == task.task_id for item in self._queue
                ):
                    raise LogicalSubagentError("subagent_task_id_conflict")
                entry = _QueuedTask(
                    task=task,
                    executor=executor,
                    queue_position=self._next_position,
                    submitted_at=time.monotonic(),
                    cancellation=threading.Event(),
                    done=threading.Event(),
                )
                self._next_position += 1
                self._queue.append(entry)
                entries.append(entry)
            self._condition.notify_all()
        deadline = time.monotonic() + self.policy.task_timeout_seconds
        for entry in entries:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not entry.done.wait(timeout=remaining):
                self.cancel(entry.task.task_id, owner_id=entry.task.owner_id)
                raise LogicalSubagentError("subagent_task_timeout")
        return tuple(cast(LogicalSubagentOutcome, entry.outcome) for entry in entries)

    def cancel(self, task_id: str, *, owner_id: str) -> JsonObject:
        with self._condition:
            for index, queued in enumerate(self._queue):
                if queued.task.task_id == task_id:
                    if queued.task.owner_id != owner_id:
                        raise LogicalSubagentError("subagent_owner_denied")
                    self._queue.pop(index)
                    queued.cancellation.set()
                    gpu = GpuAvailability.unavailable("cancelled-before-gpu-probe")
                    outcome = self._make_outcome(
                        queued,
                        state="CANCELLED",
                        result=None,
                        failure_category="subagent_cancelled_while_queued",
                        queue_wait_seconds=time.monotonic() - queued.submitted_at,
                        elapsed_seconds=0.0,
                        gpu=gpu,
                    )
                    queued.outcome = outcome
                    self._remember_completed_locked(task_id, outcome)
                    queued.done.set()
                    self._condition.notify_all()
                    return {"task_id": task_id, "state": "CANCELLED", "queued": True}
            active = self._active.get(task_id)
            if active is not None:
                if active.task.owner_id != owner_id:
                    raise LogicalSubagentError("subagent_owner_denied")
                active.cancellation.set()
                return {"task_id": task_id, "state": "CANCEL_REQUESTED", "queued": False}
            completed = self._completed.get(task_id)
            if completed is not None:
                return {"task_id": task_id, "state": completed.state, "queued": False}
        raise LogicalSubagentError("subagent_task_not_found")

    def snapshot(self, *, owner_id: str | None = None) -> JsonObject:
        with self._condition:
            queued = [item for item in self._queue if owner_id is None or item.task.owner_id == owner_id]
            active = [item for item in self._active.values() if owner_id is None or item.task.owner_id == owner_id]
            completed = [item for item in self._completed.values() if owner_id is None or item.owner_id == owner_id]
            return {
                "execution_mode": self.policy.execution_mode.value,
                "gpu_execution_limit": self.policy.active_limit,
                "queued": len(queued),
                "running": len(active),
                "completed": len(completed),
                "queue_capacity": self.policy.queue_capacity,
                "max_spawn_depth": self.policy.max_spawn_depth,
                "fairness": "fifo",
                "real_concurrency_enabled": self.policy.execution_mode is SubagentExecutionMode.REAL_CONCURRENT,
                "queued_task_ids": [item.task.task_id for item in queued],
                "running_task_ids": [item.task.task_id for item in active],
                "completed_cache_size": self.policy.completed_cache_size,
            }

    def page_result(
        self,
        *,
        task: LogicalSubagentTask,
        result_id: str,
        offset: int = 0,
        max_bytes: int = 24_000,
    ) -> JsonObject:
        """Read a durable logical-subagent result under the task scope."""

        if self.result_store is None:
            raise LogicalSubagentError("logical_result_store_unconfigured")
        try:
            return self.result_store.page(
                user_id=task.owner_id,
                repo_id=task.project_id,
                session_id=task.task_id,
                result_id=result_id,
                artifact="result.json",
                offset=offset,
                max_bytes=max_bytes,
            )
        except ZetsuResultError as exc:
            raise LogicalSubagentError(exc.category) from exc

    def close(self, *, wait_timeout_seconds: float = 1.0) -> None:
        if not 0.01 <= wait_timeout_seconds <= 30:
            raise LogicalSubagentError("subagent_close_timeout_invalid")
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for queued in self._queue:
                queued.cancellation.set()
            self._condition.notify_all()
        self._worker.join(timeout=wait_timeout_seconds)


LogicalSubagentScheduler = GpuAwareSubagentScheduler
