from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from research_workspace.laplace_core import LaplaceCore
from research_workspace.logical_subagents import (
    GpuAvailability,
    GpuAwareSubagentScheduler,
    LogicalSubagentError,
    LogicalSubagentOutcome,
    LogicalSubagentTask,
    SubagentExecutionMode,
    SubagentOutOfMemoryError,
    SubagentSchedulerPolicy,
)


def _gpu(*, free_mib: int = 12_000) -> GpuAvailability:
    return GpuAvailability(
        status="AVAILABLE",
        total_mib=16_000,
        used_mib=16_000 - free_mib,
        free_mib=free_mib,
        source="test-observation",
        captured_at_utc="2026-08-24T20:00:00Z",
    )


def _task(task_id: str, *, owner_id: str = "user-a", depth: int = 1) -> LogicalSubagentTask:
    return LogicalSubagentTask(
        task_id=task_id,
        parent_task_id="parent-a",
        owner_id=owner_id,
        project_id="project-a",
        goal=f"inspect {task_id}",
        context_digest="a" * 64,
        depth=depth,
    )


def test_serial_and_queued_logical_modes_are_fifo_and_gpu_bounded() -> None:
    observed: list[str] = []

    def executor(task: LogicalSubagentTask, _: threading.Event) -> dict[str, object]:
        observed.append(task.task_id)
        return {"summary": task.task_id}

    for mode in (SubagentExecutionMode.SERIAL, SubagentExecutionMode.QUEUED_LOGICAL):
        scheduler = GpuAwareSubagentScheduler(
            policy=SubagentSchedulerPolicy(execution_mode=mode),
            gpu_probe=lambda: _gpu(),
        )
        try:
            outcomes = scheduler.run_batch(tuple(_task(f"task-{index}") for index in range(3)), executor)
            assert [outcome.task_id for outcome in outcomes] == ["task-0", "task-1", "task-2"]
            assert all(outcome.state == "SUCCEEDED" for outcome in outcomes)
            assert scheduler.snapshot()["gpu_execution_limit"] == 1
        finally:
            scheduler.close()
    assert observed == ["task-0", "task-1", "task-2", "task-0", "task-1", "task-2"]


def test_running_cancellation_and_fifo_fairness() -> None:
    scheduler = GpuAwareSubagentScheduler(gpu_probe=lambda: _gpu())
    started = threading.Event()
    release = threading.Event()
    execution_order: list[str] = []
    first_result: list[LogicalSubagentOutcome] = []

    def blocking(task: LogicalSubagentTask, cancellation: threading.Event) -> dict[str, object]:
        execution_order.append(task.task_id)
        started.set()
        while not release.is_set():
            if cancellation.is_set():
                break
            time.sleep(0.005)
        return {"cancelled_seen": cancellation.is_set()}

    def run_first() -> None:
        first_result.extend(scheduler.run_batch((_task("first"),), blocking))

    thread = threading.Thread(target=run_first)
    thread.start()
    assert started.wait(timeout=2)
    assert scheduler.cancel("first", owner_id="user-a")["state"] == "CANCEL_REQUESTED"
    release.set()
    thread.join(timeout=2)
    assert first_result[0].state == "CANCELLED"
    assert execution_order == ["first"]
    assert scheduler.snapshot(owner_id="user-a")["running"] == 0
    scheduler.close()


def test_gpu_scarcity_queues_then_expires_without_executor_or_slot_leak() -> None:
    blocked_calls: list[str] = []
    blocked = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(queue_timeout_seconds=0.03),
        gpu_probe=lambda: GpuAvailability.unavailable("uncertain-owner"),
    )
    try:
        def should_not_run(task: LogicalSubagentTask, _: threading.Event) -> dict[str, object]:
            blocked_calls.append(task.task_id)
            return {}

        outcome = blocked.run_batch((_task("blocked"),), should_not_run)[0]
        assert outcome.state == "FAILED"
        assert outcome.failure_category == "subagent_queue_timeout"
        assert blocked_calls == []
    finally:
        blocked.close()

    calls: list[str] = []
    recovering = GpuAwareSubagentScheduler(gpu_probe=lambda: _gpu())

    def oom_then_pass(task: LogicalSubagentTask, _: threading.Event) -> dict[str, object]:
        calls.append(task.task_id)
        if task.task_id == "oom":
            raise SubagentOutOfMemoryError("gpu_oom_recoverable")
        return {"status": "ok"}

    try:
        outcomes = recovering.run_batch((_task("oom"), _task("after-oom")), oom_then_pass)
        assert [outcome.state for outcome in outcomes] == ["RECOVERABLE", "SUCCEEDED"]
        assert calls == ["oom", "after-oom"]
        assert recovering.snapshot()["running"] == 0
    finally:
        recovering.close()


def test_real_concurrency_requires_explicit_policy_and_remains_gpu_limited() -> None:
    with pytest.raises(LogicalSubagentError, match="real_concurrency_not_certified"):
        SubagentSchedulerPolicy(
            execution_mode=SubagentExecutionMode.REAL_CONCURRENT,
            max_gpu_concurrency=2,
        )
    scheduler = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(
            execution_mode=SubagentExecutionMode.REAL_CONCURRENT,
            max_gpu_concurrency=2,
            allow_real_concurrency=True,
        ),
        gpu_probe=lambda: _gpu(free_mib=12_000),
    )
    active = 0
    peak = 0
    lock = threading.Lock()

    def concurrent_executor(task: LogicalSubagentTask, _: threading.Event) -> dict[str, object]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"task": task.task_id}

    try:
        outcomes = scheduler.run_batch((_task("parallel-1"), _task("parallel-2")), concurrent_executor)
        assert all(outcome.state == "SUCCEEDED" for outcome in outcomes)
        assert peak <= 2
        assert scheduler.snapshot()["real_concurrency_enabled"] is True
    finally:
        scheduler.close()


def test_scope_and_depth_gates_and_core_adapter(tmp_path: Path) -> None:
    scheduler = GpuAwareSubagentScheduler(gpu_probe=lambda: _gpu())
    try:
        with pytest.raises(LogicalSubagentError, match="subagent_batch_scope_mismatch"):
            scheduler.run_batch((_task("a"), _task("b", owner_id="user-b")), lambda *_: {})
        with pytest.raises(LogicalSubagentError, match="subagent_spawn_depth_exceeded"):
            scheduler.run_batch((_task("deep", depth=2),), lambda *_: {})
        core = LaplaceCore(tmp_path / "core", object(), object(), logical_subagents=scheduler)  # type: ignore[arg-type]
        outcome = core.run_logical_subagents((_task("core-task"),), lambda task, _: {"task": task.task_id})
        assert outcome[0].state == "SUCCEEDED"
    finally:
        scheduler.close()
