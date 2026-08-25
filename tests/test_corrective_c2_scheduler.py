from __future__ import annotations

import threading
import time

import pytest

from research_workspace.logical_subagents import (
    GpuAvailability,
    GpuAwareSubagentScheduler,
    LogicalSubagentError,
    LogicalSubagentTask,
    SubagentExecutionMode,
    SubagentSchedulerPolicy,
)


def _task(task_id: str, *, required_free_mib: int = 4_096) -> LogicalSubagentTask:
    return LogicalSubagentTask(
        task_id=task_id,
        parent_task_id="parent",
        owner_id="owner",
        project_id="project",
        goal=task_id,
        context_digest="b" * 64,
        required_free_mib=required_free_mib,
    )


def _gpu(free_mib: int = 16_000) -> GpuAvailability:
    return GpuAvailability(
        status="AVAILABLE",
        total_mib=16_000,
        used_mib=16_000 - free_mib,
        free_mib=free_mib,
        source="c2-test",
        captured_at_utc="2026-08-25T00:00:00Z",
    )


def test_temporary_gpu_scarcity_queues_and_wakes_without_executor_calls() -> None:
    observations = [GpuAvailability.unavailable("temporary"), _gpu()]
    calls: list[str] = []
    scheduler = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(queue_timeout_seconds=1.0),
        gpu_probe=lambda: observations.pop(0) if observations else _gpu(),
    )
    try:
        outcome = scheduler.run_batch(
            (_task("wakes"),),
            lambda task, _: calls.append(task.task_id) or {"ok": True},
        )[0]
        assert outcome.state == "SUCCEEDED"
        assert calls == ["wakes"]
        assert outcome.queue_wait_seconds >= 0.04
    finally:
        scheduler.close()


def test_queued_cancellation_and_queue_full_are_bounded() -> None:
    started = threading.Event()
    release = threading.Event()
    scheduler = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(
            max_logical_children=2, queue_capacity=2, queue_timeout_seconds=1.0
        ),
        gpu_probe=_gpu,
    )
    first_result: list[object] = []
    second_result: list[object] = []

    def blocking(task: LogicalSubagentTask, _: threading.Event) -> dict[str, object]:
        started.set()
        release.wait(1)
        return {"task": task.task_id}

    def run_first() -> None:
        first_result.extend(scheduler.run_batch((_task("first"),), blocking))

    first_thread = threading.Thread(target=run_first)
    first_thread.start()
    assert started.wait(1)
    queued_thread = threading.Thread(
        target=lambda: second_result.extend(
            scheduler.run_batch((_task("second"),), lambda *_: {"unexpected": True})
        )
    )
    queued_thread.start()
    deadline = time.monotonic() + 1
    while "second" not in scheduler.snapshot()["queued_task_ids"] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert scheduler.snapshot()["queued_task_ids"] == ["second"]
    with pytest.raises(LogicalSubagentError, match="subagent_queue_full"):
        scheduler.run_batch((_task("third"),), lambda *_: {"unexpected": True})
    assert scheduler.cancel("second", owner_id="owner")["queued"] is True
    queued_thread.join(1)
    release.set()
    first_thread.join(1)
    assert second_result[0].state == "CANCELLED"
    assert first_result[0].state == "SUCCEEDED"
    scheduler.close()


def test_real_concurrent_admission_reserves_vram_atomically_and_releases() -> None:
    started: list[str] = []
    first_started = threading.Event()
    release = threading.Event()
    scheduler = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(
            execution_mode=SubagentExecutionMode.REAL_CONCURRENT,
            max_gpu_concurrency=2,
            allow_real_concurrency=True,
            minimum_free_mib=1,
            gpu_headroom_mib=128,
            queue_timeout_seconds=1.0,
        ),
        gpu_probe=lambda: _gpu(9_000),
    )

    def executor(task: LogicalSubagentTask, _: threading.Event) -> dict[str, object]:
        started.append(task.task_id)
        if task.task_id == "one":
            first_started.set()
            release.wait(1)
        return {"task": task.task_id}

    result: list[object] = []
    thread = threading.Thread(
        target=lambda: result.extend(
            scheduler.run_batch(
                (_task("one", required_free_mib=6_000), _task("two", required_free_mib=6_000)),
                executor,
            )
        )
    )
    thread.start()
    assert first_started.wait(1)
    time.sleep(0.05)
    snapshot = scheduler.snapshot()
    assert started == ["one"]
    assert snapshot["reserved_gpu_mib"] == 6_128
    assert snapshot["queued_task_ids"] == ["two"]
    release.set()
    thread.join(2)
    assert [item.state for item in result] == ["SUCCEEDED", "SUCCEEDED"]
    assert scheduler.snapshot()["reserved_gpu_mib"] == 0
    scheduler.close()


def test_each_task_gets_an_independent_execution_deadline() -> None:
    scheduler = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(task_timeout_seconds=0.12, queue_timeout_seconds=1.0),
        gpu_probe=_gpu,
    )
    try:
        def executor(_: LogicalSubagentTask, __: threading.Event) -> dict[str, object]:
            time.sleep(0.07)
            return {"ok": True}

        outcomes = scheduler.run_batch((_task("first"), _task("second")), executor)
        assert [outcome.state for outcome in outcomes] == ["SUCCEEDED", "SUCCEEDED"]
        assert outcomes[1].elapsed_seconds >= 0.06
    finally:
        scheduler.close()


def test_expired_queue_and_restart_start_with_no_reservations() -> None:
    blocked = GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(queue_timeout_seconds=0.03),
        gpu_probe=lambda: GpuAvailability.unavailable("temporary"),
    )
    try:
        outcome = blocked.run_batch((_task("expires"),), lambda *_: {"unexpected": True})[0]
        assert outcome.state == "FAILED"
        assert outcome.failure_category == "subagent_queue_timeout"
        assert blocked.snapshot()["reserved_gpu_mib"] == 0
    finally:
        blocked.close()
    restarted = GpuAwareSubagentScheduler(gpu_probe=_gpu)
    try:
        assert restarted.snapshot()["reserved_gpu_mib"] == 0
        assert restarted.snapshot()["gpu_reservations"] == {}
    finally:
        restarted.close()
