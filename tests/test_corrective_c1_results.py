from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import pytest

from research_workspace.logical_subagents import (
    GpuAvailability,
    GpuAwareSubagentScheduler,
    LogicalSubagentError,
    LogicalSubagentTask,
    SubagentSchedulerPolicy,
)
from research_workspace.zetsu_results import ZetsuResultStore


def _task(task_id: str, *, owner_id: str = "owner") -> LogicalSubagentTask:
    return LogicalSubagentTask(
        task_id=task_id,
        parent_task_id="parent",
        owner_id=owner_id,
        project_id="project",
        goal=f"inspect {task_id}",
        context_digest="a" * 64,
    )


def _gpu() -> GpuAvailability:
    return GpuAvailability(
        status="AVAILABLE",
        total_mib=16_000,
        used_mib=0,
        free_mib=16_000,
        source="c1-test",
        captured_at_utc="2026-08-25T00:00:00Z",
    )


def _scheduler(root: Path, *, cache_size: int = 64) -> GpuAwareSubagentScheduler:
    return GpuAwareSubagentScheduler(
        policy=SubagentSchedulerPolicy(completed_cache_size=cache_size),
        gpu_probe=_gpu,
        result_store=ZetsuResultStore(root / "results"),
    )


def _reconstruct(scheduler: GpuAwareSubagentScheduler, task: LogicalSubagentTask, result_id: str) -> bytes:
    result = bytearray()
    offset = 0
    while True:
        page = scheduler.page_result(task=task, result_id=result_id, offset=offset, max_bytes=17)
        result.extend(base64.b64decode(str(page["content_base64"])))
        next_offset = page["next_offset"]
        if next_offset is None:
            return bytes(result)
        offset = int(next_offset)


def test_below_limit_result_is_unchanged_and_durable(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    task = _task("small")
    expected = {"summary": "ok", "unicode": "αβγ"}
    try:
        outcome = scheduler.run_batch((task,), lambda *_: expected)[0]
        assert outcome.state == "SUCCEEDED"
        assert outcome.result == expected
        assert outcome.delivery_status == "INLINE"
        assert outcome.result_id
        assert json.loads(_reconstruct(scheduler, task, str(outcome.result_id))) == expected
    finally:
        scheduler.close()


def test_oversized_result_is_success_and_exactly_paged(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    task = _task("large")
    expected = {"text": "αβγ" * 70_000, "tail": ["終", 1, True]}
    calls = 0

    def executor(_: LogicalSubagentTask, __: threading.Event) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return expected

    try:
        outcome = scheduler.run_batch((task,), executor)[0]
        assert calls == 1
        assert outcome.state == "SUCCEEDED"
        assert outcome.delivery_status == "PAGED"
        assert isinstance(outcome.result, dict)
        assert outcome.result["execution_status"] == "SUCCEEDED"
        assert outcome.result["delivery_status"] == "PAGED"
        reconstructed = _reconstruct(scheduler, task, str(outcome.result_id))
        assert reconstructed == json.dumps(
            expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert json.loads(reconstructed) == expected
        repeat = scheduler.page_result(task=task, result_id=str(outcome.result_id), max_bytes=17)
        assert repeat["offset"] == 0
        assert calls == 1
    finally:
        scheduler.close()


def test_result_authorization_cursor_and_restart_are_fail_closed(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    task = _task("protected")
    try:
        outcome = scheduler.run_batch((task,), lambda *_: {"value": "α" * 100_000})[0]
        result_id = str(outcome.result_id)
        foreign = _task("protected", owner_id="foreign")
        with pytest.raises(LogicalSubagentError, match="zetsu_result_not_found"):
            scheduler.page_result(task=foreign, result_id=result_id)
        with pytest.raises(LogicalSubagentError, match="zetsu_result_offset_invalid"):
            scheduler.page_result(task=task, result_id=result_id, offset=-1)
        restarted = ZetsuResultStore(tmp_path / "results")
        page = restarted.page(
            user_id=task.owner_id,
            repo_id=task.project_id,
            session_id=task.task_id,
            result_id=result_id,
            artifact="result.json",
            offset=0,
            max_bytes=64,
        )
        assert page["result_id"] == result_id
    finally:
        scheduler.close()


def test_cache_eviction_does_not_delete_durable_result(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, cache_size=1)
    first = _task("first")
    second = _task("second")
    try:
        first_outcome = scheduler.run_batch((first,), lambda *_: {"value": "α" * 100_000})[0]
        scheduler.run_batch((second,), lambda *_: {"value": "second"})
        snapshot = scheduler.snapshot()
        assert snapshot["completed"] == 1
        assert snapshot["completed_cache_size"] == 1
        assert json.loads(_reconstruct(scheduler, first, str(first_outcome.result_id)))["value"].startswith("α")
    finally:
        scheduler.close()
