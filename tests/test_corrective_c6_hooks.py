from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from research_workspace.hooks import (
    HookContext,
    HookError,
    HookReport,
    HookResult,
    HookSecurityError,
    HookService,
    HookStage,
)


def _emit(service: HookService, key: str, stage: HookStage = HookStage.PRE_MUTATION) -> HookReport:
    return service.emit(
        stage,
        owner_id="owner",
        project_id="project",
        session_id="session",
        task_id="task",
        idempotency_key=key,
        payload={"key": key},
    )


def test_cooperative_timeout_signals_context_and_records_late_completion(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")
    stopped = threading.Event()

    def cooperative(context: HookContext) -> None:
        # Wait for the dispatcher's explicit cooperative cancellation signal.
        # Checking the clock here made the test race the worker startup under
        # load: a callback scheduled after its short deadline could return
        # before ``Future.result`` observed the timeout.
        while not context.cancellation_requested:
            time.sleep(0.001)
        stopped.set()

    service.register("cooperative", HookStage.PRE_MUTATION, cooperative, timeout_seconds=0.005)
    with pytest.raises(HookSecurityError, match="hook_timeout") as failure:
        _emit(service, "cooperative-timeout")
    assert failure.value.evidence["cooperative_cancel_requested"] is True
    assert failure.value.evidence["physical_termination_guaranteed"] is False
    assert stopped.wait(1)
    deadline = time.monotonic() + 1
    while service.diagnostics()["late_completion_count"] < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service.diagnostics()["late_completion_count"] == 1
    assert service.diagnostics()["physical_termination_guaranteed"] is False
    service.close()


def test_ignoring_callback_cannot_publish_late_authoritative_result(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")
    authoritative: list[str] = []

    def ignores_deadline(context: HookContext) -> HookResult:
        time.sleep(0.03)
        if not context.deadline_exceeded and not context.cancellation_requested:
            authoritative.append("late")
        return HookResult(context_modification="must-be-discarded")

    service.register("ignores", HookStage.PRE_MUTATION, ignores_deadline, timeout_seconds=0.001)
    with pytest.raises(HookSecurityError, match="hook_timeout"):
        _emit(service, "ignored-timeout")
    time.sleep(0.06)
    assert authoritative == []
    diagnostics = service.diagnostics()
    assert diagnostics["late_completion_count"] == 1
    assert diagnostics["late_completions"][0]["authoritative_result_discarded"] is True
    service.close()


def test_repeated_ignored_timeouts_use_bounded_workers(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")

    def ignored(_: HookContext) -> None:
        time.sleep(0.025)

    service.register("ignored", HookStage.PRE_MUTATION, ignored, timeout_seconds=0.001)
    for index in range(20):
        with pytest.raises(HookError, match="hook_timeout"):
            _emit(service, f"repeated-{index}")
    time.sleep(0.08)
    diagnostics = service.diagnostics()
    assert diagnostics["worker_limit"] == 4
    assert diagnostics["peak_active_callbacks"] <= 4
    assert diagnostics["active_callbacks"] == 0
    service.close()


def test_post_timeout_remains_observational(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")

    def slow(_: HookContext) -> None:
        time.sleep(0.02)

    service.register("post-slow", HookStage.POST_VERIFY, slow, timeout_seconds=0.001)
    report = _emit(service, "post-timeout", HookStage.POST_VERIFY)
    assert report.continued is True
    assert report.blocked is False
    assert report.failures[0].category == "hook_timeout"
    service.close()


def test_shutdown_with_active_hook_is_explicit_and_rejects_new_work(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")
    started = threading.Event()
    release = threading.Event()
    result: list[HookReport] = []

    def active(_: HookContext) -> None:
        started.set()
        release.wait(1)

    service.register("active", HookStage.TASK_START, active, timeout_seconds=1.0)
    def run_active_hook() -> None:
        result.append(_emit(service, "active-hook", HookStage.TASK_START))

    thread = threading.Thread(target=run_active_hook)
    thread.start()
    assert started.wait(1)
    service.close()
    assert service.diagnostics()["closed"] is True
    with pytest.raises(HookError, match="hook_service_closed"):
        _emit(service, "after-close", HookStage.TASK_START)
    release.set()
    thread.join(1)
    assert result[0].continued is True
