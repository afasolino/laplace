from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from research_workspace.hooks import (
    HookCallback,
    HookContext,
    HookError,
    HookFailurePolicy,
    HookResult,
    HookReport,
    HookSecurityError,
    HookService,
    HookStage,
    HookBlockedError,
    HookCancelledError,
)
from research_workspace.laplace_core import LaplaceCore


def _emit(
    service: HookService, key: str, *, owner: str = "alice", project: str = "lab"
) -> HookReport:
    return service.emit(
        HookStage.PRE_MUTATION,
        owner_id=owner,
        project_id=project,
        session_id="session-1",
        task_id="task-1",
        idempotency_key=key,
        payload={"path": "src/example.py"},
    )


def _append(values: list[str], value: str) -> HookCallback:
    def callback(_: HookContext) -> None:
        values.append(value)

    return callback


def test_order_disabled_and_duplicate_replay_are_deterministic(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")
    order: list[str] = []
    service.register("late", HookStage.PRE_MUTATION, _append(order, "late"), priority=2)
    service.register("early", HookStage.PRE_MUTATION, _append(order, "early"), priority=-1)
    service.register("disabled", HookStage.PRE_MUTATION, _append(order, "disabled"), enabled=False)
    first = _emit(service, "mutation-1")
    replay = _emit(service, "mutation-1")
    assert order == ["early", "late"]
    assert first.executed == ("early", "late")
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.executed == first.executed


def test_all_declared_stages_are_typed_and_owner_project_scoped(tmp_path: Path) -> None:
    assert {stage.value for stage in HookStage} == {
        "TASK_START",
        "TASK_RESUME",
        "TASK_CANCEL",
        "PRE_RETRIEVAL",
        "POST_RETRIEVAL",
        "PRE_MEMORY_WRITE",
        "POST_MEMORY_WRITE",
        "PRE_MUTATION",
        "POST_MUTATION",
        "PRE_VERIFY",
        "POST_VERIFY",
        "TASK_COMPLETE",
        "TASK_FAILURE",
        "IDLE_START",
        "IDLE_END",
    }
    service = HookService(tmp_path / "hooks")
    calls: list[str] = []
    service.register("global", HookStage.PRE_MUTATION, _append(calls, "global"))
    service.register(
        "alice",
        HookStage.PRE_MUTATION,
        _append(calls, "alice"),
        owner_id="alice",
    )
    service.register(
        "alice-lab",
        HookStage.PRE_MUTATION,
        _append(calls, "alice-lab"),
        owner_id="alice",
        project_id="lab",
    )
    service.register(
        "bob",
        HookStage.PRE_MUTATION,
        _append(calls, "bob"),
        owner_id="bob",
    )
    _emit(service, "alice-event")
    _emit(service, "bob-event", owner="bob", project="other")
    assert calls == ["global", "alice", "alice-lab", "global", "bob"]
    with pytest.raises(HookError, match="project_hook_requires_owner"):
        service.register("bad-scope", HookStage.TASK_START, lambda _: None, project_id="lab")


def test_security_pre_failure_timeout_and_block_fail_closed(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")

    def fail(_: HookContext) -> None:
        raise ValueError("bad hook")

    service.register("fails", HookStage.PRE_MUTATION, fail)
    with pytest.raises(HookSecurityError, match="hook_exception"):
        _emit(service, "exception")

    restarted = HookService(tmp_path / "hooks")
    with pytest.raises(HookSecurityError, match="hook_replayed_failure"):
        restarted.emit(
            HookStage.PRE_MUTATION,
            owner_id="alice",
            project_id="lab",
            session_id="session-1",
            task_id="task-1",
            idempotency_key="exception",
            payload={"path": "src/example.py"},
        )

    timeout_service = HookService(tmp_path / "timeout")

    def slow(_: HookContext) -> None:
        time.sleep(0.05)

    timeout_service.register(
        "slow",
        HookStage.PRE_MUTATION,
        slow,
        timeout_seconds=0.005,
    )
    with pytest.raises(HookSecurityError, match="hook_timeout"):
        _emit(timeout_service, "timeout")

    block_service = HookService(tmp_path / "block")
    block_service.register("block", HookStage.PRE_MUTATION, lambda _: HookResult(cancel=True))
    with pytest.raises(HookBlockedError, match="hook_blocked"):
        _emit(block_service, "blocked")


def test_post_failures_are_observability_only_and_nonsecurity_pre_can_continue(
    tmp_path: Path,
) -> None:
    service = HookService(tmp_path / "hooks")
    seen: list[str] = []

    def post_failure(_: HookContext) -> None:
        raise RuntimeError("metrics sink unavailable")

    def after(_: HookContext) -> HookResult:
        seen.append("after")
        return HookResult(cancel=True)

    service.register("post", HookStage.POST_VERIFY, post_failure)
    service.register(
        "after",
        HookStage.POST_VERIFY,
        after,
    )
    report = service.emit(
        HookStage.POST_VERIFY,
        owner_id="alice",
        project_id="lab",
        session_id="session-1",
        task_id="task-1",
        idempotency_key="verify-1",
        payload={"returncode": 0},
    )
    assert report.continued is True
    assert report.blocked is False
    assert len(report.failures) == 1
    assert seen == ["after"]

    pre_observe = HookService(tmp_path / "pre-observe")
    pre_observe.register(
        "optional",
        HookStage.TASK_START,
        post_failure,
        security_critical=False,
        failure_policy=HookFailurePolicy.OBSERVE,
    )
    assert pre_observe.emit(
        HookStage.TASK_START,
        owner_id="alice",
        project_id="lab",
        session_id="session-1",
        task_id="task-1",
        idempotency_key="start-1",
    ).continued is True
    with pytest.raises(HookError, match="post_hook_must_be_observability_only"):
        pre_observe.register(
            "bad-post",
            HookStage.POST_VERIFY,
            lambda _: None,
            security_critical=True,
        )


def test_cancellation_task_cancel_hook_runs_and_other_hooks_stop(tmp_path: Path) -> None:
    service = HookService(tmp_path / "hooks")
    signal = threading.Event()
    signal.set()
    called: list[str] = []
    service.register("pre", HookStage.PRE_VERIFY, _append(called, "pre"))
    with pytest.raises(HookCancelledError, match="hook_cancelled"):
        service.emit(
            HookStage.PRE_VERIFY,
            owner_id="alice",
            project_id="lab",
            session_id="session-1",
            task_id="task-1",
            idempotency_key="cancelled-pre",
            cancel_event=signal,
        )
    assert called == []

    service.register("cancel", HookStage.TASK_CANCEL, _append(called, "cancel"))
    report = service.emit(
        HookStage.TASK_CANCEL,
        owner_id="alice",
        project_id="lab",
        session_id="session-1",
        task_id="task-1",
        idempotency_key="cancel-event",
        cancel_event=signal,
    )
    assert report.continued is True
    assert called == ["cancel"]


def test_restart_resume_and_idempotency_conflict_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "hooks"
    service = HookService(root)
    calls: list[str] = []
    service.register("resume", HookStage.TASK_RESUME, _append(calls, "resume"))
    service.emit(
        HookStage.TASK_RESUME,
        owner_id="alice",
        project_id="lab",
        session_id="session-1",
        task_id="task-1",
        idempotency_key="resume-1",
        payload={"checkpoint": "c1"},
    )
    restarted = HookService(root)
    restarted.register("resume", HookStage.TASK_RESUME, _append(calls, "replayed"))
    assert restarted.emit(
        HookStage.TASK_RESUME,
        owner_id="alice",
        project_id="lab",
        session_id="session-1",
        task_id="task-1",
        idempotency_key="resume-1",
        payload={"checkpoint": "c1"},
    ).replayed is True
    assert calls == ["resume"]
    with pytest.raises(HookError, match="hook_idempotency_conflict"):
        restarted.emit(
            HookStage.TASK_RESUME,
            owner_id="alice",
            project_id="lab",
            session_id="session-1",
            task_id="task-1",
            idempotency_key="resume-1",
            payload={"checkpoint": "different"},
        )


def test_core_exposes_shared_hooks_without_zetsu(tmp_path: Path) -> None:
    core = LaplaceCore(tmp_path, object(), object())  # type: ignore[arg-type]
    observed: list[str] = []
    core.hooks.register("start", HookStage.TASK_START, _append(observed, "start"))
    report = core.emit_hook(
        HookStage.TASK_START,
        owner_id="alice",
        project_id="lab",
        session_id="session-1",
        task_id="task-1",
        idempotency_key="core-start",
    )
    assert report.continued is True
    assert observed == ["start"]
    assert core.hooks is core.hooks
