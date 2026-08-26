from __future__ import annotations

import base64
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from research_workspace.agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentToolPolicy,
)
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.operator_api import OperatorApiSettings, OperatorAuth, create_operator_app
from research_workspace.operator_service import OperatorService
from research_workspace.service_tiers import LanePolicy, ModelLane, ModelRoute
from research_workspace.zetsu_results import ZetsuResultError, ZetsuResultStore
from research_workspace.zetsu_agent import ZetsuAgentCoordinator
from research_workspace.zetsu_scheduler import (
    CERTIFIED_AGENT_CAPACITIES,
    AgentCapacityPolicy,
    AgentSchedulerError,
    AgentTaskScheduler,
)
from research_workspace import zetsu_runtime


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    (root / "value.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    return _git(root, "rev-parse", "HEAD")


def _manager(
    tmp_path: Path, *, quota: int = 2
) -> tuple[AgentSandboxManager, RepositoryAuthorizationStore, str]:
    repository = tmp_path / "repository"
    revision = _repository(repository)
    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    authorizations.register("repo", repository)
    authorizations.grant("owner", "repo", base_revision=revision)
    return (
        AgentSandboxManager(
            tmp_path / "worktrees",
            authorizations,
            per_user_quota=quota,
            global_quota=max(quota, 4),
        ),
        authorizations,
        revision,
    )


def _binding(manager: AgentSandboxManager, session_id: str) -> Path:
    binding = manager.create(
        user_id="owner",
        repo_id="repo",
        session_id=session_id,
        tool_policy=AgentToolPolicy("bounded", ("apply_patch", "run_tests")),
    )
    return Path(binding.worktree_root)


def _persist_and_release(
    manager: AgentSandboxManager,
    session_id: str,
    *,
    content: bytes,
) -> tuple[str, bytes]:
    patch = manager.patch(session_id, user_id="owner")
    store = ZetsuResultStore(manager.sandbox_root / "zetsu_agent_results")
    delivery = store.persist(
        user_id="owner",
        repo_id="repo",
        session_id=session_id,
        status="SUCCESS",
        summary="verified",
        artifacts={"result.json": content, "handoff.patch": patch},
    )
    manager.record_result(
        session_id,
        user_id="owner",
        command_count=1,
        verification_summary="PASSED:fixture",
        terminal=True,
    )
    released = manager.authorize_terminal_cleanup(
        session_id,
        user_id="owner",
        result_id=str(delivery["result_id"]),
    )
    assert released["action"] in {"RELEASED", "ALREADY_RELEASED"}
    return str(delivery["result_id"]), patch


def test_more_sequential_terminal_tasks_than_quota_release_physical_worktrees(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path, quota=2)
    for index in range(20):
        session_id = f"session-{index}"
        root = _binding(manager, session_id)
        (root / "value.txt").write_text(f"value-{index}\n", encoding="utf-8")
        result_id, _ = _persist_and_release(
            manager,
            session_id,
            content=json.dumps({"index": index}).encode("utf-8"),
        )
        assert not root.exists()
        historical = manager.status(session_id, user_id="owner")
        assert historical["status"] == "TERMINAL_RELEASED"
        assert historical["result_id"] == result_id


def test_real_live_quota_and_resumable_interruption_are_retained_across_restart(
    tmp_path: Path,
) -> None:
    manager, authorizations, _ = _manager(tmp_path, quota=2)
    first = _binding(manager, "live-one")
    _binding(manager, "live-two")
    manager.start_task(
        "live-one", user_id="owner", lane="quality", sanitized_model_name="qwen"
    )
    checkpoint = (
        manager.sandbox_root
        / "zetsu_agent_checkpoints"
        / f"{hashlib.sha256(b'live-one').hexdigest()}.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "session_id": "live-one",
                "user_id_sha256": hashlib.sha256(b"owner").hexdigest(),
                "repo_id": "repo",
                "base_revision": manager.require_active(
                    "live-one", user_id="owner"
                ).base_revision,
                "worktree_head": _git(first, "rev-parse", "HEAD"),
                "worktree_status_sha256": hashlib.sha256(b"").hexdigest(),
                "changed_paths": [],
                "consumed_wall_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    manager.record_interrupted("live-one", user_id="owner", reason="operator_restart")
    with pytest.raises(AgentSandboxError, match="per_user_worktree_quota"):
        _binding(manager, "live-three")

    recovered = AgentSandboxManager(
        tmp_path / "worktrees",
        authorizations,
        per_user_quota=2,
        global_quota=4,
    )
    status = recovered.status("live-one", user_id="owner")
    assert status["lifecycle_state"] == "INTERRUPTED_RESUMABLE"
    assert first.is_dir()
    scheduler = AgentTaskScheduler(
        recovered.sandbox_root / "zetsu_agent_scheduler.sqlite3",
        recovered,
        AgentCapacityPolicy("full", 1, 4, "fixture"),
    )
    with pytest.raises(AgentSchedulerError, match="agent_queue_wait_timeout"):
        scheduler.wait_for_admission(
            user_id="owner",
            repo_id="repo",
            session_id="live-three",
            instruction_digest="c" * 64,
            wait_timeout_seconds=0.05,
        )
    assert not (recovered.sandbox_root / "owner" / "live-three").exists()


def test_restart_recovers_cleanup_after_durable_terminal_commit(tmp_path: Path) -> None:
    manager, authorizations, _ = _manager(tmp_path)
    root = _binding(manager, "restart-cleanup")
    (root / "value.txt").write_text("terminal\n", encoding="utf-8")
    patch = manager.patch("restart-cleanup", user_id="owner")
    store = ZetsuResultStore(manager.sandbox_root / "zetsu_agent_results")
    delivery = store.persist(
        user_id="owner",
        repo_id="repo",
        session_id="restart-cleanup",
        status="SUCCESS",
        summary="durable before crash",
        artifacts={"result.json": b"{}\n", "handoff.patch": patch},
    )
    manager.record_result(
        "restart-cleanup",
        user_id="owner",
        command_count=1,
        verification_summary="PASSED",
        terminal=True,
    )
    # Simulate a crash after cleanup authorization was durably committed.
    with manager._connect() as connection:
        connection.execute(
            """
            UPDATE worktree_sessions SET cleanup_eligible=1, result_id=?
            WHERE session_id='restart-cleanup'
            """,
            (delivery["result_id"],),
        )
    assert root.is_dir()

    recovered = AgentSandboxManager(tmp_path / "worktrees", authorizations)
    assert not root.exists()
    status = recovered.status("restart-cleanup", user_id="owner")
    assert status["status"] == "TERMINAL_RELEASED"


def test_three_expired_active_and_five_failed_are_reconciled_before_admission(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path, quota=8)
    roots: list[Path] = []
    for index in range(8):
        session_id = f"saturated-{index}"
        roots.append(_binding(manager, session_id))
    for index in range(3, 8):
        manager.record_result(
            f"saturated-{index}",
            user_id="owner",
            command_count=0,
            verification_summary="FAILED:fixture",
            failed=True,
        )
    with manager._connect() as connection:
        connection.execute(
            """
            UPDATE worktree_sessions SET created_at_utc='2020-01-01T00:00:00+00:00',
                updated_at_utc='2020-01-01T00:00:00+00:00',
                wall_deadline_utc='2020-01-01T00:30:00+00:00'
            WHERE session_id IN ('saturated-0','saturated-1','saturated-2')
            """
        )

    report = manager.collect_garbage(dry_run=False, user_id="owner", limit=16)
    assert report["released"] == 8, report
    assert all(not root.exists() for root in roots)
    assert manager.capacity_snapshot(user_id="owner")["user_live_resumable_worktrees"] == 0
    next_root = _binding(manager, "after-reconciliation")
    assert next_root.is_dir()
    for index in range(8):
        historical = manager.status(f"saturated-{index}", user_id="owner")
        assert historical["status"] == "TERMINAL_RELEASED"
        assert isinstance(historical["result_id"], str)


def test_fifo_queue_waits_without_allocating_worktree_then_wakes_on_release(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path, quota=2)
    scheduler = AgentTaskScheduler(
        manager.sandbox_root / "zetsu_agent_scheduler.sqlite3",
        manager,
        AgentCapacityPolicy("full", execution_slots=1, queue_capacity=4, certification="fixture"),
    )
    first = scheduler.wait_for_admission(
        user_id="owner",
        repo_id="repo",
        session_id="queue-first",
        instruction_digest="a" * 64,
        wait_timeout_seconds=10,
    )
    first_root = _binding(manager, "queue-first")
    admitted: list[str] = []
    failures: list[BaseException] = []

    def wait_second() -> None:
        try:
            ticket = scheduler.wait_for_admission(
                user_id="owner",
                repo_id="repo",
                session_id="queue-second",
                instruction_digest="b" * 64,
                wait_timeout_seconds=10,
            )
            admitted.append(ticket.session_id)
            scheduler.finish(ticket, state="SUCCEEDED")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    thread = threading.Thread(target=wait_second)
    thread.start()
    deadline = time.monotonic() + 5
    while scheduler.snapshot()["queued_agent_tasks"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert scheduler.task_status(user_id="owner", session_id="queue-second")["state"] == "QUEUED"
    assert not (manager.sandbox_root / "owner" / "queue-second").exists()
    assert scheduler.snapshot()["running_agent_tasks"] == 1

    manager.close_if_clean("queue-first", user_id="owner")
    assert not first_root.exists()
    scheduler.finish(first, state="SUCCEEDED")
    thread.join(timeout=5)
    assert failures == []
    assert admitted == ["queue-second"]


def test_queue_cancellation_timeout_full_and_restart_do_not_duplicate(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path, quota=2)
    database = manager.sandbox_root / "zetsu_agent_scheduler.sqlite3"
    policy = AgentCapacityPolicy("nocodev", 1, 1, "fixture")
    scheduler = AgentTaskScheduler(database, manager, policy)
    running = scheduler.wait_for_admission(
        user_id="owner",
        repo_id="repo",
        session_id="running",
        instruction_digest="a" * 64,
        wait_timeout_seconds=10,
    )
    outcomes: list[str] = []

    def wait(session_id: str, timeout: float) -> None:
        try:
            scheduler.wait_for_admission(
                user_id="owner",
                repo_id="repo",
                session_id=session_id,
                instruction_digest="b" * 64,
                wait_timeout_seconds=timeout,
            )
            outcomes.append("ADMITTED")
        except Exception as exc:
            outcomes.append(str(exc))

    queued = threading.Thread(target=wait, args=("cancel-me", 10.0))
    queued.start()
    deadline = time.monotonic() + 5
    while scheduler.snapshot()["queued_agent_tasks"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(AgentSchedulerError, match="agent_queue_full"):
        scheduler.wait_for_admission(
            user_id="owner",
            repo_id="repo",
            session_id="queue-full",
            instruction_digest="c" * 64,
            wait_timeout_seconds=1,
        )
    assert scheduler.cancel(user_id="owner", session_id="cancel-me")["cancelled"] is True
    queued.join(timeout=5)
    assert outcomes == ["agent_queue_cancelled"]

    timed = threading.Thread(target=wait, args=("time-out", 0.05))
    timed.start()
    timed.join(timeout=5)
    assert outcomes[-1] == "agent_queue_wait_timeout"
    recovered = AgentTaskScheduler(database, manager, policy)
    assert recovered.task_status(user_id="owner", session_id="running")["state"] == "FAILED"
    with pytest.raises(AgentSchedulerError, match="agent_queue_task_not_found"):
        recovered.task_status(user_id="foreign-owner", session_id="running")
    scheduler.finish(running, state="SUCCEEDED")


def test_certified_topology_capacity_is_conservative_without_matching_gpu() -> None:
    full = CERTIFIED_AGENT_CAPACITIES["full"]
    nocodev = CERTIFIED_AGENT_CAPACITIES["nocodev"]
    assert full.execution_slots == 1
    assert nocodev.execution_slots == full.execution_slots
    assert full.queue_capacity == nocodev.queue_capacity == 16
    assert "not_transferable" in nocodev.certification


@pytest.mark.parametrize("codev_enabled", [True, False], ids=["full", "nocodev"])
def test_topology_saturation_queues_before_worktree_or_model_use_and_is_fifo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codev_enabled: bool,
) -> None:
    manager, _, _ = _manager(tmp_path, quota=4)
    tiered = SimpleNamespace(
        sandboxes=manager,
        lane_policy=SimpleNamespace(codev_enabled=codev_enabled),
    )
    coordinator = ZetsuAgentCoordinator(cast(Any, tiered))
    initial_status = coordinator.scheduler_status(user_id="owner")
    assert initial_status["runtime_topology"] == (
        "full" if codev_enabled else "nocodev"
    )
    assert initial_status["agent_execution_slot_limit"] == 1
    assert initial_status["per_user_worktree_safety_limit"] == 4
    first_release = threading.Event()
    entered: list[str] = []
    results: list[str] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        session_id = str(kwargs["session_id"])
        entered.append(session_id)
        if session_id == "fifo-one":
            assert first_release.wait(timeout=5)
        return {"status": "SUCCESS", "session_id": session_id}

    monkeypatch.setattr(coordinator, "_run_unlocked", fake_run)

    def run(session_id: str) -> None:
        coordinator.run(
            user_id="owner",
            repo_id="repo",
            session_id=session_id,
            instruction=f"task {session_id}",
            wait_timeout_seconds=10,
        )
        results.append(session_id)

    first = threading.Thread(target=run, args=("fifo-one",))
    second = threading.Thread(target=run, args=("fifo-two",))
    third = threading.Thread(target=run, args=("fifo-three",))
    first.start()
    deadline = time.monotonic() + 5
    while entered != ["fifo-one"] and time.monotonic() < deadline:
        time.sleep(0.01)
    second.start()
    deadline = time.monotonic() + 5
    while coordinator.scheduler_status(user_id="owner")["queued_agent_tasks"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    third.start()
    deadline = time.monotonic() + 5
    while coordinator.scheduler_status(user_id="owner")["queued_agent_tasks"] != 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert entered == ["fifo-one"]
    assert coordinator.scheduler_status(user_id="owner")["running_agent_tasks"] == 1
    assert not (manager.sandbox_root / "owner" / "fifo-two").exists()
    assert not (manager.sandbox_root / "owner" / "fifo-three").exists()
    first_release.set()
    for thread in (first, second, third):
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert entered == ["fifo-one", "fifo-two", "fifo-three"]
    assert sorted(results) == ["fifo-one", "fifo-three", "fifo-two"]


def test_concurrent_queue_worktree_result_release_stress_has_no_accumulation(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path, quota=4)
    scheduler = AgentTaskScheduler(
        manager.sandbox_root / "zetsu_agent_scheduler.sqlite3",
        manager,
        AgentCapacityPolicy("full", 3, 32, "deterministic_non_gpu_fixture"),
    )
    failures: list[str] = []

    def execute(index: int) -> None:
        session_id = f"stress-{index:02d}"
        admission = None
        try:
            admission = scheduler.wait_for_admission(
                user_id="owner",
                repo_id="repo",
                session_id=session_id,
                instruction_digest=f"{index:064x}",
                wait_timeout_seconds=20,
            )
            root = _binding(manager, session_id)
            (root / "value.txt").write_text(f"stress-{index}\n", encoding="utf-8")
            result_id, _ = _persist_and_release(
                manager,
                session_id,
                content=json.dumps({"index": index}).encode(),
            )
            scheduler.finish(admission, state="SUCCEEDED", result_id=result_id)
        except Exception as exc:  # pragma: no cover - asserted below
            if admission is not None:
                scheduler.finish(
                    admission,
                    state="FAILED",
                    failure_category=type(exc).__name__,
                )
            failures.append(f"{session_id}:{exc}")

    threads = [threading.Thread(target=execute, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert failures == []
    assert manager.capacity_snapshot(user_id="owner")["user_live_resumable_worktrees"] == 0
    assert scheduler.snapshot()["running_agent_tasks"] == 0
    assert scheduler.snapshot()["queued_agent_tasks"] == 0


def test_scheduler_path_and_owner_status_are_fail_closed(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    foreign = tmp_path / "foreign.sqlite3"
    scheduler_path = manager.sandbox_root / "zetsu_agent_scheduler.sqlite3"
    scheduler_path.symlink_to(foreign)
    with pytest.raises(AgentSchedulerError, match="agent_scheduler_path_invalid"):
        AgentTaskScheduler(
            scheduler_path,
            manager,
            AgentCapacityPolicy("full", 1, 4, "fixture"),
        )
    assert not foreign.exists()


def test_gc_protects_dirty_unfinalized_and_forged_ownership_metadata(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    root = _binding(manager, "protected")
    (root / "value.txt").write_text("dirty\n", encoding="utf-8")
    manager.record_result(
        "protected",
        user_id="owner",
        command_count=1,
        verification_summary="PASSED",
        terminal=True,
    )
    # Terminal history alone never authorizes deletion; exact result persistence is required.
    assert manager.collect_garbage(dry_run=False)["examined"] == 0
    assert root.is_dir()

    store = ZetsuResultStore(manager.sandbox_root / "zetsu_agent_results")
    delivery = store.persist(
        user_id="owner",
        repo_id="repo",
        session_id="protected",
        status="SUCCESS",
        summary="saved",
        artifacts={
            "result.json": b"{}\n",
            "handoff.patch": b"forged-but-owner-bound-test-artifact\n",
        },
    )
    with manager._connect() as connection:
        connection.execute(
            """
            UPDATE worktree_sessions SET cleanup_eligible=1, result_id=?
            WHERE session_id='protected'
            """,
            (delivery["result_id"],),
        )
    marker = manager._ownership_path("protected")
    original = marker.read_text(encoding="utf-8")
    forged = json.loads(original)
    forged["user_id"] = "foreign-owner"
    marker.write_text(json.dumps(forged), encoding="utf-8")
    report = manager.collect_garbage(dry_run=False)
    assert report["items"][0]["action"] == "PROTECTED"
    assert report["items"][0]["reason"] == "ownership_proof_invalid"
    assert root.is_dir()

    marker.unlink()
    marker.symlink_to(root / "value.txt")
    report = manager.collect_garbage(dry_run=False)
    assert report["items"][0]["action"] == "PROTECTED"
    assert root.is_dir()


def test_gc_never_deletes_ordinary_or_foreign_layout_even_with_forged_row_and_marker(
    tmp_path: Path,
) -> None:
    manager, authorizations, _ = _manager(tmp_path)
    agent_root = _binding(manager, "forged-layout")
    manager.record_result(
        "forged-layout",
        user_id="owner",
        command_count=0,
        verification_summary="PASSED",
        terminal=True,
    )
    store = ZetsuResultStore(manager.sandbox_root / "zetsu_agent_results")
    delivery = store.persist(
        user_id="owner",
        repo_id="repo",
        session_id="forged-layout",
        status="SUCCESS",
        summary="saved",
        artifacts={"result.json": b"{}\n"},
    )
    ordinary = authorizations.repository("repo").canonical_root
    with manager._connect() as connection:
        row = connection.execute(
            "SELECT ownership_token, base_revision FROM worktree_sessions WHERE session_id=?",
            ("forged-layout",),
        ).fetchone()
        assert row is not None
        connection.execute(
            """
            UPDATE worktree_sessions
            SET worktree_root=?, cleanup_eligible=1, result_id=?
            WHERE session_id='forged-layout'
            """,
            (str(ordinary), delivery["result_id"]),
        )
    marker = manager._ownership_path("forged-layout")
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "forged-layout",
                "user_id": "owner",
                "repo_id": "repo",
                "canonical_repository_root": str(ordinary),
                "worktree_root": str(ordinary),
                "base_revision": str(row["base_revision"]),
                "ownership_token": str(row["ownership_token"]),
            }
        ),
        encoding="utf-8",
    )
    report = manager.collect_garbage(dry_run=False)
    assert report["items"][0]["action"] == "PROTECTED"
    assert report["items"][0]["reason"] == "worktree_path_not_owned_layout"
    assert ordinary.is_dir()
    assert (ordinary / ".git").exists()
    assert agent_root.is_dir()


def test_verified_clean_legacy_zetsu_terminal_migrates_before_release(
    tmp_path: Path,
) -> None:
    manager, authorizations, revision = _manager(tmp_path)
    binding = manager.create(
        user_id="owner",
        repo_id="repo",
        session_id="zetsu-legacy-terminal",
        tool_policy=AgentToolPolicy("bounded", ("apply_patch", "run_tests")),
        task_title="Zetsu Qwen delegated task",
    )
    manager.record_result(
        binding.session_id,
        user_id="owner",
        command_count=0,
        verification_summary="PASSED:legacy",
    )
    checkpoint = (
        manager.sandbox_root
        / "zetsu_agent_checkpoints"
        / f"{hashlib.sha256(binding.session_id.encode()).hexdigest()}.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "session_id": binding.session_id,
                "user_id_sha256": hashlib.sha256(b"owner").hexdigest(),
                "repo_id": "repo",
                "base_revision": revision,
                "changed_paths": [],
            }
        ),
        encoding="utf-8",
    )
    manager._ownership_path(binding.session_id).unlink()
    with manager._connect() as connection:
        connection.execute(
            "UPDATE worktree_sessions SET ownership_token=NULL WHERE session_id=?",
            (binding.session_id,),
        )

    recovered = AgentSandboxManager(
        manager.sandbox_root,
        authorizations,
        recover_cleanup=False,
    )
    dry_run = recovered.collect_garbage(dry_run=True)
    assert dry_run["items"][0]["action"] == "WOULD_MIGRATE_AND_RELEASE"
    assert Path(binding.worktree_root).is_dir()
    actual = recovered.collect_garbage(dry_run=False)
    assert actual["released"] == 1
    assert not Path(binding.worktree_root).exists()
    historical = recovered.status(binding.session_id, user_id="owner")
    assert historical["result_id"]

def test_result_pages_are_exact_bounded_and_authorization_isolated_after_cleanup(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path)
    root = _binding(manager, "large-result")
    (root / "value.txt").write_text("changed\n", encoding="utf-8")
    payload = ("αβγ\n" * 20_000).encode("utf-8")
    result_id, patch = _persist_and_release(manager, "large-result", content=payload)
    assert not root.exists()
    store = ZetsuResultStore(manager.sandbox_root / "zetsu_agent_results")

    reconstructed = bytearray()
    offset = 0
    while True:
        page = store.page(
            user_id="owner",
            repo_id="repo",
            session_id="large-result",
            result_id=result_id,
            artifact="result.json",
            offset=offset,
            max_bytes=4097,
        )
        reconstructed.extend(base64.b64decode(str(page["content_base64"])))
        next_offset = page["next_offset"]
        if next_offset is None:
            break
        offset = int(next_offset)
    assert bytes(reconstructed) == payload
    patch_page = store.page(
        user_id="owner",
        repo_id="repo",
        session_id="large-result",
        result_id=result_id,
        artifact="handoff.patch",
        offset=0,
        max_bytes=65_536,
    )
    assert base64.b64decode(str(patch_page["content_base64"])) == patch
    with pytest.raises(ZetsuResultError, match="zetsu_result_not_found"):
        store.page(
            user_id="foreign",
            repo_id="repo",
            session_id="large-result",
            result_id=result_id,
            artifact="result.json",
            offset=0,
            max_bytes=512,
        )
    with pytest.raises(ZetsuResultError, match="zetsu_result_artifact_invalid"):
        store.page(
            user_id="owner",
            repo_id="repo",
            session_id="large-result",
            result_id=result_id,
            artifact="../manifest.json",
            offset=0,
            max_bytes=512,
        )
    with pytest.raises(ZetsuResultError, match="zetsu_result_not_found"):
        store.page(
            user_id="owner",
            repo_id="foreign-repo",
            session_id="large-result",
            result_id=result_id,
            artifact="result.json",
            offset=0,
            max_bytes=512,
        )


def _runtime_record(state: Path, repository: Path, runtime_id: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "runtime_id": runtime_id,
        "boot_id": zetsu_runtime._boot_id(),
        "repository": str(repository.resolve()),
        "state_root": str(state.resolve()),
        "topology": "nocodev",
        "codev": "intentionally_disabled",
        "services": {},
    }


def test_stop_recovers_surviving_worker_after_supervisor_exit_and_is_idempotent(
    tmp_path: Path,
) -> None:
    state = (tmp_path / "state").resolve()
    repository = (tmp_path / "repository").resolve()
    state.mkdir()
    repository.mkdir()
    runtime_id = "r" * 64
    environment = {
        **os.environ,
        "LAPLACE_ZETSU_RUNTIME_ID": runtime_id,
        "LAPLACE_ZETSU_STATE_ROOT": str(state),
        "LAPLACE_ZETSU_REPOSITORY": str(repository),
    }
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
                "raise SystemExit(0)"
            ),
        ],
        env=environment,
        start_new_session=True,
    )
    supervisor.wait(timeout=10)
    record = _runtime_record(state, repository, runtime_id)
    deadline = time.monotonic() + 10
    while not zetsu_runtime._runtime_owned_processes(record) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert zetsu_runtime._runtime_owned_processes(record)
    record_path = state / "run/zetsu_services.json"
    zetsu_runtime._atomic_json(record_path, record)

    stopped = zetsu_runtime.stop_local_runtime(state)
    assert stopped["status"] == "STOPPED"
    assert stopped["diagnostics"]["initial_owned_pids"]
    assert zetsu_runtime._runtime_owned_processes(record) == {}
    assert zetsu_runtime.stop_local_runtime(state)["status"] == "STOPPED"


def test_stop_never_signals_foreign_or_pid_reuse_identity(tmp_path: Path) -> None:
    state = (tmp_path / "state").resolve()
    repository = (tmp_path / "repository").resolve()
    state.mkdir()
    repository.mkdir()
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        record = _runtime_record(state, repository, "owned-token-that-foreign-does-not-have")
        record["services"] = {
            "qwen": {"process": zetsu_runtime.asdict(zetsu_runtime._process_identity(foreign.pid))}
        }
        zetsu_runtime._atomic_json(state / "run/zetsu_services.json", record)
        stopped = zetsu_runtime.stop_local_runtime(state)
        assert stopped["status"] == "STOPPED"
        assert stopped["diagnostics"]["initial_owned_pids"] == []
        assert foreign.poll() is None
    finally:
        foreign.send_signal(signal.SIGTERM)
        foreign.wait(timeout=10)


@pytest.mark.anyio
async def test_nocodev_readiness_is_ready_with_only_qwen_and_operator(
    tmp_path: Path, monkeypatch
) -> None:
    qwen_model = "fixture-qwen"
    routes = {
        ModelLane.QUALITY: ModelRoute(
            ModelLane.QUALITY, qwen_model, "http://127.0.0.1:8207", 0
        ),
        ModelLane.STANDARD: ModelRoute(
            ModelLane.STANDARD, qwen_model, "http://127.0.0.1:8207", 10
        ),
        ModelLane.ECONOMY: ModelRoute(
            ModelLane.ECONOMY, "fixture-codev", "http://127.0.0.1:8103", 20
        ),
    }
    tiered = SimpleNamespace(lane_policy=LanePolicy(routes, codev_enabled=False))
    body = json.dumps({"data": [{"id": qwen_model}]}).encode("utf-8")

    class Reader:
        async def readuntil(self, _separator: bytes) -> bytes:
            return f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()

        async def readexactly(self, _length: int) -> bytes:
            return body

        async def read(self, _length: int) -> bytes:
            return body

    class Writer:
        def write(self, _content: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    async def open_connection(_host: str, port: int) -> tuple[Reader, Writer]:
        assert port == 8207
        return Reader(), Writer()

    class Registry:
        snapshot = object()

    class Sessions:
        @staticmethod
        def active_count() -> int:
            return 0

    registered = SimpleNamespace(registry=Registry(), sessions=Sessions())
    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    app = create_operator_app(
        OperatorService(Path(__file__).resolve().parents[1], tmp_path / "operator"),
        OperatorAuth({"read-token-000000000000000000000000": "read"}),
        settings=OperatorApiSettings(
            fixture_mode=False,
            bearer_api_enabled=True,
            codev_enabled=False,
        ),
        tiered=cast(Any, tiered),
        registered_auth=cast(Any, registered),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        response = await client.get("/api/v1/readiness")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "READY"
    assert payload["codev"] == "intentionally_disabled"
    assert not any(reason.endswith(":economy") for reason in payload["reasons"])


def test_bounded_persistent_yield_remains_resumable(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path, quota=2)
    root = _binding(manager, "persistent-yield")

    checkpoint = (
        manager.sandbox_root
        / "zetsu_agent_checkpoints"
        / f"{hashlib.sha256(b'persistent-yield').hexdigest()}.json"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "session_id": "persistent-yield",
                "user_id_sha256": hashlib.sha256(b"owner").hexdigest(),
                "repo_id": "repo",
                "base_revision": manager.require_active(
                    "persistent-yield", user_id="owner"
                ).base_revision,
                "worktree_head": _git(root, "rev-parse", "HEAD"),
                "worktree_status_sha256": hashlib.sha256(b"").hexdigest(),
                "changed_paths": [],
                "consumed_wall_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    result_id = "res_" + ("1" * 32)
    recorded = manager.record_result(
        "persistent-yield",
        user_id="owner",
        command_count=12,
        verification_summary="INCOMPLETE:max_steps_exhausted",
        resumable=True,
        result_id=result_id,
    )

    assert recorded["state"] == "INTERRUPTED_RESUMABLE"
    assert recorded["result_id"] == result_id

    status = manager.status("persistent-yield", user_id="owner")
    assert status["lifecycle_state"] == "INTERRUPTED_RESUMABLE"
    assert status["result_id"] == result_id
    assert root.is_dir()

    with manager._connect() as connection:
        event = connection.execute(
            """
            SELECT event, state
            FROM worktree_events
            WHERE session_id=?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            ("persistent-yield",),
        ).fetchone()

    assert event is not None
    assert event["event"] == "TASK_YIELDED_RESUMABLE"
    assert event["state"] == "INTERRUPTED_RESUMABLE"

    reconciled = manager.reconcile(
        dry_run=False,
        user_id="owner",
        session_id="persistent-yield",
        limit=1,
    )
    assert reconciled["released"] == 0
    assert root.is_dir()
    assert (
        manager.status("persistent-yield", user_id="owner")["lifecycle_state"]
        == "INTERRUPTED_RESUMABLE"
    )


def test_record_result_rejects_conflicting_resumable_failure_state(
    tmp_path: Path,
) -> None:
    manager, _, _ = _manager(tmp_path)
    _binding(manager, "invalid-yield")

    with pytest.raises(AgentSandboxError, match="result_state_invalid"):
        manager.record_result(
            "invalid-yield",
            user_id="owner",
            command_count=1,
            verification_summary="FAILED:fixture",
            failed=True,
            resumable=True,
        )
