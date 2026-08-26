from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from research_workspace import zetsu_cli
from research_workspace.agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentSessionBinding,
    AgentToolPolicy,
)
from research_workspace.repository_authorization import (
    RepositoryAuthorizationError,
    RepositoryAuthorizationStore,
)
from research_workspace.service_tiers import ModelLane, ServiceTierError
from research_workspace.zetsu_agent import AgentRunContext, ZetsuAgentCoordinator


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
    tmp_path: Path, *, quota: int = 4
) -> tuple[AgentSandboxManager, RepositoryAuthorizationStore, Path, str]:
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
            global_quota=max(quota, 8),
        ),
        authorizations,
        repository,
        revision,
    )


def _policy() -> AgentToolPolicy:
    return AgentToolPolicy("bounded", ("apply_patch", "run_tests"))


def _binding(manager: AgentSandboxManager, session_id: str, *, user_id: str = "owner") -> Path:
    return Path(
        manager.create(
            user_id=user_id,
            repo_id="repo",
            session_id=session_id,
            tool_policy=_policy(),
        ).worktree_root
    )


def _context(binding: AgentSessionBinding) -> AgentRunContext:
    return AgentRunContext(
        user_id=binding.user_id,
        session_id=binding.session_id,
        repo_id=binding.repo_id,
        lane=ModelLane.QUALITY,
        binding=binding,
        worktree=Path(binding.worktree_root),
        max_steps=12,
        max_chars=8_000,
        compaction_ratio=0.8,
        model_id="fixture-model",
        context_limit=131_072,
        required_verification_argv=None,
        run_started=0.0,
        remaining_wall_seconds=60.0,
    )


def test_repository_readiness_registered_authorized_clean_and_current(tmp_path: Path) -> None:
    manager, authorizations, repository, revision = _manager(tmp_path)
    del manager
    readiness = authorizations.readiness("owner", repository)
    assert readiness == {
        "canonical_root": str(repository.resolve()),
        "repo_id": "repo",
        "registered": True,
        "authorized": True,
        "granted_revision": revision,
        "grant_revision": 1,
        "current_head": revision,
        "working_tree_clean": True,
        "agent_task_ready": True,
        "state": "ready",
        "revision_sync_required": False,
    }


def test_repository_readiness_reports_unregistered_and_unauthorized(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _repository(repository)
    unregistered = RepositoryAuthorizationStore(tmp_path / "unregistered.sqlite3")
    assert unregistered.readiness("owner", repository)["state"] == "repository_not_registered"

    authorized = RepositoryAuthorizationStore(tmp_path / "authorized.sqlite3")
    authorized.register("repo", repository)
    assert authorized.readiness("owner", repository)["state"] == "repository_not_authorized"


def test_repository_identity_change_preserves_security_failure(tmp_path: Path) -> None:
    manager, authorizations, repository, _ = _manager(tmp_path)
    del manager
    original = repository.stat()
    moved = repository.with_name("moved-repository")
    shutil.move(repository, moved)
    _repository(repository)
    if repository.stat().st_ino == original.st_ino:
        (repository / "inode-bump").mkdir()
        shutil.rmtree(repository)
        _repository(repository)
    with pytest.raises(RepositoryAuthorizationError, match="registered_repository_identity_changed"):
        authorizations.repository("repo")


def test_clean_head_advance_synchronizes_grant_audit_and_exact_worktree(tmp_path: Path) -> None:
    manager, authorizations, repository, revision_a = _manager(tmp_path)
    (repository / "committed.txt").write_text("committed\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "advance")
    revision_b = _git(repository, "rev-parse", "HEAD")

    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="revision-b", tool_policy=_policy()
    )
    assert binding.base_revision == revision_b
    assert _git(Path(binding.worktree_root), "rev-parse", "HEAD") == revision_b
    grant = authorizations.require_grant("owner", "repo")
    assert grant.base_revision == revision_b
    assert grant.revision == 2
    events = authorizations.revision_events("owner", "repo")
    assert len(events) == 1
    assert events[0]["old_base_revision"] == revision_a
    assert events[0]["new_base_revision"] == revision_b
    history = manager.history("revision-b", user_id="owner")
    assert history[0]["event"] == "CREATED"
    assert history[0]["details"]["revision_sync"]["synchronized"] is True


def test_concurrent_new_sessions_share_one_consistent_revision(tmp_path: Path) -> None:
    manager, _, repository, _ = _manager(tmp_path, quota=8)
    (repository / "new.txt").write_text("new\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "new revision")
    expected = _git(repository, "rev-parse", "HEAD")
    managers = [
        manager,
        AgentSandboxManager(
            tmp_path / "worktrees",
            manager.authorizations,
            per_user_quota=8,
            global_quota=8,
        ),
    ]
    bindings: list[AgentSessionBinding] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def create(index: int) -> None:
        try:
            value = managers[index % len(managers)].create(
                user_id="owner",
                repo_id="repo",
                session_id=f"concurrent-{index}",
                tool_policy=_policy(),
            )
            with lock:
                bindings.append(value)
        except BaseException as exc:  # pragma: no cover - asserted below
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=create, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert failures == []
    assert len(bindings) == 6
    assert {binding.base_revision for binding in bindings} == {expected}
    assert {
        _git(Path(binding.worktree_root), "rev-parse", "HEAD") for binding in bindings
    } == {expected}


def test_dirty_advanced_checkout_is_not_materialized_or_copied(tmp_path: Path) -> None:
    manager, _, repository, revision_a = _manager(tmp_path)
    (repository / "committed.txt").write_text("committed\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "advance")
    (repository / "unrelated-dirty.txt").write_text("dirty\n", encoding="utf-8")
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="dirty-advanced", tool_policy=_policy()
    )
    assert binding.base_revision == revision_a
    assert not (Path(binding.worktree_root) / "committed.txt").exists()
    assert (repository / "unrelated-dirty.txt").is_file()


def test_dirty_caller_only_read_returns_precise_materialization_error(tmp_path: Path) -> None:
    manager, _, repository, revision = _manager(tmp_path)
    (repository / "caller-only.py").write_text("dirty = True\n", encoding="utf-8")
    binding_record = manager.create(
        user_id="owner", repo_id="repo", session_id="dirty-read", tool_policy=_policy()
    )
    coordinator = object.__new__(ZetsuAgentCoordinator)
    with pytest.raises(ServiceTierError, match="repository_state_not_materialized") as raised:
        coordinator._read(
            _context(binding_record),
            {"path": "caller-only.py"},
        )
    assert raised.value.evidence == {
        "canonical_root": str(repository.resolve()),
        "current_head": revision,
        "granted_revision": revision,
        "working_tree_clean": False,
        "path": "caller-only.py",
    }
    assert not (Path(binding_record.worktree_root) / "caller-only.py").exists()


def test_clean_successful_session_releases_safely(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    root = _binding(manager, "clean-success")
    result = manager.close_if_clean("clean-success", user_id="owner")
    assert result["status"] == "RELEASED_CLEAN_WORKTREE"
    assert not root.exists()
    record = manager.status("clean-success", user_id="owner")
    assert record["physical_state"] == "REMOVED"


def test_owner_events_remain_readable_after_clean_release(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    _binding(manager, "event-release")
    manager.record_progress(
        "event-release",
        user_id="owner",
        event="TURN_CANCELLED",
        details={"turn_id": "turn-00000001"},
    )
    assert manager.close_if_clean("event-release", user_id="owner")["status"] == (
        "RELEASED_CLEAN_WORKTREE"
    )
    events = manager.events("event-release", user_id="owner", after_sequence=0)
    assert any(item["event"] == "TURN_CANCELLED" for item in events)
    with pytest.raises(AgentSandboxError, match="unknown_agent_session"):
        manager.events("event-release", user_id="other", after_sequence=0)


def test_quantum_continuing_progress_event_is_state_neutral(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    _binding(manager, "quantum-progress")
    before = manager.status("quantum-progress", user_id="owner")
    manager.record_progress(
        "quantum-progress",
        user_id="owner",
        event="QUANTUM_CONTINUING",
        details={
            "quantum": 1,
            "cumulative_steps": 12,
            "progress": True,
            "directive": "reassess_finish",
        },
    )
    after = manager.status("quantum-progress", user_id="owner")
    assert after["lifecycle_state"] == before["lifecycle_state"]
    events = manager.events("quantum-progress", user_id="owner", after_sequence=0)
    assert events[-1]["event"] == "QUANTUM_CONTINUING"
    assert events[-1]["details"]["cumulative_steps"] == 12


def test_clean_failed_session_is_reclaimed_before_quota_denial(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path, quota=1)
    root = _binding(manager, "failed-clean")
    manager.record_result(
        "failed-clean", user_id="owner", command_count=0, verification_summary="FAILED:fixture", failed=True
    )
    next_root = _binding(manager, "after-failed-clean")
    assert next_root.is_dir()
    assert not root.exists()
    assert manager.status("failed-clean", user_id="owner")["physical_state"] == "REMOVED"


def test_expired_active_session_is_reconciled_before_quota_denial(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path, quota=1)
    root = _binding(manager, "expired-active")
    with manager._connect() as connection:
        connection.execute(
            "UPDATE worktree_sessions SET wall_deadline_utc='2020-01-01T00:00:00+00:00' WHERE session_id=?",
            ("expired-active",),
        )
    next_root = _binding(manager, "after-expired-active")
    assert next_root.is_dir()
    assert not root.exists()


def test_failed_dirty_and_cancelled_dirty_sessions_are_preserved(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path, quota=4)
    failed_root = _binding(manager, "failed-dirty")
    (failed_root / "changed.txt").write_text("changed\n", encoding="utf-8")
    manager.record_result(
        "failed-dirty", user_id="owner", command_count=1, verification_summary="FAILED:fixture", failed=True
    )
    cancelled_root = _binding(manager, "cancelled-dirty")
    (cancelled_root / "changed.txt").write_text("changed\n", encoding="utf-8")
    cancelled = manager.cancel("cancelled-dirty", user_id="owner")
    assert cancelled["status"] == "PRESERVED_DIRTY_WORKTREE"
    report = manager.collect_garbage(dry_run=False, user_id="owner", limit=8)
    reasons = {item.get("reason") for item in report["items"]}
    assert any(isinstance(reason, str) and reason.endswith("_dirty") for reason in reasons)
    assert "cancelled_dirty_preserved" in reasons
    assert failed_root.is_dir() and cancelled_root.is_dir()


def test_running_session_is_never_reclaimed(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path, quota=2)
    root = _binding(manager, "running")
    manager.start_task("running", user_id="owner", lane="quality", sanitized_model_name="fixture")
    report = manager.collect_garbage(dry_run=False, user_id="owner", limit=4)
    assert report["items"][0]["reason"] == "executor_live"
    assert root.is_dir()


def test_reconciliation_is_owner_scoped(tmp_path: Path) -> None:
    manager, authorizations, _, _ = _manager(tmp_path, quota=2)
    authorizations.grant("other", "repo")
    owner_root = _binding(manager, "owner-failed")
    manager.record_result(
        "owner-failed", user_id="owner", command_count=0, verification_summary="FAILED:fixture", failed=True
    )
    other_root = _binding(manager, "other-live", user_id="other")
    report = manager.collect_garbage(dry_run=False, user_id="owner", limit=4)
    assert report["released"] == 1
    assert not owner_root.exists()
    assert other_root.is_dir()


def test_valid_resume_reuses_existing_worktree_and_cross_owner_is_rejected(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    root = _binding(manager, "resume-me")
    before = manager.capacity_snapshot(user_id="owner")["user_live_resumable_worktrees"]
    resumed = manager.resume("resume-me", user_id="owner")
    assert resumed["session_id"] == "resume-me"
    assert Path(str(resumed["worktree_root"])) == root if "worktree_root" in resumed else root.is_dir()
    assert manager.capacity_snapshot(user_id="owner")["user_live_resumable_worktrees"] == before
    with pytest.raises(AgentSandboxError, match="unknown_agent_session"):
        manager.resume("resume-me", user_id="other")


def test_setup_failure_releases_fresh_clean_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _, _, _ = _manager(tmp_path)
    tiered = SimpleNamespace(
        sandboxes=manager,
        lane_policy=SimpleNamespace(codev_enabled=False),
    )
    coordinator = ZetsuAgentCoordinator(cast(Any, tiered))

    def fail_after_create(**kwargs: object) -> dict[str, object]:
        manager.create(
            user_id="owner",
            repo_id="repo",
            session_id=str(kwargs["session_id"]),
            tool_policy=_policy(),
        )
        raise ServiceTierError("repository_state_not_materialized")

    monkeypatch.setattr(coordinator, "_run_unlocked", fail_after_create)
    with pytest.raises(ServiceTierError, match="repository_state_not_materialized"):
        coordinator.run(
            user_id="owner",
            repo_id="repo",
            instruction="fixture",
            session_id="setup-failure",
            wait_timeout_seconds=1,
        )
    assert not (manager.sandbox_root / "owner" / "setup-failure").exists()
    assert manager.status("setup-failure", user_id="owner")["physical_state"] == "REMOVED"


def test_authenticated_sessions_cli_reports_quota_and_release_flags(tmp_path: Path, capsys, monkeypatch) -> None:
    state = tmp_path / "state"
    auth = RepositoryAuthorizationStore(state / "tiered_serving/repository_authorizations.sqlite3")
    repository = tmp_path / "repo"
    _repository(repository)
    auth.register("repo", repository)
    auth.grant("plus-local", "repo")
    manager = AgentSandboxManager(state / "tiered_serving/worktrees", auth)
    _binding(manager, "cli-session", user_id="plus-local")
    token_path = state / "auth/bearer_tokens.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "local-session-token": {
                        "user_id": "plus-local",
                        "capability_tier": "plus",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.chmod(token_path, 0o600)
    monkeypatch.setenv("LAPLACE_ZETSU_TOKEN", "local-session-token")
    assert zetsu_cli.main(["sessions", "--state-root", str(state), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    record = payload["sessions"][0]
    assert payload["principal_id"] == "plus-local"
    assert record["session_id"] == "cli-session"
    assert record["worktree_path"]
    assert record["counts_against_quota"] is True
    assert record["worktree_clean"] is True
