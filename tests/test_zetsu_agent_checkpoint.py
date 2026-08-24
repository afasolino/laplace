from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_workspace.agent_sandbox import AgentSessionBinding, AgentToolPolicy
from research_workspace.service_tiers import ModelLane, ServiceTierError
from research_workspace.zetsu_agent import (
    AgentCheckpointStore,
    AgentExecutionState,
    AgentRunContext,
    AgentTelemetry,
    ZetsuAgentCoordinator,
)


class _Sandboxes:
    def __init__(self, root: Path) -> None:
        self.sandbox_root = root
        self.required: list[tuple[str, str]] = []

    def require_active(self, session_id: str, *, user_id: str):
        self.required.append((session_id, user_id))
        return object()


class _Tiered:
    def __init__(self, root: Path) -> None:
        self.sandboxes = _Sandboxes(root)
        self.cancelled = False
        self.chat_calls: list[dict[str, object]] = []

    def agent_session_status(self, *, user_id: str, session_id: str):
        return {"status": "CANCELLED" if self.cancelled else "RUNNING"}

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return {
            "response": {
                "content": '{"action":"finish","summary":"compact semantic history"}',
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }
        }


class _Corpus:
    def __init__(self) -> None:
        self.users: list[str] = []
        self._lock = threading.Lock()

    def search(self, owner_user_id: str, query: str, *, limit: int = 8):
        with self._lock:
            self.users.append(owner_user_id)
        return {"results": []}


def _binding(root: Path, *, session: str, user: str, max_commands: int = 4) -> AgentSessionBinding:
    return AgentSessionBinding(
        session_id=session,
        user_id=user,
        repo_id="repo",
        canonical_repository_root=str(root),
        worktree_root=str(root),
        base_revision="a" * 40,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="test",
            allowed_tools=("apply_patch", "run_tests"),
            max_commands=max_commands,
            max_wall_seconds=60,
        ),
        environment={},
        created_at_utc="2026-08-22T00:00:00+00:00",
    )


def _ctx(
    root: Path,
    *,
    session: str,
    user: str,
    max_commands: int = 4,
    model_id: str = "test-model",
    context_limit: int = 131_072,
    verification_argv: tuple[str, ...] | None = None,
) -> AgentRunContext:
    binding = _binding(root, session=session, user=user, max_commands=max_commands)
    return AgentRunContext(
        user_id=user,
        session_id=session,
        repo_id="repo",
        lane=ModelLane.QUALITY,
        binding=binding,
        worktree=root,
        max_steps=12,
        max_chars=8_000,
        compaction_ratio=0.80,
        model_id=model_id,
        context_limit=context_limit,
        required_verification_argv=verification_argv,
        run_started=time.monotonic(),
        remaining_wall_seconds=60,
    )


def test_concurrent_agent_contexts_do_not_share_owner_identity(tmp_path: Path) -> None:
    tiered = _Tiered(tmp_path)
    corpus = _Corpus()
    coordinator = ZetsuAgentCoordinator(
        tiered, corpus, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    assert not hasattr(coordinator, "_active_user_id")
    barrier = threading.Barrier(2)

    def retrieve(user: str, session: str) -> None:
        ctx = _ctx(tmp_path, session=session, user=user)
        state = AgentExecutionState(objective="x")
        barrier.wait()
        coordinator._retrieve(ctx, state, {"query": "evidence"})

    threads = [
        threading.Thread(target=retrieve, args=("user-a", "session-a")),
        threading.Thread(target=retrieve, args=("user-b", "session-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert sorted(corpus.users) == ["user-a", "user-b"]


def test_checkpoint_resume_restores_exact_and_semantic_state(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(
        coordinator,
        "_worktree_state",
        lambda _root: ("b" * 40, "1" * 64, ["src/a.py"]),
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    state = AgentExecutionState(
        objective="objective",
        step=5,
        summary="semantic summary",
        recent_observations=["obs"],
        validation_history=[{"passed": False, "returncode": 1}],
        unresolved_failures=["failed-test"],
        evidence_refs=["chk_" + "a" * 32],
        next_state="verify_latest_mutation",
        mutation_epoch=2,
        last_verified_epoch=1,
        command_count=3,
        telemetry=AgentTelemetry(tool_calls=3),
    )
    coordinator._checkpoint(ctx, state)
    restored = coordinator._restore(ctx, "objective")
    assert restored.step == 5
    assert restored.summary == "semantic summary"
    assert restored.recent_observations == ["obs"]
    assert restored.changed_paths == ["src/a.py"]
    assert restored.validation_history[-1]["passed"] is False
    assert restored.unresolved_failures == ["failed-test"]
    assert restored.mutation_epoch == 2
    assert restored.last_verified_epoch == 1
    assert restored.command_count == 3


def test_checkpoint_resume_rejects_worktree_drift(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    monkeypatch.setattr(
        coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, [])
    )
    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))
    monkeypatch.setattr(
        coordinator,
        "_worktree_state",
        lambda _root: ("b" * 40, "2" * 64, ["changed.py"]),
    )
    with pytest.raises(ServiceTierError, match="zetsu_agent_checkpoint_worktree_drift"):
        coordinator._restore(ctx, "objective")


def test_same_session_concurrent_execution_fails_busy_before_resume(tmp_path: Path) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    lock = coordinator._session_lock("same-session")
    assert lock.acquire(blocking=False) is True
    try:
        with pytest.raises(ServiceTierError, match="zetsu_agent_session_busy"):
            coordinator.run(
                user_id="user-a",
                repo_id="repo",
                instruction="objective",
                session_id="same-session",
            )
    finally:
        lock.release()


def test_mutating_finish_requires_verification_after_latest_mutation() -> None:
    coordinator = object.__new__(ZetsuAgentCoordinator)
    state = AgentExecutionState(objective="x", mutation_epoch=1, last_verified_epoch=-1)
    assert coordinator._finish_allowed(state) is False
    required = ["pytest", "tests/test_x.py", "-q"]
    assert coordinator._verification_qualifies(["git", "status"], required) is False
    assert coordinator._verification_qualifies(required, required) is True
    assert coordinator._verification_qualifies(["pytest", "tests/test_y.py", "-q"], required) is False
    assert coordinator._verification_qualifies(["pytest", "--collect-only"], ["pytest", "--collect-only"]) is False
    assert coordinator._verification_qualifies(["ruff", "check", "src"], ["ruff", "check", "src"]) is False
    assert coordinator._verification_qualifies(["mypy", "src"], ["mypy", "src"]) is False
    state.last_verified_epoch = 1
    assert coordinator._finish_allowed(state) is True
    state.unresolved_failures.append("verification_failed:still-open")
    assert coordinator._finish_allowed(state) is False


def test_cancellation_and_command_budget_fail_closed(tmp_path: Path) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a", max_commands=1)
    state = AgentExecutionState(objective="x")
    coordinator._consume_tool_budget(ctx, state)
    with pytest.raises(ServiceTierError, match="zetsu_agent_command_budget_exhausted"):
        coordinator._consume_tool_budget(ctx, state)
    expired = AgentRunContext(
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        repo_id=ctx.repo_id,
        lane=ctx.lane,
        binding=ctx.binding,
        worktree=ctx.worktree,
        max_steps=ctx.max_steps,
        max_chars=ctx.max_chars,
        compaction_ratio=ctx.compaction_ratio,
        model_id=ctx.model_id,
        context_limit=ctx.context_limit,
        required_verification_argv=ctx.required_verification_argv,
        run_started=time.monotonic() - 2.0,
        remaining_wall_seconds=1.0,
    )
    with pytest.raises(ServiceTierError, match="zetsu_agent_wall_budget_exhausted"):
        coordinator._ensure_active(expired, state)
    tiered.cancelled = True
    with pytest.raises(ServiceTierError, match="agent_session_cancelled"):
        coordinator._ensure_active(ctx, state)


def test_running_verification_honors_cancellation(
    tmp_path: Path, monkeypatch
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    state = AgentExecutionState(objective="x")
    executable = tmp_path / "pytest"
    executable.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n", encoding="utf-8"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        "research_workspace.zetsu_agent.shutil.which",
        lambda _name, path=None: str(executable),
    )
    monkeypatch.setattr(
        "research_workspace.zetsu_agent.AgentSandboxManager.fixed_environment",
        staticmethod(lambda _binding: dict(os.environ)),
    )
    monkeypatch.setattr(
        coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, [])
    )

    def cancel() -> None:
        time.sleep(0.25)
        tiered.cancelled = True

    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        with pytest.raises(ServiceTierError, match="agent_session_cancelled"):
            coordinator._verify(ctx, state, {"argv": ["pytest", "tests/test_x.py"]})
    finally:
        thread.join(timeout=2)
    assert state.validation_history[-1]["aborted_category"] == "agent_session_cancelled"
    assert state.validation_history[-1]["passed"] is False


def test_create_new_text_file_is_bounded_and_isolated(tmp_path: Path) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    created = coordinator._create(ctx, {"path": "new.txt", "content": "bounded\n"})
    assert created == "CREATED:new.txt"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "bounded\n"
    with pytest.raises(ServiceTierError, match="zetsu_agent_git_metadata_forbidden"):
        coordinator._create(ctx, {"path": ".git/config", "content": "x"})


def test_compaction_preserves_exact_state_and_can_continue(tmp_path: Path) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    state = AgentExecutionState(
        objective="objective",
        recent_observations=["old-1", "old-2", "old-3", "new-1"],
        changed_paths=["src/a.py"],
        validation_history=[{"passed": True, "mutation_epoch": 2}],
        mutation_epoch=2,
        last_verified_epoch=2,
        command_count=2,
    )
    coordinator._compact(ctx, state, threshold=100_000)
    assert state.summary == "compact semantic history"
    assert state.recent_observations == ["old-2", "old-3", "new-1"]
    assert state.changed_paths == ["src/a.py"]
    assert state.mutation_epoch == 2
    assert state.last_verified_epoch == 2
    assert state.telemetry.compactions == 1
    assert state.telemetry.last_model_reported_input_tokens is None


def test_verify_argv_confines_model_controlled_verification_to_worktree(tmp_path: Path) -> None:
    coordinator = object.__new__(ZetsuAgentCoordinator)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    assert coordinator._verify_argv(tmp_path, ["pytest", "tests/test_x.py", "-q"]) == [
        "pytest",
        "tests/test_x.py",
        "-q",
    ]
    for argv in (
        ["git", "status"],
        ["pytest", "../outside.py"],
        ["pytest", "/etc/passwd"],
        ["pytest", "-p", "external_plugin", "tests/test_x.py"],
        ["pytest", "-o", "pythonpath=/tmp", "tests/test_x.py"],
        ["pytest", "--override-ini=pythonpath=/tmp", "tests/test_x.py"],
        ["ruff", "check", "--config=external.toml", "tests/test_x.py"],
        ["mypy", "--python-executable=/usr/bin/python", "tests/test_x.py"],
        ["ruff", "format", "tests/test_x.py"],
        ["ruff", "check", "--fix", "tests/test_x.py"],
    ):
        with pytest.raises(ServiceTierError):
            coordinator._verify_argv(tmp_path, argv)


def test_active_state_digest_excludes_large_verification_bodies() -> None:
    state = AgentExecutionState(
        objective="objective",
        worktree_status_sha256="a" * 64,
        validation_history=[
            {
                "command_id": "abc",
                "argv": ["pytest", "tests/test_x.py", "-q"],
                "returncode": 0,
                "passed": True,
                "stdout_tail": "x" * 8_000,
                "stderr_tail": "y" * 8_000,
                "qualifies_for_mutation": True,
                "mutation_epoch": 1,
            }
        ],
    )
    digest = ZetsuAgentCoordinator._state_digest(state)
    assert "stdout_tail" not in digest
    assert "stderr_tail" not in digest
    assert '"argv"' not in digest
    assert "worktree_status_sha256" in digest
    assert "abc" in digest


def test_checkpoint_resume_rejects_budget_change(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))
    changed = AgentRunContext(
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        repo_id=ctx.repo_id,
        lane=ctx.lane,
        binding=ctx.binding,
        worktree=ctx.worktree,
        max_steps=ctx.max_steps + 1,
        max_chars=ctx.max_chars,
        compaction_ratio=ctx.compaction_ratio,
        model_id=ctx.model_id,
        context_limit=ctx.context_limit,
        required_verification_argv=ctx.required_verification_argv,
        run_started=time.monotonic(),
        remaining_wall_seconds=ctx.remaining_wall_seconds,
    )
    with pytest.raises(ServiceTierError, match="zetsu_agent_resume_budget_mismatch"):
        coordinator._restore(changed, "objective")


def test_resume_rejects_invalid_checkpoint_accounting(
    tmp_path: Path, monkeypatch
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))
    raw = coordinator.checkpoints.read(ctx.session_id)
    assert raw is not None
    raw["command_count"] = 10_000
    coordinator.checkpoints.write(ctx.session_id, raw)

    with pytest.raises(ServiceTierError, match="zetsu_agent_checkpoint_accounting_invalid"):
        coordinator._restore(ctx, "objective")


def test_checkpoint_resume_rejects_tool_policy_change(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))
    ctx = _ctx(tmp_path, session="session-a", user="user-a", max_commands=4)
    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))
    changed = _ctx(tmp_path, session="session-a", user="user-a", max_commands=5)
    with pytest.raises(ServiceTierError, match="zetsu_agent_resume_tool_policy_mismatch"):
        coordinator._restore(changed, "objective")


def test_verification_that_mutates_worktree_fails_closed(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    state = AgentExecutionState(objective="x")
    executable = tmp_path / "pytest"
    executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "research_workspace.zetsu_agent.shutil.which",
        lambda _name, path=None: str(executable),
    )
    monkeypatch.setattr(
        "research_workspace.zetsu_agent.AgentSandboxManager.fixed_environment",
        staticmethod(lambda _binding: dict(os.environ)),
    )
    states = iter(
        [
            ("a" * 40, "", []),
            ("a" * 40, " M src/changed.py\n", ["src/changed.py"]),
        ]
    )
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: next(states))

    with pytest.raises(ServiceTierError, match="zetsu_agent_verification_mutated_worktree"):
        coordinator._verify(ctx, state, {"argv": ["pytest", "tests/test_x.py", "-q"]})
    assert state.mutation_epoch == 1
    assert state.last_verified_epoch == -1
    assert any(item.startswith("latest_mutation_unverified:epoch=1") for item in state.unresolved_failures)
    assert any(item.startswith("verification_mutated_worktree:epoch=1:") for item in state.unresolved_failures)
    assert state.validation_history[-1]["worktree_mutated"] is True

def test_verify_argv_rejects_executable_path_alias(tmp_path: Path) -> None:
    with pytest.raises(ServiceTierError, match="zetsu_agent_verify_command_forbidden"):
        ZetsuAgentCoordinator._verify_argv(
            tmp_path, ["/tmp/pytest", "tests/test_x.py", "-q"]
        )
    with pytest.raises(ServiceTierError, match="zetsu_agent_verify_command_forbidden"):
        ZetsuAgentCoordinator._verify_argv(
            tmp_path, ["../pytest", "tests/test_x.py", "-q"]
        )


def test_checkpoint_resume_binds_route_and_required_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(
        coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, [])
    )
    verifier = ("pytest", "tests/test_x.py", "-q")
    ctx = _ctx(
        tmp_path,
        session="session-a",
        user="user-a",
        verification_argv=verifier,
    )
    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))

    changed_route = _ctx(
        tmp_path,
        session="session-a",
        user="user-a",
        model_id="other-model",
        verification_argv=verifier,
    )
    with pytest.raises(ServiceTierError, match="zetsu_agent_resume_route_mismatch"):
        coordinator._restore(changed_route, "objective")

    changed_verifier = _ctx(
        tmp_path,
        session="session-a",
        user="user-a",
        verification_argv=("pytest", "tests/test_y.py", "-q"),
    )
    with pytest.raises(ServiceTierError, match="zetsu_agent_resume_verifier_mismatch"):
        coordinator._restore(changed_verifier, "objective")


def test_checkpoint_rejects_malformed_semantic_state_and_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(
        coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, [])
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))
    raw = coordinator.checkpoints.read(ctx.session_id)
    assert raw is not None
    raw["summary"] = ["not", "a", "string"]
    coordinator.checkpoints.write(ctx.session_id, raw)
    with pytest.raises(ServiceTierError, match="zetsu_agent_checkpoint_state_invalid"):
        coordinator._restore(ctx, "objective")

    coordinator._checkpoint(ctx, AgentExecutionState(objective="objective"))
    raw = coordinator.checkpoints.read(ctx.session_id)
    assert raw is not None
    telemetry = raw["telemetry"]
    assert isinstance(telemetry, dict)
    telemetry["qwen_calls"] = -1
    coordinator.checkpoints.write(ctx.session_id, raw)
    with pytest.raises(ServiceTierError, match="zetsu_agent_checkpoint_telemetry_invalid"):
        coordinator._restore(ctx, "objective")
