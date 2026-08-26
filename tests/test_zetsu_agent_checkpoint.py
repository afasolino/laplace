from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from research_workspace.agent_sandbox import AgentSessionBinding, AgentToolPolicy
from research_workspace.service_tiers import LanePolicy, ModelLane, ModelRoute, ServiceTierError
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
        continuation_count=2,
        stagnation_count=1,
        exploration_only_quanta=2,
        recent_action_fingerprints=["a" * 64],
        quantum_action_fingerprints=["b" * 64],
        quantum_exploration_only=False,
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
    assert restored.continuation_count == 2
    assert restored.stagnation_count == 1
    assert restored.exploration_only_quanta == 2
    assert restored.recent_action_fingerprints == ["a" * 64]
    assert restored.quantum_action_fingerprints == ["b" * 64]
    assert restored.quantum_exploration_only is False


def test_pre_adaptive_schema3_checkpoint_resumes_at_quantum_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(
        coordinator,
        "_worktree_state",
        lambda _root: ("a" * 40, "0" * 64, []),
    )
    ctx = _ctx(tmp_path, session="legacy-session", user="user-a")
    state = AgentExecutionState(objective="objective", step=12)
    coordinator._checkpoint(ctx, state)
    raw = coordinator.checkpoints.read(ctx.session_id)
    assert raw is not None
    for key in (
        "continuation_count",
        "stagnation_count",
        "exploration_only_quanta",
        "recent_action_fingerprints",
        "quantum_action_fingerprints",
        "quantum_exploration_only",
        "quantum_step_limit",
        "absolute_step_limit",
    ):
        raw.pop(key, None)
    coordinator.checkpoints.write(ctx.session_id, raw)

    restored = coordinator._restore(ctx, "objective")
    assert restored.step == 12
    assert restored.continuation_count == 0
    assert restored.stagnation_count == 0
    assert len(restored.quantum_action_fingerprints) == 1
    assert restored.quantum_exploration_only is False


def test_checkpoint_resume_rejects_worktree_drift(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))
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
    assert (
        coordinator._verification_qualifies(["pytest", "tests/test_y.py", "-q"], required) is False
    )
    assert (
        coordinator._verification_qualifies(
            ["pytest", "--collect-only"], ["pytest", "--collect-only"]
        )
        is False
    )
    assert (
        coordinator._verification_qualifies(["ruff", "check", "src"], ["ruff", "check", "src"])
        is False
    )
    assert coordinator._verification_qualifies(["mypy", "src"], ["mypy", "src"]) is False
    state.last_verified_epoch = 1
    assert coordinator._finish_allowed(state) is True
    state.unresolved_failures.append("verification_failed:still-open")
    assert coordinator._finish_allowed(state) is False


def test_malformed_model_action_is_accounted_and_retried_without_losing_state() -> None:
    state = AgentExecutionState(objective="x", mutation_epoch=1, changed_paths=["src/a.py"])
    error = ServiceTierError(
        "response_validation_failed",
        {
            "gate_id": "valid_json",
            "reason": "malformed JSON",
            "model_reported_usage": {"prompt_tokens": 123, "completion_tokens": 17},
        },
    )
    assert ZetsuAgentCoordinator._record_recoverable_model_failure(state, 4, error) is True
    assert state.changed_paths == ["src/a.py"]
    assert state.telemetry.qwen_calls == 1
    assert state.telemetry.agent_steps == 1
    assert state.telemetry.qwen_input_tokens == 123
    assert state.telemetry.qwen_output_tokens == 17
    assert state.telemetry.qwen_usage_reported_calls == 1
    assert state.next_state == "retry_valid_json_action"
    assert "MODEL_RESPONSE_REJECTED" in state.recent_observations[-1]
    assert (
        ZetsuAgentCoordinator._record_recoverable_model_failure(
            state, 5, ServiceTierError("agent_session_cancelled")
        )
        is False
    )


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


def test_output_cap_continuation_preserves_state_without_duplicate_mutation(
    tmp_path: Path,
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="cap-session", user="user-a")
    state = AgentExecutionState(
        objective="x",
        mutation_epoch=1,
        changed_paths=["src/already_applied.py"],
        command_count=1,
    )
    error = ServiceTierError(
        "model_output_limit_reached",
        {
            "gate_id": "no_silent_truncation",
            "finish_reason": "length",
            "partial_content": '{"action":"edit","edits":[{"path":"src/already_applied.py"',
            "model_reported_usage": {"prompt_tokens": 100, "completion_tokens": 4096},
        },
    )
    for step in range(1, 5):
        assert coordinator._record_model_failure(ctx, state, step, error) is True
        assert state.mutation_epoch == 1
        assert state.command_count == 1
        assert state.changed_paths == ["src/already_applied.py"]
    assert coordinator._record_model_failure(ctx, state, 5, error) is False
    assert state.output_cap_continuations == 4
    assert len(coordinator.results.staging_artifacts("cap-session")) == 4
    assert state.next_state == "continue_after_output_cap"


def test_oversized_verifier_streams_are_persisted_exactly_not_injected(
    tmp_path: Path, monkeypatch
) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="large-verify", user="user-a")
    state = AgentExecutionState(objective="x")
    executable = tmp_path / "pytest"
    executable.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "sys.stdout.write('O' * 200000)\nsys.stderr.write('E' * 180000)\n",
        encoding="utf-8",
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
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))

    record = coordinator._verify(ctx, state, {"argv": ["pytest", "tests/test_x.py"]})
    assert record["passed"] is True
    assert len(str(record["stdout_tail"])) == 8_000
    assert len(str(record["stderr_tail"])) == 8_000
    staged = coordinator.results.staging_artifacts("large-verify")
    assert staged["verify-001.stdout"].stat().st_size == 200_000
    assert staged["verify-001.stderr"].stat().st_size == 180_000


def test_running_verification_honors_cancellation(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="session-a", user="user-a")
    state = AgentExecutionState(objective="x")
    executable = tmp_path / "pytest"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "research_workspace.zetsu_agent.shutil.which",
        lambda _name, path=None: str(executable),
    )
    monkeypatch.setattr(
        "research_workspace.zetsu_agent.AgentSandboxManager.fixed_environment",
        staticmethod(lambda _binding: dict(os.environ)),
    )
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))

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


def test_handoff_patch_preserves_tracked_and_new_file_code(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "existing.py").write_text("old = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "existing.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    (tmp_path / "existing.py").write_text("old = 2\n", encoding="utf-8")
    (tmp_path / "created.py").write_text("created = True\n", encoding="utf-8")
    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path), checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    result = coordinator._handoff_patch(
        tmp_path,
        "handoff-session",
        ["created.py", "existing.py"],
        max_chars=8_000,
    )
    assert result["patch_inline"] is False
    assert result["patch"] is None
    artifact = Path(str(result["patch_path"]))
    exact = artifact.read_text(encoding="utf-8")
    assert "-old = 1" in exact
    assert "+old = 2" in exact
    assert "+created = True" in exact
    expanded = coordinator.handoff_evidence("handoff-session", max_chars=8_000)
    assert expanded["patch_inline"] is True
    assert expanded["patch"] == exact
    assert expanded["patch_sha256"] == result["patch_sha256"]


def test_verified_handoff_promotes_once_and_rejects_target_drift(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    internal = tmp_path / "internal"
    canonical.mkdir()
    subprocess.run(["git", "init", "-q", str(canonical)], check=True)
    subprocess.run(
        ["git", "-C", str(canonical), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(canonical), "config", "user.name", "Test"], check=True)
    (canonical / "existing.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(canonical), "add", "existing.py"], check=True)
    subprocess.run(["git", "-C", str(canonical), "commit", "-qm", "base"], check=True)
    head = subprocess.run(
        ["git", "-C", str(canonical), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(canonical), "worktree", "add", "-q", "--detach", str(internal), head],
        check=True,
    )
    (internal / "existing.py").write_text("value = 2\n", encoding="utf-8")
    (internal / "created.py").write_text("created = True\n", encoding="utf-8")

    coordinator = ZetsuAgentCoordinator(
        _Tiered(internal), checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    target_head, target_status, _ = coordinator._worktree_state(canonical)
    binding = AgentSessionBinding(
        session_id="promotion-session",
        user_id="user-a",
        repo_id="repo",
        canonical_repository_root=str(canonical),
        worktree_root=str(internal),
        base_revision=head,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="test",
            allowed_tools=("apply_patch", "run_tests"),
            max_commands=4,
            max_wall_seconds=60,
        ),
        environment={},
        created_at_utc="2026-08-22T00:00:00+00:00",
    )
    ctx = AgentRunContext(
        user_id="user-a",
        session_id="promotion-session",
        repo_id="repo",
        lane=ModelLane.QUALITY,
        binding=binding,
        worktree=internal,
        max_steps=12,
        max_chars=8_000,
        compaction_ratio=0.80,
        model_id="test-model",
        context_limit=131_072,
        required_verification_argv=("pytest", "tests/test_x.py", "-q"),
        run_started=time.monotonic(),
        remaining_wall_seconds=60,
        apply_to_repository=True,
    )
    changed = ["created.py", "existing.py"]
    state = AgentExecutionState(
        objective="objective",
        changed_paths=changed,
        mutation_epoch=1,
        last_verified_epoch=1,
        target_initial_head=target_head,
        target_initial_status_sha256=target_status,
    )
    handoff = coordinator._handoff_patch(internal, "promotion-session", changed, max_chars=8_000)

    promoted = coordinator._apply_verified_handoff(ctx, state, handoff)
    assert promoted["applied"] is True
    assert promoted["already_applied"] is False
    assert (canonical / "existing.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (canonical / "created.py").read_text(encoding="utf-8") == "created = True\n"
    assert coordinator._apply_verified_handoff(ctx, state, handoff)["already_applied"] is True

    state.applied_patch_sha256 = ""
    state.target_applied_status_sha256 = ""
    recovered = coordinator._apply_verified_handoff(ctx, state, handoff)
    assert recovered["already_applied"] is True
    assert recovered["recovered_after_checkpoint_gap"] is True

    (canonical / "unrelated.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ServiceTierError, match="zetsu_agent_apply_target_drift"):
        coordinator._apply_verified_handoff(ctx, state, handoff)


def test_batched_read_and_edit_are_bounded_and_transactional(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path), checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    ctx = _ctx(tmp_path, session="batch-session", user="user-a")

    read = coordinator._read(tmp_path, {"paths": ["a.py", "b.py"]})
    assert '"a.py": "a = 1\\n"' in read
    assert '"b.py": "b = 1\\n"' in read
    edited = coordinator._edit(
        ctx,
        {
            "edits": [
                {"path": "a.py", "old_text": "a = 1", "new_text": "a = 2"},
                {"path": "b.py", "old_text": "b = 1", "new_text": "b = 2"},
            ]
        },
    )
    assert edited == "EDITED:a.py,b.py:replacements=2"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "a = 2\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "b = 2\n"


def test_agent_stops_after_bound_verifier_and_promotes_without_final_model_call(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    internal = tmp_path / "internal"
    canonical.mkdir()
    subprocess.run(["git", "init", "-q", str(canonical)], check=True)
    subprocess.run(
        ["git", "-C", str(canonical), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(canonical), "config", "user.name", "Test"], check=True)
    (canonical / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = canonical / "tests"
    tests.mkdir()
    (tests / "test_value.py").write_text(
        "from pathlib import Path\n\ndef test_value():\n"
        "    assert (Path(__file__).parents[1] / 'source.py').read_text() == 'VALUE = 2\\n'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(canonical), "add", "."], check=True)
    subprocess.run(["git", "-C", str(canonical), "commit", "-qm", "base"], check=True)
    head = subprocess.run(
        ["git", "-C", str(canonical), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(canonical), "worktree", "add", "-q", "--detach", str(internal), head],
        check=True,
    )
    binding = AgentSessionBinding(
        session_id="unused",
        user_id="user-a",
        repo_id="repo",
        canonical_repository_root=str(canonical),
        worktree_root=str(internal),
        base_revision=head,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="zetsu-qwen-agent-v2",
            allowed_tools=("apply_patch", "run_tests"),
            network_enabled=False,
            max_commands=24,
            max_wall_seconds=1_800,
        ),
        environment={},
        created_at_utc="2026-08-22T00:00:00+00:00",
    )

    class Sandboxes:
        sandbox_root = tmp_path / "sandboxes"

        @staticmethod
        def require_active(_session_id: str, *, user_id: str) -> AgentSessionBinding:
            assert user_id == "user-a"
            return binding

        @staticmethod
        def start_task(*_args: object, **_kwargs: object) -> None:
            return None

        @staticmethod
        def record_result(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"status": "DIRTY"}

    class Tiered:
        sandboxes = Sandboxes()
        lane_policy = LanePolicy(
            {
                ModelLane.QUALITY: ModelRoute(
                    ModelLane.QUALITY,
                    "test-qwen",
                    "http://127.0.0.1:1",
                    0,
                    131_072,
                    4096,
                ),
                ModelLane.STANDARD: ModelRoute(
                    ModelLane.STANDARD,
                    "test-qwen",
                    "http://127.0.0.1:1",
                    10,
                    131_072,
                    2048,
                ),
                ModelLane.ECONOMY: ModelRoute(
                    ModelLane.ECONOMY,
                    "codev",
                    "http://127.0.0.1:2",
                    20,
                    8192,
                    2048,
                ),
            }
        )

        def __init__(self) -> None:
            self.calls = 0

        @staticmethod
        def create_agent_session(**_kwargs: object) -> dict[str, object]:
            return {"status": "ACTIVE"}

        @staticmethod
        def agent_session_status(**_kwargs: object) -> dict[str, object]:
            return {"status": "RUNNING"}

        def chat(self, **_kwargs: object) -> dict[str, object]:
            actions = (
                {
                    "action": "edit",
                    "edits": [
                        {
                            "path": "source.py",
                            "old_text": "VALUE = 1",
                            "new_text": "VALUE = 2",
                        }
                    ],
                },
                {
                    "action": "verify",
                    "argv": ["pytest", "tests/test_value.py", "-q"],
                },
            )
            if self.calls >= len(actions):
                raise AssertionError("unexpected final model call")
            action = actions[self.calls]
            self.calls += 1
            return {
                "response": {
                    "content": json.dumps(action),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            }

    tiered = Tiered()
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    result = coordinator.run(
        user_id="user-a",
        repo_id="repo",
        instruction="Change VALUE to 2.",
        lane=ModelLane.QUALITY,
        max_steps=12,
        max_chars=4_000,
        verification_argv=["pytest", "tests/test_value.py", "-q"],
        apply_to_repository=True,
    )

    assert tiered.calls == 2
    assert result["status"] == "SUCCESS"
    assert result["promotion"] == {
        "requested": True,
        "applied": True,
        "already_applied": False,
        "target_status_sha256": result["promotion"]["target_status_sha256"],
    }
    assert result["verification"]["passed"] is True
    assert result["unresolved_failures"] == []
    assert (canonical / "source.py").read_text(encoding="utf-8") == "VALUE = 2\n"

    resumed = coordinator.run(
        user_id="user-a",
        repo_id="repo",
        instruction="Change VALUE to 2.",
        lane=ModelLane.QUALITY,
        session_id=result["session_id"],
        max_steps=12,
        max_chars=4_000,
        verification_argv=["pytest", "tests/test_value.py", "-q"],
        apply_to_repository=True,
    )
    assert tiered.calls == 2
    assert resumed["status"] == "SUCCESS"
    assert resumed["promotion"]["already_applied"] is True


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


def test_resume_rejects_invalid_checkpoint_accounting(tmp_path: Path, monkeypatch) -> None:
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
    assert any(
        item.startswith("latest_mutation_unverified:epoch=1") for item in state.unresolved_failures
    )
    assert any(
        item.startswith("verification_mutated_worktree:epoch=1:")
        for item in state.unresolved_failures
    )
    assert state.validation_history[-1]["worktree_mutated"] is True


def test_verify_argv_rejects_executable_path_alias(tmp_path: Path) -> None:
    with pytest.raises(ServiceTierError, match="zetsu_agent_verify_command_forbidden"):
        ZetsuAgentCoordinator._verify_argv(tmp_path, ["/tmp/pytest", "tests/test_x.py", "-q"])
    with pytest.raises(ServiceTierError, match="zetsu_agent_verify_command_forbidden"):
        ZetsuAgentCoordinator._verify_argv(tmp_path, ["../pytest", "tests/test_x.py", "-q"])


def test_quantum_progress_requires_new_action_or_host_state() -> None:
    coordinator = object.__new__(ZetsuAgentCoordinator)
    state = AgentExecutionState(objective="x", worktree_status_sha256="0" * 64)
    action = {"action": "search", "query": "needle", "glob": "*.py"}

    coordinator._record_action_progress(state, "search", action)
    first = coordinator._assess_quantum(state)
    assert first["progress"] is True
    assert first["novel_actions"] == 1
    assert state.stagnation_count == 0

    coordinator._record_action_progress(state, "search", action)
    second = coordinator._assess_quantum(state)
    assert second["progress"] is False
    assert second["novel_actions"] == 0
    assert state.stagnation_count == 1

    state.worktree_status_sha256 = "1" * 64
    coordinator._record_action_progress(state, "search", action)
    third = coordinator._assess_quantum(state)
    assert third["progress"] is True
    assert third["novel_actions"] == 1
    assert state.stagnation_count == 0


def test_adaptive_quantum_continues_same_objective_to_finish(tmp_path: Path) -> None:
    repository = tmp_path / "adaptive-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "component.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "component.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    binding = AgentSessionBinding(
        session_id="adaptive-session",
        user_id="user-a",
        repo_id="repo",
        canonical_repository_root=str(repository),
        worktree_root=str(repository),
        base_revision=head,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="adaptive-test",
            allowed_tools=("read_file",),
            max_commands=20,
            max_wall_seconds=1_800,
        ),
        environment={},
        created_at_utc="2026-08-26T00:00:00+00:00",
    )

    class Sandboxes:
        sandbox_root = tmp_path / "adaptive-sandboxes"

        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []
            self.results: list[dict[str, object]] = []

        @staticmethod
        def require_active(_session_id: str, *, user_id: str) -> AgentSessionBinding:
            assert user_id == "user-a"
            return binding

        @staticmethod
        def start_task(*_args: object, **_kwargs: object) -> None:
            return None

        def record_progress(self, _session_id: str, **kwargs: object) -> dict[str, object]:
            details = kwargs.get("details", {})
            assert isinstance(details, dict)
            self.events.append((str(kwargs["event"]), dict(details)))
            return {"sequence": len(self.events)}

        def record_result(self, *_args: object, **kwargs: object) -> dict[str, object]:
            self.results.append(dict(kwargs))
            return {"status": "ACTIVE_CLEAN"}

    class Tiered:
        sandboxes = Sandboxes()
        lane_policy = LanePolicy(
            {
                lane: ModelRoute(
                    lane,
                    f"test-{lane.value}",
                    "http://127.0.0.1:1",
                    0,
                    131_072,
                    4_096,
                )
                for lane in ModelLane
            }
        )

        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.actions = iter(
                (
                    {"action": "search", "query": "VALUE", "glob": "*.py"},
                    {"action": "search", "query": "component", "glob": "*.py"},
                    {"action": "finish", "result": "inspection complete"},
                )
            )

        @staticmethod
        def agent_session_status(**_kwargs: object) -> dict[str, object]:
            return {"status": "ACTIVE"}

        def chat(self, **kwargs: object) -> dict[str, object]:
            messages = kwargs["messages"]
            assert isinstance(messages, list | tuple)
            self.prompts.append(json.dumps(messages))
            return {
                "response": {
                    "content": json.dumps(next(self.actions)),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            }

    tiered = Tiered()
    coordinator = ZetsuAgentCoordinator(
        tiered,
        checkpoint_store=AgentCheckpointStore(tmp_path / "adaptive-checkpoints"),
    )
    result = coordinator.run_turn(
        user_id="user-a",
        repo_id="repo",
        instruction="Inspect component.py and report VALUE",
        lane=ModelLane.QUALITY,
        session_id="adaptive-session",
        max_steps=2,
        max_chars=2_000,
        verification_argv=None,
        wait_timeout_seconds=30,
    )

    assert result["status"] == "SUCCESS"
    assert result["content"] == "inspection complete"
    assert result["task_label"] == "component.py VALUE"
    assert result["cumulative_steps"] == 3
    assert result["continuation_count"] == 1
    assert result["failure_category"] is None
    continuing = [item for item in tiered.sandboxes.events if item[0] == "QUANTUM_CONTINUING"]
    assert len(continuing) == 1
    assert continuing[0][1]["cumulative_steps"] == 2
    assert continuing[0][1]["progress"] is True
    assert continuing[0][1]["directive"] == "reassess_finish"
    assert continuing[0][1]["task_label"] == "component.py VALUE"
    searches = [item for item in tiered.sandboxes.events if item[0] == "REPOSITORY_SEARCH_STARTED"]
    assert searches[0][1]["query"] == "VALUE"
    assert searches[0][1]["task_label"] == "component.py VALUE"
    assert len(tiered.sandboxes.results) == 1
    assert tiered.sandboxes.results[0]["resumable"] is False
    assert "reassess_finish" in tiered.prompts[2]

    checkpoint = coordinator.checkpoints.read("adaptive-session")
    assert checkpoint is not None
    assert checkpoint["objective"] == "Inspect component.py and report VALUE"
    assert checkpoint["step"] == 3
    assert checkpoint["continuation_count"] == 1
    assert checkpoint["next_state"] == "finished"


def test_standalone_agent_uses_independent_command_budget_across_quantum(tmp_path: Path) -> None:
    repository = tmp_path / "standalone-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "component.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "component.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class Sandboxes:
        sandbox_root = tmp_path / "standalone-sandboxes"

        def __init__(self) -> None:
            self.binding: AgentSessionBinding | None = None
            self.results: list[dict[str, object]] = []

        @staticmethod
        def has_session(_session_id: str, *, user_id: str) -> bool:
            assert user_id == "user-a"
            return False

        def require_active(self, session_id: str, *, user_id: str) -> AgentSessionBinding:
            assert user_id == "user-a"
            assert self.binding is not None
            assert self.binding.session_id == session_id
            return self.binding

        @staticmethod
        def start_task(*_args: object, **_kwargs: object) -> None:
            return None

        @staticmethod
        def record_progress(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"sequence": 1}

        def record_result(self, *_args: object, **kwargs: object) -> dict[str, object]:
            self.results.append(dict(kwargs))
            return {"status": "SUCCEEDED"}

    class Tiered:
        sandboxes = Sandboxes()
        lane_policy = LanePolicy(
            {
                lane: ModelRoute(
                    lane,
                    f"test-{lane.value}",
                    "http://127.0.0.1:1",
                    0,
                    131_072,
                    4_096,
                )
                for lane in ModelLane
            }
        )

        def __init__(self) -> None:
            self.created_policy: AgentToolPolicy | None = None
            self.actions = iter(
                (
                    {"action": "search", "query": "VALUE", "glob": "*.py"},
                    {"action": "search", "query": "component", "glob": "*.py"},
                    {"action": "finish", "result": "standalone complete"},
                )
            )

        def create_agent_session(self, **kwargs: object) -> None:
            policy = kwargs["tool_policy"]
            assert isinstance(policy, AgentToolPolicy)
            self.created_policy = policy
            session_id = str(kwargs["session_id"])
            self.sandboxes.binding = AgentSessionBinding(
                session_id=session_id,
                user_id="user-a",
                repo_id="repo",
                canonical_repository_root=str(repository),
                worktree_root=str(repository),
                base_revision=head,
                grant_revision=1,
                tool_policy=policy,
                environment={},
                created_at_utc="2026-08-26T00:00:00+00:00",
            )

        @staticmethod
        def agent_session_status(**_kwargs: object) -> dict[str, object]:
            return {"status": "ACTIVE"}

        def chat(self, **_kwargs: object) -> dict[str, object]:
            return {
                "response": {
                    "content": json.dumps(next(self.actions)),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            }

    tiered = Tiered()
    coordinator = ZetsuAgentCoordinator(
        tiered,
        checkpoint_store=AgentCheckpointStore(tmp_path / "standalone-checkpoints"),
    )
    result = coordinator.run(
        user_id="user-a",
        repo_id="repo",
        instruction="Inspect component.py and report VALUE",
        lane=ModelLane.QUALITY,
        max_steps=2,
        max_chars=2_000,
        verification_argv=None,
        wait_timeout_seconds=30,
    )

    assert tiered.created_policy is not None
    assert tiered.created_policy.max_commands == 100
    assert result["status"] == "SUCCESS"
    assert result["content"] == "standalone complete"
    assert result["cumulative_steps"] == 3
    assert result["continuation_count"] == 1
    assert len(tiered.sandboxes.results) == 1
    assert tiered.sandboxes.results[0]["terminal"] is True


def test_persistent_agent_turn_reuses_worktree_and_carries_verified_summary(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "component.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "component.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    binding = AgentSessionBinding(
        session_id="chat-agent-session",
        user_id="user-a",
        repo_id="repo",
        canonical_repository_root=str(repository),
        worktree_root=str(repository),
        base_revision=head,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="chat-test",
            allowed_tools=("read_file", "apply_patch", "run_validation"),
            max_commands=24,
            max_wall_seconds=1_800,
        ),
        environment={},
        created_at_utc="2026-08-25T00:00:00+00:00",
    )

    class Sandboxes:
        sandbox_root = tmp_path / "sandboxes"

        def __init__(self) -> None:
            self.terminal_flags: list[bool] = []
            self.resumable_flags: list[bool] = []

        @staticmethod
        def require_active(_session_id: str, *, user_id: str) -> AgentSessionBinding:
            assert user_id == "user-a"
            return binding

        @staticmethod
        def start_task(*_args: object, **_kwargs: object) -> None:
            return None

        def record_result(self, *_args: object, **kwargs: object) -> dict[str, object]:
            self.terminal_flags.append(bool(kwargs["terminal"]))
            self.resumable_flags.append(bool(kwargs.get("resumable", False)))
            return {"status": "ACTIVE_CLEAN"}

    class Tiered:
        sandboxes = Sandboxes()
        lane_policy = LanePolicy(
            {
                lane: ModelRoute(
                    lane,
                    f"test-{lane.value}",
                    "http://127.0.0.1:1",
                    0,
                    131_072,
                    4_096,
                )
                for lane in ModelLane
            }
        )

        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.results = iter(("first inspected result", "second follow-up result"))
            self.yield_mode = False

        @staticmethod
        def agent_session_status(**_kwargs: object) -> dict[str, object]:
            return {"status": "ACTIVE"}

        def chat(self, **kwargs: object) -> dict[str, object]:
            messages = kwargs["messages"]
            assert isinstance(messages, list | tuple)
            prompt = json.dumps(messages)
            self.prompts.append(prompt)
            if self.yield_mode:
                return {
                    "response": {
                        "content": json.dumps(
                            {"action": "search", "query": "fixture-never-matches", "glob": "*.py"}
                        ),
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    }
                }
            result = next(self.results)
            return {
                "response": {
                    "content": json.dumps({"action": "finish", "result": result}),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            }

    tiered = Tiered()
    coordinator = ZetsuAgentCoordinator(
        tiered,
        checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints"),
    )
    first = coordinator.run_turn(
        user_id="user-a",
        repo_id="repo",
        instruction="Inspect component.py",
        lane=ModelLane.QUALITY,
        session_id="chat-agent-session",
        max_steps=4,
        max_chars=2_000,
        verification_argv=None,
        wait_timeout_seconds=30,
    )
    second = coordinator.run_turn(
        user_id="user-a",
        repo_id="repo",
        instruction="Follow up on that result",
        lane=ModelLane.QUALITY,
        session_id="chat-agent-session",
        max_steps=4,
        max_chars=2_000,
        verification_argv=("pytest", "-q"),
        wait_timeout_seconds=30,
    )

    tiered.yield_mode = True
    third = coordinator.run_turn(
        user_id="user-a",
        repo_id="repo",
        instruction="Continue inspecting without finishing yet",
        lane=ModelLane.QUALITY,
        session_id="chat-agent-session",
        max_steps=4,
        max_chars=2_000,
        verification_argv=("pytest", "-q"),
        wait_timeout_seconds=30,
    )

    assert first["session_id"] == second["session_id"] == third["session_id"] == "chat-agent-session"
    assert first["worktree_release"] == {"action": "PRESERVED_FOR_CONTINUATION"}
    assert second["worktree_release"] == {"action": "PRESERVED_FOR_CONTINUATION"}
    assert third["worktree_release"] == {"action": "PRESERVED_FOR_CONTINUATION"}
    assert third["status"] == "INCOMPLETE"
    assert third["failure_category"] == "agent_stagnation_exhausted"
    assert third["content"] == "Adaptive continuation stopped after repeated stagnant quanta."
    assert third["cumulative_steps"] == 12
    assert third["continuation_count"] == 2
    assert third["stagnation_count"] == 2
    assert tiered.sandboxes.terminal_flags == [False, False, False]
    assert tiered.sandboxes.resumable_flags == [False, False, True]
    assert "first inspected result" in tiered.prompts[1]
    assert len(tiered.prompts) == 14

    checkpoint = coordinator.checkpoints.read("chat-agent-session")
    assert checkpoint is not None
    assert checkpoint["required_verification_argv"] == ["pytest", "-q"]
    assert checkpoint["next_state"] == "stagnation_exhausted"
    assert checkpoint["step"] == 12


def test_checkpoint_resume_binds_route_and_required_verifier(tmp_path: Path, monkeypatch) -> None:
    tiered = _Tiered(tmp_path)
    coordinator = ZetsuAgentCoordinator(
        tiered, checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints")
    )
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))
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
    monkeypatch.setattr(coordinator, "_worktree_state", lambda _root: ("a" * 40, "0" * 64, []))
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
