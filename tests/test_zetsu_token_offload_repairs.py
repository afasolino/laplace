from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import types

import research_workspace.zetsu_sdk_stdio as sdk
from research_workspace.agent_sandbox import (
    AgentSandboxManager,
    AgentSessionBinding,
    AgentToolPolicy,
)
from research_workspace.service_tiers import ModelLane, ServiceTierError
from research_workspace.verification_policy import (
    normalize_verification_plan,
    verification_plan_digest,
)
from research_workspace.zetsu_agent import ZetsuAgentCoordinator
from research_workspace.zetsu_checkpoint import AgentCheckpointStore
from research_workspace.zetsu_state import AgentExecutionState, AgentRunContext


def _git(root: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *argv],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "owned.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "unrelated.py").write_text("OTHER = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    return _git(root, "rev-parse", "HEAD")


class _Sandboxes:
    def __init__(self, root: Path) -> None:
        self.sandbox_root = root


class _Tiered:
    def __init__(self, root: Path) -> None:
        self.sandboxes = _Sandboxes(root)


def _context(
    *,
    canonical: Path,
    internal: Path,
    head: str,
    session_id: str,
) -> AgentRunContext:
    binding = AgentSessionBinding(
        session_id=session_id,
        user_id="user-a",
        repo_id="repo",
        canonical_repository_root=str(canonical),
        worktree_root=str(internal),
        base_revision=head,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="test",
            allowed_tools=("apply_patch", "run_tests"),
            max_commands=8,
            max_wall_seconds=60,
        ),
        environment={},
        created_at_utc="2026-09-02T00:00:00+00:00",
    )
    return AgentRunContext(
        user_id="user-a",
        session_id=session_id,
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


def _state(
    coordinator: ZetsuAgentCoordinator,
    canonical: Path,
) -> AgentExecutionState:
    target_head, target_status, target_changed = coordinator._worktree_state(canonical)
    return AgentExecutionState(
        objective="objective",
        changed_paths=["owned.py"],
        mutation_epoch=1,
        last_verified_epoch=1,
        target_initial_head=target_head,
        target_initial_status_sha256=target_status,
        target_initial_snapshot_sha256=coordinator._target_snapshot_sha256(
            canonical,
            target_changed,
        ),
    )


def test_handoff_exact_patch_is_scoped_to_declared_changed_paths(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    head = _repo(canonical)
    internal = tmp_path / "internal"
    _git(canonical, "worktree", "add", "-q", "--detach", str(internal), head)
    (internal / "owned.py").write_text("VALUE = 2\n", encoding="utf-8")
    (internal / "unrelated.py").write_text("OTHER = 2\n", encoding="utf-8")

    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path / "sandbox"),
        checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints"),
    )
    patch = coordinator._exact_patch(internal, ["owned.py"])

    assert "owned.py" in patch
    assert "unrelated.py" not in patch


def test_promotion_preserves_preexisting_disjoint_target_changes(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    head = _repo(canonical)
    internal = tmp_path / "internal"
    _git(canonical, "worktree", "add", "-q", "--detach", str(internal), head)

    (canonical / "unrelated.py").write_text("OTHER = 7\n", encoding="utf-8")
    (internal / "owned.py").write_text("VALUE = 2\n", encoding="utf-8")

    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path / "sandbox"),
        checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints"),
    )
    state = _state(coordinator, canonical)
    ctx = _context(
        canonical=canonical,
        internal=internal,
        head=head,
        session_id="scope-session",
    )
    handoff = coordinator._handoff_patch(
        internal,
        "scope-session",
        ["owned.py"],
        max_chars=8_000,
    )

    promoted = coordinator._apply_verified_handoff(ctx, state, handoff)

    assert promoted["applied"] is True
    assert promoted["already_applied"] is False
    assert (canonical / "owned.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (canonical / "unrelated.py").read_text(encoding="utf-8") == "OTHER = 7\n"


def test_promotion_rejects_overlap_with_existing_target_change(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    head = _repo(canonical)
    internal = tmp_path / "internal"
    _git(canonical, "worktree", "add", "-q", "--detach", str(internal), head)

    (canonical / "owned.py").write_text("VALUE = 9\n", encoding="utf-8")
    (internal / "owned.py").write_text("VALUE = 2\n", encoding="utf-8")

    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path / "sandbox"),
        checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints"),
    )
    state = _state(coordinator, canonical)
    ctx = _context(
        canonical=canonical,
        internal=internal,
        head=head,
        session_id="overlap-session",
    )
    handoff = coordinator._handoff_patch(
        internal,
        "overlap-session",
        ["owned.py"],
        max_chars=8_000,
    )

    with pytest.raises(ServiceTierError, match="zetsu_agent_apply_target_overlap"):
        coordinator._apply_verified_handoff(ctx, state, handoff)

    assert (canonical / "owned.py").read_text(encoding="utf-8") == "VALUE = 9\n"


def test_promotion_recovers_checkpoint_gap_with_disjoint_dirty_baseline(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    head = _repo(canonical)
    internal = tmp_path / "internal"
    _git(canonical, "worktree", "add", "-q", "--detach", str(internal), head)

    (canonical / "unrelated.py").write_text("OTHER = 7\n", encoding="utf-8")
    (internal / "owned.py").write_text("VALUE = 2\n", encoding="utf-8")

    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path / "sandbox"),
        checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints"),
    )
    state = _state(coordinator, canonical)
    ctx = _context(
        canonical=canonical,
        internal=internal,
        head=head,
        session_id="gap-session",
    )
    handoff = coordinator._handoff_patch(
        internal,
        "gap-session",
        ["owned.py"],
        max_chars=8_000,
    )
    applied = subprocess.run(
        [
            "git",
            "-C",
            str(canonical),
            "apply",
            "--whitespace=error-all",
            str(handoff["patch_path"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert applied.returncode == 0

    recovered = coordinator._apply_verified_handoff(ctx, state, handoff)

    assert recovered["already_applied"] is True
    assert recovered["recovered_after_checkpoint_gap"] is True
    assert (canonical / "unrelated.py").read_text(encoding="utf-8") == "OTHER = 7\n"


def test_bridge_agent_task_timeout_tracks_declared_wait(monkeypatch) -> None:
    observed: list[float] = []

    class Fake:
        async def call_tool(self, _name: str, _arguments: dict[str, object]):
            return types.CallToolResult.model_validate(
                {"content": [], "isError": False}
            )

    @asynccontextmanager
    async def factory(
        _endpoint: str,
        _token: str,
        timeout: float,
    ) -> AsyncIterator[Any]:
        observed.append(timeout)
        yield Fake()

    monkeypatch.setattr(sdk, "_sdk_client", factory)
    backend = sdk.ZetsuBackend(
        "http://127.0.0.1:8765/mcp",
        "secret",
        timeout=30,
    )
    asyncio.run(
        backend.call_tool(
            "agent_task",
            {"wait_timeout_seconds": 1_800},
        )
    )

    assert observed == [1_830.0]


def test_bridge_non_agent_tool_keeps_short_timeout(monkeypatch) -> None:
    observed: list[float] = []

    class Fake:
        async def call_tool(self, _name: str, _arguments: dict[str, object]):
            return types.CallToolResult.model_validate(
                {"content": [], "isError": False}
            )

    @asynccontextmanager
    async def factory(
        _endpoint: str,
        _token: str,
        timeout: float,
    ) -> AsyncIterator[Any]:
        observed.append(timeout)
        yield Fake()

    monkeypatch.setattr(sdk, "_sdk_client", factory)
    backend = sdk.ZetsuBackend(
        "http://127.0.0.1:8765/mcp",
        "secret",
        timeout=30,
    )
    asyncio.run(backend.call_tool("search", {"query": "x"}))

    assert observed == [30.0]
def test_verification_plan_binds_cwd_and_rejects_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "fixture").mkdir()

    plan = normalize_verification_plan(
        repo,
        verification_plan=[
            {"cwd": "fixture", "argv": ["pytest", "test_public.py", "-q"]},
        ],
    )
    assert plan == (("fixture", ("pytest", "test_public.py", "-q")),)
    assert verification_plan_digest(plan) != verification_plan_digest(
        ((".", ("pytest", "test_public.py", "-q")),)
    )

    with pytest.raises(
        ServiceTierError, match="zetsu_agent_verify_cwd_forbidden"
    ):
        normalize_verification_plan(
            repo,
            verification_plan=[
                {
                    "cwd": "../outside",
                    "argv": ["pytest", "test_public.py", "-q"],
                },
            ],
        )


def test_bound_plan_runs_same_named_pytest_modules_in_own_cwds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    head = _repo(canonical)
    (canonical / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    _git(canonical, "add", ".gitignore")
    _git(canonical, "commit", "-qm", "ignore verifier caches")
    head = _git(canonical, "rev-parse", "HEAD")
    internal = tmp_path / "internal"
    _git(canonical, "worktree", "add", "-q", "--detach", str(internal), head)

    for directory, module_name, value in (
        ("async_fixture", "job_runner", 11),
        ("sqlite_fixture", "store", 22),
    ):
        root = internal / directory
        root.mkdir()
        (root / f"{module_name}.py").write_text(
            f"VALUE = {value}\n",
            encoding="utf-8",
        )
        (root / "test_public.py").write_text(
            f"from {module_name} import VALUE\n\n"
            f"def test_value():\n    assert VALUE == {value}\n",
            encoding="utf-8",
        )

    coordinator = ZetsuAgentCoordinator(
        _Tiered(tmp_path / "sandbox"),
        checkpoint_store=AgentCheckpointStore(tmp_path / "checkpoints"),
    )
    binding = AgentSessionBinding(
        session_id="plan-session",
        user_id="user-a",
        repo_id="repo",
        canonical_repository_root=str(canonical),
        worktree_root=str(internal),
        base_revision=head,
        grant_revision=1,
        tool_policy=AgentToolPolicy(
            policy_id="test",
            allowed_tools=("run_tests",),
            max_commands=8,
            max_wall_seconds=60,
        ),
        environment={},
        created_at_utc="2026-09-02T00:00:00+00:00",
    )
    plan = normalize_verification_plan(
        internal,
        verification_plan=[
            {
                "cwd": "async_fixture",
                "argv": ["pytest", "test_public.py", "-q"],
            },
            {
                "cwd": "sqlite_fixture",
                "argv": ["pytest", "test_public.py", "-q"],
            },
        ],
    )
    assert plan is not None

    ctx = AgentRunContext(
        user_id="user-a",
        session_id="plan-session",
        repo_id="repo",
        lane=ModelLane.QUALITY,
        binding=binding,
        worktree=internal,
        max_steps=12,
        max_chars=8_000,
        compaction_ratio=0.80,
        model_id="test-model",
        context_limit=131_072,
        required_verification_argv=None,
        run_started=time.monotonic(),
        remaining_wall_seconds=60,
        required_verification_plan=plan,
    )
    state = AgentExecutionState(
        objective="verify fixtures",
        mutation_epoch=1,
        last_verified_epoch=0,
        unresolved_failures=["latest_mutation_unverified:epoch=1"],
    )

    monkeypatch.setattr(
        AgentSandboxManager,
        "fixed_environment",
        staticmethod(lambda _binding: dict(os.environ)),
    )
    monkeypatch.setattr(coordinator, "_ensure_active", lambda *_args: None)
    monkeypatch.setattr(
        coordinator, "_progress", lambda *_args, **_kwargs: None
    )

    record = coordinator._verify(
        ctx,
        state,
        {
            "action": "verify",
            "argv": ["pytest", "does-not-exist.py", "-q"],
        },
    )

    assert record["passed"] is True
    assert record["argv"] is None
    assert record["step_count"] == 2
    assert record["plan"] == [
        {
            "cwd": "async_fixture",
            "argv": ["pytest", "test_public.py", "-q"],
        },
        {
            "cwd": "sqlite_fixture",
            "argv": ["pytest", "test_public.py", "-q"],
        },
    ]
    assert len(record["steps"]) == 2
    assert all(step["passed"] is True for step in record["steps"])
    assert state.last_verified_epoch == 1
    assert state.unresolved_failures == []
