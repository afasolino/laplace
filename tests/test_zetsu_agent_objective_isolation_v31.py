from research_workspace.zetsu_agent import (
    AgentExecutionState,
    AgentTelemetry,
    ZetsuAgentCoordinator,
)


def _prior(*, verified: bool) -> AgentExecutionState:
    return AgentExecutionState(
        objective="old repository task",
        step=11,
        summary="pytest passed for the old task",
        recent_observations=["OLD PYTEST RESULT", "old semantic conclusion"],
        changed_paths=["tests/test_old.py"],
        worktree_head="abc123",
        worktree_status_sha256="status123",
        lane="quality",
        model_id="old-model",
        context_limit=32000,
        required_verification_argv=["pytest", "tests/test_old.py"],
        validation_history=[{"passed": True, "old": True}],
        unresolved_failures=["old_reasoning_failure"],
        evidence_refs=["old_evidence"],
        mutation_epoch=2,
        last_verified_epoch=2 if verified else 1,
        command_count=17,
        consumed_wall_seconds=21.5,
        target_initial_head="initial",
        target_initial_status_sha256="initial-status",
        target_applied_status_sha256="applied-status",
        applied_patch_sha256="patch-sha",
        output_cap_continuations=3,
        continuation_count=4,
        stagnation_count=2,
        exploration_only_quanta=5,
        recent_action_fingerprints=["old-action"],
        quantum_action_fingerprints=["old-q"],
        quantum_exploration_only=False,
        telemetry=AgentTelemetry(qwen_calls=0),
    )


def test_restart_resets_semantic_state_but_preserves_execution_safety() -> None:
    prior = _prior(verified=True)
    state = ZetsuAgentCoordinator._restart_objective_state(
        prior,
        instruction="What is 2 + 2?",
        lane="quality",
        model_id="new-model",
        context_limit=64000,
        required_verification_argv=("pytest", "tests/test_old.py"),
        required_verification_plan=None,
    )
    assert state.objective == "What is 2 + 2?"
    assert state.summary == ""
    assert state.recent_observations == []
    assert state.validation_history == []
    assert state.evidence_refs == []
    assert state.unresolved_failures == []
    assert state.step == 0
    assert state.continuation_count == 0
    assert state.stagnation_count == 0
    assert state.recent_action_fingerprints == []
    assert state.changed_paths == ["tests/test_old.py"]
    assert state.worktree_head == "abc123"
    assert state.mutation_epoch == 2
    assert state.last_verified_epoch == 2
    assert state.command_count == 17
    assert state.target_initial_head == "initial"
    assert state.target_applied_status_sha256 == "applied-status"
    assert state.applied_patch_sha256 == "patch-sha"


def test_restart_carries_only_unverified_mutation_failure_when_needed() -> None:
    prior = _prior(verified=False)
    state = ZetsuAgentCoordinator._restart_objective_state(
        prior,
        instruction="new objective",
        lane="quality",
        model_id="new-model",
        context_limit=64000,
        required_verification_argv=("pytest", "tests/test_old.py"),
        required_verification_plan=None,
    )
    assert state.unresolved_failures == ["latest_mutation_unverified:epoch=2"]
    assert "old_reasoning_failure" not in state.unresolved_failures
