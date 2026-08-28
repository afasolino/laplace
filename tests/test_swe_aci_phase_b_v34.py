from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from research_workspace.swe_aci_phase_b import (
    MIN_PHASE_B_PAIRED_TASKS,
    ACITrace,
    AgentTaskRecord,
    RecordingBaselineCoordinator,
    RecordingCandidateCoordinator,
    assess_phase_b,
    load_phase_b_tasks,
)


def record(
    task_id: str,
    kind: str,
    *,
    factor: float = 1.0,
    correct: bool = True,
    verifier: bool | None = None,
    usage_complete: bool = True,
    security: bool = True,
    model_id: str = "quality-model",
) -> AgentTaskRecord:
    return AgentTaskRecord(
        task_id=task_id,
        kind=kind,  # type: ignore[arg-type]
        correct=correct,
        input_tokens=int(1000 * factor),
        output_tokens=int(400 * factor),
        tool_rounds=max(1, int(10 * factor)),
        wall_seconds=10.0 * factor,
        failed=not correct,
        verifier_passed=verifier,
        repository_coverage=1.0,
        relevance=1.0,
        usage_complete=usage_complete,
        security_preserved=security,
        model_id=model_id,
        fixture_sha256="a" * 64,
    )


def matrix(*, factor: float = 1.0) -> list[AgentTaskRecord]:
    kinds = ["view"] * 4 + ["search"] * 4 + ["mutation"] * 4
    return [
        record(
            f"task-{index:02d}",
            kind,
            factor=factor,
            verifier=True if kind == "mutation" else None,
        )
        for index, kind in enumerate(kinds)
    ]


def test_phase_b_manifest_is_twelve_balanced_real_agent_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    tasks = load_phase_b_tasks(
        root / "benchmarks/v34_swe_agent_aci/tasks_phase_b.json"
    )
    assert len(tasks) == 12
    assert MIN_PHASE_B_PAIRED_TASKS == 10
    assert sum(task.kind == "view" for task in tasks) == 4
    assert sum(task.kind == "search" for task in tasks) == 4
    assert sum(task.kind == "mutation" for task in tasks) == 4
    assert all(
        task.verification_argv == ("pytest", "tests/test_fixture.py", "-q")
        for task in tasks
        if task.kind == "mutation"
    )


def test_phase_b_adopts_safe_material_efficiency_gain() -> None:
    result = assess_phase_b(matrix(), matrix(factor=0.80))
    assert result["decision"] == "ADOPT"
    assert result["efficiency_gain_count"] == 4


def test_phase_b_keeps_on_pairwise_correctness_regression() -> None:
    baseline = matrix()
    candidate = matrix(factor=0.80)
    item = candidate[0]
    candidate[0] = AgentTaskRecord(
        **{
            **item.__dict__,
            "correct": False,
            "failed": True,
        }
    )
    result = assess_phase_b(baseline, candidate)
    assert result["decision"] == "KEEP_LAPLACE"
    assert result["non_regression"]["pairwise_safety"] is False


def test_phase_b_blocks_incomplete_exact_model_usage() -> None:
    baseline = matrix()
    candidate = matrix(factor=0.80)
    item = candidate[0]
    candidate[0] = AgentTaskRecord(
        **{
            **item.__dict__,
            "usage_complete": False,
        }
    )
    assert assess_phase_b(baseline, candidate) == {
        "decision": "BLOCKED",
        "reason": "model_usage_incomplete",
    }


def test_phase_b_blocks_model_or_fixture_mismatch() -> None:
    baseline = matrix()
    candidate = matrix(factor=0.80)
    item = candidate[0]
    candidate[0] = AgentTaskRecord(
        **{
            **item.__dict__,
            "model_id": "different-model",
        }
    )
    assert assess_phase_b(baseline, candidate)["decision"] == "BLOCKED"

    candidate = matrix(factor=0.80)
    item = candidate[0]
    candidate[0] = AgentTaskRecord(
        **{
            **item.__dict__,
            "fixture_sha256": "b" * 64,
        }
    )
    assert assess_phase_b(baseline, candidate)["reason"] == "fixture_mismatch"


def _aci(
    coordinator_type: type[object],
    *,
    transient: bool,
    tools: frozenset[str],
) -> object:
    coordinator = object.__new__(coordinator_type)
    coordinator.v34_trace = ACITrace()  # type: ignore[attr-defined]
    coordinator.tiered = SimpleNamespace(  # type: ignore[attr-defined]
        agent_session_status=lambda **_kwargs: {"status": "ACTIVE"}
    )
    ctx = SimpleNamespace(
        worktree=Path.cwd(),
        user_id="owner",
        session_id="session",
        allow_mutation=transient,
        required_verification_argv=("pytest", "-q"),
        binding=SimpleNamespace(
            tool_policy=SimpleNamespace(allowed_tools=tools)
        ),
    )
    return coordinator._typed_aci(ctx, SimpleNamespace())  # type: ignore[attr-defined]


def test_phase_b_recorders_preserve_transient_mutation_authority() -> None:
    for coordinator_type in (
        RecordingBaselineCoordinator,
        RecordingCandidateCoordinator,
    ):
        assert _aci(
            coordinator_type,
            transient=False,
            tools=frozenset({"apply_patch"}),
        ).allow_mutation is False
        assert _aci(
            coordinator_type,
            transient=True,
            tools=frozenset({"read_file"}),
        ).allow_mutation is False
        assert _aci(
            coordinator_type,
            transient=True,
            tools=frozenset({"read_file", "apply_patch"}),
        ).allow_mutation is True


def test_trace_coverage_is_deduplicated() -> None:
    trace = ACITrace()
    trace.record("src/a.py")
    trace.record("src/a.py")
    trace.record_search_result(
        {
            "matches": [
                {"path": "src/b.py"},
                {"path": "src/b.py"},
            ]
        }
    )
    assert trace.paths == ["src/a.py", "src/b.py"]
