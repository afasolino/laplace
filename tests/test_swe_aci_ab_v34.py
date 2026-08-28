from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_workspace.bounded_aci import BoundedACIError
from research_workspace.swe_aci_ab import (
    PHASE_A_REPETITIONS,
    SWE_AGENT_VIEW_LINES,
    PrimitiveTrial,
    SweAgentABCoordinator,
    SweAgentACIShadow,
    assess_phase_a,
    load_phase_a_tasks,
)


def repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "laplace@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Laplace Test"],
        cwd=repo,
        check=True,
    )
    (repo / "a.py").write_text(
        "def alpha(value: int) -> int:\n"
        "    target = value\n"
        "    target += 1\n"
        "    return target\n",
        encoding="utf-8",
    )
    (repo / "b.py").write_text("target = 2\n", encoding="utf-8")
    (repo / "long.txt").write_text(
        "".join(
            f"line-{index:03d}\n"
            for index in range(1, 151)
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    return repo


def shadow(
    repo: Path,
    *,
    mutate: bool = False,
) -> SweAgentACIShadow:
    return SweAgentACIShadow(
        repo,
        owner_user_id="owner",
        session_id="session",
        allow_mutation=mutate,
    )


def test_read_region_caps_actual_agent_method_to_100_lines(
    tmp_path: Path,
) -> None:
    result = shadow(repo_fixture(tmp_path)).read_region(
        path="long.txt",
        start_line=1,
        end_line=150,
    )
    assert result["window_lines"] == SWE_AGENT_VIEW_LINES
    assert result["end_line"] == 100
    assert result["requested_end_line"] == 150
    assert result["window_truncated"] is True


def test_search_text_returns_unique_files_without_match_lines(
    tmp_path: Path,
) -> None:
    result = shadow(repo_fixture(tmp_path)).search_text(
        query="target",
        glob="*.py",
    )
    assert set(result["files"]) == {"a.py", "b.py"}
    assert result["file_count"] == 2
    assert "matches" not in result
    rendered = repr(result)
    assert "'line':" not in rendered
    assert "'text':" not in rendered


def test_empty_search_is_explicit(tmp_path: Path) -> None:
    result = shadow(repo_fixture(tmp_path)).search_text(
        query="not-present",
        glob="*.py",
    )
    assert result["files"] == []
    assert result["message"] == "NO_MATCHES"


def test_path_escape_remains_fail_closed(tmp_path: Path) -> None:
    repo = repo_fixture(tmp_path)
    (tmp_path / "outside.py").write_text(
        "secret = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(BoundedACIError, match="aci_path_escape"):
        shadow(repo).read_region(
            path="../outside.py",
            start_line=1,
            end_line=1,
        )


def test_git_metadata_remains_fail_closed(tmp_path: Path) -> None:
    repo = repo_fixture(tmp_path)
    with pytest.raises(BoundedACIError, match="aci_git_metadata_forbidden"):
        shadow(repo).read_region(
            path=".git/config",
            start_line=1,
            end_line=1,
        )


def test_mutation_denial_precedes_syntax_preflight(
    tmp_path: Path,
) -> None:
    repo = repo_fixture(tmp_path)
    with pytest.raises(BoundedACIError) as captured:
        shadow(repo).edit_region(
            path="a.py",
            old_text="def alpha(value: int) -> int:",
            new_text="def alpha(value: int -> int:",
        )
    assert captured.value.category == "aci_mutation_not_allowed"


def test_invalid_python_edit_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    repo = repo_fixture(tmp_path)
    before = (repo / "a.py").read_text(encoding="utf-8")
    with pytest.raises(BoundedACIError) as captured:
        shadow(repo, mutate=True).edit_region(
            path="a.py",
            old_text="def alpha(value: int) -> int:",
            new_text="def alpha(value: int -> int:",
        )
    assert captured.value.category == "aci_swe_python_syntax_invalid"
    assert (repo / "a.py").read_text(encoding="utf-8") == before
    assert (
        subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=repo,
            check=False,
        ).returncode
        == 0
    )


def test_valid_python_edit_delegates_to_bounded_mutation(
    tmp_path: Path,
) -> None:
    repo = repo_fixture(tmp_path)
    result = shadow(repo, mutate=True).edit_region(
        path="a.py",
        old_text="    target += 1\n",
        new_text="    target += 2\n",
    )
    assert result["syntax_preflight"] == "PASS"
    assert "target += 2" in (
        repo / "a.py"
    ).read_text(encoding="utf-8")
    subprocess.run(
        ["git", "diff", "--check"],
        cwd=repo,
        check=True,
    )


def _coordinator_aci(
    repo: Path,
    *,
    transient: bool,
    tools: frozenset[str],
) -> SweAgentACIShadow:
    coordinator = object.__new__(SweAgentABCoordinator)
    coordinator.tiered = SimpleNamespace(
        agent_session_status=lambda **_kwargs: {"status": "ACTIVE"}
    )
    ctx = SimpleNamespace(
        worktree=repo,
        user_id="owner",
        session_id="session",
        task_id="task",
        allow_mutation=transient,
        required_verification_argv=("pytest", "-q"),
        binding=SimpleNamespace(
            tool_policy=SimpleNamespace(allowed_tools=tools)
        ),
    )
    result = coordinator._typed_aci(ctx, SimpleNamespace())
    assert isinstance(result, SweAgentACIShadow)
    return result


def test_candidate_requires_transient_and_policy_mutation_authority(
    tmp_path: Path,
) -> None:
    repo = repo_fixture(tmp_path)
    assert _coordinator_aci(
        repo,
        transient=False,
        tools=frozenset({"apply_patch"}),
    ).allow_mutation is False
    assert _coordinator_aci(
        repo,
        transient=True,
        tools=frozenset({"read_file"}),
    ).allow_mutation is False
    allowed = _coordinator_aci(
        repo,
        transient=True,
        tools=frozenset({"read_file", "apply_patch"}),
    )
    assert allowed.allow_mutation is True
    assert allowed.required_verification_argv == ("pytest", "-q")


def trial(
    task_id: str,
    primitive: str,
    *,
    tokens: float = 100.0,
    correctness: float = 1.0,
    verifier: float = 1.0,
    safe: bool = True,
) -> PrimitiveTrial:
    return PrimitiveTrial(
        task_id=task_id,
        primitive=primitive,  # type: ignore[arg-type]
        correctness=correctness,
        context_tokens=tokens,
        wall_seconds=1.0,
        failure=0.0,
        verifier_success=verifier,
        repository_coverage=1.0,
        relevance=1.0,
        security_preserved=safe,
    )


def matrix(*, candidate: bool = False) -> list[PrimitiveTrial]:
    primitives = (
        ["view"] * 3
        + ["search"] * 4
        + ["syntax_preflight"] * 2
        + ["governance"] * 3
    )
    return [
        trial(
            f"task-{index:02d}",
            primitive,
            tokens=80.0 if candidate and primitive == "search" else 100.0,
        )
        for index, primitive in enumerate(primitives)
    ]


def test_phase_a_assessment_promotes_safe_measured_primitive() -> None:
    result = assess_phase_a(matrix(), matrix(candidate=True))
    assert result["assessment"] == "PROMISING"
    assert result["promising_primitives"] == ["search"]
    assert result["final_adoption_decision"] is None


def test_phase_a_assessment_blocks_security_regression() -> None:
    baseline = matrix()
    candidate = matrix(candidate=True)
    last = candidate[-1]
    candidate[-1] = PrimitiveTrial(
        **{
            **last.__dict__,
            "correctness": 0.0,
            "security_preserved": False,
        }
    )
    result = assess_phase_a(baseline, candidate)
    assert result["assessment"] == "NOT_PROMISING"
    assert result["governance_safe"] is False


def test_phase_a_requires_complete_paired_matrix() -> None:
    result = assess_phase_a(matrix(), matrix(candidate=True)[:-1])
    assert result["assessment"] == "BLOCKED"


def test_frozen_manifest_has_twelve_balanced_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    tasks = load_phase_a_tasks(
        root / "benchmarks/v34_swe_agent_aci/tasks.json"
    )
    assert len(tasks) == 12
    assert PHASE_A_REPETITIONS == 5
    assert sum(task.primitive == "view" for task in tasks) == 3
    assert sum(task.primitive == "search" for task in tasks) == 4
    assert sum(task.primitive == "syntax_preflight" for task in tasks) == 2
    assert sum(task.primitive == "governance" for task in tasks) == 3
