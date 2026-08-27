from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from research_workspace.upstream_ab import (
    AIDER_REPOMAP_REVISION,
    V33_BASELINE_REVISION,
    TrialMetrics,
    UpstreamABError,
    assess_phase_a,
    compare_trials,
    load_phase_a_tasks,
    materialize_head_snapshot,
    runtime_executable_path,
    runtime_output_path,
)


def _trial(task: str, *, provider: str, context: int, correctness: float = 1.0) -> TrialMetrics:
    return TrialMetrics(
        task_id=task,
        provider=provider,
        laplace_revision=V33_BASELINE_REVISION,
        candidate_revision=(AIDER_REPOMAP_REVISION if provider == "aider-repomap-v33" else None),
        task_manifest_sha256="a" * 64,
        snapshot_sha256="b" * 64,
        correctness=correctness,
        context_tokens=context,
        completion_tokens=100,
        tool_rounds=4,
        wall_time_seconds=10.0,
        failed=False,
        verifier_success=True,
        repository_coverage=1.0,
        relevance=1.0,
    )


def test_final_decision_requires_roadmap_minimum_ten_tasks() -> None:
    baseline = tuple(_trial(str(i), provider="laplace", context=1000) for i in range(3))
    candidate = tuple(
        _trial(str(i), provider="aider-repomap-v33", context=800) for i in range(3)
    )
    with pytest.raises(UpstreamABError, match="ab_trial_count_insufficient"):
        compare_trials(baseline, candidate)


def test_final_decision_adopts_only_after_paired_agent_efficiency_gain() -> None:
    baseline = tuple(_trial(str(i), provider="laplace", context=1000) for i in range(10))
    candidate = tuple(
        _trial(str(i), provider="aider-repomap-v33", context=800) for i in range(10)
    )
    result = compare_trials(baseline, candidate)
    assert result.decision == "ADOPT"
    assert "context_tokens_improved" in result.reasons


def test_final_decision_rejects_provenance_mismatch() -> None:
    baseline = tuple(_trial(str(i), provider="laplace", context=1000) for i in range(10))
    candidate = [
        _trial(str(i), provider="aider-repomap-v33", context=800) for i in range(10)
    ]
    candidate[0] = TrialMetrics(
        **{**candidate[0].to_json(), "snapshot_sha256": "c" * 64}
    )
    with pytest.raises(UpstreamABError, match="ab_trial_provenance_mismatch"):
        compare_trials(baseline, tuple(candidate))


def test_nonfinite_trial_metric_is_rejected() -> None:
    raw = _trial("x", provider="laplace", context=1000).to_json()
    raw["wall_time_seconds"] = math.nan
    with pytest.raises(UpstreamABError, match="ab_trial_wall_time_seconds_invalid"):
        TrialMetrics.from_json(raw)


def test_phase_a_never_returns_adopt() -> None:
    rows = []
    for index in range(10):
        rows.append({
            "task_id": str(index),
            "snapshot_match": True,
            "baseline": {
                "context_tokens": 1000,
                "path_recall": 1.0,
                "term_recall": 1.0,
                "wall_time_seconds": 1.0,
            },
            "candidate": {
                "context_tokens": 700,
                "path_recall": 1.0,
                "term_recall": 1.0,
                "wall_time_seconds": 1.0,
            },
        })
    result = assess_phase_a(rows)
    assert result["assessment"] == "PROMISING"
    assert result["final_adoption_decision"] is None


def test_manifest_freezes_twelve_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    tasks = load_phase_a_tasks(root / "benchmarks/v33_aider_repomap/tasks.json")
    assert len(tasks) == 12
    assert {task.task_id for task in tasks} == {
        "repo-map-builder",
        "agent-mutation-authority",
        "chat-agent-turn",
        "prompt-toolkit-input",
        "verification-sidecar",
        "gradio-controller",
        "mcp-sdk-bridge",
        "codex-status",
        "operator-agent-request",
        "conversation-service",
        "bounded-edit-diff-check",
        "task-label",
    }


def test_snapshot_uses_committed_bytes_not_dirty_or_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "v33@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "v33"], check=True)
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "untracked.py").write_text("secret = True\n", encoding="utf-8")

    target = repo / ".runtime" / "v33-aider" / "snapshot"
    actual_revision, digest = materialize_head_snapshot(
        repo, target, expected_revision=revision
    )
    assert actual_revision == revision
    assert len(digest) == 64
    assert (target / "tracked.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not (target / "untracked.py").exists()


def test_output_must_remain_under_runtime_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(UpstreamABError, match="v33_output_outside_runtime_results"):
        runtime_output_path(repo, repo / "outside.json", filename="unused.json")
    valid = runtime_output_path(repo, None, filename="phase-a.json")
    assert valid.parent == repo / ".runtime" / "v33-aider" / "results"


def test_runtime_executable_path_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    venv = repo / ".runtime" / "v33-aider" / "aider-venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python_link = bin_dir / "python"
    python_link.symlink_to(Path(sys.executable).resolve())

    selected = runtime_executable_path(
        repo,
        python_link,
        runtime_root=venv,
    )

    assert selected == python_link.absolute()
    assert selected.is_symlink()
    assert selected.resolve() == Path(sys.executable).resolve()
