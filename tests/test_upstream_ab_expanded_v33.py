from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.upstream_ab import UpstreamABError
from research_workspace.upstream_ab_expanded import (
    EXPANDED_BUDGETS,
    EXPANDED_REPETITIONS,
    EXPANDED_TASK_COUNT,
    ExpandedPhaseATask,
    assess_expanded_phase_a,
    expanded_quality,
    load_expanded_tasks,
    timing_summary,
    validate_candidate_repeat_payload,
)


def _row(
    task_id: str,
    budget: int,
    *,
    category: str = "orientation",
    focus_mode: str = "NO_FOCUS",
    baseline_quality: float = 1.0,
    candidate_quality: float = 1.0,
    baseline_tokens: int = 1000,
    candidate_tokens: int = 800,
    baseline_wall: float = 1.0,
    candidate_wall: float = 0.8,
) -> dict[str, object]:
    def metrics(
        quality: float,
        tokens: int,
        wall: float,
    ) -> dict[str, object]:
        return {
            "context_tokens": tokens,
            "path_recall": quality,
            "path_precision": quality,
            "term_recall": quality,
            "symbol_recall": quality,
            "relevance_score": quality,
            "relevance_per_1k_tokens": quality * 1000 / tokens,
            "timing": {
                "samples": [wall] * EXPANDED_REPETITIONS,
                "median_seconds": wall,
                "p95_seconds": wall,
            },
        }

    return {
        "task_id": task_id,
        "token_budget": budget,
        "category": category,
        "focus_mode": focus_mode,
        "snapshot_match": True,
        "baseline": metrics(baseline_quality, baseline_tokens, baseline_wall),
        "candidate": metrics(candidate_quality, candidate_tokens, candidate_wall),
    }


def _full_matrix(
    *,
    candidate_quality: float = 1.0,
    candidate_tokens: int = 800,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    categories = [
        "direct_symbol",
        "cross_file",
        "orientation",
        "ambiguous",
        "change_impact",
        "governance",
    ]
    for index in range(EXPANDED_TASK_COUNT):
        category = categories[index % len(categories)]
        focus_mode = "NO_FOCUS" if index >= 10 else "FOCUSED"
        for budget in EXPANDED_BUDGETS:
            rows.append(
                _row(
                    f"task-{index:02d}",
                    budget,
                    category=category,
                    focus_mode=focus_mode,
                    candidate_quality=candidate_quality,
                    candidate_tokens=candidate_tokens,
                )
            )
    return rows


def test_manifest_has_thirty_balanced_tasks_and_four_budgets() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/v33_aider_repomap/tasks_expanded.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = load_expanded_tasks(path)

    assert len(tasks) == EXPANDED_TASK_COUNT == 30
    assert raw["budgets"] == list(EXPANDED_BUDGETS)
    assert raw["timed_repetitions"] == EXPANDED_REPETITIONS
    assert sum(task.focus_mode == "NO_FOCUS" for task in tasks) >= 12
    assert sum(task.focus_mode == "FOCUSED" for task in tasks) >= 8
    categories = {task.category for task in tasks}
    assert categories == {
        "direct_symbol",
        "cross_file",
        "orientation",
        "ambiguous",
        "change_impact",
        "governance",
    }


def test_expanded_quality_scores_path_precision_and_recall() -> None:
    task = ExpandedPhaseATask(
        task_id="x",
        category="orientation",
        focus_mode="NO_FOCUS",
        query="x",
        focus_paths=(),
        expected_paths=("a.py", "b.py"),
        expected_terms=("alpha",),
        expected_symbols=("Symbol",),
    )
    metrics = expanded_quality(
        "a.py\nother.py\nalpha\nSymbol\n",
        task,
        ("a.py", "b.py", "other.py"),
    )
    assert metrics["path_recall"] == 0.5
    assert metrics["path_precision"] == 0.5
    assert metrics["term_recall"] == 1.0
    assert metrics["symbol_recall"] == 1.0


def test_timing_requires_exact_five_samples() -> None:
    with pytest.raises(UpstreamABError, match="timing_repetition_mismatch"):
        timing_summary([1.0, 2.0])


def test_expanded_assessment_can_be_promising_overall() -> None:
    result = assess_expanded_phase_a(_full_matrix())
    assert result["assessment"] == "PROMISING"
    assert result["overall_promising"] is True
    assert result["final_adoption_decision"] is None


def test_expanded_assessment_rejects_quality_regression_despite_efficiency() -> None:
    result = assess_expanded_phase_a(
        _full_matrix(candidate_quality=0.70, candidate_tokens=500)
    )
    assert result["assessment"] == "NOT_PROMISING"
    assert result["overall_promising"] is False
    assert result["advantage_strata"] == []


def test_expanded_assessment_requires_complete_task_budget_matrix() -> None:
    rows = _full_matrix()
    with pytest.raises(UpstreamABError, match="trial_count_invalid"):
        assess_expanded_phase_a(rows[:-1])


def test_expanded_assessment_rejects_duplicate_task_budget_pair() -> None:
    rows = _full_matrix()
    rows[-1] = rows[0]
    with pytest.raises(UpstreamABError, match="trial_duplicate"):
        assess_expanded_phase_a(rows)


def test_candidate_repeat_payload_requires_same_process_stability() -> None:
    raw = {
        "text": "map",
        "repeat_count": EXPANDED_REPETITIONS,
        "within_process_stable": False,
        "unique_map_hashes": 2,
        "wall_time_seconds_samples": [0.1] * EXPANDED_REPETITIONS,
    }
    with pytest.raises(
        UpstreamABError,
        match="candidate_within_process_nondeterministic",
    ):
        validate_candidate_repeat_payload(raw)


def test_candidate_repeat_payload_accepts_five_stable_measured_calls() -> None:
    raw = {
        "text": "map",
        "repeat_count": EXPANDED_REPETITIONS,
        "within_process_stable": True,
        "unique_map_hashes": 1,
        "wall_time_seconds_samples": [0.1, 0.2, 0.1, 0.3, 0.2],
    }
    result = validate_candidate_repeat_payload(raw)
    assert result.text == "map"
    assert result.unique_map_hashes == 1
    assert result.wall_times == (0.1, 0.2, 0.1, 0.3, 0.2)


def test_candidate_repeat_payload_rejects_wrong_sample_count() -> None:
    raw = {
        "text": "map",
        "repeat_count": EXPANDED_REPETITIONS,
        "within_process_stable": True,
        "unique_map_hashes": 1,
        "wall_time_seconds_samples": [0.1, 0.2],
    }
    with pytest.raises(
        UpstreamABError,
        match="candidate_repeat_samples_invalid",
    ):
        validate_candidate_repeat_payload(raw)
