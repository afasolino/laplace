from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.evaluation import (
    GPU_BLOCKED_STATUS,
    REQUIRED_CATEGORIES,
    EvaluationError,
    load_suite,
    run_offline_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/evaluation/frozen_suite_v1.json"
FIXTURES = ROOT / "benchmarks/evaluation"


def test_frozen_suite_covers_every_required_category_and_passes() -> None:
    suite = load_suite(SUITE)
    assert {case.category for case in suite.cases} == set(REQUIRED_CATEGORIES)
    report = run_offline_evaluation(SUITE, FIXTURES)
    assert report.status == "PASS"
    assert report.fixture_only is True
    assert report.fixture_task_quality["passed_cases"] == len(suite.cases)
    assert report.live_model_quality["status"] == GPU_BLOCKED_STATUS
    assert report.infrastructure_correctness["provider_contacted"] is False


def test_suite_rejects_missing_category_unknown_fields_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    raw = json.loads(SUITE.read_text(encoding="utf-8"))
    raw["cases"] = raw["cases"][:-1]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EvaluationError, match="categories missing"):
        load_suite(missing)

    raw = json.loads(SUITE.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(unknown)

    raw = json.loads(SUITE.read_text(encoding="utf-8"))
    raw["cases"][1]["case_id"] = raw["cases"][0]["case_id"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EvaluationError, match="unique"):
        load_suite(duplicate)


def test_patch_path_escape_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(SUITE.read_text(encoding="utf-8"))
    for case in raw["cases"]:
        if case["category"] == "python_patch_applicability":
            case["inputs"]["file"] = "../outside.py"
            break
    malicious = tmp_path / "malicious.json"
    malicious.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EvaluationError, match="unsafe"):
        run_offline_evaluation(malicious, FIXTURES)

