from __future__ import annotations

import csv
import sys
from pathlib import Path

from research_workspace.engineering import normalize_task_spec
from research_workspace.paired_benchmark import (
    _BENCHMARK_TASKS,
    _codex_environment,
    _lane_prompt,
    _patch_from_lane,
    _shared_task_specification,
    _task_spec,
    _write_reports,
)
from research_workspace.quality_improvement import _UNSEEN_TASKS


def test_all_six_public_benchmark_specs_normalize() -> None:
    root = Path(__file__).resolve().parents[1]
    assert [task.domain for task in _BENCHMARK_TASKS].count("python") == 4
    assert [task.domain for task in _BENCHMARK_TASKS].count("systemverilog") == 2
    for task in _BENCHMARK_TASKS:
        normalized = normalize_task_spec(root, task.domain, _task_spec(task))
        assert normalized["task_id"] == task.task_id


def test_all_six_unseen_quality_specs_normalize() -> None:
    root = Path(__file__).resolve().parents[1]
    assert [task.domain for task in _UNSEEN_TASKS].count("python") == 4
    assert [task.domain for task in _UNSEEN_TASKS].count("systemverilog") == 2
    for task in _UNSEEN_TASKS:
        normalized = normalize_task_spec(root, task.domain, _task_spec(task))
        assert normalized["quality_contract"]["repair_budget"] == 2


def test_codex_lane_receives_the_same_two_repair_cycle_budget() -> None:
    task = _BENCHMARK_TASKS[0]
    assert "at most two repair cycles" in _lane_prompt(task, _task_spec(task))


def test_codex_lane_pins_the_control_plane_python_environment() -> None:
    environment = _codex_environment(sys.executable)
    assert environment["PATH"].split(":")[0].endswith(".venv/bin")
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_python_lanes_receive_absolute_shared_interpreter_commands() -> None:
    task = _BENCHMARK_TASKS[0]
    specification = _shared_task_specification(task, sys.executable)
    commands = specification["verification_commands"]
    assert isinstance(commands, list)
    assert all(str(Path(sys.executable).absolute()) in command for command in commands)


def test_lane_patch_paths_do_not_keep_the_git_b_prefix(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "allowed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "allowed.py"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    source.write_text("value = 2\n", encoding="utf-8")
    _, changed = _patch_from_lane(tmp_path, base, tmp_path / "lane.patch")
    assert changed == ["allowed.py"]


def test_comparison_report_flattens_each_lane_and_keeps_quality_dimensions(tmp_path: Path) -> None:
    payload = {
        "status": "MEASURED",
        "reason": "fixture",
        "tasks": [
            {
                "task_id": "py_fixture",
                "domain": "python",
                "valid_run": True,
                "execution_overlap_proven": True,
                "held_out_exposed_to_lanes": False,
                "lanes": {
                    "codex_direct": {
                        "status": "COMPLETE",
                        "wall_time_s": 1.0,
                        "repair_cycles": None,
                        "human_intervention": 0,
                        "score": {
                            "objective_score_0_100": 100,
                            "functional_correctness": True,
                            "held_out_correctness": True,
                            "typing_and_lint_quality": True,
                            "robustness": True,
                        },
                    }
                },
            }
        ],
    }
    paths = _write_reports(tmp_path, payload)
    with Path(str(paths["csv"])).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["lane"] == "codex_direct"
    assert rows[0]["held_out_correctness"] == "True"
    assert "Codex-direct" in Path(str(paths["markdown"])).read_text(encoding="utf-8")
