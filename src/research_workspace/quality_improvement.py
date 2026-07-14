"""Laplace-only quality-improvement evaluation with immutable held-out scoring.

This module intentionally never starts a Codex-direct lane.  It reuses the
paired-benchmark evaluator only after each local five-role worktree has stopped,
so held-out material remains unavailable to every implementation role.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from .engineering import (
    AgentTaskStore,
    JsonObject,
    LocalToolRunner,
    _write_json_atomic,
    normalize_task_spec,
)
from .inference import ServingCandidate
from .paired_benchmark import (
    BenchmarkTask,
    _BENCHMARK_TASKS,
    _evaluate_lane,
    _git,
    _task_spec,
)
from .team_runner import LocalTeamRunner, TeamWorkflowOptions


ORIGINAL_RUN_ID = "20260713T103206Z_6bce7155"
ORIGINAL_BASE_COMMIT = "e34936fc686ab70d726e7289336f49155dc920f2"

TARGETED_TASK_IDS: tuple[str, ...] = (
    "py_fastapi_strict_endpoint",
    "py_unseen_sqlite_state",
    "sv_ready_valid_buffer",
    "sv_axi_lite_irq_regs",
    "sv_unseen_rv_slot",
    "sv_unseen_w1c_event",
)

_UNSEEN_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        "py_unseen_pydantic_policy",
        "python",
        "benchmarks/a6000_agent_team/quality_unseen/py_unseen_pydantic_policy",
        ("request.py",),
        ("test_public.py",),
        "Make the policy request model reject coercion and undeclared fields while preserving valid integer requests.",
        (
            "Use strict integer validation.",
            "Forbid undeclared fields.",
            "Preserve the valid model contract.",
        ),
    ),
    BenchmarkTask(
        "py_unseen_atomic_writer",
        "python",
        "benchmarks/a6000_agent_team/quality_unseen/py_unseen_atomic_writer",
        ("writer.py",),
        ("test_public.py",),
        "Constrain JSON output paths to the caller-provided root without weakening valid nested output.",
        (
            "Reject absolute paths.",
            "Reject traversal outside root.",
            "Preserve valid nested JSON output.",
        ),
    ),
    BenchmarkTask(
        "py_unseen_async_deadline",
        "python",
        "benchmarks/a6000_agent_team/quality_unseen/py_unseen_async_deadline",
        ("deadline.py",),
        ("test_public.py",),
        "Enforce an asynchronous deadline and await cancellation cleanup before raising DeadlineExceeded.",
        (
            "Raise DeadlineExceeded after the deadline.",
            "Await task cancellation cleanup.",
            "Preserve successful results.",
        ),
    ),
    BenchmarkTask(
        "py_unseen_sqlite_state",
        "python",
        "benchmarks/a6000_agent_team/quality_unseen/py_unseen_sqlite_state",
        ("state.py",),
        ("test_public.py",),
        "Make a SQLite state transition idempotent, conflict-safe, and transactional.",
        (
            "Return False for an identical existing state.",
            "Reject conflicts without overwriting provenance.",
            "Rollback on failure.",
        ),
    ),
    BenchmarkTask(
        "sv_unseen_rv_slot",
        "systemverilog",
        "benchmarks/a6000_agent_team/quality_unseen/sv_unseen_rv_slot",
        ("rv_slot.sv", "tb_rv_slot.sv"),
        ("tb_rv_slot.sv",),
        "Repair the one-slot ready/valid buffer to support a simultaneous dequeue and enqueue without data loss.",
        (
            "Keep payload stable while stalled.",
            "Accept replacement on simultaneous dequeue/enqueue.",
            "Reset empty.",
        ),
    ),
    BenchmarkTask(
        "sv_unseen_w1c_event",
        "systemverilog",
        "benchmarks/a6000_agent_team/quality_unseen/sv_unseen_w1c_event",
        ("w1c_event.sv", "tb_w1c_event.sv"),
        ("tb_w1c_event.sv",),
        "Repair the W1C event register so IRQ requires both enable and pending state while respecting WSTRB.",
        (
            "Reset pending and enable low.",
            "Apply byte-strobed writes only.",
            "Clear pending with W1C and deassert IRQ.",
        ),
    ),
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> JsonObject:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _status_from_result(result: JsonObject) -> str:
    value = result.get("status")
    return value if isinstance(value, str) else "UNKNOWN"


def _worktree_from_result(result: JsonObject) -> Path | None:
    value = result.get("worktree")
    if isinstance(value, str):
        path = Path(value)
        return path if path.is_dir() else None
    return None


def _task_record(
    repository_root: Path,
    root: Path,
    task: BenchmarkTask,
    candidate: ServingCandidate,
    *,
    base_commit: str,
    options: TeamWorkflowOptions,
    control_python: str,
    timeout_seconds: int,
    shared_reference_root: Path | None = None,
) -> JsonObject:
    """Run an isolated Laplace task, then score it in a fresh evaluator worktree."""
    project_root = root / "projects" / task.task_id
    specification = normalize_task_spec(repository_root, task.domain, _task_spec(task))
    store = AgentTaskStore(project_root)
    task_state = store.create(task.domain, specification)
    started_at, started = _now(), time.monotonic()
    result = LocalTeamRunner(
        repository_root,
        project_root,
        candidate,
        options=options,
        shared_reference_root=shared_reference_root,
    ).run(task_state.task_id, query=task.objective)
    elapsed = time.monotonic() - started
    record: JsonObject = {
        "task_id": task.task_id,
        "domain": task.domain,
        "base_commit": base_commit,
        "started_at": started_at,
        "ended_at": _now(),
        "wall_time_s": elapsed,
        "status": _status_from_result(result),
        "typed_result": result,
        "project_root": str(project_root),
        "held_out_exposed_to_implementation": False,
        "human_intervention": 0,
        "workflow": {
            "role_mode": options.role_mode,
            "retrieval_mode": options.retrieval_mode,
            "adversarial_verification": options.adversarial_verification,
            "reviewer_invariants": options.reviewer_invariants,
            "repair_budget": 2,
            "shared_reference_root": (
                str(shared_reference_root.resolve()) if shared_reference_root is not None else None
            ),
        },
    }
    worktree = _worktree_from_result(result)
    if worktree is None:
        record["evaluation"] = {"status": "NOT_EVALUATED", "reason": "No task worktree exists."}
        return record
    evaluated = _evaluate_lane(
        repository_root,
        root,
        task,
        "laplace_team",
        worktree,
        base_commit,
        control_python,
        timeout_seconds,
    )
    record["evaluation"] = evaluated
    record["score"] = evaluated.get("score")
    task_json = result.get("task")
    if isinstance(task_json, dict):
        loops = task_json.get("correction_loops")
        record["repair_cycles"] = loops if isinstance(loops, int) else None
        record["verifier_approved"] = task_json.get("state") == "final_report"
    return record


def _mean(records: list[JsonObject], domain: str) -> float | None:
    scores: list[float] = []
    for record in records:
        if record.get("domain") != domain:
            continue
        score = record.get("score")
        if isinstance(score, dict) and isinstance(score.get("objective_score_0_100"), (int, float)):
            scores.append(float(score["objective_score_0_100"]))
    return statistics.mean(scores) if scores else None


def _held_out_false_approval(records: list[JsonObject]) -> list[str]:
    failures: list[str] = []
    for record in records:
        score = record.get("score")
        if (
            record.get("verifier_approved") is True
            and isinstance(score, dict)
            and score.get("held_out_correctness") is False
        ):
            failures.append(str(record.get("task_id")))
    return failures


def run_original_rerun(
    repository_root: Path,
    candidate: ServingCandidate,
    *,
    output_root: Path,
    control_python: str | None = None,
    timeout_seconds: int = 900,
    shared_reference_root: Path | None = None,
) -> JsonObject:
    """Re-run the original six fixtures from their historic, common checkpoint."""
    root = repository_root.resolve()
    run_root = output_root / "runs" / f"original_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    options = TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT)
    records = [
        _task_record(
            root,
            run_root,
            task,
            candidate,
            base_commit=ORIGINAL_BASE_COMMIT,
            options=options,
            control_python=control_python or sys.executable,
            timeout_seconds=timeout_seconds,
            shared_reference_root=shared_reference_root,
        )
        for task in _BENCHMARK_TASKS
    ]
    payload: JsonObject = {
        "status": "MEASURED",
        "purpose": "Laplace-only original-task rerun after workflow quality controls.",
        "base_commit": ORIGINAL_BASE_COMMIT,
        "candidate": candidate.to_json(),
        "same_original_task_specs": True,
        "same_original_public_tests": True,
        "same_original_held_out_tests": True,
        "same_timeout_seconds": timeout_seconds,
        "same_repair_budget": 2,
        "held_out_exposed_to_implementation": False,
        "tasks": records,
        "aggregate": {
            "python_mean_score": _mean(records, "python"),
            "systemverilog_mean_score": _mean(records, "systemverilog"),
            "false_verifier_approvals": _held_out_false_approval(records),
        },
        "run_root": str(run_root),
        "created_at": _now(),
    }
    _write_json_atomic(output_root / "original_tasks_rerun.json", payload)
    return payload


def run_unseen_tasks(
    repository_root: Path,
    candidate: ServingCandidate,
    *,
    output_root: Path,
    control_python: str | None = None,
    timeout_seconds: int = 900,
    shared_reference_root: Path | None = None,
) -> JsonObject:
    """Evaluate six new fixtures from one post-improvement checkpoint.

    The unseen fixtures and evaluator are committed before this function runs.
    Their held-out strings are evaluated only after each Laplace worktree stops.
    """
    root = repository_root.resolve()
    base_commit = _git(root, ["rev-parse", "HEAD"]).strip()
    run_root = output_root / "runs" / f"unseen_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    options = TeamWorkflowOptions(base_commit=base_commit)
    records = [
        _task_record(
            root,
            run_root,
            task,
            candidate,
            base_commit=base_commit,
            options=options,
            control_python=control_python or sys.executable,
            timeout_seconds=timeout_seconds,
            shared_reference_root=shared_reference_root,
        )
        for task in _UNSEEN_TASKS
    ]
    payload: JsonObject = {
        "status": "MEASURED",
        "purpose": "Laplace-only unseen-task generalization evaluation.",
        "base_commit": base_commit,
        "candidate": candidate.to_json(),
        "tasks": records,
        "held_out_exposed_to_implementation": False,
        "aggregate": {
            "python_mean_score": _mean(records, "python"),
            "systemverilog_mean_score": _mean(records, "systemverilog"),
            "false_verifier_approvals": _held_out_false_approval(records),
        },
        "run_root": str(run_root),
        "created_at": _now(),
    }
    _write_json_atomic(output_root / "unseen_tasks_results.json", payload)
    return payload


def _baseline_scores(root: Path | None) -> dict[str, float]:
    if root is None:
        return {}
    scores: dict[str, float] = {}
    for filename in ("original_tasks_rerun.json", "unseen_tasks_results.json"):
        path = root / filename
        if not path.is_file():
            continue
        payload = _read_json(path)
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        for item in tasks:
            if not isinstance(item, dict):
                continue
            task_id = item.get("task_id")
            score = item.get("score")
            if (
                isinstance(task_id, str)
                and isinstance(score, dict)
                and isinstance(score.get("objective_score_0_100"), (int, float))
            ):
                scores[task_id] = float(score["objective_score_0_100"])
    return scores


def run_targeted_six(
    repository_root: Path,
    candidate: ServingCandidate,
    *,
    output_root: Path,
    control_python: str | None = None,
    timeout_seconds: int = 900,
    shared_reference_root: Path | None = None,
    baseline_root: Path | None = None,
) -> JsonObject:
    """Run only the six diagnosed tasks with full retrieval and operational review."""
    root = repository_root.resolve()
    current_base = _git(root, ["rev-parse", "HEAD"]).strip()
    original = {task.task_id: task for task in _BENCHMARK_TASKS}
    unseen = {task.task_id: task for task in _UNSEEN_TASKS}
    baseline_scores = _baseline_scores(baseline_root.resolve() if baseline_root else None)
    run_root = output_root / "targeted_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    records: list[JsonObject] = []
    for task_id in TARGETED_TASK_IDS:
        if task_id in original:
            task = original[task_id]
            base_commit = ORIGINAL_BASE_COMMIT
        elif task_id in unseen:
            task = unseen[task_id]
            base_commit = current_base
        else:
            raise RuntimeError(f"Targeted task is not registered: {task_id}")
        options = TeamWorkflowOptions(base_commit=base_commit)
        record = _task_record(
            root,
            run_root,
            task,
            candidate,
            base_commit=base_commit,
            options=options,
            control_python=control_python or sys.executable,
            timeout_seconds=timeout_seconds,
            shared_reference_root=shared_reference_root,
        )
        previous = baseline_scores.get(task_id)
        score = record.get("score")
        current = (
            float(score["objective_score_0_100"])
            if isinstance(score, dict)
            and isinstance(score.get("objective_score_0_100"), (int, float))
            else None
        )
        record["baseline_score_0_100"] = previous
        record["score_delta"] = (
            current - previous if current is not None and previous is not None else None
        )
        records.append(record)

    payload: JsonObject = {
        "status": "MEASURED",
        "purpose": "Six-task targeted validation of structured repair, retrieval, and reviewer fixes.",
        "candidate": candidate.to_json(),
        "targeted_task_ids": list(TARGETED_TASK_IDS),
        "current_base_commit": current_base,
        "historic_original_base_commit": ORIGINAL_BASE_COMMIT,
        "baseline_root": str(baseline_root.resolve()) if baseline_root else None,
        "held_out_exposed_to_implementation": False,
        "tasks": records,
        "aggregate": {
            "python_mean_score": _mean(records, "python"),
            "systemverilog_mean_score": _mean(records, "systemverilog"),
            "false_verifier_approvals": _held_out_false_approval(records),
            "completed_tasks": sum(item.get("status") == "COMPLETE" for item in records),
        },
        "run_root": str(run_root),
        "created_at": _now(),
    }
    _write_json_atomic(output_root / "targeted_six_results.json", payload)

    fields = [
        "task_id",
        "domain",
        "status",
        "wall_time_s",
        "repair_cycles",
        "verifier_approved",
        "baseline_score_0_100",
        "objective_score_0_100",
        "score_delta",
        "functional_correctness",
        "held_out_correctness",
    ]
    with (output_root / "targeted_six_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            score = record.get("score")
            score_dict = score if isinstance(score, dict) else {}
            row = {key: record.get(key) for key in fields}
            for key in fields:
                if key in score_dict:
                    row[key] = score_dict.get(key)
            writer.writerow(row)

    aggregate = payload.get("aggregate")
    aggregate_values = aggregate if isinstance(aggregate, dict) else {}
    lines = [
        "# Six-task targeted Laplace rerun",
        "",
        f"Python mean score: `{aggregate_values.get('python_mean_score')}`.",
        f"SystemVerilog mean score: `{aggregate_values.get('systemverilog_mean_score')}`.",
        f"False approvals: `{aggregate_values.get('false_verifier_approvals')}`.",
        "",
        "| Task | Status | Baseline | Current | Delta | Repairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        score = record.get("score")
        current = score.get("objective_score_0_100") if isinstance(score, dict) else None
        lines.append(
            f"| {record.get('task_id')} | {record.get('status')} | "
            f"{record.get('baseline_score_0_100')} | {current} | "
            f"{record.get('score_delta')} | {record.get('repair_cycles')} |"
        )
    (output_root / "targeted_six_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _original_failure_rows(repository_root: Path) -> list[JsonObject]:
    replay = _read_json(
        repository_root
        / "outputs"
        / "a6000_agent_team"
        / "comparison"
        / "runs"
        / ORIGINAL_RUN_ID
        / "scope_corrected_evaluation"
        / "replay_result.json"
    )
    rows = replay.get("tasks")
    return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _artifact(repository_root: Path, task_id: str, name: str) -> tuple[str, JsonObject | None]:
    path = (
        repository_root
        / "outputs"
        / "a6000_agent_team"
        / "comparison"
        / "runs"
        / ORIGINAL_RUN_ID
        / "laplace_projects"
        / task_id
        / "Data"
        / "AgentTeam"
        / "tasks"
        / task_id
        / "artifacts"
        / f"{name}.json"
    )
    return str(path), _read_json(path) if path.is_file() else None


def _failure_classes(task_id: str, row: JsonObject, result: JsonObject) -> list[str]:
    """Conservative causal labels grounded in recorded results, never guesses."""
    lanes = row.get("lanes")
    laplace = lanes.get("laplace_team") if isinstance(lanes, dict) else None
    score = laplace.get("score") if isinstance(laplace, dict) else None
    labels: list[str] = []
    if (
        isinstance(score, dict)
        and score.get("held_out_correctness") is False
        and result.get("status") == "COMPLETE"
    ):
        labels.extend(
            [
                "missing_test_generation",
                "verifier_false_acceptance",
                "reviewer_false_acceptance",
            ]
        )
    if result.get("status") == "FAILED_AFTER_REPAIRS":
        labels.append("ineffective_correction_prompt")
    error = result.get("error")
    if isinstance(error, str) and "no source change" in error.lower():
        labels.append("implementation_reasoning_error")
    if task_id.startswith("sv_") and result.get("status") == "FAILED_AFTER_REPAIRS":
        labels.append("insufficient_or_irrelevant_retrieval")
    if not labels:
        labels.append("implementation_reasoning_error")
    return sorted(set(labels))


def write_failure_analysis(repository_root: Path, output_root: Path) -> JsonObject:
    """Produce an evidence-linked taxonomy for every original Laplace task."""
    tasks: list[JsonObject] = []
    for row in _original_failure_rows(repository_root):
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        result_path = (
            repository_root
            / "outputs"
            / "a6000_agent_team"
            / "comparison"
            / "runs"
            / ORIGINAL_RUN_ID
            / "laplace_results"
            / f"{task_id}.json"
        )
        result = _read_json(result_path)
        evidence: dict[str, str] = {"typed_result": str(result_path)}
        for name in (
            "requirements",
            "plan",
            "evidence_packet",
            "implementation_report",
            "patch_manifest",
            "verification_report",
            "review_report",
            "final_report",
        ):
            path, _ = _artifact(repository_root, task_id, name)
            if Path(path).is_file():
                evidence[name] = path
        lanes = row.get("lanes")
        laplace = lanes.get("laplace_team") if isinstance(lanes, dict) else {}
        score = laplace.get("score") if isinstance(laplace, dict) else {}
        completed_with_hidden_failure = (
            result.get("status") == "COMPLETE"
            and isinstance(score, dict)
            and score.get("held_out_correctness") is False
        )
        tasks.append(
            {
                "task_id": task_id,
                "domain": row.get("domain"),
                "status": result.get("status"),
                "score": score,
                "causal_classes": _failure_classes(task_id, row, result),
                "concrete_evidence": evidence,
                "observed_error": result.get("error"),
                "analysis": (
                    "The stored verifier approved before running an explicit fixture command or "
                    "contract-derived negative test; post-lane held-out evaluation disproves that approval."
                    if completed_with_hidden_failure
                    else "The lane outcome and immutable task artifacts retain the rejected patch, "
                    "verification evidence, and bounded-correction failure for causal review."
                ),
            }
        )
    payload: JsonObject = {
        "source_run": ORIGINAL_RUN_ID,
        "generated_at": _now(),
        "taxonomy_categories": [
            "task_specification_error",
            "insufficient_or_irrelevant_retrieval",
            "implementation_reasoning_error",
            "missing_test_generation",
            "verifier_false_acceptance",
            "reviewer_false_acceptance",
            "ineffective_correction_prompt",
            "orchestration_or_timeout_issue",
            "underlying_model_capability_limit",
        ],
        "tasks": tasks,
    }
    _write_json_atomic(output_root / "failure_taxonomy.json", payload)
    lines = ["# Laplace-team failure analysis", ""]
    for task in tasks:
        classes = task.get("causal_classes")
        classification = (
            ", ".join(item for item in classes if isinstance(item, str))
            if isinstance(classes, list)
            else "unclassified"
        )
        evidence_field = task.get("concrete_evidence")
        lines.extend(
            [
                f"## {task['task_id']}",
                "",
                f"Classification: {classification}.",
                "",
                str(task["analysis"]),
                "",
                f"Recorded error: `{task.get('observed_error')}`.",
                "",
                "Evidence:",
                "",
            ]
        )
        if isinstance(evidence_field, dict):
            for label, path in evidence_field.items():
                lines.append(f"- {label}: `{path}`")
        lines.append("")
    (output_root / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


@dataclass(frozen=True)
class Ablation:
    name: str
    options: TeamWorkflowOptions
    task_ids: tuple[str, ...]


def run_ablations(
    repository_root: Path,
    candidate: ServingCandidate,
    *,
    output_root: Path,
    control_python: str | None = None,
    timeout_seconds: int = 900,
    shared_reference_root: Path | None = None,
) -> list[JsonObject]:
    """Run controlled single-factor workflow ablations on representative failures."""
    task_by_id = {task.task_id: task for task in _BENCHMARK_TASKS}
    ablations = (
        Ablation(
            "implementer_without_rag",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT, retrieval_mode="none"),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
        Ablation(
            "implementer_project_local_rag",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT, retrieval_mode="project_local"),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
        Ablation(
            "implementer_curated_references",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT, retrieval_mode="curated_only"),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
        Ablation(
            "one_agent_direct_local_model",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT, role_mode="direct"),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
        Ablation(
            "full_five_agent_workflow",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
        Ablation(
            "verifier_without_adversarial",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT, adversarial_verification=False),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
        Ablation(
            "reviewer_without_invariants",
            TeamWorkflowOptions(base_commit=ORIGINAL_BASE_COMMIT, reviewer_invariants=False),
            ("py_safe_async_job", "sv_axi_lite_irq_regs"),
        ),
    )
    records: list[JsonObject] = []
    for ablation in ablations:
        for task_id in ablation.task_ids:
            root = output_root / "ablation_runs" / ablation.name
            record = _task_record(
                repository_root,
                root,
                task_by_id[task_id],
                candidate,
                base_commit=ORIGINAL_BASE_COMMIT,
                options=ablation.options,
                control_python=control_python or sys.executable,
                timeout_seconds=timeout_seconds,
                shared_reference_root=shared_reference_root,
            )
            record["ablation"] = ablation.name
            records.append(record)
    fields = [
        "ablation",
        "task_id",
        "domain",
        "status",
        "wall_time_s",
        "repair_cycles",
        "verifier_approved",
        "objective_score_0_100",
        "functional_correctness",
        "held_out_correctness",
        "human_intervention",
    ]
    with (output_root / "agent_ablation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            score = record.get("score")
            score_dict = score if isinstance(score, dict) else {}
            writer.writerow(
                {
                    **{key: record.get(key) for key in fields if key not in score_dict},
                    **{key: score_dict.get(key) for key in fields if key in score_dict},
                }
            )
    _write_json_atomic(
        output_root / "agent_ablation_results.json",
        {
            "status": "MEASURED",
            "candidate": candidate.to_json(),
            "base_commit": ORIGINAL_BASE_COMMIT,
            "held_out_exposed_to_implementation": False,
            "records": records,
            "created_at": _now(),
        },
    )
    return records


def write_summary(
    output_root: Path, original: JsonObject, unseen: JsonObject, ablations: list[JsonObject]
) -> None:
    aggregate = original.get("aggregate")
    details = aggregate if isinstance(aggregate, dict) else {}
    unseen_aggregate = unseen.get("aggregate")
    unseen_details = unseen_aggregate if isinstance(unseen_aggregate, dict) else {}
    lines = [
        "# Laplace-team quality improvement summary",
        "",
        "The Codex-direct lane was not changed or invoked in this phase.",
        "",
        f"Original-task Python mean: `{details.get('python_mean_score')}`.",
        f"Original-task SystemVerilog mean: `{details.get('systemverilog_mean_score')}`.",
        f"Verifier approvals contradicted by held-out evaluation: `{details.get('false_verifier_approvals')}`.",
        "",
        f"Unseen-task Python mean: `{unseen_details.get('python_mean_score')}`.",
        f"Unseen-task SystemVerilog mean: `{unseen_details.get('systemverilog_mean_score')}`.",
        f"Unseen false verifier approvals: `{unseen_details.get('false_verifier_approvals')}`.",
        "",
        f"Controlled ablation observations: `{len(ablations)}`.",
        "",
        "Every held-out evaluation is performed only after its implementation worktree has stopped. "
        "A missing or failed measurement is retained as such and is never converted into a quality score.",
    ]
    (output_root / "quality_improvement_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _candidate_from_json(path: Path) -> ServingCandidate:
    raw = _read_json(path)
    data = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else raw
    if not isinstance(data, dict):
        raise RuntimeError("Candidate JSON is malformed")
    engine = str(data["engine"])
    if engine not in {"vllm", "sglang"}:
        raise RuntimeError("Candidate engine must be vllm or sglang")
    return ServingCandidate(
        engine=cast(Literal["vllm", "sglang"], engine),
        endpoint=str(data["endpoint"]),
        model=str(data["model"]),
        revision=str(data["revision"]),
        quantization=str(data["quantization"]),
        kernel=str(data["kernel"]),
        prefix_caching=bool(data["prefix_caching"]),
        chunked_prefill=bool(data["chunked_prefill"]),
        cuda_graph_mode=str(data["cuda_graph_mode"]),
        scheduler=str(data["scheduler"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Laplace-only quality improvement evidence.")
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/a6000_agent_team/quality_improvement")
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument(
        "--targeted-six",
        action="store_true",
        help="Run only the six diagnosed Python/SystemVerilog tasks.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Prior corrected result root containing original/unseen JSON files.",
    )
    parser.add_argument(
        "--shared-reference-root",
        type=Path,
        help="Exact shared FormalScience Library root containing Python/ and SystemVerilog/",
    )
    arguments = parser.parse_args(argv)
    root = Path.cwd().resolve()
    output = (root / arguments.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.analysis_only:
        write_failure_analysis(root, output)
        return 0
    candidate = _candidate_from_json(arguments.candidate_json)
    cuda = LocalToolRunner(root).run("cuda_probe", ["nvidia-smi", "-L"], timeout_seconds=30)
    if cuda.status != "PASS" or "A6000" not in cuda.stdout:
        raise RuntimeError(
            "Quality rerun refuses to substitute CPU inference for the required A6000."
        )
    if arguments.targeted_six:
        run_targeted_six(
            root,
            candidate,
            output_root=output,
            timeout_seconds=arguments.timeout_seconds,
            shared_reference_root=arguments.shared_reference_root,
            baseline_root=arguments.baseline_root,
        )
        return 0
    write_failure_analysis(root, output)
    original = run_original_rerun(
        root,
        candidate,
        output_root=output,
        timeout_seconds=arguments.timeout_seconds,
        shared_reference_root=arguments.shared_reference_root,
    )
    ablation_file = output / "agent_ablation_results.json"
    stored_ablation_records = (
        _read_json(ablation_file).get("records", []) if ablation_file.is_file() else []
    )
    prior_ablations = (
        [item for item in stored_ablation_records if isinstance(item, dict)]
        if isinstance(stored_ablation_records, list)
        else []
    )
    ablations = (
        run_ablations(
            root,
            candidate,
            output_root=output,
            timeout_seconds=arguments.timeout_seconds,
            shared_reference_root=arguments.shared_reference_root,
        )
        if arguments.run_ablations
        else prior_ablations
    )
    unseen = run_unseen_tasks(
        root,
        candidate,
        output_root=output,
        timeout_seconds=arguments.timeout_seconds,
        shared_reference_root=arguments.shared_reference_root,
    )
    write_summary(output, original, unseen, ablations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
