#!/usr/bin/env python3
"""Run the expanded, stratified Step-3.3 RepoMap screening experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import TypeAlias, cast

from research_workspace.repository_context import RepositoryContextService
from research_workspace.upstream_ab import (
    AIDER_REPOMAP_REVISION,
    V33_BASELINE_REVISION,
    manifest_sha256,
    materialize_head_snapshot,
    require_v33_baseline,
    runtime_executable_path,
    runtime_output_path,
)
from research_workspace.upstream_ab_expanded import (
    EXPANDED_BUDGETS,
    EXPANDED_REPETITIONS,
    ExpandedPhaseATask,
    assess_expanded_phase_a,
    expanded_quality,
    load_expanded_tasks,
    timing_summary,
    validate_candidate_repeat_payload,
    validate_expanded_task_paths,
)

JsonObject: TypeAlias = dict[str, object]


def _candidate_environment(
    python: Path,
    *,
    expected_prefix: Path,
) -> JsonObject:
    code = (
        "import importlib.metadata,json,sys;"
        "import aider.repomap,grep_ast;"
        "print(json.dumps({"
        "'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,"
        "'aider_module':aider.repomap.__file__,"
        "'grep_ast_module':grep_ast.__file__,"
        "'grep_ast_version':importlib.metadata.version('grep-ast')"
        "},sort_keys=True))"
    )
    result = subprocess.run(
        [os.fspath(python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:800]
        raise RuntimeError(f"v33_expanded_candidate_environment_failed:{detail}")
    try:
        raw: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v33_expanded_candidate_environment_json_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v33_expanded_candidate_environment_invalid")
    environment = cast(JsonObject, raw)
    prefix = environment.get("prefix")
    base_prefix = environment.get("base_prefix")
    if not isinstance(prefix, str) or not isinstance(base_prefix, str):
        raise TypeError("v33_expanded_candidate_prefix_invalid")
    if Path(prefix).resolve(strict=True) != expected_prefix.resolve(strict=True):
        raise RuntimeError("v33_expanded_candidate_prefix_mismatch")
    if Path(prefix).resolve(strict=True) == Path(base_prefix).resolve(strict=True):
        raise RuntimeError("v33_expanded_candidate_not_virtualenv")
    if environment.get("grep_ast_version") != "0.9.0":
        raise RuntimeError("v33_expanded_grep_ast_version_mismatch")
    return environment


def _candidate_repeated(
    *,
    python: Path,
    probe: Path,
    checkout: Path,
    snapshot: Path,
    task: ExpandedPhaseATask,
    budget: int,
) -> tuple[JsonObject, list[float], float]:
    argv = [
        os.fspath(python),
        os.fspath(probe),
        "--repo",
        os.fspath(snapshot),
        "--aider-checkout",
        os.fspath(checkout),
        "--query",
        task.query,
        "--token-budget",
        str(budget),
        "--repeat-count",
        str(EXPANDED_REPETITIONS),
    ]
    for focus in task.focus_paths:
        argv.extend(["--focus", focus])

    process_started = time.perf_counter()
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    process_elapsed = time.perf_counter() - process_started
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:800]
        raise RuntimeError(f"v33_expanded_aider_probe_failed:{detail}")
    try:
        raw: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v33_expanded_aider_probe_json_invalid") from exc

    repeated = validate_candidate_repeat_payload(raw)
    return cast(JsonObject, raw), list(repeated.wall_times), process_elapsed

def _baseline_repeated(
    *,
    snapshot: Path,
    task: ExpandedPhaseATask,
    budget: int,
) -> tuple[str, list[float]]:
    service = RepositoryContextService(snapshot)
    # Warm-up is deliberately excluded from timings.
    warm = service.build_repo_map(
        query=task.query,
        focus_paths=task.focus_paths,
        token_budget=budget,
    ).text
    timings: list[float] = []
    for _ in range(EXPANDED_REPETITIONS):
        started = time.perf_counter()
        text = service.build_repo_map(
            query=task.query,
            focus_paths=task.focus_paths,
            token_budget=budget,
        ).text
        timings.append(time.perf_counter() - started)
        if text != warm:
            raise RuntimeError("v33_expanded_baseline_map_nondeterministic")
    return warm, timings



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--aider-python", type=Path, required=True)
    parser.add_argument("--aider-checkout", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("benchmarks/v33_aider_repomap/tasks_expanded.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    require_v33_baseline(repo)
    tasks_path = args.tasks if args.tasks.is_absolute() else repo / args.tasks
    tasks = load_expanded_tasks(tasks_path)
    manifest_digest = manifest_sha256(tasks_path)

    runtime = repo / ".runtime" / "v33-aider"
    venv = runtime / "aider-venv"
    aider_python = runtime_executable_path(
        repo,
        args.aider_python,
        runtime_root=venv,
    )
    checkout = args.aider_checkout.resolve(strict=True)
    candidate_environment = _candidate_environment(
        aider_python,
        expected_prefix=venv,
    )
    aider_module = candidate_environment.get("aider_module")
    if not isinstance(aider_module, str):
        raise TypeError("v33_expanded_aider_module_invalid")
    try:
        Path(aider_module).resolve(strict=True).relative_to(checkout)
    except ValueError as exc:
        raise RuntimeError("v33_expanded_aider_module_outside_pin") from exc

    snapshots = runtime / "snapshots-expanded"
    baseline_snapshot = snapshots / "laplace"
    candidate_snapshot = snapshots / "aider"
    baseline_revision, baseline_snapshot_sha = materialize_head_snapshot(
        repo,
        baseline_snapshot,
    )
    candidate_revision, candidate_snapshot_sha = materialize_head_snapshot(
        repo,
        candidate_snapshot,
    )
    if (
        baseline_revision != candidate_revision
        or baseline_snapshot_sha != candidate_snapshot_sha
    ):
        raise RuntimeError("v33_expanded_provider_snapshots_not_identical")

    all_repo_paths = validate_expanded_task_paths(baseline_snapshot, tasks)
    candidate_paths = validate_expanded_task_paths(candidate_snapshot, tasks)
    if all_repo_paths != candidate_paths:
        raise RuntimeError("v33_expanded_snapshot_paths_not_identical")

    probe = Path(__file__).resolve().with_name("v3_3_aider_repomap_probe.py")
    trials: list[JsonObject] = []
    for task in tasks:
        for budget in EXPANDED_BUDGETS:
            baseline_text, baseline_times = _baseline_repeated(
                snapshot=baseline_snapshot,
                task=task,
                budget=budget,
            )
            candidate_raw, candidate_times, candidate_process_elapsed = (
                _candidate_repeated(
                    python=aider_python,
                    probe=probe,
                    checkout=checkout,
                    snapshot=candidate_snapshot,
                    task=task,
                    budget=budget,
                )
            )
            if candidate_raw.get("upstream_revision") != AIDER_REPOMAP_REVISION:
                raise RuntimeError("v33_expanded_aider_revision_mismatch")
            if candidate_raw.get("grep_ast_version") != "0.9.0":
                raise RuntimeError("v33_expanded_candidate_grep_ast_mismatch")
            candidate_text = candidate_raw.get("text")
            if not isinstance(candidate_text, str):
                raise TypeError("v33_expanded_candidate_text_invalid")

            baseline_metrics = {
                **expanded_quality(baseline_text, task, all_repo_paths),
                "timing": timing_summary(baseline_times),
            }
            candidate_metrics = {
                **expanded_quality(candidate_text, task, all_repo_paths),
                "timing": timing_summary(candidate_times),
                "candidate_batch_process_seconds": candidate_process_elapsed,
                "within_process_stable": candidate_raw["within_process_stable"],
                "unique_map_hashes": candidate_raw["unique_map_hashes"],
            }
            trials.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "focus_mode": task.focus_mode,
                    "query": task.query,
                    "focus_paths": list(task.focus_paths),
                    "expected_paths": list(task.expected_paths),
                    "token_budget": budget,
                    "snapshot_sha256": baseline_snapshot_sha,
                    "snapshot_match": True,
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                }
            )

    assessment = assess_expanded_phase_a(trials, margin=args.margin)
    output = runtime_output_path(
        repo,
        args.output,
        filename="phase-a-expanded-repomap.json",
    )
    payload: JsonObject = {
        "schema": "laplace-v33-aider-repomap-expanded-results-v1",
        "phase": "A_EXPANDED_MAP_ONLY",
        "laplace_revision": V33_BASELINE_REVISION,
        "aider_revision": AIDER_REPOMAP_REVISION,
        "task_manifest_sha256": manifest_digest,
        "snapshot_sha256": baseline_snapshot_sha,
        "candidate_environment": candidate_environment,
        "token_metric": "deterministic_char4_for_both_providers",
        "timing_method": (
            "one warm-up then five measured map calls per provider; Aider "
            "repetitions execute inside one isolated candidate process; map "
            "time excludes process startup and the batch process time is "
            "recorded separately"
        ),
        "prior_screening_result": ".runtime/v33-aider/results/phase-a-repomap.json",
        "trials": trials,
        "assessment": assessment,
        "final_adoption_decision": None,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": "A_EXPANDED_MAP_ONLY",
                "assessment": assessment["assessment"],
                "tasks": len(tasks),
                "budgets": list(EXPANDED_BUDGETS),
                "samples": len(trials),
                "timed_repetitions": EXPANDED_REPETITIONS,
                "advantage_strata": assessment["advantage_strata"],
                "output": str(output),
                "laplace_revision": V33_BASELINE_REVISION,
                "aider_revision": AIDER_REPOMAP_REVISION,
                "snapshot_sha256": baseline_snapshot_sha,
                "candidate_python_prefix": candidate_environment["prefix"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
