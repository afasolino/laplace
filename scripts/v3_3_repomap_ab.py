#!/usr/bin/env python3
"""Run Step 3.3 Phase-A Laplace/Aider RepoMap comparison."""

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
    PhaseATask,
    assess_phase_a,
    load_phase_a_tasks,
    manifest_sha256,
    map_quality,
    materialize_head_snapshot,
    require_v33_baseline,
    runtime_executable_path,
    runtime_output_path,
)

JsonObject: TypeAlias = dict[str, object]


def _candidate_python_environment(
    python: Path,
    *,
    expected_prefix: Path,
) -> JsonObject:
    code = (
        "import json,sys;"
        "print(json.dumps({"
        "'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,"
        "'executable':sys.executable"
        "},sort_keys=True))"
    )
    result = subprocess.run(
        [os.fspath(python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:500]
        raise RuntimeError(f"v33_aider_python_probe_failed:{detail}")
    try:
        raw: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v33_aider_python_probe_json_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v33_aider_python_probe_result_invalid")
    environment = cast(JsonObject, raw)
    prefix_raw = environment.get("prefix")
    base_prefix_raw = environment.get("base_prefix")
    if not isinstance(prefix_raw, str) or not isinstance(base_prefix_raw, str):
        raise TypeError("v33_aider_python_prefix_invalid")
    prefix = Path(prefix_raw).resolve(strict=True)
    base_prefix = Path(base_prefix_raw).resolve(strict=True)
    if prefix != expected_prefix.resolve(strict=True) or prefix == base_prefix:
        raise RuntimeError("v33_aider_python_not_expected_virtualenv")
    return environment


def _candidate(
    *,
    python: Path,
    probe: Path,
    checkout: Path,
    snapshot: Path,
    task: PhaseATask,
) -> JsonObject:
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
        str(task.token_budget),
    ]
    for focus in task.focus_paths:
        argv.extend(["--focus", focus])
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:500]
        raise RuntimeError(f"v33_aider_probe_failed:{detail}")
    try:
        raw: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v33_aider_probe_json_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v33_aider_probe_result_invalid")
    return cast(JsonObject, raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--aider-python", type=Path, required=True)
    parser.add_argument("--aider-checkout", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("benchmarks/v33_aider_repomap/tasks.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    require_v33_baseline(repo)
    tasks_path = args.tasks if args.tasks.is_absolute() else repo / args.tasks
    tasks = load_phase_a_tasks(tasks_path)
    manifest_digest = manifest_sha256(tasks_path)

    runtime = repo / ".runtime" / "v33-aider"
    aider_venv = runtime / "aider-venv"
    aider_python = runtime_executable_path(
        repo,
        args.aider_python,
        runtime_root=aider_venv,
    )
    candidate_environment = _candidate_python_environment(
        aider_python,
        expected_prefix=aider_venv,
    )
    snapshots = runtime / "snapshots"
    laplace_snapshot = snapshots / "laplace"
    aider_snapshot = snapshots / "aider"
    laplace_revision, laplace_snapshot_sha = materialize_head_snapshot(
        repo, laplace_snapshot
    )
    aider_revision, aider_snapshot_sha = materialize_head_snapshot(repo, aider_snapshot)
    if (
        laplace_revision != aider_revision
        or laplace_snapshot_sha != aider_snapshot_sha
    ):
        raise RuntimeError("v33_provider_snapshots_not_identical")

    service = RepositoryContextService(laplace_snapshot)
    baseline_outputs: dict[str, JsonObject] = {}
    for task in tasks:
        started = time.perf_counter()
        repo_map = service.build_repo_map(
            query=task.query,
            focus_paths=task.focus_paths,
            token_budget=task.token_budget,
        )
        elapsed = time.perf_counter() - started
        baseline_outputs[task.task_id] = {
            "provider": "laplace",
            "parser_contract": repo_map.to_json()["parser_contract"],
            "wall_time_seconds": elapsed,
            "text": repo_map.text,
            **map_quality(repo_map.text, task),
        }

    probe = Path(__file__).resolve().with_name("v3_3_aider_repomap_probe.py")
    trials: list[JsonObject] = []
    for task in tasks:
        candidate_raw = _candidate(
            python=aider_python,
            probe=probe,
            checkout=args.aider_checkout.resolve(strict=True),
            snapshot=aider_snapshot,
            task=task,
        )
        if candidate_raw.get("upstream_revision") != AIDER_REPOMAP_REVISION:
            raise RuntimeError("v33_aider_probe_revision_mismatch")
        candidate_text = candidate_raw.get("text")
        if not isinstance(candidate_text, str):
            raise TypeError("v33_aider_probe_text_invalid")
        candidate = {
            **candidate_raw,
            **map_quality(candidate_text, task),
        }
        trial: JsonObject = {
            "task_id": task.task_id,
            "query": task.query,
            "focus_paths": list(task.focus_paths),
            "token_budget": task.token_budget,
            "snapshot_sha256": laplace_snapshot_sha,
            "snapshot_match": True,
            "baseline": baseline_outputs[task.task_id],
            "candidate": candidate,
        }
        trials.append(trial)

    assessment = assess_phase_a(trials, margin=args.margin)
    payload: JsonObject = {
        "schema": "laplace-v33-aider-repomap-phase-a-v2",
        "phase": "A_MAP_ONLY",
        "laplace_revision": V33_BASELINE_REVISION,
        "aider_revision": AIDER_REPOMAP_REVISION,
        "task_manifest_sha256": manifest_digest,
        "snapshot_sha256": laplace_snapshot_sha,
        "candidate_environment": candidate_environment,
        "token_metric": "deterministic_char4_for_both_providers",
        "trials": trials,
        "assessment": assessment,
        "final_adoption_decision": None,
        "note": (
            "Phase A cannot adopt Aider. PROMISING only authorizes a separately "
            "guarded paired agent-level Phase B."
        ),
    }
    output = runtime_output_path(
        repo, args.output, filename="phase-a-repomap.json"
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "phase": "A_MAP_ONLY",
        "output": str(output),
        "assessment": assessment["assessment"],
        "tasks": len(tasks),
        "laplace_revision": V33_BASELINE_REVISION,
        "aider_revision": AIDER_REPOMAP_REVISION,
        "snapshot_sha256": laplace_snapshot_sha,
        "candidate_python_prefix": candidate_environment["prefix"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
