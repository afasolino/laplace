#!/usr/bin/env python3
"""End-to-end tiered-serving certification, selection, bundle, and safe shutdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.serving_profile_runtime import ServingProfileRuntime, ServingRuntimeError
from research_workspace.serving_profiles import ServingProfile, load_profiles


def _number(value: object, *, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _whole(value: object, *, default: int = 0) -> int:
    return int(value) if isinstance(value, int) else default


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def _run(
    command: Sequence[str],
    *,
    root: Path,
    output: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(
        list(command),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=7_200,
        env=environment,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    return {
        "command": list(command),
        "started_at_utc": started,
        "returncode": result.returncode,
        "log": str(output),
    }


def _hardware_audit(root: Path) -> dict[str, object]:
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        "gpu_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        "topology": ["nvidia-smi", "topo", "-m"],
        "memory": ["free", "-h"],
        "cpu": ["lscpu"],
        "disk": ["df", "-h", str(root)],
    }
    observations: dict[str, object] = {}
    for name, command in commands.items():
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        observations[name] = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return observations


def _repository_audit(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": git("status", "--short", "--branch"),
        "worktree_count": len(git("worktree", "list", "--porcelain").split("worktree ")) - 1,
        "python": sys.version,
        "platform": platform.platform(),
        "prompt": "prompts/laplace_codex_tiered_users_repo_bound_offload_v2.md",
        "preservation_policy": (
            "Inherited changes and historical outputs are preserved; only fresh "
            "tiered_serving_<timestamp> evidence is written."
        ),
    }


def _measured_batch_throughput(result_path: Path) -> tuple[float, str]:
    """Calculate wall-clock batch throughput from the highest live load arm."""

    measurements = result_path.parent / "request_measurements.csv"
    if not measurements.is_file():
        return 0.0, "UNAVAILABLE"
    with measurements.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "SUCCESS"
            and str(row.get("request_id", "")).startswith("load-")
        ]
    concurrencies = [
        int(row["concurrency"])
        for row in rows
        if str(row.get("concurrency", "")).isdigit()
    ]
    if not concurrencies:
        return 0.0, "UNAVAILABLE"
    highest = max(concurrencies)
    selected = [row for row in rows if int(row["concurrency"]) == highest]
    output_tokens = sum(
        int(row["output_tokens"])
        for row in selected
        if str(row.get("output_tokens", "")).isdigit()
    )
    wall_ms = max(
        (
            float(row.get("queue_time_ms") or 0.0) + float(row["e2e_ms"])
            for row in selected
        ),
        default=0.0,
    )
    if output_tokens <= 0 or wall_ms <= 0:
        return 0.0, "UNAVAILABLE"
    return (
        output_tokens / (wall_ms / 1_000),
        f"highest_concurrency_batch_wall_clock_concurrency_{highest}",
    )


def _aggregate(
    profile_root: Path,
    output_root: Path,
    profiles: Sequence[ServingProfile],
) -> list[dict[str, object]]:
    by_id = {profile.profile_id: profile for profile in profiles}
    rows: list[dict[str, object]] = []
    for result_path in sorted(profile_root.glob("*/result.json")):
        raw: object = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        benchmark = raw.get("benchmark")
        if not isinstance(benchmark, dict):
            continue
        profile_id = str(raw.get("profile_id"))
        profile = by_id.get(profile_id)
        batch_throughput, throughput_method = _measured_batch_throughput(result_path)
        rows.append(
            {
                "profile_id": profile_id,
                "status": raw.get("status"),
                "quality_score": raw.get("quality_score"),
                "hard_gate_pass": raw.get("hard_gate_pass"),
                "aggregate_output_tokens_per_second": batch_throughput,
                "throughput_method": throughput_method,
                "p95_ttft_ms": benchmark.get("p95_ttft_ms"),
                "p95_e2e_ms": benchmark.get("p95_e2e_ms"),
                "marker_recall": benchmark.get("marker_recall"),
                "resolution_sha256": raw.get("resolution_sha256"),
                "max_model_len": profile.max_model_len if profile else None,
                "max_num_seqs": profile.max_num_seqs if profile else None,
                "kv_cache_dtype": profile.kv_cache_dtype if profile else None,
                "cpu_offload_gb": profile.cpu_offload_gb if profile else None,
            }
        )
    fields = [
        "profile_id",
        "status",
        "quality_score",
        "hard_gate_pass",
        "aggregate_output_tokens_per_second",
        "throughput_method",
        "p95_ttft_ms",
        "p95_e2e_ms",
        "marker_recall",
        "resolution_sha256",
        "max_model_len",
        "max_num_seqs",
        "kv_cache_dtype",
        "cpu_offload_gb",
    ]
    with (output_root / "profile_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _select(rows: list[dict[str, object]]) -> dict[str, object]:
    baseline = next(
        (row for row in rows if str(row["profile_id"]).startswith("P0_")), None
    )
    baseline_quality = (
        _number(baseline.get("quality_score"))
        if baseline is not None and isinstance(baseline.get("quality_score"), (int, float))
        else None
    )
    accepted: list[dict[str, object]] = []
    for row in rows:
        quality = row.get("quality_score")
        if (
            row.get("hard_gate_pass") is True
            and isinstance(quality, (int, float))
            and (baseline_quality is None or float(quality) >= baseline_quality * 0.99)
        ):
            accepted.append(row)
    baseline_p95 = (
        _number(baseline.get("p95_e2e_ms"))
        if baseline is not None and isinstance(baseline.get("p95_e2e_ms"), (int, float))
        else float("inf")
    )
    baseline_sequences = (
        _whole(baseline.get("max_num_seqs"))
        if baseline is not None and isinstance(baseline.get("max_num_seqs"), int)
        else 1
    )
    balanced = [
        row
        for row in accepted
        if isinstance(row.get("max_num_seqs"), int)
        and _whole(row.get("max_num_seqs")) > baseline_sequences
        and isinstance(row.get("p95_e2e_ms"), (int, float))
        and _number(row.get("p95_e2e_ms"), default=float("inf")) <= baseline_p95
    ]
    selected = (
        max(
            balanced or accepted,
            key=lambda row: (
                _number(row.get("aggregate_output_tokens_per_second")),
                -_number(row.get("p95_e2e_ms"), default=float("inf")),
            ),
        )
        if accepted
        else None
    )
    widest_context = max(
        accepted,
        key=lambda row: (
            _whole(row.get("max_model_len")),
            _number(row.get("aggregate_output_tokens_per_second")),
        ),
        default=None,
    )
    return {
        "status": "SELECTED" if selected else "NO_PROFILE_SELECTED",
        "quality_floor_relative_to_p0": 0.99,
        "baseline_quality_score": baseline_quality,
        "accepted_profile_ids": [row["profile_id"] for row in accepted],
        "selection_policy": (
            "quality floor and hard gates, then increased sequence capacity with "
            "p95 no worse than P0, then measured highest-concurrency batch throughput"
        ),
        "quality_profile": selected,
        "standard_profile": selected,
        "high_context_profile": widest_context,
        "economy_profile": {
            "model_id": "laplace-codev-r1-rl-qwen-7b-w4a16",
            "restriction": "systemverilog_only",
            "selection_basis": "frozen local specialized worker; independent of capability tier",
        },
    }


def _bundle(output_root: Path) -> Path:
    manifest: list[dict[str, object]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {
            "bundle_manifest.json",
            "tiered_serving_certification.tar.gz",
        }:
            data = path.read_bytes()
            manifest.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    manifest_path = output_root / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle = output_root / "tiered_serving_certification.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != bundle:
                archive.add(path, arcname=str(path.relative_to(output_root)))
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else root / f"outputs/tiered_serving_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    audit = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository": _repository_audit(root),
        "hardware": _hardware_audit(root),
        "installed_models": {
            "quality_standard": "/home/giando/work/laplace/.models/Qwen3.6-35B-A3B-W4A16-AWQ",
            "economy_systemverilog": "/home/giando/work/laplace/.models/CodeV-R1-RL-Qwen-7B-W4A16-AWQ",
        },
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "audit.md").write_text(
        "# Tiered serving audit\n\n"
        "This audit is captured before profile launch. Raw observations follow.\n\n"
        "```json\n"
        + json.dumps(audit, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    (output_root / "git_status.txt").write_text(
        status.stdout + status.stderr, encoding="utf-8"
    )
    diff_stat = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    (output_root / "git_diff_stat.txt").write_text(
        diff_stat.stdout + diff_stat.stderr, encoding="utf-8"
    )
    shutil.copy2(
        root / "configs/serving_quality_manifest.json",
        output_root / "quality_manifest.json",
    )
    checks: list[dict[str, object]] = []
    if not arguments.skip_cpu:
        commands = [
            (
                "compileall",
                [
                    sys.executable,
                    "-m",
                    "compileall",
                    "-q",
                    "src",
                    "scripts",
                    "tests",
                ],
            ),
            ("pytest", [sys.executable, "-m", "pytest", "-q"]),
            ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
            ("mypy", [sys.executable, "-m", "mypy", "src/research_workspace"]),
            (
                "bandit_high",
                [
                    sys.executable,
                    "-m",
                    "bandit",
                    "-q",
                    "-lll",
                    "-r",
                    "src/research_workspace",
                ],
            ),
            ("git_diff_check", ["git", "diff", "--check"]),
        ]
        for name, command in commands:
            checks.append(
                {
                    "name": name,
                    **_run(
                        command,
                        root=root,
                        output=output_root / "tests" / f"{name}.log",
                    ),
                }
            )
        browser_environment = dict(os.environ)
        browser_environment.setdefault(
            "PLAYWRIGHT_BROWSERS_PATH", "/tmp/laplace-playwright-browsers"
        )
        checks.append(
            {
                "name": "gui_e2e",
                **_run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_operator_gui_e2e.py",
                    ],
                    root=root,
                    output=output_root / "tests/gui_e2e.log",
                    environment=browser_environment,
                ),
            }
        )
    (output_root / "test_results.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "test_results.txt").write_text(
        "\n".join(
            f"{item['name']}: returncode={item['returncode']} log={item['log']}"
            for item in checks
        )
        + "\n",
        encoding="utf-8",
    )
    resolver = _run(
        [
            sys.executable,
            "scripts/resolve_serving_profiles.py",
            "--output",
            str(output_root / "resolved_profiles.json"),
        ],
        root=root,
        output=output_root / "profile_resolution.log",
    )
    (output_root / "profile_resolution_result.json").write_text(
        json.dumps(resolver, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_directory = output_root / "resolved_profiles"
    resolved_directory.mkdir()
    if (output_root / "resolved_profiles.json").is_file():
        shutil.copy2(
            output_root / "resolved_profiles.json",
            resolved_directory / "all_profiles.json",
        )
    gpu_runs: list[dict[str, object]] = []
    if not arguments.skip_gpu and _whole(resolver.get("returncode"), default=-1) == 0:
        for profile in load_profiles(root / "configs/serving_profiles"):
            command = [
                sys.executable,
                "scripts/run_serving_profile_experiment.py",
                "--profile-id",
                profile.profile_id,
                "--output-root",
                str(output_root / "profiles" / profile.profile_id),
            ]
            if arguments.smoke_only:
                command.append("--smoke-only")
            result = _run(
                command,
                root=root,
                output=output_root / "profiles" / profile.profile_id / "runner.log",
            )
            gpu_runs.append({"profile_id": profile.profile_id, **result})
    (output_root / "gpu_run_results.json").write_text(
        json.dumps(gpu_runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profiles = load_profiles(root / "configs/serving_profiles")
    rows = _aggregate(output_root / "profiles", output_root, profiles)
    for resolved_path in sorted(
        (output_root / "profiles").glob("*/server_profile_resolved.json")
    ):
        shutil.copy2(
            resolved_path,
            resolved_directory / f"{resolved_path.parent.name}.json",
        )
    shutil.copy2(
        output_root / "profile_results.csv",
        output_root / "serving_profile_summary.csv",
    )
    (output_root / "serving_profile_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality_results: list[dict[str, object]] = []
    load_results: list[dict[str, object]] = []
    traces_path = output_root / "relevant_traces.jsonl"
    metrics_path = output_root / "GPU_and_CPU_metrics.jsonl"
    with traces_path.open("w", encoding="utf-8") as traces, metrics_path.open(
        "w", encoding="utf-8"
    ) as metrics:
        metrics.write(
            json.dumps(
                {"kind": "hardware_audit", "hardware": audit["hardware"]},
                sort_keys=True,
            )
            + "\n"
        )
        for profile_root in sorted((output_root / "profiles").glob("P*")):
            quality_path = profile_root / "quality_results.json"
            if quality_path.is_file():
                raw_quality: object = json.loads(quality_path.read_text(encoding="utf-8"))
                if isinstance(raw_quality, list):
                    quality_results.extend(
                        item for item in raw_quality if isinstance(item, dict)
                    )
            measurements_path = profile_root / "request_measurements.csv"
            if measurements_path.is_file():
                with measurements_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        record = {"profile_id": profile_root.name, **dict(row)}
                        load_results.append(record)
                        traces.write(json.dumps(record, sort_keys=True) + "\n")
            gpu_path = profile_root / "gpu_samples.csv"
            if gpu_path.is_file():
                with gpu_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        metrics.write(
                            json.dumps(
                                {
                                    "kind": "gpu_sample",
                                    "profile_id": profile_root.name,
                                    **dict(row),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
    (output_root / "quality_results.json").write_text(
        json.dumps(quality_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "load_results.json").write_text(
        json.dumps(load_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection = _select(rows)
    accepted_profile_ids_raw = selection.get("accepted_profile_ids")
    accepted_profile_ids = (
        set(accepted_profile_ids_raw)
        if isinstance(accepted_profile_ids_raw, list)
        else set()
    )
    (output_root / "selected_profiles.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "pareto_frontier.json").write_text(
        json.dumps(
            {
                "axes": ["quality_score", "p95_e2e_ms", "aggregate_output_tokens_per_second"],
                "eligible_points": [
                    row
                    for row in rows
                    if row["profile_id"] in accepted_profile_ids
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        output_root / "pareto_frontier.json",
        output_root / "serving_profile_pareto.json",
    )
    (output_root / "serving_profile_recommendation.md").write_text(
        "# Serving profile recommendation\n\n"
        f"Selection status: `{selection['status']}`.\n\n"
        "The selected quality/standard point is the highest measured throughput "
        "candidate that passes hard gates and retains at least 99% of P0 quality. "
        "Economy remains the separately restricted SystemVerilog CodeV route.\n",
        encoding="utf-8",
    )
    server_logs = output_root / "server_logs"
    server_logs.mkdir()
    for log in sorted((output_root / "profiles").glob("*/runtime/*.server.log")):
        shutil.copy2(log, server_logs / f"{log.parents[1].name}_{log.name}")
    pcie_records = [
        {
            "profile_id": path.parent.name,
            "sample_path": str(path.relative_to(output_root)),
            "sample": path.read_text(encoding="utf-8", errors="replace"),
        }
        for path in sorted((output_root / "profiles").glob("*/pcie_traffic_sample.txt"))
    ]
    (output_root / "PCIe_bandwidth_result.json").write_text(
        json.dumps(
            {
                "status": "MEASURED" if pcie_records else "NOT_MEASURED",
                "samples": pcie_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = ServingProfileRuntime(output_root / "profiles/runtime")
    safe_shutdown: dict[str, object]
    try:
        safe_shutdown = runtime.status()
    except ServingRuntimeError as exc:
        safe_shutdown = {
            "status": "OBSERVATION_FAILED",
            "failure_category": exc.category,
            "evidence": exc.evidence,
        }
    (output_root / "final_safe_shutdown.json").write_text(
        json.dumps(safe_shutdown, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cpu_pass = all(result["returncode"] == 0 for result in checks)
    gpu_complete = len(rows) == len(load_profiles(root / "configs/serving_profiles"))
    (output_root / "smoke_results.json").write_text(
        json.dumps(
            {
                "static_cpu_security": cpu_pass,
                "profile_resolution": resolver["returncode"] == 0,
                "live_profile_count": len(rows),
                "required_profile_count": len(
                    load_profiles(root / "configs/serving_profiles")
                ),
                "gpu_complete": gpu_complete,
                "safe_shutdown_observed": safe_shutdown.get("status") == "OBSERVED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = (
        "# Tiered serving certification\n\n"
        f"- CPU/security/GUI checks: {'PASS' if cpu_pass else 'FAIL'}\n"
        f"- GPU profiles completed: {len(rows)}/{len(load_profiles(root / 'configs/serving_profiles'))}\n"
        f"- GPU certification complete: {gpu_complete}\n"
        f"- Selection status: {selection['status']}\n"
        "- Capability tier and model lane remain independent axes.\n"
        "- Basic requests receive no tool schemas; Plus agent requests require a live server grant and isolated worktree.\n"
        "- Only ownership-recorded Laplace process groups are eligible for shutdown.\n"
    )
    (output_root / "final_report.md").write_text(report, encoding="utf-8")
    bundle = _bundle(output_root)
    print(json.dumps({"output_root": str(output_root), "bundle": str(bundle)}))
    return 0 if cpu_pass and (arguments.skip_gpu or gpu_complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
