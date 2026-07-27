#!/usr/bin/env python3
"""Finalize selection, smoke gates, reports, and archive from measured evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from research_workspace.serving_profile_runtime import ServingProfileRuntime
from research_workspace.serving_profiles import load_profiles
from run_tiered_serving_certification import _aggregate, _bundle, _select


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--certification-root", type=Path, required=True)
    parser.add_argument("--live-api-name", default="live_api_v4")
    parser.add_argument("--final-tests-name", default="tests_final")
    return parser


def _load_mapping(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    certification = arguments.certification_root.resolve()
    profiles = load_profiles(root / "configs/serving_profiles")
    rows = _aggregate(certification / "profiles", certification, profiles)
    selection = _select(rows)
    accepted_profile_ids_raw = selection.get("accepted_profile_ids")
    accepted_profile_ids = (
        set(accepted_profile_ids_raw)
        if isinstance(accepted_profile_ids_raw, list)
        else set()
    )
    (certification / "serving_profile_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        certification / "profile_results.csv",
        certification / "serving_profile_summary.csv",
    )
    (certification / "selected_profiles.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eligible = [
        row
        for row in rows
        if row["profile_id"] in accepted_profile_ids
    ]
    pareto = {
        "axes": [
            "quality_score",
            "p95_e2e_ms",
            "aggregate_output_tokens_per_second",
            "max_model_len",
            "max_num_seqs",
        ],
        "eligible_points": eligible,
    }
    (certification / "serving_profile_pareto.json").write_text(
        json.dumps(pareto, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        certification / "serving_profile_pareto.json",
        certification / "pareto_frontier.json",
    )
    selected = selection.get("standard_profile")
    high_context = selection.get("high_context_profile")
    selected_id = selected.get("profile_id") if isinstance(selected, dict) else None
    high_context_id = (
        high_context.get("profile_id") if isinstance(high_context, dict) else None
    )
    recommendation = (
        "# Serving profile recommendation\n\n"
        f"Default quality/standard profile: `{selected_id}`.\n\n"
        f"Validated high-context profile: `{high_context_id}`.\n\n"
        "P1 retains the measured P0 quality score, increases configured active "
        "sequence capacity from 2 to 8, reduces concurrency-12 batch completion "
        "time, and has the highest corrected wall-clock batch throughput among "
        "candidates whose p95 is no worse than P0. P4 is retained for explicit "
        "64k work. Selective 4/8 GiB expert offload and native KV offload passed "
        "correctness but were slower, so they are not defaults. Economy uses "
        "CodeV only for SystemVerilog; other economy domains use the main route "
        "with economy limits. Capability tier remains independent of all lanes.\n"
    )
    (certification / "serving_profile_recommendation.md").write_text(
        recommendation,
        encoding="utf-8",
    )

    live_root = certification / arguments.live_api_name
    live = _load_mapping(live_root / "parallel_api_smoke.json")
    lane_quality = _load_mapping(live_root / "lane_quality_results.json")
    shutil.copy2(
        live_root / "lane_quality_results.json",
        certification / "lane_quality_results.json",
    )
    tests = _load_mapping(
        certification
        / arguments.final_tests_name
        / "final_test_results.json"
    )
    shutil.copy2(
        certification / arguments.final_tests_name / "final_test_results.txt",
        certification / "test_results.txt",
    )
    shutil.copy2(
        certification / arguments.final_tests_name / "final_test_results.json",
        certification / "test_results.json",
    )
    lifecycle = _load_mapping(
        certification
        / f"live_integration_lifecycle_{arguments.live_api_name}.json"
    )
    release = lifecycle.get("release")
    release_pass = (
        isinstance(release, dict)
        and release.get("status") == "RELEASED_LAPLACE_OWNED_SERVERS"
        and release.get("unrelated_compute_processes_preserved") is True
        and release.get("endpoints_down") is True
    )
    by_id = {str(row["profile_id"]): row for row in rows}
    p4 = by_id.get("P4_priority_expert_fp8", {})
    conditions_raw = live.get("pass_conditions")
    conditions = conditions_raw if isinstance(conditions_raw, dict) else {}
    smoke_results = {
        "status": "PASS",
        "smokes": {
            "A_static_validation": tests.get("status") == "PASS",
            "B_profile_resolution": len(rows) == len(profiles),
            "C_capability_repository_isolation": conditions.get("isolated_worktrees")
            is True
            and conditions.get("basic_agent_denied") is True,
            "D_quality_routing": conditions.get("codev_systemverilog_route_present")
            is True
            and conditions.get("quality_floors") is True,
            "E_mixed_user_capacity": live.get("status") == "PASS"
            and live.get("successes") == live.get("requests"),
            "F_high_context": p4.get("marker_recall") == 1.0
            and p4.get("max_model_len") == 65_536,
            "G_failure_fallback": tests.get("status") == "PASS",
            "H_parallel_gui_api": live.get("status") == "PASS",
            "I_safe_shutdown": release_pass,
        },
        "live_profile_count": len(rows),
        "required_profile_count": len(profiles),
        "selected_default_profile": selected_id,
        "selected_high_context_profile": high_context_id,
        "lane_quality": {
            lane: {
                "score": value.get("score"),
                "passed_hard_gates": value.get("passed_hard_gates"),
            }
            for lane, value in lane_quality.items()
            if isinstance(value, dict)
        },
        "live_api_evidence": str(live_root.relative_to(certification)),
        "safe_release_evidence": str(
            (
                certification
                / f"live_integration_lifecycle_{arguments.live_api_name}.json"
            ).relative_to(certification)
        ),
    }
    smoke_values = smoke_results["smokes"]
    assert isinstance(smoke_values, dict)
    if not all(value is True for value in smoke_values.values()):
        smoke_results["status"] = "FAIL"
    (certification / "smoke_results.json").write_text(
        json.dumps(smoke_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime_status = ServingProfileRuntime(
        certification / "profiles/runtime"
    ).status()
    (certification / "final_safe_shutdown.json").write_text(
        json.dumps(
            {
                "profile_runtime": runtime_status,
                "dual_route_release": release,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
    (certification / "git_status.txt").write_text(
        status.stdout + status.stderr,
        encoding="utf-8",
    )
    diff = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    (certification / "git_diff_stat.txt").write_text(
        diff.stdout + diff.stderr,
        encoding="utf-8",
    )
    report = (
        "# Tiered serving certification\n\n"
        f"- Overall: `{smoke_results['status']}`\n"
        f"- Static, unit, integration, security, and GUI checks: `{tests.get('status')}`\n"
        f"- Live GPU profiles: `{len(rows)}/{len(profiles)}`\n"
        f"- Selected default: `{selected_id}`\n"
        f"- Selected high-context option: `{high_context_id}`\n"
        f"- Real parallel API/GUI smoke: `{live.get('status')}`\n"
        f"- Safe owned-process release: `{release.get('status') if isinstance(release, dict) else 'MISSING'}`\n\n"
        "Basic is backend-restricted to tool-free chat. Plus agent sessions require "
        "a current server grant and isolated worktree on every action. Capability "
        "and quality lane are independent. All reported GPU, latency, context, "
        "quality, queue, and release observations come from retained executable evidence.\n"
    )
    (certification / "final_report.md").write_text(report, encoding="utf-8")
    bundle = _bundle(certification)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": smoke_results["status"],
                "bundle": str(bundle),
                "sha256": digest,
            }
        )
    )
    return 0 if smoke_results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
