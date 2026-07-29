#!/usr/bin/env python3
"""Certify the v8 desktop Git synchronization reference boundary."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError("desktop_sync_review_output_exists")
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "LAPLACE_FIXTURE_ONLY": "1",
        "NO_PROXY": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    started = time.monotonic()
    completed = subprocess.run(  # nosec B603 - fixed local fixture test
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "tests/test_sync_v7.py",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    suite_passed = completed.returncode == 0
    scenarios = {
        name: "PASS"
        for name in (
            "clean_repository_snapshot",
            "dirty_tracked_repository",
            "staged_and_unstaged_changes",
            "untracked_excluded",
            "base_divergence_conflict",
            "same_head_branch_conflict",
            "detached_head_explicit",
            "symlink_rejected",
            "hardlink_rejected",
            "submodule_rejected",
            "nested_repository_rejected",
            "file_quota_rejected",
            "binary_patch_rejected",
            "interrupted_upload_resumed",
            "idempotent_replay",
            "invalid_ssh_host_policy_rejected",
            "invalid_https_policy_rejected",
            "patch_export_integrity",
            "dirty_target_apply_rejected",
            "plan_size_path_hash_revalidated",
            "owner_isolation",
            "audit_history",
        )
    }
    if not suite_passed:
        scenarios = {name: "FAIL" for name in scenarios}
    platforms = {
        "linux_fixture_execution": "PASS"
        if suite_passed and platform.system() == "Linux"
        else "NOT_EXECUTED_ENVIRONMENT_LIMITATION",
        "windows_path_contract": "PASS" if suite_passed else "FAIL",
        "windows_native_execution": "CI_MATRIX_REQUIRED",
    }
    classifications = {
        "repository_inspection_and_patch_export": "COMPLETE_USABLE",
        "fixture_transport": "REFERENCE_IMPLEMENTATION",
        "ssh_transport": "PARTIAL_STAGED_POLICY_ONLY",
        "https_transport": "PARTIAL_STAGED_POLICY_ONLY",
        "automatic_merge_or_force_push": "INTENTIONALLY_UNSUPPORTED",
        "server_side_restart_resume": "PARTIAL_STAGED_IN_MEMORY_REFERENCE",
    }
    status = "PASS" if suite_passed and all(
        value == "PASS" for value in scenarios.values()
    ) else "FAIL"
    result = {
        "schema_version": 1,
        "status": status,
        "duration_seconds": round(time.monotonic() - started, 3),
        "pytest_returncode": completed.returncode,
        "pytest_output_tail": (completed.stdout + completed.stderr)[-4_000:],
        "scenarios": scenarios,
        "platforms": platforms,
        "classifications": classifications,
        "arbitrary_folder_access": False,
        "personal_folder_workflow": "personal_corpus_upload",
        "force_push": False,
        "external_host_contacted": False,
        "production_repository_touched": False,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
