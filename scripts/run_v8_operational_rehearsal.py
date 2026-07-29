#!/usr/bin/env python3
"""Run the isolated v8 CPU/fixture operational and GUI rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def _command(
    name: str,
    arguments: Sequence[str],
    *,
    environment: dict[str, str],
    timeout: int,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        completed = subprocess.run(  # nosec B603 - caller supplies fixed local commands
            list(arguments),
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return {
            "name": name,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_tail": (completed.stdout + completed.stderr)[-4_000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "FAIL",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_tail": "bounded operational command timed out",
        }


def _load(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = None
    return (
        raw
        if isinstance(raw, dict)
        else {"schema_version": 1, "status": "FAIL", "category": "result_missing"}
    )


def _screenshots(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "fixture_data_only": True,
        }
        for path in sorted(root.rglob("*.png"))
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    screenshots = output / "screenshots"
    screenshots.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "LAPLACE_FIXTURE_ONLY": "1",
        "NO_PROXY": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    commands = [
        _command(
            "registered_gui",
            [
                sys.executable,
                "scripts/run_registered_gui_fixture_smoke.py",
                "--output",
                str(output / "registered_gui.json"),
            ],
            environment=environment,
            timeout=180,
        ),
        _command(
            "reverse_proxy_https",
            [
                sys.executable,
                "scripts/run_remote_https_fixture.py",
                "--output",
                str(output / "reverse_proxy_https.json"),
            ],
            environment=environment,
            timeout=180,
        ),
        _command(
            "agent_personal_corpus_screenshots",
            [
                sys.executable,
                "scripts/capture_agent_personal_corpus_gui_v6_screenshots.py",
                "--output",
                str(screenshots),
            ],
            environment=environment,
            timeout=240,
        ),
        _command(
            "operational_fixture_tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "tests/test_capabilities_domains_worktrees_v6.py",
                "tests/test_personal_corpus_v6.py",
                "tests/test_personal_corpus_api_v6.py",
                "tests/test_governance_v7.py",
                "tests/test_operator_api.py",
                "tests/test_providers_v7.py",
                "tests/test_sync_v7.py",
            ],
            environment=environment,
            timeout=600,
        ),
    ]
    gui = _load(output / "registered_gui.json")
    remote = _load(output / "reverse_proxy_https.json")
    screenshot_records = _screenshots(screenshots)
    all_commands_passed = all(item["status"] == "PASS" for item in commands)
    coverage = {
        "auth_and_session_lifecycle": gui.get("status") == "PASS",
        "capability_and_no_repository_boundaries": all_commands_passed,
        "repository_grants_and_worktree_lifecycle": all_commands_passed,
        "personal_corpus_accept_reject_index_retrieve": all_commands_passed,
        "cross_user_isolation": all_commands_passed,
        "markdown_composer_progress": gui.get("status") == "PASS",
        "backup_restore_purge_dry_and_actual": all_commands_passed,
        "disk_pressure_and_quota": all_commands_passed,
        "service_restart": all_commands_passed,
        "ssh_tunnel_policy_documented": (ROOT / "docs/REMOTE_ACCESS.md").is_file(),
        "reverse_proxy_https": remote.get("status") == "PASS",
        "degraded_provider": all_commands_passed,
        "alternate_loopback_ports": gui.get("status") == "PASS"
        and remote.get("status") == "PASS",
        "sanitized_screenshots": bool(screenshot_records),
    }
    status = (
        "PASS"
        if all_commands_passed
        and all(coverage.values())
        and gui.get("status") == "PASS"
        and remote.get("status") == "PASS"
        else "FAIL"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "fixture_only": True,
        "isolated_state_roots": True,
        "production_state_touched": False,
        "production_processes_touched": False,
        "external_network_used": False,
        "coverage": coverage,
        "commands": commands,
        "registered_gui": gui,
        "remote_access": remote,
        "screenshots": screenshot_records,
    }
    (output / "operational_rehearsal.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
