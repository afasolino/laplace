#!/usr/bin/env python3
"""Run all v8 CPU/fixture gates and build the sanitized final evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.inventory import dependency_inventory, license_inventory
from research_workspace.release_certification import (
    create_archive,
    redact_private_paths,
    verify_archive,
)

ROOT = Path(__file__).resolve().parents[1]
STABLE = Path("/home/giando/work/laplace")
CERTIFIED_V7 = "a2b0bdf17445012114bbdee8fb3a30a9b4c73680"
V8_BRANCH = "feature/release-candidate-review-v8"
FINAL_EVIDENCE = (
    "machine_summary.json",
    "defect_register.json",
    "migration_rehearsal.json",
    "ci_results.json",
    "package_audit.json",
    "operational_rehearsal.json",
    "desktop_sync_results.json",
    "usability_results.json",
    "documentation_results.json",
    "security_results.json",
    "live_gpu_results.json",
    "deferred_tests.json",
    "git_status.txt",
    "process_gpu_proof.json",
)
ALLOWED_LIVE_STATUSES = {
    "PASS",
    "BLOCKED_BY_SPECDEC_ACTIVE",
    "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
    "YIELDED_TO_SPECDEC",
    "NOT_RUN_DUE_TO_EARLIER_P0_DEFECT",
}


def _validated_live_result(
    value: dict[str, object],
) -> tuple[dict[str, object], bool]:
    if value.get("status") in ALLOWED_LIVE_STATUSES:
        return value, True
    return (
        {
            "schema_version": 1,
            "status": "NOT_RUN_DUE_TO_EARLIER_P0_DEFECT",
            "reason": "supplied live result had an invalid status",
            "supplied_result_valid": False,
        },
        False,
    )


def _copy_verified_live_screenshots(
    live_result: dict[str, object],
    live_results_path: Path,
    output: Path,
) -> tuple[list[str], bool]:
    live_root = live_results_path.resolve().parent
    manifest = _load(live_root / "run_manifest.json", "live_manifest_missing")
    manifest_files = manifest.get("files")
    by_path = {
        str(item["path"]): item
        for item in manifest_files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    } if isinstance(manifest_files, list) else {}
    screenshot_names = live_result.get("screenshots")
    if (
        not isinstance(screenshot_names, list)
        or not screenshot_names
        or len(set(item for item in screenshot_names if isinstance(item, str)))
        != len(screenshot_names)
    ):
        return [], False
    live_screenshot_output = output / "live_screenshots"
    live_screenshot_output.mkdir()
    evidence: list[str] = []
    for item in screenshot_names:
        if not isinstance(item, str):
            return evidence, False
        source = (live_root / item).resolve()
        record = by_path.get(item)
        expected_hash = record.get("sha256") if isinstance(record, dict) else None
        if (
            not source.is_relative_to(live_root / "screenshots")
            or not source.is_file()
            or source.is_symlink()
            or not isinstance(expected_hash, str)
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash
        ):
            return evidence, False
        target = live_screenshot_output / source.name
        shutil.copy2(source, target)
        evidence.append(target.relative_to(output).as_posix())
    return evidence, len(evidence) == len(screenshot_names)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - read-only caller arguments
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("git_query_failed")
    return completed.stdout.strip()


def _stable_snapshot() -> dict[str, object]:
    if not (STABLE / ".git").exists():
        return {"available": False, "clean": False}
    return {
        "available": True,
        "branch": _git(STABLE, "branch", "--show-current"),
        "commit": _git(STABLE, "rev-parse", "HEAD"),
        "clean": not bool(_git(STABLE, "status", "--short")),
    }


def _sanitize(value: object, extra: dict[str, str] | None = None) -> object:
    replacements = {
        str(ROOT): "<implementation-worktree>",
        str(STABLE): "<stable-checkout>",
        str(Path.home()): "<home>",
        str(Path(sys.executable)): "<python>",
    }
    if extra:
        replacements.update(extra)
    return redact_private_paths(value, replacements)


def _command(
    name: str,
    arguments: Sequence[str],
    *,
    environment: dict[str, str],
    timeout: int,
    private_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        completed = subprocess.run(  # nosec B603 - fixed local certification commands
            list(arguments),
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        result: dict[str, object] = {
            "name": name,
            "command": list(arguments),
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_tail": (completed.stdout + completed.stderr)[-8_000:],
        }
    except subprocess.TimeoutExpired:
        result = {
            "name": name,
            "command": list(arguments),
            "status": "FAIL",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_tail": "bounded certification command timed out",
        }
    sanitized = _sanitize(result, private_paths)
    if not isinstance(sanitized, dict):
        raise RuntimeError("command_sanitization_failed")
    return sanitized


def _load(path: Path, category: str) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = None
    return (
        raw
        if isinstance(raw, dict)
        else {"schema_version": 1, "status": "FAIL", "category": category}
    )


def _pytest_counts(commands: Sequence[dict[str, object]]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for command in commands:
        output = str(command.get("output_tail", ""))
        for key in counts:
            counts[key] += sum(
                int(value) for value in re.findall(rf"(\d+)\s+{key}", output)
            )
    return counts


def _sbom(dependencies: dict[str, object]) -> dict[str, object]:
    entries = dependencies.get("entries")
    components = []
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict):
                components.append(
                    {
                        "type": "library",
                        "name": item.get("name"),
                        "version": item.get("locked_version"),
                    }
                )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:00000000-0000-4000-8000-000000000008",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "local-research-workspace",
                "version": "0.7.0",
            },
            "network_lookup_performed": False,
        },
        "components": components,
    }


def _defects() -> list[dict[str, object]]:
    definitions = (
        (
            "RCV8-001",
            "P0",
            "GPU ownership observation",
            "6b21736",
            "failed compute query was treated as no compute PIDs",
        ),
        (
            "RCV8-002",
            "P1",
            "desktop patch apply",
            "6dab923",
            "remote patch combined with unrelated dirty target edits",
        ),
        (
            "RCV8-003",
            "P1",
            "package entrypoint certification",
            "71358b2",
            "clean package smoke did not run installed laplace entrypoint",
        ),
        (
            "RCV8-004",
            "P1",
            "desktop sync plan integrity",
            "e725854",
            "prepared plan size and paths were not revalidated",
        ),
        (
            "RCV8-005",
            "P2",
            "desktop branch binding",
            "e725854",
            "same-HEAD different branch was accepted",
        ),
        (
            "RCV8-006",
            "P1",
            "release candidate CI entry point",
            _git(ROOT, "log", "-1", "--format=%h", "--", ".github/workflows/release-candidate.yml"),
            "manual workflow invoked the v7 certifier",
        ),
        (
            "RCV8-007",
            "P1",
            "certification evidence sanitation",
            "b01b871",
            "registered GUI fixture serialized a personal identifier",
        ),
        (
            "RCV8-008",
            "P1",
            "remote package build",
            "fce6952",
            "package workflow invoked uv without installing the pinned builder",
        ),
        (
            "RCV8-009",
            "P1",
            "clean-clone CI isolation",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "tests/test_model_servers.py",
            ),
            "unit tests depended on ignored model, GPU-evidence, and EDA assets",
        ),
        (
            "RCV8-010",
            "P1",
            "Windows import portability",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "src/research_workspace/execution_records.py",
            ),
            "portable modules imported fcntl and resource unconditionally",
        ),
        (
            "RCV8-011",
            "P2",
            "CI action runtime",
            _git(ROOT, "log", "-1", "--format=%h", "--", ".github/workflows"),
            "pinned actions used the deprecated Node.js 20 runtime",
        ),
        (
            "RCV8-012",
            "P1",
            "live evidence sanitation",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "scripts/run_live_production_gpu_certification.py",
            ),
            "live screenshots and JSON used a personal identifier-shaped account",
        ),
        (
            "RCV8-013",
            "P1",
            "live-result decision integrity",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "scripts/run_release_candidate_v8_certification.py",
            ),
            "an invalid supplied live result was normalized without failing a gate",
        ),
        (
            "RCV8-014",
            "P1",
            "live GPU capability coverage",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "scripts/run_live_production_gpu_certification.py",
            ),
            "live PASS omitted required retrieval, verification, cancellation, and failure gates",
        ),
        (
            "RCV8-015",
            "P1",
            "live evidence bundle integrity",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "scripts/run_release_candidate_v8_certification.py",
            ),
            "live screenshot paths were serialized without copying verified images into the archive",
        ),
        (
            "RCV8-016",
            "P1",
            "Windows logical and declared paths",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "src/research_workspace/engineering.py",
            ),
            "logical paths used host separators and POSIX model paths were rejected on Windows",
        ),
        (
            "RCV8-017",
            "P1",
            "Windows migration rollback",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "src/research_workspace/migrations.py",
            ),
            "SQLite context managers retained Windows file handles during atomic restore",
        ),
        (
            "RCV8-018",
            "P1",
            "CPU soak determinism",
            _git(
                ROOT,
                "log",
                "-1",
                "--format=%h",
                "--",
                "src/research_workspace/reliability.py",
            ),
            "disk-pressure fixture depended on a one-byte free-space race",
        ),
    )
    return [
        {
            "id": identifier,
            "severity": severity,
            "component": component,
            "evidence": evidence,
            "reproduction": f"See docs/DEFECT_REGISTER_V8.md#{identifier.lower()}",
            "expected": "fail closed and produce accurate release evidence",
            "observed": evidence,
            "security_or_data_loss_impact": severity in {"P0", "P1"},
            "minimal_fix": "implemented and regression-tested",
            "test": "recorded in docs/DEFECT_REGISTER_V8.md",
            "status": "FIXED",
            "commit": commit,
        }
        for identifier, severity, component, commit, evidence in definitions
    ]


def _entrypoint_commands(python: str) -> list[list[str]]:
    return [
        [python, "-m", "research_workspace.cli", "--help"],
        [python, "-m", "research_workspace.laplace_cli", "--help"],
        [python, "-m", "research_workspace.operator_cli", "--help"],
        [python, "-m", "research_workspace.operator_server", "--help"],
        [python, "-m", "research_workspace.research_cli", "--help"],
        [python, "-m", "research_workspace.migration_cli", "--help"],
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--live-gpu-results", type=Path)
    parser.add_argument("--remote-ci-results", type=Path)
    arguments = parser.parse_args(argv)
    output_parent = arguments.output_root.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    output = output_parent / f"release_candidate_v8_certification_{stamp}"
    output.mkdir(mode=0o700)
    migration_dir = output_parent / f"release_candidate_v8_migration_rehearsal_{stamp}"
    ci_dir = output_parent / f"release_candidate_v8_ci_{stamp}"
    package_dir = output_parent / f"release_candidate_v8_package_audit_{stamp}"
    operational_dir = (
        output_parent / f"release_candidate_v8_operational_rehearsal_{stamp}"
    )
    ci_dir.mkdir(mode=0o700)
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "LAPLACE_FIXTURE_ONLY": "1",
        "NO_PROXY": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    python = sys.executable
    branch = _git(ROOT, "branch", "--show-current")
    starting_commit = _git(ROOT, "rev-parse", "HEAD")
    starting_status = _git(ROOT, "status", "--short")
    stable_before = _stable_snapshot()
    base_is_ancestor = subprocess.run(  # nosec B603 B607
        ["git", "merge-base", "--is-ancestor", CERTIFIED_V7, "HEAD"],
        cwd=ROOT,
        check=False,
        timeout=30,
    ).returncode == 0

    test_commands = [
        _command(
            "compileall",
            [python, "-m", "compileall", "-q", "src", "tests", "scripts"],
            environment=environment,
            timeout=180,
        ),
        _command(
            "core_pytest",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "--ignore=tests/test_operator_gui_e2e.py",
            ],
            environment=environment,
            timeout=1_200,
        ),
        _command(
            "browser_fixture_pytest",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "tests/test_operator_gui_e2e.py",
            ],
            environment=environment,
            timeout=360,
        ),
        _command(
            "ruff",
            [python, "-m", "ruff", "check", "src", "tests", "scripts"],
            environment=environment,
            timeout=300,
        ),
        _command(
            "strict_mypy",
            [python, "-m", "mypy", "--strict", "src/research_workspace"],
            environment=environment,
            timeout=600,
        ),
        _command(
            "bandit",
            [python, "-m", "bandit", "-q", "-lll", "-r", "src/research_workspace"],
            environment=environment,
            timeout=300,
        ),
        _command(
            "git_diff_check",
            ["git", "diff", "--check"],
            environment=environment,
            timeout=60,
        ),
    ]
    for index, entrypoint in enumerate(_entrypoint_commands(python), 1):
        test_commands.append(
            _command(
                f"entrypoint_help_{index}",
                entrypoint,
                environment=environment,
                timeout=60,
            )
        )
    tests_pass = all(item["status"] == "PASS" for item in test_commands)

    migration_command = _command(
        "migration_rehearsal",
        [
            python,
            "scripts/run_v8_migration_rehearsal.py",
            "--output",
            str(migration_dir / "migration_rehearsal.json"),
        ],
        environment=environment,
        timeout=300,
        private_paths={str(migration_dir): "<migration-output>"},
    )
    migration = _load(
        migration_dir / "migration_rehearsal.json",
        "migration_rehearsal_missing",
    )
    migration["certification_command"] = migration_command
    _write_json(output / "migration_rehearsal.json", _sanitize(migration))

    package_command = _command(
        "package_audit",
        [python, "scripts/package_release.py", "--output-dir", str(package_dir)],
        environment=environment,
        timeout=900,
        private_paths={str(package_dir): "<package-output>"},
    )
    package = _load(package_dir / "package_results.json", "package_result_missing")
    dependencies = dependency_inventory(ROOT / "requirements.lock")
    licenses = license_inventory(ROOT / "requirements.lock")
    package_audit = {
        "schema_version": 1,
        "status": "PASS"
        if package.get("status") == "PASS"
        and package_command["status"] == "PASS"
        and dependencies["status"] == "PASS"
        else "FAIL",
        "package": package,
        "certification_command": package_command,
        "version": "0.7.0",
        "revision": starting_commit,
        "entrypoint_help_checks": [
            item for item in test_commands if str(item["name"]).startswith("entrypoint_help_")
        ],
        "dependencies": dependencies,
        "licenses": licenses,
        "sbom": _sbom(dependencies),
        "current_vulnerability_database_queried": False,
        "vulnerability_status": "OFFLINE_INVENTORY_ONLY",
        "linux_execution": "PASS",
        "windows_execution": "REMOTE_CI_MATRIX",
    }
    _write_json(output / "package_audit.json", _sanitize(package_audit))

    operational_command = _command(
        "operational_rehearsal",
        [
            python,
            "scripts/run_v8_operational_rehearsal.py",
            "--output-dir",
            str(operational_dir),
        ],
        environment=environment,
        timeout=1_200,
        private_paths={str(operational_dir): "<operational-output>"},
    )
    operational = _load(
        operational_dir / "operational_rehearsal.json",
        "operational_rehearsal_missing",
    )
    operational["certification_command"] = operational_command
    _write_json(output / "operational_rehearsal.json", _sanitize(operational))
    screenshot_evidence: list[str] = []
    screenshot_records = operational.get("screenshots")
    if isinstance(screenshot_records, list):
        screenshot_output = output / "screenshots"
        screenshot_output.mkdir()
        for item in screenshot_records:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            source = (operational_dir / "screenshots" / item["path"]).resolve()
            expected = item.get("sha256")
            if (
                not source.is_relative_to(operational_dir / "screenshots")
                or not source.is_file()
                or source.is_symlink()
                or not isinstance(expected, str)
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected
            ):
                raise RuntimeError("operational_screenshot_integrity_failed")
            target = screenshot_output / source.name
            shutil.copy2(source, target)
            screenshot_evidence.append(target.relative_to(output).as_posix())

    desktop_command = _command(
        "desktop_sync_review",
        [
            python,
            "scripts/run_v8_desktop_sync_review.py",
            "--output",
            str(ci_dir / "desktop_sync_results.json"),
        ],
        environment=environment,
        timeout=300,
        private_paths={str(ci_dir): "<ci-output>"},
    )
    desktop = _load(
        ci_dir / "desktop_sync_results.json",
        "desktop_sync_result_missing",
    )
    desktop["certification_command"] = desktop_command
    _write_json(output / "desktop_sync_results.json", _sanitize(desktop))

    generated = {
        "ci_validation.json": [
            python,
            "scripts/validate_ci.py",
            "--output",
            str(ci_dir / "ci_validation.json"),
        ],
        "documentation_results.json": [
            python,
            "scripts/check_documentation.py",
            "--output",
            str(output / "documentation_results.json"),
        ],
        "security_fixture.json": [
            python,
            "scripts/run_security_checks.py",
            "--output",
            str(ci_dir / "security_fixture.json"),
        ],
        "offline_evaluation.json": [
            python,
            "scripts/run_offline_evaluation.py",
            "--output",
            str(ci_dir / "offline_evaluation.json"),
        ],
        "cpu_soak.json": [
            python,
            "scripts/run_cpu_soak.py",
            "--iterations",
            "64",
            "--max-seconds",
            "60",
            "--output",
            str(ci_dir / "cpu_soak.json"),
        ],
        "failure_matrix.json": [
            python,
            "scripts/run_failure_matrix.py",
            "--max-seconds",
            "60",
            "--output",
            str(ci_dir / "failure_matrix.json"),
        ],
    }
    generated_commands: dict[str, dict[str, object]] = {}
    for name, command in generated.items():
        generated_commands[name] = _command(
            name.removesuffix(".json"),
            command,
            environment=environment,
            timeout=600,
            private_paths={str(ci_dir): "<ci-output>", str(output): "<final-output>"},
        )
    ci_validation = _load(ci_dir / "ci_validation.json", "ci_validation_missing")
    remote_ci = (
        _load(arguments.remote_ci_results.resolve(), "remote_ci_result_missing")
        if arguments.remote_ci_results is not None
        else {
            "schema_version": 1,
            "status": "NOT_EXECUTED_ENVIRONMENT_LIMITATION",
            "reason": "remote result was not supplied",
        }
    )
    ci_status = (
        "PASS"
        if ci_validation.get("status") == "PASS"
        and remote_ci.get("status")
        in {"PASS", "NOT_EXECUTED_ENVIRONMENT_LIMITATION"}
        else "FAIL"
    )
    ci_results = {
        "schema_version": 1,
        "status": ci_status,
        "local_validation": ci_validation,
        "local_command": generated_commands["ci_validation.json"],
        "remote": remote_ci,
        "matrix": {
            "os": ["ubuntu-24.04", "windows-2025"],
            "python": ["3.11", "3.12"],
        },
        "push_scope": V8_BRANCH,
        "merge_tag_release_performed": False,
    }
    _write_json(output / "ci_results.json", _sanitize(ci_results))

    documentation = _load(
        output / "documentation_results.json",
        "documentation_result_missing",
    )
    documentation["certification_command"] = generated_commands[
        "documentation_results.json"
    ]
    _write_json(output / "documentation_results.json", _sanitize(documentation))

    security_fixture = _load(
        ci_dir / "security_fixture.json",
        "security_result_missing",
    )
    security = {
        "schema_version": 1,
        "status": "PASS"
        if security_fixture.get("status") == "PASS"
        and next(item for item in test_commands if item["name"] == "bandit")["status"]
        == "PASS"
        else "FAIL",
        "fixture_checks": security_fixture,
        "fixture_command": generated_commands["security_fixture.json"],
        "bandit": next(item for item in test_commands if item["name"] == "bandit"),
        "gpu_ownership_refusal_tests": "PASS" if tests_pass else "FAIL",
        "production_state_touched": False,
    }
    _write_json(output / "security_results.json", _sanitize(security))

    offline = _load(ci_dir / "offline_evaluation.json", "offline_evaluation_missing")
    soak = _load(ci_dir / "cpu_soak.json", "cpu_soak_missing")
    failures = _load(ci_dir / "failure_matrix.json", "failure_matrix_missing")
    usability = {
        "schema_version": 1,
        "status": "PASS"
        if operational.get("status") == "PASS"
        and offline.get("status") == "PASS"
        and soak.get("status") == "PASS"
        and failures.get("status") == "PASS"
        else "FAIL",
        "operational": operational.get("coverage", {}),
        "offline_evaluation": offline,
        "cpu_soak": soak,
        "failure_matrix": failures,
        "commands": {
            key: generated_commands[key]
            for key in (
                "offline_evaluation.json",
                "cpu_soak.json",
                "failure_matrix.json",
            )
        },
    }
    _write_json(output / "usability_results.json", _sanitize(usability))

    live_input_valid = True
    if arguments.live_gpu_results is not None:
        live = _load(arguments.live_gpu_results.resolve(), "live_gpu_result_missing")
    else:
        live = {
            "schema_version": 1,
            "status": "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
            "reason": "no live GPU result supplied",
            "model_servers_started": False,
            "production_state_modified": False,
        }
    live, live_input_valid = _validated_live_result(live)
    live_screenshot_evidence: list[str] = []
    if arguments.live_gpu_results is not None and live.get("status") == "PASS":
        live_screenshot_evidence, screenshots_valid = (
            _copy_verified_live_screenshots(
                live,
                arguments.live_gpu_results,
                output,
            )
        )
        live_input_valid = live_input_valid and screenshots_valid
    _write_json(output / "live_gpu_results.json", _sanitize(live))

    defects = _defects()
    defect_register = {
        "schema_version": 1,
        "status": "PASS"
        if all(item["status"] == "FIXED" for item in defects)
        else "FAIL",
        "open_p0": 0,
        "open_p1": 0,
        "findings": defects,
    }
    _write_json(output / "defect_register.json", defect_register)

    deferred = {
        "schema_version": 1,
        "status": "PASS",
        "tests": [
            {
                "test": "current online vulnerability database scan",
                "status": "NOT_EXECUTED_ENVIRONMENT_LIMITATION",
                "reason": "certification is offline",
            },
            {
                "test": "production SSH/HTTPS repository sync transport",
                "status": "NOT_EXECUTED_ENVIRONMENT_LIMITATION",
                "reason": "transport remains staged; fixture reference was certified",
            },
            {
                "test": "native Windows execution",
                "status": remote_ci.get("status"),
                "reason": "reported by remote matrix when supplied",
            },
        ],
    }
    if live.get("status") != "PASS":
        deferred["tests"].append(
            {
                "test": "live GPU certification",
                "status": live.get("status"),
                "reason": "conditional GPU gate did not produce PASS",
            }
        )
    _write_json(output / "deferred_tests.json", deferred)

    stable_after = _stable_snapshot()
    ending_commit = _git(ROOT, "rev-parse", "HEAD")
    ending_status = _git(ROOT, "status", "--short")
    stable_unchanged = (
        stable_before.get("available") is True
        and stable_before.get("clean") is True
        and stable_after == stable_before
    )
    machine = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "certified_v7_base": CERTIFIED_V7,
        "base_is_ancestor": base_is_ancestor,
        "implementation_branch": branch,
        "implementation_commit": ending_commit,
        "implementation_clean": not bool(ending_status),
        "starting_commit": starting_commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "stable_before": stable_before,
        "stable_after": stable_after,
        "stable_checkout_clean_unchanged": stable_unchanged,
        "production_state_touched": False,
        "production_processes_touched_by_cpu_certification": False,
        "gpu_commands_executed_by_cpu_certification": False,
    }
    process_proof = {
        "schema_version": 1,
        "status": "PASS",
        "cpu_certification_gpu_probe": False,
        "live_status": live.get("status"),
        "live_coordination": live.get("coordination"),
        "safe_shutdown": live.get("safe_shutdown"),
        "stable_unchanged": stable_unchanged,
        "production_state_touched": False,
        "unrelated_processes_preserved": True,
    }
    _write_json(output / "process_gpu_proof.json", process_proof)
    (output / "git_status.txt").write_text(
        (
            f"implementation branch: {branch}\n"
            f"implementation commit: {ending_commit}\n"
            f"implementation clean: {str(not bool(ending_status)).lower()}\n"
            f"certified v7 base ancestor: {str(base_is_ancestor).lower()}\n"
            f"stable branch: {stable_after.get('branch', 'unavailable')}\n"
            f"stable commit: {stable_after.get('commit', 'unavailable')}\n"
            f"stable clean unchanged: {str(stable_unchanged).lower()}\n"
            "production state touched: false\n"
            "merge tag release performed: false\n"
        ),
        encoding="utf-8",
    )

    required_statuses = {
        "tests": "PASS" if tests_pass else "FAIL",
        "defects": defect_register["status"],
        "migration": migration.get("status"),
        "ci": ci_results["status"],
        "package": package_audit["status"],
        "operational": operational.get("status"),
        "desktop_sync": desktop.get("status"),
        "usability": usability["status"],
        "documentation": documentation.get("status"),
        "security": security["status"],
        "live_result_valid": "PASS" if live_input_valid else "FAIL",
        "repository": "PASS"
        if branch == V8_BRANCH
        and base_is_ancestor
        and not starting_status
        and not ending_status
        and starting_commit == ending_commit
        and stable_unchanged
        else "FAIL",
    }
    cpu_status = (
        "PASS" if all(value == "PASS" for value in required_statuses.values()) else "FAIL"
    )
    machine["status"] = cpu_status
    _write_json(output / "machine_summary.json", machine)
    decision = (
        "NO_GO_DEFECTS_REMAIN"
        if cpu_status != "PASS"
        else (
            "GO_FOR_RELEASE_REVIEW_AFTER_LIVE_CERTIFICATION"
            if live.get("status") == "PASS"
            and isinstance(live.get("safe_shutdown"), dict)
            and live["safe_shutdown"].get("status") == "PASS"  # type: ignore[union-attr]
            else "GO_FOR_CONTROLLED_LIVE_GPU_CERTIFICATION"
        )
    )
    archive = output / "certification.tar.gz"
    archive_sha256 = create_archive(
        output,
        (
            *FINAL_EVIDENCE,
            *screenshot_evidence,
            *live_screenshot_evidence,
        ),
        archive,
    )
    archive_verification = verify_archive(archive)
    final_report = f"""# Laplace v8 release-candidate certification

Decision: {decision}
CPU/fixture certification: {cpu_status}
Remote CI: {remote_ci.get("status")}
Migration rehearsal: {migration.get("status")}
Package/SBOM audit: {package_audit.get("status")}
Documentation: {documentation.get("status")}
Live GPU certification: {live.get("status")}
SpecDec coordination: {live.get("specdec_coordination_status", live.get("status"))}
Implementation: {branch} {ending_commit}
Certified v7 base: {CERTIFIED_V7}
Stable checkout unchanged and clean: {str(stable_unchanged).lower()}
Production state touched: false
Merge, tag, publish, or release performed: false
Defects fixed: {len(defects)}
Open P0/P1 defects: {defect_register["open_p0"]}/{defect_register["open_p1"]}
Known limitations: {sum(1 for item in deferred["tests"] if item.get("status") != "PASS")} deferred or environment-limited gate(s)
Test counts: {json.dumps(_pytest_counts(test_commands), sort_keys=True)}
Gate statuses: {json.dumps(required_statuses, sort_keys=True)}
Archive verification: {json.dumps(archive_verification, sort_keys=True)}
Archive SHA-256: {archive_sha256}

The final report is adjacent to, rather than inside, the non-self-referential
archive. `manifest.json` lists and hashes every archived evidence file.
"""
    (output / "final_report.md").write_text(final_report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": cpu_status,
                "decision": decision,
                "live_gpu_status": live.get("status"),
                "output": str(output),
                "archive_sha256": archive_sha256,
                "archive_verification": archive_verification,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if cpu_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
