#!/usr/bin/env python3
"""Run every v7 non-GPU gate and create a sanitized certification archive."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Sequence

from research_workspace.inventory import dependency_inventory, license_inventory
from research_workspace.migrations import (
    MigrationError,
    create_synthetic_v0_state,
    migrate,
    preflight,
    rollback_to_backup,
)
from research_workspace.release_certification import (
    create_archive,
    redact_private_paths,
    sha256_file,
    verify_archive,
)

ROOT = Path(__file__).resolve().parents[1]
STABLE = Path("/home/giando/work/laplace")
CERTIFIED_BASE = "0d9a1d54f445ad25a8bb84d3133c3377f446c476"
GPU_STATUS = "BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE"
EVIDENCE_NAMES = (
    "machine_summary.json",
    "architecture_audit.json",
    "test_results.json",
    "ci_validation.json",
    "package_results.json",
    "migration_results.json",
    "offline_evaluation.json",
    "soak_results.json",
    "failure_matrix.json",
    "security_results.json",
    "documentation_results.json",
    "dependency_inventory.json",
    "license_inventory.json",
    "git_status.txt",
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git query failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def _stable_snapshot() -> dict[str, object]:
    if not (STABLE / ".git").exists():
        return {"available": False, "clean": False}
    status = _git(STABLE, "status", "--short")
    return {
        "available": True,
        "branch": _git(STABLE, "branch", "--show-current"),
        "commit": _git(STABLE, "rev-parse", "HEAD"),
        "clean": not bool(status),
        "status": status.splitlines(),
    }


def _sanitized(value: object, temporary: Path | None = None) -> object:
    replacements = {
        str(ROOT): "<implementation-worktree>",
        str(STABLE): "<stable-checkout>",
        str(Path(sys.executable)): "<python>",
    }
    if temporary is not None:
        replacements[str(temporary)] = "<temporary>"
    return redact_private_paths(value, replacements)


def _command(
    name: str,
    arguments: Sequence[str],
    *,
    environment: dict[str, str],
    timeout: int,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = completed.stdout + completed.stderr
        result: dict[str, object] = {
            "name": name,
            "command": list(arguments),
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_tail": output[-12_000:],
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
    return _sanitized(result)  # type: ignore[return-value]


def _pytest_counts(commands: Sequence[dict[str, object]]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for command in commands:
        output = str(command.get("output_tail", ""))
        for key in counts:
            matches = re.findall(rf"(\d+)\s+{key}", output)
            counts[key] += sum(int(value) for value in matches)
    return counts


def _migration_evidence(root: Path) -> dict[str, object]:
    state = root / "migration-fixture"
    interrupted = root / "migration-interrupted"
    try:
        create_synthetic_v0_state(state, state_id="certification-fixture")
        before = preflight(state)
        dry_run = migrate(
            state,
            expected_state_id="certification-fixture",
            dry_run=True,
        )
        applied = migrate(
            state,
            expected_state_id="certification-fixture",
            dry_run=False,
        )
        after = preflight(state)
        rollback = rollback_to_backup(
            state,
            expected_state_id="certification-fixture",
            backup_id=str(applied["backup_id"]),
        )
        create_synthetic_v0_state(interrupted, state_id="interrupted-fixture")
        interrupted_rolled_back = False
        try:
            migrate(
                interrupted,
                expected_state_id="interrupted-fixture",
                dry_run=False,
                fail_after=2,
            )
        except MigrationError as exc:
            interrupted_rolled_back = str(exc) == "migration_failed_rolled_back"
        status = (
            "PASS"
            if before.status == "PASS"
            and dry_run["status"] == "DRY_RUN_PASS"
            and applied["status"] == "PASS"
            and after.status == "PASS"
            and rollback["status"] == "ROLLED_BACK"
            and interrupted_rolled_back
            else "FAIL"
        )
        return {
            "schema_version": 1,
            "status": status,
            "store_count": len(before.stores),
            "dry_run": dry_run,
            "apply": applied,
            "post_integrity": after.public(),
            "rollback": rollback,
            "interruption_rolled_back": interrupted_rolled_back,
            "fixture_only": True,
            "production_state_touched": False,
        }
    except (MigrationError, OSError) as exc:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "category": str(exc),
            "fixture_only": True,
            "production_state_touched": False,
        }


def _architecture_audit(branch: str, commit: str) -> dict[str, object]:
    audit = ROOT / "docs/ARCHITECTURE_AUDIT_V7.md"
    schema_manifest = ROOT / "schemas/v7/manifest.json"
    protocols = (
        "ModelProvider",
        "EmbeddingProvider",
        "ConversationStore",
        "CorpusStore",
        "RetrievalService",
        "ArtifactStore",
        "ProvenanceStore",
        "RepositoryService",
        "WorktreeService",
        "IdentityService",
        "CapabilityService",
        "JobService",
        "AuditService",
        "ConfigurationService",
    )
    architecture_text = (ROOT / "src/research_workspace/architecture.py").read_text(
        encoding="utf-8"
    )
    base_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CERTIFIED_BASE, "HEAD"],
        cwd=ROOT,
        check=False,
        timeout=30,
    ).returncode == 0
    missing = [name for name in protocols if f"class {name}" not in architecture_text]
    schemas: object = json.loads(schema_manifest.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if audit.is_file() and schema_manifest.is_file() and base_check and not missing
            else "FAIL"
        ),
        "certified_base": CERTIFIED_BASE,
        "base_is_ancestor": base_check,
        "implementation_branch": branch,
        "implementation_commit": commit,
        "architecture_audit_sha256": sha256_file(audit),
        "provider_and_store_protocols": list(protocols),
        "missing_protocols": missing,
        "schema_manifest": schemas,
        "gpu_or_provider_probe_performed": False,
    }


def _load_result(path: Path, fallback_category: str) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = None
    if not isinstance(raw, dict):
        return {"schema_version": 1, "status": "FAIL", "category": fallback_category}
    return raw


def _report(
    *,
    status: str,
    branch: str,
    commit: str,
    stable_clean: bool,
    counts: dict[str, int],
    archive_sha256: str,
    output_name: str,
) -> str:
    return f"""# Laplace v7 architecture release certification

CPU/fixture certification: {status}
GPU/live-model certification: {GPU_STATUS}
production state touched: false
production model processes touched: false
stable checkout clean: {str(stable_clean).lower()}
implementation branch and commit: {branch} {commit}
certified base: {CERTIFIED_BASE}
test counts: {json.dumps(counts, sort_keys=True)}
archive: {output_name}/certification.tar.gz
archive SHA-256: {archive_sha256}

## Known limitations

- Fixture task quality is not live-model quality.
- Offline dependency audit does not query a current vulnerability database.
- Backup encryption, SSH/HTTPS sync transport, Windows runners and live provider
  behavior require their deployment/CI environments.
- The adjacent final report is excluded from the evidence archive so it can contain
  the archive's non-self-referential SHA-256. Every archived evidence hash is verified.

## Deferred live tests

- exact served-model identity and readiness for every configured route;
- live generation, streaming, tool/structured-output and timeout behavior;
- selected 4B Q4 model at 8k operational context on the NVIDIA GPU;
- measured VRAM, latency, tokens/s, concurrency and quality;
- model lifecycle ownership, safe start and safe shutdown;
- optional vision unload/restore behavior and optional Ryzen AI auxiliary path.

No GPU, model endpoint, listening-port or process query was executed by this
certification.

## Future separately authorized live command

```bash
PYTHONPATH=src .venv/bin/python scripts/run_live_production_gpu_certification.py \\
  --output-root /explicit/new/live-certification-output
```
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs",
        help="Parent directory; a timestamped certification directory is created below it.",
    )
    arguments = parser.parse_args(argv)
    output_parent = arguments.output_root.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / f"architecture_release_v7_certification_{_timestamp()}"
    output.mkdir(mode=0o700)
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "LAPLACE_FIXTURE_ONLY": "1",
        "NO_PROXY": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    branch = _git(ROOT, "branch", "--show-current")
    commit = _git(ROOT, "rev-parse", "HEAD")
    stable_before = _stable_snapshot()

    commands = [
        _command(
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"],
            environment=environment,
            timeout=180,
        ),
        _command(
            "core_pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "--ignore=tests/test_operator_gui_e2e.py",
                "--ignore=tests/test_migrations_v7.py",
            ],
            environment=environment,
            timeout=900,
        ),
        _command(
            "browser_fixture_pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "tests/test_operator_gui_e2e.py",
            ],
            environment=environment,
            timeout=300,
        ),
        _command(
            "migration_pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "tests/test_migrations_v7.py",
            ],
            environment=environment,
            timeout=300,
        ),
        _command(
            "ruff",
            [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
            environment=environment,
            timeout=300,
        ),
        _command(
            "strict_mypy",
            [sys.executable, "-m", "mypy", "--strict", "src/research_workspace"],
            environment=environment,
            timeout=600,
        ),
        _command(
            "bandit",
            [
                sys.executable,
                "-m",
                "bandit",
                "-q",
                "-lll",
                "-r",
                "src/research_workspace",
            ],
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
    counts = _pytest_counts(commands)
    _write_json(
        output / "test_results.json",
        {
            "schema_version": 1,
            "status": (
                "PASS" if all(item["status"] == "PASS" for item in commands) else "FAIL"
            ),
            "counts": counts,
            "commands": commands,
            "fixture_only": True,
            "external_network_used": False,
            "gpu_commands_executed": False,
        },
    )

    with tempfile.TemporaryDirectory(prefix="laplace-v7-certification-") as temporary_name:
        temporary = Path(temporary_name)
        migration = _migration_evidence(temporary)
        _write_json(output / "migration_results.json", _sanitized(migration, temporary))

        package_dir = temporary / "package"
        package_command = _command(
            "reproducible_package",
            [
                sys.executable,
                "scripts/package_release.py",
                "--output-dir",
                str(package_dir),
            ],
            environment=environment,
            timeout=900,
        )
        package = _load_result(package_dir / "package_results.json", "package_result_missing")
        package["certification_command"] = package_command
        _write_json(output / "package_results.json", _sanitized(package, temporary))

        generated_commands = {
            "offline_evaluation.json": [
                sys.executable,
                "scripts/run_offline_evaluation.py",
                "--output",
                str(output / "offline_evaluation.json"),
            ],
            "soak_results.json": [
                sys.executable,
                "scripts/run_cpu_soak.py",
                "--iterations",
                "32",
                "--max-seconds",
                "30",
                "--output",
                str(output / "soak_results.json"),
            ],
            "failure_matrix.json": [
                sys.executable,
                "scripts/run_failure_matrix.py",
                "--max-seconds",
                "30",
                "--output",
                str(output / "failure_matrix.json"),
            ],
            "security_results.json": [
                sys.executable,
                "scripts/run_security_checks.py",
                "--output",
                str(output / "security_results.json"),
            ],
            "documentation_results.json": [
                sys.executable,
                "scripts/check_documentation.py",
                "--output",
                str(output / "documentation_results.json"),
            ],
            "ci_validation.json": [
                sys.executable,
                "scripts/validate_ci.py",
                "--output",
                str(output / "ci_validation.json"),
            ],
        }
        generated_results: dict[str, dict[str, object]] = {}
        for name, command in generated_commands.items():
            command_result = _command(
                name.removesuffix(".json"),
                command,
                environment=environment,
                timeout=600,
            )
            loaded = _load_result(output / name, f"{name}_missing")
            loaded["certification_command"] = command_result
            sanitized_loaded = _sanitized(loaded, temporary)
            assert isinstance(sanitized_loaded, dict)
            generated_results[name] = sanitized_loaded
            _write_json(output / name, sanitized_loaded)

    dependencies = dependency_inventory(ROOT / "requirements.lock")
    licenses = license_inventory(ROOT / "requirements.lock")
    _write_json(output / "dependency_inventory.json", _sanitized(dependencies))
    _write_json(output / "license_inventory.json", _sanitized(licenses))
    architecture = _architecture_audit(branch, commit)
    _write_json(output / "architecture_audit.json", architecture)

    stable_after = _stable_snapshot()
    stable_clean = bool(
        stable_before.get("clean")
        and stable_after.get("clean")
        and stable_before.get("commit") == stable_after.get("commit")
        and stable_before.get("branch") == stable_after.get("branch")
    )
    implementation_status = _git(ROOT, "status", "--short")
    machine = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "certified_base": CERTIFIED_BASE,
        "implementation_branch": branch,
        "implementation_commit": commit,
        "implementation_clean": not bool(implementation_status),
        "python": platform.python_version(),
        "platform": platform.system(),
        "stable_before": stable_before,
        "stable_after": stable_after,
        "stable_checkout_clean_unchanged": stable_clean,
        "production_state_touched": False,
        "production_model_processes_touched": False,
        "gpu_status": GPU_STATUS,
        "gpu_queries_executed": False,
        "model_endpoint_queries_executed": False,
        "process_queries_executed": False,
    }
    _write_json(output / "machine_summary.json", machine)
    (output / "git_status.txt").write_text(
        (
            f"implementation branch: {branch}\n"
            f"implementation commit: {commit}\n"
            f"implementation clean: {str(not bool(implementation_status)).lower()}\n"
            f"stable branch: {stable_after.get('branch', 'unavailable')}\n"
            f"stable commit: {stable_after.get('commit', 'unavailable')}\n"
            f"stable clean: {str(stable_clean).lower()}\n"
            "production state touched: false\n"
            "production model processes touched: false\n"
        ),
        encoding="utf-8",
    )

    structured_statuses = [
        _load_result(output / name, "missing").get("status")
        for name in EVIDENCE_NAMES
        if name.endswith(".json")
        and name not in {"machine_summary.json", "license_inventory.json"}
    ]
    cpu_status: Literal["PASS", "FAIL"] = (
        "PASS"
        if all(status == "PASS" for status in structured_statuses)
        and dependencies["status"] == "PASS"
        and architecture["status"] == "PASS"
        and stable_clean
        and not implementation_status
        else "FAIL"
    )
    archive = output / "certification.tar.gz"
    archive_sha256 = create_archive(output, EVIDENCE_NAMES, archive)
    archive_verification = verify_archive(archive)
    if archive_verification["status"] != "PASS":
        cpu_status = "FAIL"
    report = _report(
        status=cpu_status,
        branch=branch,
        commit=commit,
        stable_clean=stable_clean,
        counts=counts,
        archive_sha256=archive_sha256,
        output_name=output.name,
    )
    (output / "final_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": cpu_status,
                "output": str(output),
                "archive_sha256": archive_sha256,
                "archive_verification": archive_verification,
                "gpu_status": GPU_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if cpu_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

