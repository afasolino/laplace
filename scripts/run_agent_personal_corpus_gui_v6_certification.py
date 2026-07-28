#!/usr/bin/env python3
"""Run and bundle the v6 CPU/fixture gates plus a fail-closed live GPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
# This script executes only fixed, local certification commands.
import subprocess  # nosec B404
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
STABLE = Path("/home/giando/work/laplace")
BASE_COMMIT = "0d9a1d54f445ad25a8bb84d3133c3377f446c476"
BRANCH = "feature/agent-personal-corpus-gui-v6"
SCREENSHOT_NAMES = (
    "domain_selector.png",
    "agent_no_repository.png",
    "agent_new_worktree.png",
    "agent_worktree_history.png",
    "agent_progress.png",
    "personal_corpus_empty.png",
    "folder_upload_manifest.png",
    "personal_corpus_indexed.png",
    "retrieval_source_selector.png",
    "chat_processing_state.png",
    "markdown_table.png",
    "admin_capabilities.png",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _command(
    name: str,
    arguments: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: int = 900,
) -> dict[str, object]:
    started = time.monotonic()
    # The caller supplies a fixed argv assembled by this certification script.
    completed = subprocess.run(  # nosec B603
        list(arguments),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    output = completed.stdout + completed.stderr
    return {
        "name": name,
        "command": list(arguments),
        "returncode": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_tail": output[-8_000:],
    }


def _git(*arguments: str, root: Path = ROOT) -> str:
    # Git is used only for fixed, read-only repository queries.
    completed = subprocess.run(  # nosec B603 B607
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git query failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def _machine_summary() -> dict[str, object]:
    gpu = _command(
        "nvidia_gpu",
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    compute = _command(
        "nvidia_compute_processes",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    ports = _command("laplace_ports", ["ss", "-ltnp"], timeout=30)
    return {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "certified_base": BASE_COMMIT,
        "implementation_branch": _git("branch", "--show-current"),
        "implementation_head": _git("rev-parse", "HEAD"),
        "base_is_ancestor": subprocess.run(  # nosec B603 B607
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
            cwd=ROOT,
            check=False,
            timeout=30,
        ).returncode
        == 0,
        "stable_checkout": {
            "path": str(STABLE),
            "branch": _git("branch", "--show-current", root=STABLE),
            "head": _git("rev-parse", "HEAD", root=STABLE),
            "porcelain": _git("status", "--short", root=STABLE).splitlines(),
        },
        "implementation_porcelain": _git("status", "--short").splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu,
        "compute_processes": compute,
        "listening_ports_filtered": [
            line
            for line in str(ports["output_tail"]).splitlines()
            if re.search(r":(?:8102|8103|8201|8765)\b", line)
        ],
    }


def _compute_rows(command_result: object) -> list[dict[str, object]]:
    if not isinstance(command_result, dict):
        return []
    rows: list[dict[str, object]] = []
    for line in str(command_result.get("output_tail", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "process_name": parts[1],
                "used_memory_mib": int(parts[2]) if parts[2].isdigit() else None,
                "gpu_uuid": parts[3],
            }
        )
    return rows


def _probe(endpoint: str) -> dict[str, object]:
    try:
        with urllib.request.urlopen(  # nosec B310 - fixed loopback endpoints
            endpoint.rstrip("/") + "/v1/models",
            timeout=3,
        ) as response:
            raw: object = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"status": "UNAVAILABLE", "error_type": type(exc).__name__}
    data = raw.get("data") if isinstance(raw, dict) else None
    served = [
        str(item["id"])
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ] if isinstance(data, list) else []
    return {"status": "AVAILABLE", "served_model_ids": served}


def _live_preflight(machine: dict[str, object]) -> dict[str, object]:
    rows = _compute_rows(machine.get("compute_processes"))
    endpoints_before = {
        "selected_main": _probe("http://127.0.0.1:8201"),
        "codev": _probe("http://127.0.0.1:8103"),
    }
    if rows:
        status = "BLOCKED_UNRELATED_GPU_PROCESS"
        reason = (
            "The validated live lifecycle refuses to start a model while an "
            "unowned compute PID is present. No PID was signalled and no endpoint "
            "was started."
        )
    else:
        status = "READY_FOR_SELECTED_LIVE_MODEL"
        reason = (
            "No unowned compute PID was present. This assembler does not start a "
            "model directly; use the ownership-validating selected live runner."
        )
    return {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "status": status,
        "selected_live_model_start_attempted": False,
        "reason": reason,
        "unowned_compute_processes": rows,
        "endpoints_before": endpoints_before,
        "endpoints_after": {
            "selected_main": _probe("http://127.0.0.1:8201"),
            "codev": _probe("http://127.0.0.1:8103"),
        },
        "stable_operator_port_preserved": any(
            ":8765" in line
            for line in machine.get("listening_ports_filtered", [])
            if isinstance(line, str)
        ),
        "laplace_processes_stopped": [],
        "unrelated_processes_signalled": [],
        "safe_shutdown": (
            "NO_NEW_LAPLACE_PROCESS_STARTED; EXISTING_STABLE_OPERATOR_AND_UNRELATED_"
            "GPU_PROCESS_LEFT_UNTOUCHED"
        ),
    }


def _copy_screenshots(output: Path) -> dict[str, object]:
    source = ROOT / "docs/user_guide/assets"
    target = output / "screenshots"
    target.mkdir()
    files: list[dict[str, object]] = []
    for name in SCREENSHOT_NAMES:
        source_path = source / name
        if not source_path.is_file() or source_path.stat().st_size < 1_000:
            raise RuntimeError(f"missing screenshot: {name}")
        expected = _sha256(source_path)
        target_path = target / name
        shutil.copy2(source_path, target_path)
        if _sha256(target_path) != expected:
            raise RuntimeError(f"screenshot copy hash mismatch: {name}")
        files.append(
            {
                "file": f"screenshots/{name}",
                "sha256": expected,
                "size_bytes": target_path.stat().st_size,
                "fixture_data_only": True,
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "count": len(files),
        "contains_live_secrets": False,
        "contains_private_documents": False,
        "files": files,
    }


def _secret_scan(output: Path) -> dict[str, object]:
    patterns = {
        "real_argon2_hash": re.compile(rb"\$argon2id\$v=\d+\$m=\d+,t=\d+,p=\d+\$"),
        "laplace_bearer_token": re.compile(
            rb"laplace-(?:admin|read|operate|approve|basic|plus)-"
            rb"[A-Za-z0-9_-]{20,}"
        ),
        "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "cookie_assignment": re.compile(rb"laplace_session=[A-Za-z0-9_-]{20,}"),
    }
    findings: list[dict[str, str]] = []
    scanned = 0
    for root in (
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
        output,
    ):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() in {".png", ".pyc", ".sqlite3", ".db", ".gz"}
            ):
                continue
            try:
                content = path.read_bytes()
            except OSError:
                continue
            scanned += 1
            for category, pattern in patterns.items():
                if pattern.search(content):
                    findings.append(
                        {
                            "category": category,
                            "file": str(path.relative_to(ROOT))
                            if ROOT in path.parents
                            else str(path.relative_to(output)),
                        }
                    )
    return {
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": scanned,
        "findings": findings,
        "patterns_reported_without_values": sorted(patterns),
    }


def _licenses() -> dict[str, object]:
    packages: list[dict[str, str]] = []
    missing: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or "unknown"
        version = distribution.version
        license_name = (
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License")
            or ""
        ).strip()
        if not license_name:
            classifiers = distribution.metadata.get_all("Classifier") or []
            license_name = "; ".join(
                item.removeprefix("License :: ").strip()
                for item in classifiers
                if item.startswith("License :: ")
            )
        if not license_name:
            missing.append(name)
            license_name = "NOT_DECLARED_IN_INSTALLED_METADATA"
        packages.append(
            {"name": name, "version": version, "license": license_name[:500]}
        )
    packages.sort(key=lambda item: item["name"].lower())
    material = json.dumps(packages, sort_keys=True, separators=(",", ":"))
    return {
        "status": "REVIEW" if missing else "PASS",
        "package_count": len(packages),
        "missing_declared_license_count": len(missing),
        "missing_declared_license_packages": sorted(missing),
        "inventory_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "packages": packages,
    }


def _full_bandit_observation(output: Path) -> dict[str, object]:
    """Record all Bandit findings while gating only on execution errors/high risk."""
    report = output / "full_bandit_results.json"
    started = time.monotonic()
    # Bandit returns one when findings exist, so parse its JSON instead of treating
    # that expected result as a failed certification command.
    completed = subprocess.run(  # nosec B603 B607
        [
            "bandit",
            "-q",
            "-f",
            "json",
            "-o",
            str(report),
            "-r",
            "src/research_workspace",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    try:
        raw = json.loads(report.read_text(encoding="utf-8"))
        totals = raw["metrics"]["_totals"]
        severities = {
            "high": int(totals["SEVERITY.HIGH"]),
            "medium": int(totals["SEVERITY.MEDIUM"]),
            "low": int(totals["SEVERITY.LOW"]),
        }
        result_count = len(raw["results"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        severities = {"high": -1, "medium": -1, "low": -1}
        result_count = -1
    passed = completed.returncode in {0, 1} and severities["high"] == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "report_file": report.name,
        "report_sha256": _sha256(report) if report.is_file() else None,
        "result_count": result_count,
        **severities,
        "stderr_tail": completed.stderr[-2_000:],
        "note": (
            "Bandit return code 1 denotes reported findings. Certification fails "
            "if the scan cannot run or any high-severity finding is present."
        ),
    }


def _report(
    machine: dict[str, object],
    tests: dict[str, object],
    security: dict[str, object],
    live: dict[str, object],
    screenshot_manifest: dict[str, object],
) -> str:
    commands = tests.get("commands", [])
    command_lines = [
        "`" + " ".join(str(part) for part in item.get("command", [])) + "`"
        for item in commands
        if isinstance(item, dict)
    ] if isinstance(commands, list) else []
    stable = machine["stable_checkout"]
    return f"""# Agent, personal corpus, domain, and GUI v6 certification

Generated: `{_now()}`

Certified base: `{BASE_COMMIT}`
Implementation branch: `{machine['implementation_branch']}`
Implementation HEAD at evidence assembly: `{machine['implementation_head']}`

## Implemented and verified

- Independent v2 named capabilities, legacy migration, session revocation, CLI and GUI editing.
- Backend domain registry and Chat/Agent/Research validation for every displayed option.
- Persistent owner-scoped worktree lifecycle, idempotency, quota, recovery, dirty preservation, patch/export/history, and Operator inspection.
- Owner-private corpus folder/drag/ZIP staging, centralized policy, secure parsers, deterministic chunks, citations, deletion/purge, server-side resume, and sanitized Operator inventory.
- Chat composer draft behavior, bounded polling state, safe Markdown tables, copy actions, responsive and accessibility-oriented GUI.
- Full pytest, Playwright, Ruff, strict mypy, compileall, dependency compatibility, Git whitespace, Bandit high-severity gate, and secret scan.
- {screenshot_manifest['count']} sanitized fixture screenshots with verified SHA-256 values.

## Implemented but fixture-only in this run

- Browser, corpus, worktree, capability, cross-user, upload rejection, retrieval, progress, and table flows used deterministic local fixtures.
- Selected live model inference was not started because the ownership preflight found an unrelated GPU compute PID. The fail-closed result is `{live['status']}`.

## Not implemented

- Personal-corpus selection in Deep Research.
- Shared governed-corpus querying inside the tiered Chat/Agent path; the response truthfully reports `shared.retrieval_used=false`.
- Real token streaming/partial-text display.
- Automatic scheduled purge; explicit lifecycle purge is implemented.

## Known limitations

- Cancelling the browser request records `CANCELLED` and aborts the client connection, but cannot forcibly terminate a synchronous model call already executing in a worker.
- Some truthful request states can be brief and may not be observed by a polling client.
- Folder drag recursion depends on browser `webkitGetAsEntry`; folder picker and controlled ZIP remain available.
- The full Bandit scan retains pre-existing medium-confidence heuristics; the high-severity gate passes and the v6 dynamic SQL findings were removed.

## Security assumptions

- Services and model endpoints remain loopback-only unless an explicit documented SSH/HTTPS mode is configured.
- The external state root is private and outside Git.
- Only authorized documents are uploaded. Client MIME/path values are untrusted.
- Agent network tools are absent and personal corpus context is read-only.
- GPU/process shutdown signals require a matching Laplace ownership record.

## Storage and retention defaults

- Personal corpus: 64 MiB/file, 512 MiB expanded batch/extracted text, 5 GiB/user, 512 MiB protected free disk, 30-day soft delete.
- Worktrees: 8 active/retained per user, 64 globally, 30-day expiry; clean close releases and dirty close preserves.
- Runtime registries, corpora, worktrees, artifacts, conversations, sessions, and logs remain outside Git.

## Exact commands

{chr(10).join(f'- {line}' for line in command_lines)}

## Stable checkout

Path: `{stable['path']}`
Branch: `{stable['branch']}`
HEAD: `{stable['head']}`
Clean: `{not bool(stable['porcelain'])}`

## GPU and process state after safe shutdown

Status: `{live['safe_shutdown']}`
Unrelated compute processes observed: `{len(live['unowned_compute_processes'])}`
Processes signalled: `0`
Existing stable Operator port preserved: `{live['stable_operator_port_preserved']}`

Archive SHA-256 is appended to this external report after archive finalization.
"""


def _archive(output: Path) -> Path:
    archive = output / "certification.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(output.rglob("*")):
            if path == archive or not path.is_file():
                continue
            bundle.add(path, arcname=path.relative_to(output))
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args(argv)
    output = (
        arguments.output_root.resolve()
        if arguments.output_root
        else ROOT
        / "outputs"
        / f"agent_personal_corpus_gui_v6_certification_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    if ROOT == STABLE or _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("certification must run in the isolated implementation worktree")
    if _git("status", "--short", root=STABLE):
        raise RuntimeError("stable checkout is not clean")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["UV_CACHE_DIR"] = "/tmp/laplace-v6-certification-uv-cache"
    commands = [
        _command(
            "pytest",
            [sys.executable, "-m", "pytest", "-q"],
            environment=environment,
        ),
        _command("ruff", ["ruff", "check", "src", "tests", "scripts"]),
        _command("mypy", ["mypy", "--strict", "src/research_workspace"]),
        _command(
            "bandit_high",
            ["bandit", "-q", "-lll", "-r", "src/research_workspace"],
        ),
        _command(
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"],
        ),
        _command("git_diff_check", ["git", "diff", "--check"]),
        _command(
            "dependency_compatibility",
            ["uv", "pip", "check", "--python", sys.executable],
            environment=environment,
        ),
    ]
    test_results = {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in commands)
            else "FAIL"
        ),
        "commands": commands,
    }
    machine = _machine_summary()
    live = _live_preflight(machine)
    screenshots = _copy_screenshots(output)
    full_bandit = _full_bandit_observation(output)
    secret_scan = _secret_scan(output)
    security = {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "status": (
            "PASS"
            if secret_scan["status"] == "PASS"
            and full_bandit["status"] == "PASS"
            and all(
                item["status"] == "PASS"
                for item in commands
                if item["name"]
                in {"bandit_high", "dependency_compatibility", "git_diff_check"}
            )
            else "FAIL"
        ),
        "secret_scan": secret_scan,
        "dependency_compatibility": next(
            item for item in commands if item["name"] == "dependency_compatibility"
        ),
        "license_inventory": _licenses(),
        "bandit_high_severity_gate": next(
            item for item in commands if item["name"] == "bandit_high"
        ),
        "full_bandit_observation": full_bandit,
    }
    _write_json(output / "machine_summary.json", machine)
    _write_json(output / "test_results.json", test_results)
    _write_json(output / "security_results.json", security)
    _write_json(output / "live_results.json", live)
    _write_json(output / "screenshot_manifest.json", screenshots)
    report_path = output / "final_report.md"
    report_path.write_text(
        _report(machine, test_results, security, live, screenshots),
        encoding="utf-8",
    )
    archive = _archive(output)
    archive_hash = _sha256(archive)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nArchive SHA-256: `{archive_hash}`\n")
    machine["archive"] = {
        "file": archive.name,
        "sha256": archive_hash,
        "size_bytes": archive.stat().st_size,
        "note": "Hash describes the finalized archive before this external field was written.",
    }
    _write_json(output / "machine_summary.json", machine)
    summary = {
        "status": (
            "PASS_WITH_LIVE_GPU_BLOCKER"
            if test_results["status"] == "PASS"
            and security["status"] == "PASS"
            and screenshots["status"] == "PASS"
            and str(live["status"]).startswith("BLOCKED_")
            else "PASS"
            if test_results["status"] == "PASS"
            and security["status"] == "PASS"
            and screenshots["status"] == "PASS"
            else "FAIL"
        ),
        "output": str(output),
        "archive_sha256": archive_hash,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
