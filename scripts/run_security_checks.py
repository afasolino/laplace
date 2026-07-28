#!/usr/bin/env python3
"""Run offline Bandit, security fixtures, secret scan, and dependency checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from research_workspace.inventory import dependency_inventory

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".venv-vllm",
    ".models",
    ".tools",
    ".runtime",
    "outputs",
    "__pycache__",
}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    "openai_style_key": re.compile(rb"\bsk-[A-Za-z0-9]{32,}\b"),
}


def secret_scan() -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        for category, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                findings.append(
                    {"path": path.relative_to(ROOT).as_posix(), "category": category}
                )
    return findings


def _run(command: list[str], timeout: int) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "PATH": (
                str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ.get("PATH", "")
            ),
            "PYTHONPATH": "src",
        },
    )
    output = (completed.stdout + completed.stderr).replace(str(ROOT), "<repository>")
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "output": output[-8000:],
    }


def run_checks() -> dict[str, object]:
    bandit = _run(
        [sys.executable, "-m", "bandit", "-q", "-lll", "-r", "src/research_workspace"],
        120,
    )
    security_tests = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_security_fuzz_v7.py",
            "tests/test_auth_provenance_security.py",
            "tests/test_providers_v7.py",
            "tests/test_sync_v7.py",
        ],
        180,
    )
    secrets = secret_scan()
    dependencies = dependency_inventory(ROOT / "requirements.lock")
    status = (
        "PASS"
        if bandit["status"] == "PASS"
        and security_tests["status"] == "PASS"
        and not secrets
        and dependencies["status"] == "PASS"
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "seed": 7003,
        "bandit": bandit,
        "security_tests": security_tests,
        "secret_scan": {"status": "PASS" if not secrets else "FAIL", "findings": secrets},
        "dependency_audit": dependencies,
        "external_network_used": False,
        "gpu_commands_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = run_checks()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
