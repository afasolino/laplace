#!/usr/bin/env python3
"""Focused deterministic certification for the Operator-backed chat UI."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str], timeout: int = 600) -> dict[str, object]:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-5000:],
        "stderr_tail": completed.stderr[-5000:],
    }


def find_pytest_python() -> str | None:
    candidates = [
        sys.executable,
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / ".venv-vllm-cu129" / "bin" / "python"),
        shutil.which("python3"),
        shutil.which("python"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not Path(candidate).exists():
            continue
        seen.add(candidate)
        probe = subprocess.run(
            [
                candidate,
                "-c",
                "import fastapi, pydantic, pypdf, pytest; print(pytest.__version__)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if probe.returncode == 0:
            return candidate
    return None


def main() -> int:
    python = find_pytest_python()
    checks: list[dict[str, object]] = []
    if python is None:
        checks.append(
            {
                "argv": ["pytest-discovery"],
                "returncode": 2,
                "stdout_tail": "",
                "stderr_tail": "No existing repository Python environment with pytest was found.",
            }
        )
    else:
        checks.append(
            run(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_chat_operator_client_v2.py",
                    "tests/test_chat_session_v2.py",
                    "tests/test_chat_cli_v2.py",
                    "tests/test_laplace_chat_entrypoint.py",
                    "tests/test_operator_agent_conversation.py",
                ]
            )
        )
    checks.append(run(["git", "diff", "--check"], timeout=60))
    report = {
        "schema_version": 2,
        "component": "laplace_chat_operator_client",
        "status": "PASS" if all(item["returncode"] == 0 for item in checks) else "FAIL",
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
