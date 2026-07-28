#!/usr/bin/env python3
"""Run the complete post-implementation deterministic certification checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    commands = [
        (
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        ),
        (
            "targeted_pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_tiered_serving.py",
                "tests/test_model_servers.py",
                "tests/test_operator_api.py",
                "tests/test_auth_provenance_security.py",
                "tests/test_production_robustness.py",
                "tests/test_operator_gui_e2e.py",
            ],
        ),
        ("full_pytest", [sys.executable, "-m", "pytest", "-q"]),
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
    results: list[dict[str, object]] = []
    for name, command in commands:
        started = datetime.now(UTC).isoformat()
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=7_200,
        )
        log = output / f"{name}.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        results.append(
            {
                "name": name,
                "command": command,
                "started_at_utc": started,
                "returncode": completed.returncode,
                "log": str(log),
            }
        )
    summary = {
        "status": "PASS" if all(item["returncode"] == 0 for item in results) else "FAIL",
        "results": results,
    }
    (output / "final_test_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "final_test_results.txt").write_text(
        "\n".join(
            f"{item['name']}: returncode={item['returncode']} log={item['log']}"
            for item in results
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output), "status": summary["status"]}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
