#!/usr/bin/env python3
"""Run fixture pytest and emit a sanitized GitHub annotation on failure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence


def _annotation(value: str) -> str:
    sanitized = value[-7_500:].replace("%", "%25")
    sanitized = sanitized.replace("\r", "%0D").replace("\n", "%0A")
    return sanitized


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytest_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    pytest_arguments = list(arguments.pytest_arguments)
    if pytest_arguments[:1] == ["--"]:
        pytest_arguments = pytest_arguments[1:]
    completed = subprocess.run(  # nosec B603 - fixed interpreter/module
        [sys.executable, "-m", "pytest", "-q", *pytest_arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=2_100,
    )
    output = completed.stdout + completed.stderr
    print(output, end="")
    if completed.returncode != 0:
        print(
            "::error title=Laplace fixture pytest failed::"
            + _annotation(output or f"pytest exit code {completed.returncode}")
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
