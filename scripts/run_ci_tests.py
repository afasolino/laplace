#!/usr/bin/env python3
"""Run fixture pytest and emit a sanitized GitHub annotation on failure."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Sequence


def _annotation(value: str) -> str:
    sanitized = value.replace("%", "%25")
    sanitized = sanitized.replace("\r", "%0D").replace("\n", "%0A")
    return sanitized


def _failure_annotations(output: str) -> list[str]:
    lines = [
        line
        for line in output.splitlines()
        if re.match(r"^(?:FAILED|ERROR) ", line)
        or " ERROR collecting " in line
    ]
    summary = "\n".join(lines) or output[-3_000:]
    return [
        summary[index : index + 3_000]
        for index in range(0, len(summary), 3_000)
    ]


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
        for index, annotation in enumerate(
            _failure_annotations(
                output or f"pytest exit code {completed.returncode}"
            ),
            start=1,
        ):
            print(
                f"::error title=Laplace fixture pytest failures {index}::"
                + _annotation(annotation)
            )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
