#!/usr/bin/env python3
"""Run bounded CPU-only concurrency and resource-pressure fixture scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from research_workspace.reliability import run_cpu_soak, write_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = run_cpu_soak(
        iterations=arguments.iterations,
        max_seconds=arguments.max_seconds,
    )
    print(write_report(report, arguments.output))
    return 0 if report.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

