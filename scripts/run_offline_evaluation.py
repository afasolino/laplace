#!/usr/bin/env python3
"""Run the frozen provider-independent v7 evaluation suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from research_workspace.evaluation import run_offline_evaluation

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmarks/evaluation/frozen_suite_v1.json",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "benchmarks/evaluation",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = run_offline_evaluation(arguments.suite, arguments.fixtures)
    rendered = report.model_dump_json(indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

