#!/usr/bin/env python3
"""Score paired Phase-B agent evidence; this is not used for Phase A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_workspace.upstream_ab import compare_trials, load_trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-tasks", type=int, default=10)
    parser.add_argument("--efficiency-margin", type=float, default=0.05)
    args = parser.parse_args()
    result = compare_trials(
        load_trials(args.baseline),
        load_trials(args.candidate),
        minimum_tasks=args.minimum_tasks,
        efficiency_margin=args.efficiency_margin,
    )
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
