#!/usr/bin/env python3
"""Validate or materialize the single Qwen3.8 P8 selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from research_workspace.production_model import select


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["qwen38"])
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        selected = select(arguments.profile, arguments.repository_root, arguments.output)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "SELECTED", "path": str(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
