#!/usr/bin/env python3
"""Generate the offline v7 dependency inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research_workspace.inventory import dependency_inventory

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = dependency_inventory(ROOT / "requirements.lock")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

