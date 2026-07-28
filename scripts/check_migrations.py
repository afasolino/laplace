#!/usr/bin/env python3
"""Run migration preflight and dry-run for one explicitly identified state root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research_workspace.migrations import MigrationError, migrate, preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = {
            "schema_version": 1,
            "status": "PASS",
            "preflight": preflight(arguments.state_root).public(),
            "dry_run": migrate(
                arguments.state_root,
                expected_state_id=arguments.state_id,
                dry_run=True,
            ),
        }
    except MigrationError as exc:
        result = {"schema_version": 1, "status": "FAIL", "category": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
