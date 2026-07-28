#!/usr/bin/env python3
"""Print a sanitized storage summary for an explicit fixture governance state."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from research_workspace.governance import GovernancePolicy, GovernanceStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-state", required=True, type=Path)
    arguments = parser.parse_args(argv)
    store = GovernanceStore(
        arguments.fixture_state.resolve(),
        policy=GovernancePolicy(),
        namespace_secret=b"fixture-storage-report-secret",
    )
    print(store.summary().model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

