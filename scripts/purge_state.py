#!/usr/bin/env python3
"""Plan or execute retention purge against an explicit fixture state."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from research_workspace.governance import (
    PURGE_CONFIRMATION,
    GovernancePolicy,
    GovernanceStore,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-state", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    arguments = parser.parse_args(argv)
    store = GovernanceStore(
        arguments.fixture_state.resolve(),
        policy=GovernancePolicy(),
        namespace_secret=b"fixture-purge-state-secret",
    )
    plan = store.plan_purge()
    if not arguments.execute:
        print(plan.model_dump_json(indent=2))
        return 0
    if arguments.confirmation != PURGE_CONFIRMATION:
        parser.error(f"--confirmation must be exactly {PURGE_CONFIRMATION}")
    if any(candidate.requires_export for candidate in plan.candidates):
        parser.error("export-before-delete candidates require programmatic export receipts")
    count = store.tombstone(plan, confirmation=arguments.confirmation)
    print(f"tombstoned fixture assets: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

