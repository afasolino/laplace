"""Command-line interface for explicit, identity-bound state migrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .migrations import (
    MigrationError,
    migrate,
    preflight,
    recover_interrupted,
    rollback_to_backup,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-migrate")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--apply", action="store_true")
    actions.add_argument("--recover", action="store_true")
    actions.add_argument("--rollback-backup-id")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the explicitly identified apply/rollback operation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.check:
            result = preflight(arguments.state_root).public()
        elif arguments.dry_run:
            result = migrate(
                arguments.state_root,
                expected_state_id=arguments.state_id,
                dry_run=True,
            )
        elif arguments.recover:
            result = recover_interrupted(
                arguments.state_root,
                expected_state_id=arguments.state_id,
            )
        elif arguments.rollback_backup_id:
            if not arguments.yes:
                raise MigrationError("rollback_requires_yes")
            result = rollback_to_backup(
                arguments.state_root,
                expected_state_id=arguments.state_id,
                backup_id=arguments.rollback_backup_id,
            )
        else:
            if not arguments.yes:
                raise MigrationError("migration_apply_requires_yes")
            result = migrate(
                arguments.state_root,
                expected_state_id=arguments.state_id,
                dry_run=False,
            )
    except MigrationError as exc:
        print(
            json.dumps(
                {"schema_version": 1, "status": "FAIL", "category": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
