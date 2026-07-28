#!/usr/bin/env python3
"""Inspect, dry-plan or explicitly export a desktop repository patch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from research_workspace.sync_client import RepositoryInspector
from research_workspace.sync_protocol import SyncError, confirmation_for


def _write_new(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SyncError("patch_export_target_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--action", choices=("inspect", "plan", "export"), required=True)
    parser.add_argument("--patch-output", type=Path)
    parser.add_argument("--confirmation")
    arguments = parser.parse_args(argv)
    inspector = RepositoryInspector()
    try:
        if arguments.action == "inspect":
            result: dict[str, object] = {
                "status": "PASS",
                "snapshot": inspector.snapshot(
                    arguments.repository,
                    logical_repository_id=arguments.repository_id,
                ).model_dump(mode="json"),
            }
        else:
            plan, patch = inspector.plan_upload(
                arguments.repository,
                logical_repository_id=arguments.repository_id,
            )
            result = {
                "status": "DRY_RUN_PASS",
                "plan": plan.model_dump(mode="json"),
                "confirmation": confirmation_for(plan.plan_id),
                "untracked_files_included": False,
                "network_contacted": False,
            }
            if arguments.action == "export":
                if arguments.patch_output is None:
                    raise SyncError("patch_output_required")
                if arguments.confirmation != confirmation_for(plan.plan_id):
                    raise SyncError("sync_confirmation_required")
                _write_new(arguments.patch_output.resolve(), patch)
                result["status"] = "EXPORTED"
                result["patch_output"] = arguments.patch_output.name
    except (OSError, ValueError, SyncError) as exc:
        print(json.dumps({"status": "FAIL", "category": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
