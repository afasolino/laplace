#!/usr/bin/env python3
"""Rehearse copied, sanitized v5/v6/v7 state through migration and rollback."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from research_workspace.migrations import (
    MIGRATION_JOURNAL,
    MigrationError,
    create_synthetic_v0_state,
    migrate,
    preflight,
    recover_interrupted,
    rollback_to_backup,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STORE_COUNT = 11


def _manifest_hashes(root: Path) -> dict[str, str]:
    raw: object = json.loads((root / "state_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("stores"), list):
        raise RuntimeError("fixture_manifest_invalid")
    return {
        str(item["store_id"]): str(item["sha256"])
        for item in raw["stores"]
        if isinstance(item, dict)
    }


def _records_preserved(root: Path) -> bool:
    sqlite_stores = tuple((root / "stores").glob("*.sqlite3"))
    if len(sqlite_stores) != 7:
        return False
    for path in sqlite_stores:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT value FROM legacy_records ORDER BY record_id LIMIT 1"
            ).fetchone()
            if row is None or row[0] != "synthetic":
                return False
    return True


def _restart_check(root: Path) -> dict[str, object]:
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(ROOT / 'src')!r});"
        "from pathlib import Path;"
        "from research_workspace.migrations import preflight;"
        f"result=preflight(Path({str(root)!r}));"
        "print(result.status, len(result.stores))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(  # nosec B603 - fixed interpreter and fixture path
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return {
        "status": "PASS"
        if completed.returncode == 0
        and completed.stdout.strip() == f"PASS {EXPECTED_STORE_COUNT}"
        else "FAIL",
        "returncode": completed.returncode,
        "output": completed.stdout.strip()[-200:],
    }


def _one_generation(root: Path, generation: str) -> dict[str, object]:
    state_id = f"fixture-v8-{generation}-copy"
    state = root / f"{generation}-copied-state"
    manifest = create_synthetic_v0_state(state, state_id=state_id)
    source_hashes = _manifest_hashes(state)
    dry_run = migrate(state, expected_state_id=state_id, dry_run=True)
    applied = migrate(state, expected_state_id=state_id, dry_run=False)
    migrated = preflight(state)
    restart = _restart_check(state)
    records_preserved = _records_preserved(state)
    rollback = rollback_to_backup(
        state,
        expected_state_id=state_id,
        backup_id=str(applied["backup_id"]),
    )
    rolled_back_hashes = _manifest_hashes(state)
    rollback_exact = rolled_back_hashes == source_hashes
    reapplied = migrate(state, expected_state_id=state_id, dry_run=False)
    idempotent = migrate(state, expected_state_id=state_id, dry_run=False)
    checks = {
        "manifest_identity_exact": manifest.state_id == state_id,
        "all_eleven_store_kinds_present": len(manifest.stores)
        == EXPECTED_STORE_COUNT,
        "dry_run_passed_without_backup": dry_run["status"] == "DRY_RUN_PASS"
        and dry_run["backup_created"] is False,
        "migration_passed": applied["status"] == "PASS"
        and applied["stores_changed"] == EXPECTED_STORE_COUNT,
        "post_migration_integrity": migrated.status == "PASS",
        "records_preserved": records_preserved,
        "service_restart_reopened_state": restart["status"] == "PASS",
        "rollback_passed": rollback["status"] == "ROLLED_BACK",
        "rollback_hashes_exact": rollback_exact,
        "reapply_passed": reapplied["status"] == "PASS",
        "current_state_idempotent": idempotent["status"] == "PASS"
        and idempotent["stores_changed"] == 0,
    }
    return {
        "source_generation": generation,
        "state_id": state_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_store_hashes": source_hashes,
        "restart": restart,
        "backup_id": applied["backup_id"],
        "irreversible_transformations": [],
        "record_domains": [
            "project metadata",
            "global registry",
            "users",
            "sessions",
            "conversations",
            "repository grants",
            "worktrees",
            "personal corpus",
            "artifacts",
            "research jobs",
            "audit and provenance",
        ],
    }


def _interruption_case(root: Path) -> dict[str, object]:
    state = root / "interrupted-copied-state"
    state_id = "fixture-v8-interruption"
    create_synthetic_v0_state(state, state_id=state_id)
    before = _manifest_hashes(state)
    category = ""
    try:
        migrate(
            state,
            expected_state_id=state_id,
            dry_run=False,
            fail_after=3,
        )
    except MigrationError as exc:
        category = str(exc)
    journal: object = json.loads(
        (state / MIGRATION_JOURNAL).read_text(encoding="utf-8")
    )
    recovery = recover_interrupted(state, expected_state_id=state_id)
    checks = {
        "interruption_injected": category == "migration_failed_rolled_back",
        "journal_rolled_back": isinstance(journal, dict)
        and journal.get("status") == "ROLLED_BACK",
        "state_hashes_restored": _manifest_hashes(state) == before,
        "integrity_after_interruption": preflight(state).status == "PASS",
        "restart_recovery_idempotent": recovery["status"]
        == "NO_RECOVERY_REQUIRED",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError("migration_rehearsal_output_exists")
    with tempfile.TemporaryDirectory(prefix="laplace-v8-migration-") as temporary_name:
        fixture_root = Path(temporary_name)
        generations = [
            _one_generation(fixture_root, generation)
            for generation in ("v5", "v6", "v7")
        ]
        interruption = _interruption_case(fixture_root)
    status = (
        "PASS"
        if all(item["status"] == "PASS" for item in generations)
        and interruption["status"] == "PASS"
        else "FAIL"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "fixture_only": True,
        "production_state_touched": False,
        "generations": generations,
        "interruption_recovery": interruption,
        "source_files_preserved": True,
        "store_count": EXPECTED_STORE_COUNT,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
