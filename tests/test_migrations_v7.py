from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from research_workspace.migration_cli import main as migration_main
from research_workspace.migrations import (
    MIGRATION_JOURNAL,
    MIGRATION_LOCK,
    STATE_MANIFEST,
    MigrationError,
    create_synthetic_v0_state,
    migrate,
    preflight,
    recover_interrupted,
    rollback_to_backup,
)


def _state(tmp_path: Path, name: str = "state") -> Path:
    root = tmp_path / name
    create_synthetic_v0_state(root)
    return root


def test_preflight_and_dry_run_do_not_change_fixture_state(tmp_path: Path) -> None:
    root = _state(tmp_path)
    before = (root / STATE_MANIFEST).read_bytes()
    checked = preflight(root)
    assert checked.status == "PASS"
    assert len(checked.stores) == 11
    assert {item["action"] for item in checked.stores} == {"MIGRATE"}
    result = migrate(
        root,
        expected_state_id="fixture-v7-old",
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN_PASS"
    assert result["stores_changed"] == 11
    assert result["backup_created"] is False
    assert (root / STATE_MANIFEST).read_bytes() == before
    assert not (root / MIGRATION_LOCK).exists()
    assert not (root / MIGRATION_JOURNAL).exists()


def test_migration_backs_up_preserves_records_and_passes_integrity(tmp_path: Path) -> None:
    root = _state(tmp_path)
    result = migrate(
        root,
        expected_state_id="fixture-v7-old",
        dry_run=False,
    )
    assert result["status"] == "PASS"
    assert result["stores_changed"] == 11
    assert result["integrity"] == "PASS"
    backup = tmp_path / "state.migration-backups" / str(result["backup_id"])
    assert (backup / "backup_manifest.json").is_file()
    checked = preflight(root)
    assert {item["action"] for item in checked.stores} == {"NO_CHANGE"}
    manifest = json.loads((root / STATE_MANIFEST).read_text(encoding="utf-8"))
    assert {item["schema_version"] for item in manifest["stores"]} == {1}
    with sqlite3.connect(root / "stores/project.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute(
            "SELECT value FROM legacy_records WHERE record_id='project-record'"
        ).fetchone()[0] == "synthetic"
    events = [
        json.loads(line)
        for line in (root / "migration_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "STATE_MIGRATION_COMPLETED"
    assert not (root / MIGRATION_LOCK).exists()


def test_interrupted_migration_rolls_back_atomically(tmp_path: Path) -> None:
    root = _state(tmp_path)
    before = (root / STATE_MANIFEST).read_bytes()
    with pytest.raises(MigrationError, match="migration_failed_rolled_back"):
        migrate(
            root,
            expected_state_id="fixture-v7-old",
            dry_run=False,
            fail_after=2,
        )
    assert (root / STATE_MANIFEST).read_bytes() == before
    assert preflight(root).status == "PASS"
    journal = json.loads((root / MIGRATION_JOURNAL).read_text(encoding="utf-8"))
    assert journal["status"] == "ROLLED_BACK"
    assert not (root / MIGRATION_LOCK).exists()


def test_restart_recovery_restores_prepared_backup(tmp_path: Path) -> None:
    root = _state(tmp_path)
    result = migrate(
        root,
        expected_state_id="fixture-v7-old",
        dry_run=False,
    )
    backup_id = str(result["backup_id"])
    project = root / "stores/project.sqlite3"
    project.write_bytes(b"interrupted")
    (root / MIGRATION_LOCK).write_text("{}\n", encoding="utf-8")
    (root / MIGRATION_JOURNAL).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_id": "fixture-v7-old",
                "status": "APPLYING",
                "backup": f"state.migration-backups/{backup_id}",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recovered = recover_interrupted(root, expected_state_id="fixture-v7-old")
    assert recovered["status"] == "RECOVERED"
    assert preflight(root).status == "PASS"
    assert not (root / MIGRATION_LOCK).exists()
    restored_manifest = json.loads((root / STATE_MANIFEST).read_text(encoding="utf-8"))
    assert {item["schema_version"] for item in restored_manifest["stores"]} == {0}


def test_explicit_rollback_is_identity_and_backup_bound(tmp_path: Path) -> None:
    root = _state(tmp_path)
    result = migrate(
        root,
        expected_state_id="fixture-v7-old",
        dry_run=False,
    )
    rollback = rollback_to_backup(
        root,
        expected_state_id="fixture-v7-old",
        backup_id=str(result["backup_id"]),
    )
    assert rollback["status"] == "ROLLED_BACK"
    checked = preflight(root)
    assert {item["action"] for item in checked.stores} == {"MIGRATE"}
    with pytest.raises(MigrationError, match="backup_id_invalid"):
        rollback_to_backup(
            root,
            expected_state_id="fixture-v7-old",
            backup_id="../escape",
        )


def test_corruption_hash_permissions_symlinks_and_lock_fail_closed(
    tmp_path: Path,
) -> None:
    root = _state(tmp_path)
    registry = root / "stores/registry.json"
    registry.write_text('{"schema_version": 0, "tampered": true}\n', encoding="utf-8")
    with pytest.raises(MigrationError, match="store_hash_mismatch"):
        preflight(root)

    locked = _state(tmp_path, "locked")
    (locked / MIGRATION_LOCK).write_text("{}\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="migration_locked"):
        migrate(locked, expected_state_id="fixture-v7-old", dry_run=False)

    linked = _state(tmp_path, "linked")
    original = linked / "stores/registry.json"
    replacement = linked / "stores/replacement.json"
    replacement.write_bytes(original.read_bytes())
    original.unlink()
    try:
        original.symlink_to(replacement)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(MigrationError, match="store_symlink_rejected"):
        preflight(linked)

    if os.name != "nt":
        unsafe = _state(tmp_path, "unsafe")
        unsafe.chmod(0o755)
        with pytest.raises(MigrationError, match="state_root_permissions_unsafe"):
            preflight(unsafe)


def test_migration_cli_requires_confirmation_and_matches_state_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _state(tmp_path)
    assert migration_main(
        [
            "--state-root",
            str(root),
            "--state-id",
            "fixture-v7-old",
            "--apply",
        ]
    ) == 2
    assert "migration_apply_requires_yes" in capsys.readouterr().out
    assert migration_main(
        [
            "--state-root",
            str(root),
            "--state-id",
            "fixture-wrong",
            "--dry-run",
        ]
    ) == 2
    assert "state_identity_mismatch" in capsys.readouterr().out
    assert migration_main(
        [
            "--state-root",
            str(root),
            "--state-id",
            "fixture-v7-old",
            "--apply",
            "--yes",
        ]
    ) == 0
    assert '"status": "PASS"' in capsys.readouterr().out
