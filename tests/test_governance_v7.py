from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_workspace.governance import (
    BACKUP_CONFIRMATION,
    PURGE_CONFIRMATION,
    AssetCategory,
    AssetStatus,
    BackupEntry,
    BackupManifest,
    GovernanceError,
    GovernancePolicy,
    GovernanceStore,
)


def policy(**overrides: object) -> GovernancePolicy:
    values: dict[str, object] = {
        "per_user_bytes": 64,
        "global_bytes": 128,
        "minimum_free_bytes": 0,
        "retention_days": {category: 0 for category in AssetCategory},
    }
    values.update(overrides)
    return GovernancePolicy.model_validate(values)


class FixtureBackupProvider:
    provider_id = "fixture-copy"

    def export(self, source: Path, destination: Path, *, key_reference: str) -> str:
        assert key_reference == "fixture-key-ref"
        shutil.copytree(source, destination)
        return "fixture-export-receipt"

    def validate(self, destination: Path, *, key_reference: str) -> bool:
        return destination.is_dir() and key_reference == "fixture-key-ref"


def make_store(tmp_path: Path, *, configured: GovernancePolicy | None = None) -> GovernanceStore:
    return GovernanceStore(
        (tmp_path / "state").resolve(),
        policy=configured or policy(),
        namespace_secret=b"a-test-secret-with-enough-bytes",
    )


def test_quota_pressure_owner_namespace_and_cross_user_hash_isolation(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.register_account("user-a")
    store.register_account("user-b")
    first = store.store_asset(
        "user-a",
        "a1",
        AssetCategory.ATTACHMENT,
        b"same",
        provenance_id="prov-a",
    )
    second = store.store_asset(
        "user-b",
        "b1",
        AssetCategory.ATTACHMENT,
        b"same",
        provenance_id="prov-b",
    )
    assert first.scoped_sha256 != second.scoped_sha256
    assert "user-a" not in str(store.data_root)
    assert not list(store.data_root.rglob("*user-a*"))
    with pytest.raises(GovernanceError, match="per-user"):
        store.store_asset(
            "user-a",
            "large",
            AssetCategory.ATTACHMENT,
            b"x" * 61,
            provenance_id="prov-large",
        )
    pressured = make_store(
        tmp_path / "pressure",
        configured=policy(minimum_free_bytes=shutil.disk_usage(tmp_path).free + 1),
    )
    pressured.register_account("user-c")
    with pytest.raises(GovernanceError, match="disk-pressure"):
        pressured.store_asset(
            "user-c",
            "c1",
            AssetCategory.ATTACHMENT,
            b"x",
            provenance_id="prov-c",
        )
    with pytest.raises(GovernanceError, match="raw email"):
        store.register_account("person@example.test")


def test_retention_export_tombstone_explicit_purge_and_provenance(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.register_account("owner")
    store.store_asset(
        "owner",
        "draft-1",
        AssetCategory.DRAFT,
        b"draft",
        provenance_id="prov-draft",
        now=datetime(2020, 1, 1, tzinfo=UTC),
    )
    plan = store.plan_purge(now=datetime(2020, 1, 2, tzinfo=UTC))
    assert plan.reclaimable_bytes == 5
    assert plan.candidates[0].requires_export is True
    with pytest.raises(GovernanceError, match="confirmation"):
        store.tombstone(plan, confirmation="yes")
    with pytest.raises(GovernanceError, match="receipt"):
        store.tombstone(plan, confirmation=PURGE_CONFIRMATION)
    assert (
        store.tombstone(
            plan,
            confirmation=PURGE_CONFIRMATION,
            export_receipts={"draft-1": "export-1"},
        )
        == 1
    )
    tombstone = store.asset("owner", "draft-1")
    assert tombstone.status is AssetStatus.TOMBSTONED
    assert tombstone.provenance_id == "prov-draft"
    assert store.purge_tombstones(confirmation=PURGE_CONFIRMATION) == 1
    purged = store.asset("owner", "draft-1")
    assert purged.status is AssetStatus.PURGED
    assert purged.provenance_id == "prov-draft"


def test_disable_deletion_and_ownership_transfer_policy(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.register_account("source")
    store.register_account("target")
    store.store_asset(
        "source",
        "artifact-1",
        AssetCategory.ARTIFACT,
        b"result",
        provenance_id="prov-result",
    )
    assert store.transfer_ownership("source", "target", ["artifact-1"]) == 1
    transferred = store.asset("target", "artifact-1")
    assert transferred.provenance_id == "prov-result"
    store.disable_account("target")
    with pytest.raises(GovernanceError, match="disabled"):
        store.store_asset(
            "target",
            "new",
            AssetCategory.ARTIFACT,
            b"x",
            provenance_id="prov-new",
        )
    store.request_account_deletion("source")
    with pytest.raises(GovernanceError, match="disabled"):
        store.admit("source", 1)


def test_backup_manifest_validation_and_path_rejection(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.register_account("owner")
    store.store_asset(
        "owner",
        "artifact",
        AssetCategory.ARTIFACT,
        b"evidence",
        provenance_id="prov-evidence",
    )
    destination = tmp_path / "encrypted-fixture"
    manifest = store.create_backup_manifest(
        "backup-1",
        provider=FixtureBackupProvider(),
        key_reference="fixture-key-ref",
        destination=destination,
        confirmation=BACKUP_CONFIRMATION,
    )
    store.verify_backup_manifest(manifest, destination)
    assert manifest.provenance_chain == ("prov-evidence",)
    target = destination / manifest.entries[0].logical_path
    target.write_bytes(b"tampered")
    with pytest.raises(GovernanceError, match="integrity"):
        store.verify_backup_manifest(manifest, destination)
    with pytest.raises(ValueError):
        BackupEntry(logical_path="../escape", byte_count=0, sha256="0" * 64)
    with pytest.raises(ValueError):
        BackupManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "unexpected": True,
            }
        )

