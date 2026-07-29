from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research_workspace.contracts import RepositoryGrantV1
from research_workspace.fixture_services import FixtureRepositoryService
from research_workspace.sync_client import (
    DesktopSyncClient,
    RepositoryInspector,
    SyncOperationStore,
    apply_confirmed_patch,
)
from research_workspace.sync_protocol import (
    SyncError,
    SyncReceiptV1,
    SyncTransport,
    TransportPolicyV1,
    confirmation_for,
)
from research_workspace.sync_service import FixtureSyncService, RegisteredSyncRepository

NOW = "2026-07-28T00:00:00+00:00"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _repository(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "module.py")
    _git(root, "commit", "-q", "-m", "fixture base")
    return root


def _service(root: Path) -> FixtureSyncService:
    grant = RepositoryGrantV1(
        grant_id="grant-1",
        user_id="owner-a",
        repository_id="repo-a",
        revision=1,
        active=True,
        created_at_utc=NOW,
    )
    return FixtureSyncService(
        repositories=FixtureRepositoryService([grant]),
        registrations=(RegisteredSyncRepository("repo-a", root),),
    )


def test_repository_snapshot_excludes_untracked_and_redacts_remote(tmp_path: Path) -> None:
    root = _repository(tmp_path, "project")
    _git(
        root,
        "remote",
        "add",
        "origin",
        "https://user:password@example.invalid/private/repository.git",
    )
    (root / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "untracked.txt").write_text("not silently included\n", encoding="utf-8")
    snapshot = RepositoryInspector().snapshot(root, logical_repository_id="repo-a")
    assert snapshot.branch in {"main", "master"}
    assert len(snapshot.head) == 40
    assert snapshot.dirty is True
    assert snapshot.remotes == ("https://example.invalid/<redacted-path>",)
    changes = {item.logical_path: item for item in snapshot.changes}
    assert changes["module.py"].included is True
    assert changes["untracked.txt"].included is False
    assert snapshot.untracked_included is False
    plan, patch = RepositoryInspector().plan_upload(
        root,
        logical_repository_id="repo-a",
    )
    assert plan.changed_paths == ("module.py",)
    assert b"untracked" not in patch
    assert plan.force_push is False
    assert plan.untracked_included is False


def test_confirmed_resumable_fixture_upload_and_patch_export(tmp_path: Path) -> None:
    root = _repository(tmp_path, "server")
    (root / "module.py").write_text("VALUE = 2\n" + ("# padding\n" * 200), encoding="utf-8")
    plan, patch = RepositoryInspector().plan_upload(root, logical_repository_id="repo-a")
    service = _service(root)

    class InterruptAfterFirstReceipt:
        def __init__(self, delegate: FixtureSyncService) -> None:
            self.delegate = delegate
            self.interrupted = False

        def upload_chunk(
            self,
            *,
            owner_id: str,
            logical_repository_id: str,
            operation_id: str,
            base_head: str,
            offset: int,
            content: bytes,
            final: bool,
            expected_sha256: str,
        ) -> SyncReceiptV1:
            receipt = self.delegate.upload_chunk(
                owner_id=owner_id,
                logical_repository_id=logical_repository_id,
                operation_id=operation_id,
                base_head=base_head,
                offset=offset,
                content=content,
                final=final,
                expected_sha256=expected_sha256,
            )
            if not self.interrupted:
                self.interrupted = True
                raise RuntimeError("synthetic disconnect")
            return receipt

        def download_patch(
            self,
            *,
            owner_id: str,
            logical_repository_id: str,
            operation_id: str,
        ) -> bytes:
            return self.delegate.download_patch(
                owner_id=owner_id,
                logical_repository_id=logical_repository_id,
                operation_id=operation_id,
            )

    transport = InterruptAfterFirstReceipt(service)
    assert isinstance(transport, SyncTransport)
    store = SyncOperationStore(tmp_path / "sync/operations.sqlite3")
    client = DesktopSyncClient(
        owner_id="owner-a",
        operations=store,
        transport=transport,
        chunk_bytes=1024,
    )
    operation = client.prepare(plan, patch)
    with pytest.raises(SyncError, match="sync_confirmation_required"):
        client.upload(operation.operation_id, confirmation="yes")
    with pytest.raises(RuntimeError, match="synthetic disconnect"):
        client.upload(
            operation.operation_id,
            confirmation=confirmation_for(plan.plan_id),
        )
    resumed = client.upload(
        operation.operation_id,
        confirmation=confirmation_for(plan.plan_id),
    )
    assert resumed.state.value == "TRANSFERRED"
    assert resumed.transferred_bytes == len(patch)
    history = store.history("owner-a", operation.operation_id)
    assert history[0]["state"] == "PLANNED"
    assert history[-1]["state"] == "TRANSFERRED"
    assert service.download_patch(
        owner_id="owner-a",
        logical_repository_id="repo-a",
        operation_id=operation.operation_id,
    ) == patch
    safe = service.sanitized_operation("owner-a", operation.operation_id)
    assert safe["canonical_path_exposed"] is False
    assert str(root) not in str(safe)
    exported = client.export_patch(operation.operation_id, tmp_path / "exports/change.patch")
    assert exported.read_bytes() == patch
    with pytest.raises(SyncError, match="sync_repository_not_authorized"):
        service.download_patch(
            owner_id="owner-b",
            logical_repository_id="repo-a",
            operation_id=operation.operation_id,
        )


def test_patch_application_requires_confirmation_and_detects_base_conflict(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path, "source")
    target = tmp_path / "target"
    _git(tmp_path, "clone", "-q", str(source), str(target))
    (source / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
    plan, patch = RepositoryInspector().plan_upload(source, logical_repository_id="repo-a")
    with pytest.raises(SyncError, match="sync_confirmation_required"):
        apply_confirmed_patch(target, plan=plan, patch=patch, confirmation="yes")
    assert apply_confirmed_patch(
        target,
        plan=plan,
        patch=patch,
        confirmation=confirmation_for(plan.plan_id),
    ) == "APPLIED"
    assert (target / "module.py").read_text(encoding="utf-8") == "VALUE = 3\n"

    conflict = tmp_path / "conflict"
    _git(tmp_path, "clone", "-q", str(source), str(conflict))
    _git(conflict, "config", "user.email", "fixture@example.invalid")
    _git(conflict, "config", "user.name", "Fixture")
    (conflict / "other.txt").write_text("conflict\n", encoding="utf-8")
    _git(conflict, "add", "other.txt")
    _git(conflict, "commit", "-q", "-m", "different head")
    with pytest.raises(SyncError, match="sync_base_conflict"):
        apply_confirmed_patch(
            conflict,
            plan=plan,
            patch=patch,
            confirmation=confirmation_for(plan.plan_id),
        )


def test_patch_application_rejects_dirty_target_without_combining_changes(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path, "source")
    (source / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    _git(source, "add", "other.py")
    _git(source, "commit", "-q", "-m", "add second tracked file")
    target = tmp_path / "target"
    _git(tmp_path, "clone", "-q", str(source), str(target))
    (source / "module.py").write_text("VALUE = 5\n", encoding="utf-8")
    plan, patch = RepositoryInspector().plan_upload(
        source,
        logical_repository_id="repo-a",
    )
    (target / "other.py").write_text("OTHER = 2\n", encoding="utf-8")

    with pytest.raises(SyncError, match="sync_target_not_clean"):
        apply_confirmed_patch(
            target,
            plan=plan,
            patch=patch,
            confirmation=confirmation_for(plan.plan_id),
        )
    assert (target / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (target / "other.py").read_text(encoding="utf-8") == "OTHER = 2\n"


def test_prepare_revalidates_patch_size_and_path_plan(tmp_path: Path) -> None:
    source = _repository(tmp_path, "source")
    (source / "module.py").write_text("VALUE = 6\n", encoding="utf-8")
    plan, patch = RepositoryInspector().plan_upload(
        source,
        logical_repository_id="repo-a",
    )
    client = DesktopSyncClient(
        owner_id="owner-a",
        operations=SyncOperationStore(tmp_path / "sync/operations.sqlite3"),
        transport=_service(source),
    )
    with pytest.raises(SyncError, match="sync_patch_plan_mismatch"):
        client.prepare(
            plan.model_copy(update={"patch_size_bytes": len(patch) + 1}),
            patch,
        )
    with pytest.raises(SyncError, match="sync_patch_plan_mismatch"):
        client.prepare(
            plan.model_copy(update={"changed_paths": ("different.py",)}),
            patch,
        )


def test_patch_application_rejects_same_head_on_different_branch(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path, "source")
    target = tmp_path / "target"
    _git(tmp_path, "clone", "-q", str(source), str(target))
    _git(target, "switch", "-q", "-c", "other-branch")
    (source / "module.py").write_text("VALUE = 7\n", encoding="utf-8")
    plan, patch = RepositoryInspector().plan_upload(
        source,
        logical_repository_id="repo-a",
    )
    with pytest.raises(SyncError, match="sync_branch_conflict"):
        apply_confirmed_patch(
            target,
            plan=plan,
            patch=patch,
            confirmation=confirmation_for(plan.plan_id),
        )


def test_clean_and_detached_repository_snapshots_are_explicit(tmp_path: Path) -> None:
    root = _repository(tmp_path, "source")
    clean = RepositoryInspector().snapshot(root, logical_repository_id="repo-a")
    assert clean.dirty is False
    assert clean.changes == ()
    _git(root, "checkout", "-q", "--detach")
    detached = RepositoryInspector().snapshot(root, logical_repository_id="repo-a")
    assert detached.branch == "DETACHED"


def test_staged_unstaged_and_quota_states_are_explicit(tmp_path: Path) -> None:
    root = _repository(tmp_path, "source")
    (root / "module.py").write_text("VALUE = 8\n", encoding="utf-8")
    (root / "staged.py").write_text("STAGED = True\n", encoding="utf-8")
    _git(root, "add", "staged.py")
    (root / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    snapshot = RepositoryInspector().snapshot(root, logical_repository_id="repo-a")
    changes = {item.logical_path: item for item in snapshot.changes}
    assert changes["module.py"].staged is False
    assert changes["staged.py"].staged is True
    assert changes["untracked.py"].included is False
    with pytest.raises(SyncError, match="sync_file_count_exceeded"):
        RepositoryInspector(maximum_files=2).snapshot(
            root,
            logical_repository_id="repo-a",
        )


def test_changed_links_nested_repositories_submodules_and_binary_are_rejected(
    tmp_path: Path,
) -> None:
    linked = _repository(tmp_path, "linked")
    (linked / "link.py").symlink_to("module.py")
    _git(linked, "add", "link.py")
    with pytest.raises(SyncError, match="symlink_rejected"):
        RepositoryInspector().snapshot(linked, logical_repository_id="repo-a")

    nested = _repository(tmp_path, "nested")
    nested_file = nested / "vendor/code.py"
    nested_file.parent.mkdir()
    nested_file.write_text("VALUE = 1\n", encoding="utf-8")
    _git(nested, "add", "vendor/code.py")
    _git(nested, "commit", "-q", "-m", "track vendor")
    (nested / "vendor/.git").mkdir()
    nested_file.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SyncError, match="nested_repository_rejected"):
        RepositoryInspector().snapshot(nested, logical_repository_id="repo-a")

    child = _repository(tmp_path, "child")
    parent = _repository(tmp_path, "parent")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(child),
        "vendor/child",
    )
    with pytest.raises(SyncError, match="submodules_not_supported"):
        RepositoryInspector().snapshot(parent, logical_repository_id="repo-a")

    binary = _repository(tmp_path, "binary")
    (binary / "module.py").write_bytes(bytes(range(256)) * 16)
    with pytest.raises(SyncError, match="sync_patch.*rejected"):
        RepositoryInspector().plan_upload(
            binary,
            logical_repository_id="repo-a",
        )


def test_selection_path_links_submodules_and_transport_policy_fail_closed(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path, "project")
    nested = root / "nested"
    nested.mkdir()
    with pytest.raises(SyncError, match="repository_selection_not_git_top_level"):
        RepositoryInspector().snapshot(nested, logical_repository_id="repo-a")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(SyncError, match="git_inspection_failed"):
        RepositoryInspector().snapshot(plain, logical_repository_id="repo-a")

    (root / "module.py").write_text("VALUE = 4\n", encoding="utf-8")
    hardlink = root / "hardlink.py"
    try:
        os.link(root / "module.py", hardlink)
    except OSError:
        pytest.skip("hardlinks unavailable")
    with pytest.raises(SyncError, match="hardlink_rejected"):
        RepositoryInspector().snapshot(root, logical_repository_id="repo-a")

    TransportPolicyV1(
        transport="ssh",
        host="git.example.invalid",
        host_key_verification=True,
        tls_verification=False,
    )
    TransportPolicyV1(
        transport="https",
        host="git.example.invalid",
        host_key_verification=False,
        tls_verification=True,
    )
    with pytest.raises(ValueError, match="host-key"):
        TransportPolicyV1(
            transport="ssh",
            host="git.example.invalid",
            host_key_verification=False,
            tls_verification=False,
        )
    with pytest.raises(ValueError, match="TLS"):
        TransportPolicyV1(
            transport="https",
            host="git.example.invalid",
            host_key_verification=False,
            tls_verification=False,
        )
