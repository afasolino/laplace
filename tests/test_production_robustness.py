from __future__ import annotations

import errno
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research_workspace.artifact_registry import ArtifactRegistry
from research_workspace.auth_registry import RegisteredUser, RegisteredUserRegistry, hash_secret, write_registry
from research_workspace.auth_sessions import AuthAuditLog, RegisteredEmailAuth, SessionStore
from research_workspace.research_admission import ResearchAdmissionError, ResearchAdmissionStore
from research_workspace.user_capabilities import CapabilityTier


def _user() -> RegisteredUser:
    return RegisteredUser(
        email="restart@example.test",
        user_id="usr_restart",
        display_name="Restart User",
        enabled=True,
        capability_tier=CapabilityTier.BASIC,
        role="user",
        default_lane="standard",
        authorized_repo_ids=(),
        password_hash=hash_secret(
            "restart fixture password with sufficient length",
            password_policy=True,
        ),
        must_change_password=False,
    )


def test_three_research_jobs_queue_and_promote_deterministically(tmp_path: Path) -> None:
    store = ResearchAdmissionStore(tmp_path / "admission.sqlite3")
    first = store.create("usr_one", "job-one")
    second = store.create("usr_two", "job-two")
    third = store.create("usr_three", "job-three")
    assert first["state"] == "ADMITTED"
    assert second["state"] == "ADMITTED"
    assert third == {
        **third,
        "state": "QUEUED",
        "queue_reason": "global_deep_research_limit",
        "queue_position": 1,
    }
    store.begin("usr_one", "job-one")
    with pytest.raises(ResearchAdmissionError, match="capacity_guardrail"):
        store.begin("usr_three", "job-three")
    store.finish("usr_one", "job-one")
    assert store.status("usr_three", "job-three")["state"] == "ADMITTED"


def test_session_survives_service_restart_and_busy_lock_recovers(tmp_path: Path) -> None:
    registry_path = tmp_path / "auth/registered_users.yaml"
    write_registry(registry_path, [_user()])
    registry = RegisteredUserRegistry(registry_path)
    session_path = tmp_path / "auth/sessions.sqlite3"
    first_store = SessionStore(session_path)
    auth = RegisteredEmailAuth(
        registry,
        first_store,
        AuthAuditLog(tmp_path / "auth/audit.jsonl"),
    )
    created = auth.login(
        "restart@example.test",
        "restart fixture password with sufficient length",
        client_ip="127.0.0.1",
        trace_id="trace-restart",
    )
    restarted_store = SessionStore(session_path)
    assert restarted_store.resolve(created.identifier, registry).user_id == "usr_restart"

    locked = sqlite3.connect(session_path, timeout=1)
    locked.execute("BEGIN EXCLUSIVE")
    completed: list[str] = []

    def rotate() -> None:
        completed.append(restarted_store.rotate_csrf(created.identifier))

    thread = threading.Thread(target=rotate)
    thread.start()
    time.sleep(0.15)
    assert thread.is_alive()
    locked.rollback()
    locked.close()
    thread.join(timeout=3)
    assert len(completed) == 1
    assert len(completed[0]) >= 22


def test_concurrent_artifact_ids_and_registry_rows_are_consistent(tmp_path: Path) -> None:
    registry = ArtifactRegistry(
        tmp_path / "registry.sqlite3",
        tmp_path / "content",
        tmp_path / "events.jsonl",
        tmp_path / "owner.key",
    )

    def create(index: int) -> str:
        record = registry.create(
            owner_user_id="usr_concurrent",
            content=f"artifact {index}\n".encode(),
            relative_path=f"results/artifact-{index}.txt",
            source_state_fingerprint=f"{index:064x}",
            generator_model_route="quality",
            capability_tier="plus",
            trace_id=f"trace-{index}",
            repo_id="repo",
        )
        return record.artifact_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        identifiers = list(pool.map(create, range(32)))
    assert len(set(identifiers)) == 32
    assert len(registry.compact_operator_export()) == 32
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 32


def test_disk_full_does_not_publish_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ArtifactRegistry(
        tmp_path / "registry.sqlite3",
        tmp_path / "content",
        tmp_path / "events.jsonl",
        tmp_path / "owner.key",
    )
    real_open = os.open

    def fail_content_open(path: object, flags: int, mode: int = 0o777) -> int:
        if ".tmp" in str(path) and "content" in str(path):
            raise OSError(errno.ENOSPC, "fixture disk full")
        return real_open(path, flags, mode)

    monkeypatch.setattr("research_workspace.artifact_registry.os.open", fail_content_open)
    with pytest.raises(OSError) as error:
        registry.create(
            owner_user_id="usr_disk",
            content=b"must not publish",
            relative_path="result.txt",
            source_state_fingerprint="d" * 64,
            generator_model_route="quality",
            capability_tier="plus",
            trace_id="trace-disk",
        )
    assert error.value.errno == errno.ENOSPC
    assert registry.compact_operator_export() == []
    assert not list((tmp_path / "content").rglob("result.txt"))
