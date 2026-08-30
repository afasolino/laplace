from dataclasses import replace
from pathlib import Path

import pytest

from research_workspace.task_evidence import (
    TaskEvidenceError,
    TaskEvidenceStore,
    TaskOutcomeEvidence,
)


def _item() -> TaskOutcomeEvidence:
    return TaskOutcomeEvidence(
        task_id="task-1",
        owner_id="owner-1",
        project_id="project-1",
        outcome="unverified_candidate",
        roles=("implementation",),
        model_ids=("qwen-quality",),
        manager_decision="bypass",
        specialist_decision="not_selected",
        failure_classes=("verification_failed",),
        turns=3,
        tool_calls=4,
        wall_time_ms=500,
    )


def test_evidence_is_scoped_immutable_and_round_trips(tmp_path) -> None:
    store = TaskEvidenceStore(tmp_path)
    item = _item()
    store.append(item)
    assert store.load(owner_id="owner-1", project_id="project-1", task_id="task-1") == replace(
        item, publication_sequence=1
    )
    with pytest.raises(FileExistsError):
        store.append(item)


def test_evidence_survives_store_restart_and_policy_read_is_non_mutating(tmp_path: Path) -> None:
    item = _item()
    first = TaskEvidenceStore(tmp_path)
    path = first.append(item)
    before = path.read_bytes()

    restarted = TaskEvidenceStore(tmp_path)
    loaded = restarted.load(
        owner_id=item.owner_id, project_id=item.project_id, task_id=item.task_id
    )

    assert loaded == replace(item, publication_sequence=1)
    assert path.read_bytes() == before
    assert list(tmp_path.rglob("*.json")) == [path]


def test_evidence_scope_does_not_leak_between_owners_or_projects(tmp_path: Path) -> None:
    store = TaskEvidenceStore(tmp_path)
    item = _item()
    store.append(item)

    with pytest.raises(TaskEvidenceError, match="task_evidence_not_found"):
        store.load(owner_id="owner-2", project_id=item.project_id, task_id=item.task_id)
    with pytest.raises(TaskEvidenceError, match="task_evidence_not_found"):
        store.load(owner_id=item.owner_id, project_id="project-2", task_id=item.task_id)


def test_corrupt_evidence_fails_closed(tmp_path: Path) -> None:
    store = TaskEvidenceStore(tmp_path)
    item = _item()
    path = store.append(item)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(TaskEvidenceError, match="task_evidence_corrupt"):
        store.load(owner_id=item.owner_id, project_id=item.project_id, task_id=item.task_id)


def test_oversized_and_unbounded_evidence_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence_list_invalid"):
        TaskOutcomeEvidence(
            task_id="task-1",
            owner_id="owner-1",
            project_id="project-1",
            outcome="failure",
            roles=tuple(f"role-{index}" for index in range(17)),
        )

    path = tmp_path / "owner-1" / "project-1" / "task-1.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (32 * 1024 + 1))
    store = TaskEvidenceStore(tmp_path)
    with pytest.raises(TaskEvidenceError, match="task_evidence_record_too_large"):
        store.load(owner_id="owner-1", project_id="project-1", task_id="task-1")


def test_recent_uses_immutable_publication_sequence_not_task_id_order(tmp_path: Path) -> None:
    store = TaskEvidenceStore(tmp_path)
    first = _item()
    second = replace(first, task_id="task-0", outcome="verified_success")
    store.append(first)
    store.append(second)

    recent = store.recent(owner_id="owner-1", project_id="project-1", limit=2)

    assert [record.task_id for record in recent] == ["task-1", "task-0"]
    assert [record.publication_sequence for record in recent] == [1, 2]


def test_recent_fails_closed_for_legacy_records_without_chronology(tmp_path: Path) -> None:
    path = tmp_path / "owner-1" / "project-1" / "task-1.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"failure_classes":[],"manager_decision":"unknown","outcome":"failure",'
        '"owner_id":"owner-1","project_id":"project-1","roles":[],"runtime_profiles":[],'
        '"schema_version":1,"specialist_decision":"unknown","task_id":"task-1",'
        '"verifier_digest":""}',
        encoding="utf-8",
    )

    with pytest.raises(TaskEvidenceError, match="task_evidence_legacy_chronology_unknown"):
        TaskEvidenceStore(tmp_path).recent(owner_id="owner-1", project_id="project-1")
