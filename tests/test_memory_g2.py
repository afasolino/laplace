from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from research_workspace.memory import (
    MemoryCorruptionError,
    MemoryNotFoundError,
    MemoryProvenance,
    MemorySchemaCompatibilityError,
    MemoryService,
    MemoryValidationError,
    SQLiteMemoryBackend,
)


def _provenance(
    source_id: str,
    *,
    source_kind: str = "user_explicit",
    approved: bool = True,
) -> MemoryProvenance:
    return MemoryProvenance(
        source_kind=source_kind,  # type: ignore[arg-type]
        source_id=source_id,
        actor_id="owner-a",
        approved=approved,
        metadata={"fixture": True},
    )


def _service(path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryBackend(path))


def test_memory_isolated_by_owner_and_project_and_requires_explicit_intent(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    first = service.add(
        owner_id="owner-a",
        project_id="project-a",
        kind="semantic",
        content="The fixture uses a deterministic local memory backend.",
        provenance=_provenance("note-1"),
        write_mode="explicit_user",
    )
    assert first.provenance.source_id == "note-1"
    assert service.search(
        owner_id="owner-a", project_id="project-a", query="deterministic backend"
    )[0].memory_id == first.memory_id
    assert service.search(
        owner_id="owner-a", project_id="project-b", query="deterministic backend"
    ) == ()
    with pytest.raises(MemoryNotFoundError, match="memory_not_found"):
        service.get(owner_id="owner-b", project_id="project-a", memory_id=first.memory_id)
    with pytest.raises(MemoryValidationError, match="explicit_intent"):
        service.add(
            owner_id="owner-a",
            project_id="project-a",
            kind="episodic",
            content="implicit memory must be rejected",
            provenance=_provenance("implicit"),
            write_mode="",
        )


def test_memory_restart_preserves_provenance_and_history(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    first_service = _service(database)
    first = first_service.add(
        owner_id="owner-a",
        project_id="project-a",
        kind="episodic",
        content="The experiment completed with the fixture input.",
        provenance=_provenance("run-1", source_kind="deterministic_event"),
        write_mode="deterministic_event",
        contradiction_key="experiment.result",
    )
    replacement = first_service.update(
        owner_id="owner-a",
        project_id="project-a",
        memory_id=first.memory_id,
        content="The experiment was superseded by the corrected fixture input.",
        provenance=_provenance("review-1"),
        write_mode="explicit_user",
    )
    restarted = _service(database)
    restored = restarted.get(
        owner_id="owner-a", project_id="project-a", memory_id=replacement.memory_id
    )
    assert restored.provenance.source_id == "review-1"
    assert restored.supersedes_memory_id == first.memory_id
    assert restarted.get(
        owner_id="owner-a", project_id="project-a", memory_id=first.memory_id
    ).state == "SUPERSEDED"
    assert restarted.search(
        owner_id="owner-a", project_id="project-a", query="corrected fixture"
    )[0].memory_id == replacement.memory_id
    assert len(restarted.history(owner_id="owner-a", project_id="project-a", memory_id=first.memory_id)) >= 2
    assert restarted.health()["backend"] == "sqlite_local_lexical_v1"
    assert restarted.health()["capability"] == "deterministic_lexical_token_matching"
    assert restarted.health()["semantic_similarity"] is False


def test_contradiction_requires_explicit_supersession_and_delete_removes_searchability(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    first = service.add(
        owner_id="owner-a",
        project_id="project-a",
        kind="semantic",
        content="The selected model is the baseline model.",
        provenance=_provenance("decision-1"),
        write_mode="explicit_user",
        contradiction_key="model.selection",
    )
    with pytest.raises(MemoryValidationError, match="contradiction_requires_supersession"):
        service.add(
            owner_id="owner-a",
            project_id="project-a",
            kind="semantic",
            content="The selected model changed without a supersession record.",
            provenance=_provenance("decision-2"),
            write_mode="explicit_user",
            contradiction_key="model.selection",
        )
    second = service.supersede(
        owner_id="owner-a",
        project_id="project-a",
        memory_id=first.memory_id,
        content="The selected model is the corrected model.",
        provenance=_provenance("decision-2"),
        write_mode="explicit_user",
    )
    assert service.search(
        owner_id="owner-a", project_id="project-a", query="baseline"
    ) == ()
    assert service.search(
        owner_id="owner-a", project_id="project-a", query="corrected"
    )[0].memory_id == second.memory_id
    deleted = service.delete(owner_id="owner-a", project_id="project-a", memory_id=second.memory_id)
    assert deleted.state == "DELETED"
    assert service.search(owner_id="owner-a", project_id="project-a", query="corrected") == ()
    assert service.get(
        owner_id="owner-a", project_id="project-a", memory_id=second.memory_id
    ).state == "DELETED"


def test_model_proposals_need_approval_and_corrupt_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite3"
    service = _service(database)
    with pytest.raises(MemoryValidationError, match="requires_approval"):
        service.add(
            owner_id="owner-a",
            project_id="project-a",
            kind="semantic",
            content="unapproved model fact",
            provenance=_provenance(
                "proposal-1", source_kind="model_proposal", approved=False
            ),
            write_mode="approved_model_proposal",
        )
    approved = service.add(
        owner_id="owner-a",
        project_id="project-a",
        kind="semantic",
        content="approved model fact",
        provenance=_provenance("proposal-1", source_kind="model_proposal"),
        write_mode="approved_model_proposal",
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE memory_entries SET provenance_json=? WHERE memory_id=?",
            (json.dumps({"malformed": True}), approved.memory_id),
        )
    corrupted = _service(database)
    with pytest.raises(MemoryCorruptionError):
        corrupted.search(owner_id="owner-a", project_id="project-a", query="approved fact")
    with pytest.raises(MemoryCorruptionError):
        corrupted.health()


def test_newer_schema_fails_closed_without_rewriting_metadata(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    _service(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE memory_schema SET schema_version=99 WHERE singleton=1"
        )
    with pytest.raises(MemorySchemaCompatibilityError, match="memory_schema_newer_unsupported"):
        SQLiteMemoryBackend(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT schema_version FROM memory_schema WHERE singleton=1"
        ).fetchone()[0] == 99


def test_unsupported_older_schema_is_recoverable_and_not_advanced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "memory.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE memory_schema (singleton INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL, updated_at_utc TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO memory_schema VALUES (1, 0, '2026-01-01T00:00:00+00:00')"
        )

    def interrupted(*_: object) -> None:
        raise MemorySchemaCompatibilityError("memory_schema_migration_interrupted")

    monkeypatch.setattr(SQLiteMemoryBackend, "_migrate", staticmethod(interrupted))
    with pytest.raises(MemorySchemaCompatibilityError, match="migration_interrupted"):
        SQLiteMemoryBackend(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT schema_version FROM memory_schema WHERE singleton=1"
        ).fetchone()[0] == 0


def test_incompatible_schema_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE memory_schema (singleton INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL, updated_at_utc TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO memory_schema VALUES (1, 1, '2026-01-01T00:00:00+00:00')"
        )
        connection.execute("CREATE TABLE memory_entries (memory_id TEXT PRIMARY KEY)")
    with pytest.raises(MemorySchemaCompatibilityError, match="memory_schema_incompatible"):
        SQLiteMemoryBackend(database)
