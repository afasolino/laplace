from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.execution_records import canonical_sha256
from research_workspace.reproducibility import (
    ContextPacketBuilder,
    FrozenSkillRegistry,
    ReproducibilityError,
    write_reproducibility_locks,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_skill_loading_and_lock_hash_are_deterministic(tmp_path: Path) -> None:
    registry = FrozenSkillRegistry(REPOSITORY_ROOT / "codex_a6000" / "skills")
    first = registry.write_lock(tmp_path / "one" / "skills.lock.json")
    second = registry.write_lock(tmp_path / "two" / "skills.lock.json")

    assert first == second
    assert (tmp_path / "one" / "skills.lock.json").read_bytes() == (
        tmp_path / "two" / "skills.lock.json"
    ).read_bytes()
    assert len(first["skills"]) == 7
    assert registry.skills_for_role("review")[0].name == "reviewer-grounding"


def test_context_is_deterministic_excludes_forbidden_and_tracks_source_change(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "source.sv"
    held_out = repository / "tb_heldout.sv"
    secret = repository / ".env"
    source.write_text("module source; endmodule\n", encoding="utf-8")
    held_out.write_text("never include\n", encoding="utf-8")
    secret.write_text("API_KEY=never\n", encoding="utf-8")
    builder = ContextPacketBuilder(repository)
    inputs = {
        "role": "verification",
        "attempt": 0,
        "sections": {"task": "Verify public behavior."},
        "source_paths": [source, held_out, secret],
        "skills_lock_sha256": "a" * 64,
        "corpus_snapshot_sha256": "b" * 64,
    }

    first = builder.build(tmp_path / "run-one", **inputs)
    second = builder.build(tmp_path / "run-two", **inputs)
    assert Path(first.context_path).read_bytes() == Path(second.context_path).read_bytes()
    assert first.context_sha256 == second.context_sha256
    context = Path(first.context_path).read_text(encoding="utf-8")
    manifest = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    assert "never include" not in context
    assert "API_KEY" not in context
    assert {item["reason"] for item in manifest["excluded"]} == {
        "held_out_path_excluded",
        "secret_path_excluded",
    }

    source.write_text("module source; logic changed; endmodule\n", encoding="utf-8")
    changed = builder.build(tmp_path / "run-three", **inputs)
    assert changed.context_sha256 != first.context_sha256


def test_context_section_rejects_secret_or_held_out_material(tmp_path: Path) -> None:
    builder = ContextPacketBuilder(tmp_path)
    with pytest.raises(ReproducibilityError):
        builder.build(
            tmp_path / "run",
            role="review",
            attempt=0,
            sections={"task": "password=secret"},
            source_paths=[],
            skills_lock_sha256="a" * 64,
            corpus_snapshot_sha256="b" * 64,
        )


def test_subordinate_locks_bind_run_lock(tmp_path: Path) -> None:
    skills = {"schema_version": 1, "skills": [], "skills_lock_sha256": "a" * 64}
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "skills.lock.json").write_text(
        json.dumps(skills, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    context = run_root / "context_manifest.json"
    context.write_text(
        json.dumps(
            {"role": "review", "attempt": 0, "context_sha256": "b" * 64},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    held_out = {"identity_sha256": canonical_sha256("held-out-v1")}
    result = write_reproducibility_locks(
        run_root,
        skills_lock=skills,
        context_manifests=[context],
        corpus_identity={"snapshot_sha256": "c" * 64},
        models_identity={"main": {"model": "fixture"}},
        tools_identity={"python": {"version": "3.11"}},
        base_revision="d" * 40,
        experiment_configuration={"arm": "C"},
        request={"task": "fixture"},
        held_out_identity=held_out,
    )

    assert result["run_lock_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "run_lock_sha256"}
    )
    for name in (
        "context.lock.json",
        "corpus.lock.json",
        "models.lock.json",
        "tools.lock.json",
        "run.lock.json",
    ):
        assert (run_root / name).is_file()
