from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.laplace_core import LaplaceCore
from research_workspace.rules import (
    Rule,
    RuleProvenance,
    RulesConflictError,
    RulesValidationError,
    RuleService,
    SQLiteRuleBackend,
    new_rule,
)


def _provenance(kind: str, actor: str) -> RuleProvenance:
    return RuleProvenance(
        source_kind=kind,  # type: ignore[arg-type]
        source_id=f"source-{actor}",
        actor_id=actor,
        approved=True,
        created_at_utc="2026-08-24T00:00:00+00:00",
    )


def _rule(
    *,
    scope: str,
    key: str,
    value: str,
    owner: str | None = None,
    project: str | None = None,
    paths: tuple[str, ...] = (),
    actor: str = "owner-a",
    source_kind: str = "user_explicit",
) -> Rule:
    return new_rule(
        scope=scope,  # type: ignore[arg-type]
        key=key,
        value=value,
        owner_id=owner,
        project_id=project,
        path_globs=paths,
        provenance=_provenance(source_kind, actor),
    )


def test_precedence_path_project_user_global_and_project_switching(tmp_path: Path) -> None:
    service = RuleService(SQLiteRuleBackend(tmp_path / "rules.sqlite3"))
    service.add(_rule(scope="global", key="execution.network", value="blocked", actor="system", source_kind="system_policy"))
    service.add(_rule(scope="user", key="execution.network", value="localhost", owner="owner-a"))
    service.add(_rule(scope="project", key="execution.network", value="project-only", owner="owner-a", project="project-a"))
    service.add(
        _rule(
            scope="path",
            key="execution.network",
            value="path-only",
            owner="owner-a",
            project="project-a",
            paths=("src/**",),
        )
    )

    assert [rule.value for rule in service.resolve(user_id="owner-a", project_id="project-a", paths=("src/app.py",))] == [
        "path-only"
    ]
    assert [rule.value for rule in service.resolve(user_id="owner-a", project_id="project-a", paths=("docs/readme.md",))] == [
        "project-only"
    ]
    assert [rule.value for rule in service.resolve(user_id="owner-a", project_id="project-b", paths=("src/app.py",))] == [
        "localhost"
    ]
    assert [rule.value for rule in service.resolve(user_id="owner-b", project_id="project-a", paths=("src/app.py",))] == [
        "blocked"
    ]


def test_equal_authority_conflicts_fail_closed_and_path_matching_is_bounded(tmp_path: Path) -> None:
    service = RuleService(SQLiteRuleBackend(tmp_path / "rules.sqlite3"))
    service.add(_rule(scope="project", key="writes.allowed", value="no", owner="owner-a", project="project-a"))
    service.add(_rule(scope="project", key="writes.allowed", value="yes", owner="owner-a", project="project-a"))
    with pytest.raises(RulesConflictError, match="authoritative_rule_conflict"):
        service.resolve(user_id="owner-a", project_id="project-a")

    service = RuleService(SQLiteRuleBackend(tmp_path / "paths.sqlite3"))
    valid = _rule(
        scope="path",
        key="style",
        value="backend",
        owner="owner-a",
        project="project-a",
        paths=("src/**", "*.md"),
    )
    assert service.add(valid).rule_id == valid.rule_id
    assert service.resolve(user_id="owner-a", project_id="project-a", paths=("src/deep/file.py",))[0].key == "style"
    assert service.resolve(user_id="owner-a", project_id="project-a", paths=("docs/file.md",)) == ()
    assert service.resolve(user_id="owner-a", project_id="project-a", paths=("README.md",))[0].key == "style"


def test_malformed_or_unauthorized_rules_fail_closed(tmp_path: Path) -> None:
    service = RuleService(SQLiteRuleBackend(tmp_path / "rules.sqlite3"))
    with pytest.raises(RulesValidationError, match="global_rule_requires_system_provenance"):
        _rule(scope="global", key="x", value="y")
    with pytest.raises(RulesValidationError, match="invalid_rule_path_glob"):
        _rule(scope="path", key="x", value="y", owner="owner-a", project="project-a", paths=("../secret",))
    with pytest.raises(RulesValidationError, match="rule_write_not_authorized"):
        service.add(_rule(scope="user", key="x", value="y", owner="owner-a"), actor_id="owner-b")
    with pytest.raises(RulesValidationError, match="invalid_context_path"):
        service.resolve(user_id="owner-a", project_id="project-a", paths=("/outside",))
    with pytest.raises(RulesValidationError, match="rule_requires_approval"):
        RuleProvenance(
            source_kind="user_explicit",
            source_id="source-a",
            actor_id="owner-a",
            approved=False,
        )


def test_context_assembly_is_deterministic_and_memory_cannot_override_rules(tmp_path: Path) -> None:
    service = RuleService(SQLiteRuleBackend(tmp_path / "rules.sqlite3"))
    service.add(
        _rule(
            scope="project",
            key="execution.network",
            value="blocked",
            owner="owner-a",
            project="project-a",
        )
    )
    first = service.assemble_context(
        user_id="owner-a",
        project_id="project-a",
        paths=("src/app.py",),
        objective="review the change",
        memory=({"key": "execution.network", "value": "enabled"}, "learned advisory"),
        retrieval=({"file": "notes.md", "text": "network is enabled"},),
    )
    second = service.assemble_context(
        user_id="owner-a",
        project_id="project-a",
        paths=("src/app.py",),
        objective="review the change",
        memory=("learned advisory", {"value": "enabled", "key": "execution.network"}),
        retrieval=({"text": "network is enabled", "file": "notes.md"},),
    )
    assert first.assembly_sha256 == second.assembly_sha256
    rendered = first.render()
    assert rendered.index("AUTHORITATIVE_RULES_BEGIN") < rendered.index("LEARNED_MEMORY_ADVISORY_BEGIN")
    assert '"value":"blocked"' in rendered
    assert '"value":"enabled"' in rendered
    packet = first.to_json()
    sections = packet["sections"]
    assert isinstance(sections, dict)
    assert sections["authoritative_rules"]["overrideable"] is False

    core = LaplaceCore(tmp_path, object(), object(), rules=service)  # type: ignore[arg-type]
    assert core.assemble_context(user_id="owner-a", project_id="project-a").rules[0].key == "execution.network"


def test_rules_restart_and_disable_preserve_scope_and_inspectability(tmp_path: Path) -> None:
    path = tmp_path / "rules.sqlite3"
    first = RuleService(SQLiteRuleBackend(path))
    rule = first.add(_rule(scope="user", key="style", value="strict", owner="owner-a"))
    assert first.health()["status"] == "READY"
    second = RuleService(SQLiteRuleBackend(path))
    assert second.resolve(user_id="owner-a", project_id="project-a")[0].rule_id == rule.rule_id
    disabled = second.disable(rule.rule_id, actor_id="owner-a")
    assert disabled.enabled is False
    assert second.resolve(user_id="owner-a", project_id="project-a") == ()
    assert second.inspect(user_id="owner-a", project_id="project-a") == ()
