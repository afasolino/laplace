from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_workspace.agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentToolPolicy,
)
from research_workspace.auth_registry import (
    RegisteredUserRegistry,
    hash_secret,
    parse_registry,
    write_registry,
)
from research_workspace.domain_registry import DomainRegistry, DomainRegistryError
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.user_capabilities import (
    Capability,
    CapabilityTier,
    UserCapabilityStore,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.test")
    _git(root, "config", "user.name", "Fixture")
    (root / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "base")
    return _git(root, "rev-parse", "HEAD")


def test_legacy_registry_migrates_to_named_capabilities_and_v2_round_trip(
    tmp_path: Path,
) -> None:
    encoded = hash_secret("fixture activation credential")
    raw = f"""schema_version: 1
users:
  - email: operator@example.test
    user_id: operator
    display_name: Operator
    enabled: true
    capability_tier: operator
    role: admin
    default_lane: quality
    authorized_repo_ids: []
    password_hash: "{encoded}"
    must_change_password: true
""".encode()
    migrated = parse_registry(raw)
    user = migrated.users_by_id["operator"]
    assert Capability.OPERATOR in user.effective_capabilities
    assert Capability.AGENT not in user.effective_capabilities

    path = tmp_path / "auth/registered_users.yaml"
    write_registry(path, [user])
    written = path.read_text(encoding="utf-8")
    assert "schema_version: 2" in written
    registry = RegisteredUserRegistry(path)
    registry.update_user(
        "operator",
        capabilities=tuple(
            sorted(
                {*user.effective_capabilities, Capability.AGENT, Capability.PERSONAL_CORPUS},
                key=str,
            )
        ),
    )
    combined = registry.require_user("operator")
    assert Capability.AGENT in combined.effective_capabilities
    assert Capability.OPERATOR in combined.effective_capabilities


def test_sqlite_capability_migration_and_independent_enforcement(
    tmp_path: Path,
) -> None:
    store = UserCapabilityStore(tmp_path / "capabilities.sqlite3")
    basic = store.set_user("basic", CapabilityTier.BASIC)
    assert basic.capabilities == frozenset({Capability.CHAT})
    with pytest.raises(Exception, match="capability_denied"):
        store.require_capability("basic", Capability.AGENT)

    operator = store.set_user("operator", CapabilityTier.OPERATOR)
    combined = store.set_capabilities(
        "operator",
        frozenset({*operator.capabilities, Capability.AGENT, Capability.PERSONAL_CORPUS}),
    )
    assert combined.has(Capability.AGENT)
    assert combined.has(Capability.OPERATOR)
    prior_revision = combined.revision
    unchanged = store.set_capabilities("operator", combined.capabilities)
    assert unchanged.revision == prior_revision
    changed = store.set_capabilities(
        "operator", frozenset({Capability.CHAT, Capability.AGENT})
    )
    assert changed.revision == prior_revision + 1
    assert not changed.has(Capability.OPERATOR)


def test_domain_registry_enumerates_surfaces_and_rejects_unknown_or_unavailable() -> None:
    registry = DomainRegistry()
    public = registry.public()
    assert public["default_domain_id"] == "general"
    identifiers = {item["domain_id"] for item in public["domains"]}
    assert {"general", "python", "systemverilog"} <= identifiers
    assert registry.require("python", surface="agent").domain_id == "python"
    with pytest.raises(DomainRegistryError, match="unknown_domain"):
        registry.require("rust", surface="agent")
    with pytest.raises(DomainRegistryError, match="domain_unavailable_for_surface"):
        registry.require("general", surface="agent")


def test_persistent_worktree_lifecycle_quota_history_dirty_preservation_and_discard(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    commit = _repository(repository)
    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    authorizations.register("fixture-repo", repository)
    authorizations.grant("owner", "fixture-repo", base_revision=commit)
    manager = AgentSandboxManager(
        tmp_path / "worktrees",
        authorizations,
        per_user_quota=1,
        global_quota=2,
    )
    binding = manager.create(
        user_id="owner",
        repo_id="fixture-repo",
        session_id="session-one",
        tool_policy=AgentToolPolicy(
            "bounded-v1", ("read_file", "apply_patch", "run_validation")
        ),
        task_title="Fixture task",
        instruction_digest="a" * 64,
        idempotency_key="worktree:session-one",
    )
    assert Path(binding.worktree_root).is_dir()
    retried = manager.create(
        user_id="owner",
        repo_id="fixture-repo",
        session_id="a-different-client-retry-id",
        tool_policy=AgentToolPolicy(
            "bounded-v1", ("read_file", "apply_patch", "run_validation")
        ),
        task_title="Fixture task",
        instruction_digest="a" * 64,
        idempotency_key="worktree:session-one",
    )
    assert retried.session_id == binding.session_id
    assert manager.inspect("session-one", user_id="owner")[
        "retention_expires_at_utc"
    ]
    with pytest.raises(AgentSandboxError, match="per_user_worktree_quota"):
        manager.create(
            user_id="owner",
            repo_id="fixture-repo",
            session_id="session-two",
            tool_policy=AgentToolPolicy("bounded-v1", ("read_file",)),
        )

    (Path(binding.worktree_root) / "created.txt").write_text(
        "new worktree content\n", encoding="utf-8"
    )
    status = manager.status("session-one", user_id="owner")
    assert status["status"] == "DIRTY"
    assert status["changed_paths"] == ["created.txt"]
    assert "worktree_root" not in status
    preserved = manager.close_if_clean("session-one", user_id="owner")
    assert preserved["status"] == "PRESERVED_DIRTY_WORKTREE"
    assert Path(binding.worktree_root).is_dir()
    assert b"created.txt" in manager.patch("session-one", user_id="owner")

    recovered = AgentSandboxManager(
        tmp_path / "worktrees",
        authorizations,
        per_user_quota=1,
        global_quota=2,
    )
    record = recovered.inspect("session-one", user_id="owner")
    assert record["state"] == "DIRTY"
    assert "worktree_root" not in record
    assert any(item["event"] == "CLOSE_PRESERVED_DIRTY" for item in recovered.history("session-one", user_id="owner"))
    with pytest.raises(AgentSandboxError, match="discard_confirmation_required"):
        recovered.discard(
            "session-one", user_id="owner", confirmation="wrong"
        )
    discarded = recovered.discard(
        "session-one",
        user_id="owner",
        confirmation="discard:session-one",
    )
    assert discarded["status"] == "DISCARDED"
    assert not Path(binding.worktree_root).exists()
