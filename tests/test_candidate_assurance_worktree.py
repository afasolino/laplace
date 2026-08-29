from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from research_workspace.agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentToolPolicy,
)
from research_workspace.repository_authorization import RepositoryAuthorizationStore


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _manager(
    tmp_path: Path,
) -> tuple[AgentSandboxManager, RepositoryAuthorizationStore, Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Fixture")
    (repository / "value.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "base")
    revision = _git(repository, "rev-parse", "HEAD")
    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    authorizations.register("repo", repository)
    authorizations.grant("owner", "repo", base_revision=revision)
    return (
        AgentSandboxManager(
            tmp_path / "worktrees",
            authorizations,
            recover_cleanup=False,
        ),
        authorizations,
        repository,
        revision,
    )


def _policy() -> AgentToolPolicy:
    return AgentToolPolicy("bounded", ("apply_patch", "run_tests"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_mutation_is_allowed_without_verifier_and_status_is_unverified(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="unverified", tool_policy=_policy()
    )
    Path(binding.worktree_root, "new.txt").write_text("new\n", encoding="utf-8")

    record = manager.status("unverified", user_id="owner")

    assert record["assurance_state"] == "unverified_candidate"
    assert record["mutation_epoch"] == 1
    assert record["verified_epoch"] == 0
    assert record["verifier_digest"] is None
    assert record["promotion_eligible"] is False


def test_failed_verify_then_verifier_replacement_can_pass(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="replace", tool_policy=_policy()
    )
    worktree = Path(binding.worktree_root)
    target = worktree / "value.txt"
    target.write_text("first\n", encoding="utf-8")
    first = manager.status("replace", user_id="owner")
    verifier_one = _digest("pytest -q first")
    manager.begin_verification("replace", user_id="owner", verifier_digest=verifier_one)
    failed = manager.complete_verification(
        "replace",
        user_id="owner",
        verifier_digest=verifier_one,
        passed=False,
        summary="fixture failure",
    )
    target.write_text("second\n", encoding="utf-8")
    changed = manager.status("replace", user_id="owner")
    verifier_two = _digest("pytest -q second")
    manager.begin_verification("replace", user_id="owner", verifier_digest=verifier_two)
    passed = manager.complete_verification(
        "replace",
        user_id="owner",
        verifier_digest=verifier_two,
        passed=True,
        summary="fixture passed",
    )

    assert first["mutation_epoch"] == 1
    assert failed["assurance_state"] == "verification_failed"
    assert changed["mutation_epoch"] == 2
    assert passed["assurance_state"] == "verified_candidate"
    assert passed["mutation_epoch"] == passed["verified_epoch"] == 2
    assert passed["candidate_fingerprint"] == passed["verified_fingerprint"]
    assert passed["verifier_digest"] == verifier_two


def test_pass_then_material_mutation_is_stale_and_status_is_idempotent(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="stale", tool_policy=_policy()
    )
    target = Path(binding.worktree_root, "value.txt")
    target.write_text("verified\n", encoding="utf-8")
    before = manager.status("stale", user_id="owner")
    verifier = _digest("pytest -q")
    manager.begin_verification("stale", user_id="owner", verifier_digest=verifier)
    verified = manager.complete_verification(
        "stale", user_id="owner", verifier_digest=verifier, passed=True, summary="passed"
    )
    target.write_text("stale\n", encoding="utf-8")
    stale = manager.status("stale", user_id="owner")
    repeated = manager.status("stale", user_id="owner")

    assert before["mutation_epoch"] == 1
    assert verified["assurance_state"] == "verified_candidate"
    assert stale["assurance_state"] == "unverified_candidate"
    assert stale["mutation_epoch"] == 2
    assert repeated["mutation_epoch"] == stale["mutation_epoch"]
    assert repeated["candidate_fingerprint"] == stale["candidate_fingerprint"]


def test_unverified_candidate_survives_manager_restart(tmp_path: Path) -> None:
    manager, authorizations, _, _ = _manager(tmp_path)
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="restart", tool_policy=_policy()
    )
    Path(binding.worktree_root, "restart.txt").write_text("restart\n", encoding="utf-8")
    first = manager.status("restart", user_id="owner")

    recovered = AgentSandboxManager(
        manager.sandbox_root,
        authorizations,
        recover_cleanup=False,
    )
    second = recovered.resume("restart", user_id="owner")

    assert second["assurance_state"] == "unverified_candidate"
    assert second["mutation_epoch"] == first["mutation_epoch"] == 1
    assert second["candidate_fingerprint"] == first["candidate_fingerprint"]


def test_existing_schema_row_migrates_with_clean_assurance_defaults(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "worktrees"
    sandbox_root.mkdir()
    database = sandbox_root / "worktree_sessions.sqlite3"
    now = "2026-08-29T00:00:00+00:00"
    policy = json.dumps(
        {
            "policy_id": "bounded",
            "allowed_tools": ["read_file"],
            "network_enabled": False,
            "max_commands": 4,
            "max_wall_seconds": 60,
        }
    )
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE worktree_schema (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                schema_version INTEGER NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE worktree_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                canonical_repository_root TEXT NOT NULL,
                worktree_root TEXT NOT NULL UNIQUE,
                base_revision TEXT NOT NULL,
                grant_revision INTEGER NOT NULL,
                tool_policy_json TEXT NOT NULL,
                environment_json TEXT NOT NULL,
                task_title TEXT NOT NULL,
                current_task_label TEXT,
                instruction_digest TEXT NOT NULL,
                state TEXT NOT NULL,
                lane TEXT,
                sanitized_model_name TEXT,
                command_count INTEGER NOT NULL DEFAULT 0,
                changed_paths_json TEXT NOT NULL DEFAULT '[]',
                diff_hash TEXT,
                verification_summary TEXT,
                export_state TEXT NOT NULL DEFAULT 'NONE',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                retention_expires_at_utc TEXT,
                idempotency_key TEXT,
                ownership_token TEXT,
                cleanup_eligible INTEGER NOT NULL DEFAULT 0,
                result_id TEXT,
                physical_state TEXT NOT NULL DEFAULT 'REMOVED',
                executor_pid INTEGER,
                executor_start_ticks INTEGER,
                executor_boot_id TEXT,
                wall_deadline_utc TEXT,
                last_heartbeat_utc TEXT,
                UNIQUE(user_id, idempotency_key)
            );
            CREATE TABLE worktree_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event TEXT NOT NULL,
                state TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            INSERT INTO worktree_schema VALUES (1, 2, '2026-08-29T00:00:00+00:00');
            """
        )
        connection.execute(
            """
            INSERT INTO worktree_sessions (
                session_id, user_id, repo_id, canonical_repository_root, worktree_root,
                base_revision, grant_revision, tool_policy_json, environment_json,
                task_title, instruction_digest, state, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED_CLEAN', ?, ?)
            """,
            (
                "legacy",
                "owner",
                "repo",
                str(tmp_path / "repository"),
                str(tmp_path / "missing-worktree"),
                "a" * 40,
                1,
                policy,
                "{}",
                "Legacy",
                "",
                now,
                now,
            ),
        )

    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Fixture")
    (repository / "value.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", ".")
    revision = _git(repository, "commit", "-qm", "base")
    del revision
    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    authorizations.register("repo", repository)
    authorizations.grant("owner", "repo")

    manager = AgentSandboxManager(sandbox_root, authorizations, recover_cleanup=False)

    with manager._connect() as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(worktree_sessions)")
        }
        version = connection.execute(
            "SELECT schema_version FROM worktree_schema WHERE singleton=1"
        ).fetchone()[0]
        row = connection.execute(
            "SELECT mutation_epoch, verified_epoch, assurance_state FROM worktree_sessions "
            "WHERE session_id='legacy'"
        ).fetchone()

    assert version == 4
    assert {"mutation_epoch", "verified_epoch", "assurance_state"} <= columns
    assert tuple(row) == (0, 0, "legacy_unverified")


def test_dirty_unverified_candidate_cannot_be_promoted(tmp_path: Path) -> None:
    manager, _, _, _ = _manager(tmp_path)
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="blocked", tool_policy=_policy()
    )
    Path(binding.worktree_root, "blocked.txt").write_text("blocked\n", encoding="utf-8")

    with pytest.raises(AgentSandboxError, match="candidate_not_verified"):
        manager.require_verified_candidate(
            "blocked", user_id="owner", verifier_digest=_digest("pytest -q")
        )


def test_exact_verified_candidate_provides_existing_promotion_binding(tmp_path: Path) -> None:
    manager, _, _, revision = _manager(tmp_path)
    binding = manager.create(
        user_id="owner", repo_id="repo", session_id="promotable", tool_policy=_policy()
    )
    Path(binding.worktree_root, "promotable.txt").write_text("promotable\n", encoding="utf-8")
    manager.status("promotable", user_id="owner")
    verifier = _digest("pytest -q")
    manager.begin_verification("promotable", user_id="owner", verifier_digest=verifier)
    manager.complete_verification(
        "promotable", user_id="owner", verifier_digest=verifier, passed=True, summary="passed"
    )

    exact = manager.require_verified_candidate(
        "promotable", user_id="owner", verifier_digest=verifier
    )
    promoted = manager.mark_promoted("promotable", user_id="owner", binding=exact)

    assert exact.base_revision == revision
    assert exact.mutation_epoch == 1
    assert exact.candidate_fingerprint
    assert promoted["assurance_state"] == "promoted"
    assert promoted["currently_verified"] is True


def test_git_observation_failure_is_explicit_and_cannot_request_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _, _, _ = _manager(tmp_path)
    manager.create(user_id="owner", repo_id="repo", session_id="unknown", tool_policy=_policy())

    monkeypatch.setattr(
        manager,
        "_runner",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 1, "", "git failed"),
    )
    observed = manager.observe_worktree("unknown", user_id="owner")
    requested = manager.request_export("unknown", user_id="owner", promotion=True)

    assert observed["assurance_state"] == "observation_failed"
    assert observed["observation_error"] == "candidate_git_observation_failed"
    assert requested["status"] == "PROMOTION_REQUESTED_INELIGIBLE"
    assert requested["promotion_eligible"] is False
