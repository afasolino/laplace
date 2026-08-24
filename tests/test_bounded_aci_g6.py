from __future__ import annotations

import subprocess
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

from research_workspace.bounded_aci import BoundedACIError, BoundedRepositoryACI
from research_workspace.zetsu_agent import ZetsuAgentCoordinator


def _git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    return tmp_path


def _commit(root: Path, message: str = "base") -> None:
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", message], check=True)


def test_typed_read_search_structure_diff_and_git_state_are_bounded(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    source = root / "src.py"
    source.write_text("def answer():\n    return 41\n\nanswer()\n", encoding="utf-8")
    (root / "notes.txt").write_text("answer appears here\n", encoding="utf-8")
    _commit(root)
    aci = BoundedRepositoryACI(
        root,
        owner_user_id="owner-a",
        session_id="session-a",
        required_verification_argv=("pytest", "tests/test.py", "-q"),
    )

    state = aci.git_state()
    assert state["head"]
    region = aci.read_region(path="src.py", start_line=1, end_line=2)
    assert region["content"] == "def answer():\n    return 41\n"
    search = aci.search_text(query="answer", glob="*.py")
    assert search["matches"]
    symbol = aci.find_symbol("answer")
    assert symbol["symbols"]
    references = aci.find_references("answer")
    assert "references" in references
    repo_map = aci.repo_map(query="answer", focus_paths=("src.py",), token_budget=256)
    assert repo_map["authority"] == "advisory"
    assert aci.inspect_diff()["diff"] == ""

    with pytest.raises(BoundedACIError, match="aci_read_region_too_large"):
        aci.read_region(path="src.py", start_line=1, end_line=401)
    with pytest.raises(BoundedACIError, match="aci_git_metadata_forbidden"):
        aci.read_region(path=".git/config", start_line=1, end_line=1)


def test_typed_mutations_are_atomic_bounded_and_policy_bound(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    source = root / "src.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _commit(root)
    readonly = BoundedRepositoryACI(
        root, owner_user_id="owner-a", session_id="session-a", allow_mutation=False
    )
    with pytest.raises(BoundedACIError, match="aci_mutation_not_allowed"):
        readonly.edit_region(path="src.py", old_text="1", new_text="2")

    aci = BoundedRepositoryACI(
        root,
        owner_user_id="owner-a",
        session_id="session-a",
        allow_mutation=True,
        required_verification_argv=("pytest", "tests/test.py", "-q"),
    )
    edited = aci.edit_region(path="src.py", old_text="1", new_text="2")
    assert edited["replacements"] == 1
    created = aci.create_text_file(path="created.txt", content="created\n")
    assert created["path"] == "created.txt"
    assert source.read_text(encoding="utf-8") == "value = 2\n"
    with pytest.raises(BoundedACIError, match="aci_edit_anchor_not_unique"):
        aci.edit_region(path="src.py", old_text="missing", new_text="x")
    with pytest.raises(BoundedACIError, match="aci_create_target_invalid"):
        aci.create_text_file(path="created.txt", content="again")


def test_traversal_symlink_and_generic_shell_verifier_misuse_fail_closed(
    tmp_path: Path,
) -> None:
    root = _git_repo(tmp_path / "repo")
    (root / "src.py").write_text("value = 1\n", encoding="utf-8")
    _commit(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    aci = BoundedRepositoryACI(root, owner_user_id="owner-a", session_id="session-a")

    with pytest.raises(BoundedACIError, match="aci_path_escape"):
        aci.read_region(path="../outside.txt", start_line=1, end_line=1)
    with pytest.raises(BoundedACIError, match="aci_symlink_escape"):
        aci.read_region(path="link.txt", start_line=1, end_line=1)
    with pytest.raises(BoundedACIError, match="aci_verify_command_forbidden"):
        aci.verify(argv=("bash", "-c", "echo unsafe"))
    with pytest.raises(BoundedACIError, match="aci_verify_command_forbidden"):
        aci.verify(argv=("pytest", "--config-file", "/outside/config"))


def test_huge_diff_is_truncated_and_verification_timeout_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repo(tmp_path)
    source = root / "large.txt"
    source.write_text("base\n", encoding="utf-8")
    _commit(root)
    source.write_text("X" * 100_000 + "\n", encoding="utf-8")
    aci = BoundedRepositoryACI(
        root,
        owner_user_id="owner-a",
        session_id="session-a",
        required_verification_argv=("pytest", "tests/test.py", "-q"),
    )
    diff = aci.inspect_diff()
    assert diff["truncated"] is True
    assert len(str(diff["diff"]).encode("utf-8")) <= 64_000

    executable = root / "fake-pytest"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("research_workspace.bounded_aci.shutil.which", lambda _name: str(executable))
    result = aci.verify(argv=("pytest", "tests/test.py", "-q"), timeout_seconds=0.1)
    assert result["status"] == "FAIL"
    assert result["aborted"] == "aci_verify_timeout"
    assert result["worktree_mutated"] is False


def test_verification_cancellation_stops_process_and_preserves_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repo(tmp_path)
    (root / "src.py").write_text("value = 1\n", encoding="utf-8")
    _commit(root)
    executable = root / "fake-pytest"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("research_workspace.bounded_aci.shutil.which", lambda _name: str(executable))
    cancelled = False

    def is_cancelled() -> bool:
        return cancelled

    aci = BoundedRepositoryACI(
        root,
        owner_user_id="owner-a",
        session_id="session-a",
        required_verification_argv=("pytest", "tests/test.py", "-q"),
        is_cancelled=is_cancelled,
    )

    def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    timer = threading.Timer(0.1, cancel)
    timer.start()
    try:
        result = aci.verify(argv=("pytest", "tests/test.py", "-q"), timeout_seconds=5)
    finally:
        timer.join(timeout=2)
    assert result["aborted"] == "aci_verify_cancelled"
    assert result["worktree_mutated"] is False


def test_old_and_typed_edit_surfaces_preserve_frozen_repair_semantics(tmp_path: Path) -> None:
    old_root = _git_repo(tmp_path / "old")
    new_root = _git_repo(tmp_path / "new")
    for root in (old_root, new_root):
        (root / "src.py").write_text("value = 1\n", encoding="utf-8")
        _commit(root)

    coordinator = ZetsuAgentCoordinator.__new__(ZetsuAgentCoordinator)
    context = SimpleNamespace(
        worktree=old_root,
        binding=SimpleNamespace(tool_policy=SimpleNamespace(allowed_tools=("apply_patch",))),
    )
    old_observation = coordinator._edit(
        context,  # type: ignore[arg-type]
        {"path": "src.py", "old_text": "value = 1", "new_text": "value = 2"},
    )
    typed = BoundedRepositoryACI(
        new_root,
        owner_user_id="owner-a",
        session_id="session-a",
        allow_mutation=True,
    )
    new_observation = typed.edit_region(
        path="src.py", old_text="value = 1", new_text="value = 2"
    )
    assert old_observation == "EDITED:src.py:replacements=1"
    assert new_observation["replacements"] == 1
    assert (old_root / "src.py").read_text(encoding="utf-8") == (
        new_root / "src.py"
    ).read_text(encoding="utf-8")
