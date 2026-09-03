from __future__ import annotations

import subprocess
from pathlib import Path

from research_workspace.bounded_aci import BoundedRepositoryACI


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *argv],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark")
    return repo


def test_find_paths_discovers_tracked_hidden_skill(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    skill = repo / ".agents" / "skills" / "zetsu" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Zetsu\n", encoding="utf-8")
    (repo / "src.py").write_text("print('x')\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")

    aci = BoundedRepositoryACI(
        repo,
        owner_user_id="plus-local",
        session_id="path-discovery",
        allow_mutation=False,
    )
    result = aci.find_paths(query="skill", glob="*", limit=20)

    assert result["paths"] == [".agents/skills/zetsu/SKILL.md"]
    assert result["tracked_only"] is True
    assert result["truncated"] is False
    assert result["total_matches"] == 1


def test_find_paths_excludes_untracked_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "tracked.md").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.md")
    _git(repo, "commit", "-qm", "base")
    (repo / "untracked.md").write_text("untracked\n", encoding="utf-8")

    aci = BoundedRepositoryACI(
        repo,
        owner_user_id="plus-local",
        session_id="tracked-only",
        allow_mutation=False,
    )
    result = aci.find_paths(glob="*.md")

    assert result["paths"] == ["tracked.md"]
