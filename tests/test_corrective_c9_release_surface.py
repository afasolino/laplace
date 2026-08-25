from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from research_workspace import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_is_v2_and_matches_runtime_version() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.0.0"' in pyproject
    assert __version__ == "2.0.0"


def test_readme_describes_actual_v2_control_plane() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "standalone `LaplaceCore`",
        "authenticated optional Zetsu/Codex adapter",
        "laplace zetsu start --nocodev",
        "laplace zetsu sessions --json",
        "repository_state_not_materialized",
        "C8",
        "production/v1",
    ):
        assert phrase in readme
    assert "keeps one main Ollama generation active by default" not in readme
    assert "v0.7 architecture" not in readme


def test_zetsu_and_user_docs_match_current_commands_and_contract() -> None:
    zetsu = (ROOT / "docs/ZETSU.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/USER_GUIDE.md").read_text(encoding="utf-8")
    for text in (zetsu, guide):
        assert "laplace zetsu start --nocodev" in text
        assert "laplace zetsu sessions --json" in text
        assert "repository_not_authorized" in text
        assert "SiliconMind" in text
    assert "Zetsu schema version is 1.5" in zetsu
    assert "get_result" in zetsu


def test_documentation_reference_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_documentation.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "PASS"' in result.stdout
