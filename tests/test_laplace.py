from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace import laplace_cli
from research_workspace.laplace_cli import (
    LaplaceError,
    _ingest,
    _project_from_dir,
    detect_project,
    init_laplace,
    main,
)


def _home(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(laplace_cli, "APP_HOME", path)
    monkeypatch.setattr(laplace_cli, "REGISTRY_PATH", path / "projects.json")
    monkeypatch.setattr(laplace_cli, "CONFIG_PATH", path / "config.yaml")


def test_laplace_init_registry_and_parent_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _home(monkeypatch, tmp_path / "home")
    result = init_laplace("Demo", cwd=tmp_path)
    root = Path(result["project"])
    assert (root / ".laplace" / "project.yaml").is_file()
    monkeypatch.chdir(root / "Data" / "Parsed")
    paths, config = detect_project()
    assert paths.root == root
    assert config["project"]["name"] == "Demo"
    registry = json.loads((tmp_path / "home" / "projects.json").read_text(encoding="utf-8"))
    assert registry[0]["name"] == "Demo"


def test_laplace_registry_conflict_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _home(monkeypatch, tmp_path / "home")
    init_laplace("Same", cwd=tmp_path / "one")
    (tmp_path / "two").mkdir()
    with pytest.raises(LaplaceError, match="already registered"):
        init_laplace("Same", cwd=tmp_path / "two")


def test_laplace_ingest_dry_run_rejects_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _home(monkeypatch, tmp_path / "home")
    root = Path(init_laplace("Demo", cwd=tmp_path)["project"])
    paths, config = _project_from_dir(root)
    with pytest.raises(LaplaceError, match="cannot contain"):
        _ingest(paths, config, "MyWorks/../OtherTopics", dry_run=True)


def test_laplace_init_refuses_application_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _home(monkeypatch, tmp_path / "home")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='app'\n", encoding="utf-8")
    with pytest.raises(LaplaceError, match="application repository"):
        init_laplace("Nested", cwd=tmp_path / "child")
    assert not (tmp_path / "child" / ".laplace").exists()


def test_laplace_init_refuses_library_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _home(monkeypatch, tmp_path / "home")
    with pytest.raises(LaplaceError, match="Library"):
        init_laplace("Nested", cwd=tmp_path / "Library" / "MyWorks")


def test_laplace_validate_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _home(monkeypatch, tmp_path / "home")
    root = Path(init_laplace("Demo", cwd=tmp_path)["project"])
    assert main(["--project", str(root), "--validate"]) == 0
    assert '"valid": true' in capsys.readouterr().out
