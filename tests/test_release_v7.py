from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from research_workspace import __version__
from research_workspace.entrypoints import laplace_main as packaged_laplace_main
from research_workspace.laplace_cli import main as laplace_main
from research_workspace.versioning import git_revision, version_line, version_record

ROOT = Path(__file__).resolve().parents[1]


def test_semantic_version_and_build_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    revision = "a" * 40
    monkeypatch.setenv("LAPLACE_BUILD_REVISION", revision)
    assert __version__ == "0.7.0"
    record = version_record(ROOT)
    assert record == {
        "application_version": "0.7.0",
        "git_revision": revision,
        "build_identity": "0.7.0+git.aaaaaaaaaaaa",
    }
    assert version_line(ROOT) == f"laplace 0.7.0 ({revision})"
    monkeypatch.setenv("LAPLACE_BUILD_REVISION", "unsafe revision")
    assert git_revision(ROOT) == "unavailable"


def test_laplace_version_and_configuration_validation_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LAPLACE_BUILD_REVISION", "b" * 40)
    assert laplace_main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"laplace 0.7.0 ({'b' * 40})"
    for name in tuple(os.environ):
        if name.startswith("LAPLACE_CONFIG_"):
            monkeypatch.delenv(name)
    diagnostic = tmp_path / "configuration-diagnostic.json"
    assert laplace_main(
        [
            "--validate-config",
            str(ROOT / "configs/laplace.example.yaml"),
            "--configuration-mode",
            "desktop",
            "--configuration-state-root",
            "fixture/state",
            "--diagnostic-export",
            str(diagnostic),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"status": "PASS"' in output
    assert "/srv/laplace/state" not in output
    assert diagnostic.is_file()
    assert diagnostic.stat().st_mode & 0o077 == 0
    assert "/srv/laplace/state" not in diagnostic.read_text(encoding="utf-8")


def test_dependency_light_packaged_version_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LAPLACE_BUILD_REVISION", "c" * 40)
    assert packaged_laplace_main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"laplace 0.7.0 ({'c' * 40})"


def test_exported_schema_manifest_hashes_and_strictness() -> None:
    root = ROOT / "schemas/v7"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 15
    assert len(manifest["files"]) == 15
    for item in manifest["files"]:
        path = root / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False


def test_cpu_fixture_lock_and_manifest_exclude_runtime_state() -> None:
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").casefold()
    assert "==" in lock
    for forbidden in ("torch==", "cuda", "nvidia-", "triton==", "rocm"):
        assert forbidden not in lock
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for required in ("prune outputs", "prune data", "prune logs", "global-exclude .env"):
        assert required in manifest
