from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from research_workspace.certification_bundle import (
    REQUIRED_CERTIFICATION_FILES,
    CertificationBundleError,
    create_certification_bundle,
)


def test_certification_bundle_is_strict_and_byte_reproducible(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in REQUIRED_CERTIFICATION_FILES:
        (project / name).write_text(f"{name}\n", encoding="utf-8")
    first = create_certification_bundle(project, tmp_path / "first.tar.gz")
    second = create_certification_bundle(project, tmp_path / "second.tar.gz")

    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "first.tar.gz").read_bytes() == (
        tmp_path / "second.tar.gz"
    ).read_bytes()
    with tarfile.open(tmp_path / "first.tar.gz", "r:gz") as archive:
        names = archive.getnames()
    assert names == sorted([*REQUIRED_CERTIFICATION_FILES, "bundle_manifest.json"])


def test_certification_bundle_fails_closed_when_evidence_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(CertificationBundleError, match="missing"):
        create_certification_bundle(tmp_path, tmp_path / "bundle.tar.gz")
