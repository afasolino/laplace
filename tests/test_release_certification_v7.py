from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from research_workspace.release_certification import (
    CertificationArchiveError,
    create_archive,
    redact_private_paths,
    verify_archive,
)


def test_certification_archive_is_deterministic_safe_and_hash_verified(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "one.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    (output / "two.txt").write_text("fixture evidence\n", encoding="utf-8")
    first = output / "first.tar.gz"
    second = output / "second.tar.gz"
    first_hash = create_archive(output, ("one.json", "two.txt"), first)
    second_hash = create_archive(output, ("one.json", "two.txt"), second)
    assert first_hash == second_hash
    assert verify_archive(first) == {
        "status": "PASS",
        "member_count": 3,
        "manifest_hashes_verified": 2,
        "unsafe_archive_paths": False,
    }


def test_archive_verifier_rejects_traversal_and_manifest_mismatch(tmp_path: Path) -> None:
    malicious = tmp_path / "malicious.tar.gz"
    with malicious.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                payload = b"escape"
                info = tarfile.TarInfo("../escape")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(CertificationArchiveError, match="unsafe"):
        verify_archive(malicious)

    output = tmp_path / "mismatch"
    output.mkdir()
    (output / "evidence.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    archive = output / "bundle.tar.gz"
    create_archive(output, ("evidence.json",), archive)
    with tarfile.open(archive, "r:gz") as source:
        manifest = json.loads(source.extractfile("manifest.json").read())  # type: ignore[union-attr]
    manifest["files"][0]["sha256"] = "0" * 64
    replacement = output / "replacement.tar.gz"
    with replacement.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as target:
                payload = (output / "evidence.json").read_bytes()
                evidence = tarfile.TarInfo("evidence.json")
                evidence.size = len(payload)
                target.addfile(evidence, io.BytesIO(payload))
                manifest_payload = json.dumps(manifest).encode()
                manifest_info = tarfile.TarInfo("manifest.json")
                manifest_info.size = len(manifest_payload)
                target.addfile(manifest_info, io.BytesIO(manifest_payload))
    with pytest.raises(CertificationArchiveError, match="hash mismatch"):
        verify_archive(replacement)


def test_private_path_redaction_is_recursive() -> None:
    value = {
        "path": "/private/worktree/file",
        "nested": ["/tmp/laplace-secret-fixture/state", "safe"],
    }
    redacted = redact_private_paths(value, {"/private/worktree": "<worktree>"})
    assert redacted == {
        "path": "<worktree>/file",
        "nested": ["<temporary>", "safe"],
    }

