"""Strict, deterministic orchestration-upgrade certification bundles."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import subprocess  # nosec B404
import tarfile
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

from .execution_records import canonical_json_bytes

JsonObject: TypeAlias = dict[str, object]

REQUIRED_CERTIFICATION_FILES = (
    "summary.md",
    "result_compact.json",
    "events.jsonl",
    "skills.lock.json",
    "context.lock.json",
    "corpus.lock.json",
    "models.lock.json",
    "tools.lock.json",
    "run.lock.json",
    "verification_compact.json",
    "review_compact.json",
    "rag_compact.json",
    "relevant_model_calls.jsonl",
    "trace.jsonl",
    "metrics.json",
    "final_source.sv",
    "final_source.sha256",
    "verilator_build_log.json",
    "verilator_simulation_log.json",
    "iverilog_log.json",
    "vvp_public_log.json",
    "vvp_adversarial_log.json",
    "yosys_log.json",
    "git_status.txt",
    "git_diff_stat.txt",
    "test_results.txt",
    "external_reference_audit.md",
)


class CertificationBundleError(RuntimeError):
    """The certification directory is incomplete or unsafe to package."""


def _git(repository_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    if completed.returncode != 0:
        raise CertificationBundleError(
            f"git {' '.join(arguments)} failed with {completed.returncode}"
        )
    return completed.stdout


def prepare_support_files(
    project_root: Path,
    repository_root: Path,
    *,
    baseline_revision: str,
    test_results_path: Path,
    summary_markdown: str,
) -> None:
    """Materialize required repository/test evidence without altering sources."""

    project = project_root.resolve()
    repository = repository_root.resolve()
    if not test_results_path.is_file():
        raise CertificationBundleError("Preserved test-results evidence is missing")
    project.mkdir(parents=True, exist_ok=True)
    (project / "summary.md").write_text(
        summary_markdown.rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    (project / "git_status.txt").write_text(
        _git(repository, ["status", "--short", "--branch"]),
        encoding="utf-8",
        newline="\n",
    )
    (project / "git_diff_stat.txt").write_text(
        _git(repository, ["diff", "--stat", baseline_revision, "HEAD"]),
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(test_results_path, project / "test_results.txt")
    shutil.copy2(
        repository / "docs/external_reference_audit.md",
        project / "external_reference_audit.md",
    )


def _tar_info(path: Path, name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    stat = path.stat()
    info.size = stat.st_size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def create_certification_bundle(
    project_root: Path,
    destination: Path,
    *,
    extra_files: Mapping[str, Path] | None = None,
) -> JsonObject:
    """Create a byte-reproducible gzip tar after verifying every required file."""

    project = project_root.resolve()
    destination = destination.resolve()
    missing = [
        name for name in REQUIRED_CERTIFICATION_FILES if not (project / name).is_file()
    ]
    if missing:
        raise CertificationBundleError(
            "Certification files are missing: " + ", ".join(missing)
        )
    entries = {name: project / name for name in REQUIRED_CERTIFICATION_FILES}
    for name, path in (extra_files or {}).items():
        if (
            not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or not path.resolve().is_file()
        ):
            raise CertificationBundleError(f"Unsafe extra certification file: {name}")
        entries[name] = path.resolve()
    manifest: JsonObject = {
        "schema_version": 1,
        "files": [
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sorted(entries.items())
        ],
    }
    entries["bundle_manifest.json"] = Path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, path in sorted(entries.items()):
                    if name == "bundle_manifest.json":
                        data = canonical_json_bytes(manifest) + b"\n"
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
                    else:
                        with path.open("rb") as handle:
                            archive.addfile(_tar_info(path, name), handle)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, destination)
    return {
        "status": "CERTIFICATION_BUNDLE_READY",
        "path": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "file_count": len(entries),
        "manifest": manifest,
    }
