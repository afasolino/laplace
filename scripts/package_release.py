#!/usr/bin/env python3
"""Build twice, compare, inspect and clean-install the Laplace wheel/sdist."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence, TypedDict

ROOT = Path(__file__).resolve().parents[1]
PROJECT_VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
)
FORBIDDEN_PARTS = {
    ".env",
    ".git",
    ".models",
    ".runtime",
    ".tools",
    "data",
    "logs",
    "outputs",
}


class CommandResult(TypedDict):
    command: list[str]
    returncode: int
    status: str
    output_tail: str


class BuildResult(TypedDict):
    command: CommandResult
    artifacts: list[Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(arguments: Sequence[str], *, environment: dict[str, str]) -> CommandResult:
    completed = subprocess.run(  # nosec B603
        list(arguments),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return {
        "command": list(arguments),
        "returncode": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "output_tail": (completed.stdout + completed.stderr)[-4_000:],
    }


def _source_date_epoch() -> str:
    completed = subprocess.run(  # nosec B603 B607
        ["git", "show", "-s", "--format=%ct", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value.isdigit():
        raise RuntimeError("cannot_resolve_source_date_epoch")
    return value


def _safe_member(name: str) -> PurePosixPath:
    if "\x00" in name or "\\" in name:
        raise RuntimeError("unsafe_distribution_member")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("unsafe_distribution_member")
    lowered = {part.casefold() for part in path.parts}
    if lowered & FORBIDDEN_PARTS or any(part.startswith(".env") for part in lowered):
        raise RuntimeError("forbidden_distribution_member")
    return path


def _inspect_distribution(path: Path) -> dict[str, object]:
    members: list[str] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for zip_info in archive.infolist():
                logical = _safe_member(zip_info.filename)
                member_mode = (zip_info.external_attr >> 16) & 0o170000
                if member_mode == 0o120000:
                    raise RuntimeError("unsafe_distribution_member_type")
                if zip_info.file_size > 50 * 1024 * 1024:
                    raise RuntimeError("distribution_member_oversize")
                members.append(logical.as_posix())
    else:
        with tarfile.open(path, "r:gz") as archive:
            for tar_info in archive.getmembers():
                logical = _safe_member(tar_info.name)
                if tar_info.issym() or tar_info.islnk() or tar_info.isdev():
                    raise RuntimeError("unsafe_distribution_member_type")
                if tar_info.size > 50 * 1024 * 1024:
                    raise RuntimeError("distribution_member_oversize")
                members.append(logical.as_posix())
    if not members:
        raise RuntimeError("empty_distribution")
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "member_count": len(members),
        "safe_members": True,
    }


def _normalize_sdist(path: Path, *, epoch: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(path, "r:gz") as source, temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                    for original in sorted(source.getmembers(), key=lambda item: item.name):
                        logical = _safe_member(original.name)
                        if not (original.isfile() or original.isdir()):
                            raise RuntimeError("unsafe_distribution_member_type")
                        normalized = tarfile.TarInfo(logical.as_posix())
                        normalized.type = (
                            tarfile.DIRTYPE if original.isdir() else tarfile.REGTYPE
                        )
                        normalized.mode = 0o755 if original.isdir() else 0o644
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = "root"
                        normalized.gname = "root"
                        normalized.mtime = epoch
                        normalized.size = original.size if original.isfile() else 0
                        payload = source.extractfile(original) if original.isfile() else None
                        target.addfile(normalized, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build(target: Path, environment: dict[str, str]) -> BuildResult:
    target.mkdir()
    command = _command(
        [
            "uv",
            "build",
            "--offline",
            "--no-build-isolation",
            "--no-python-downloads",
            "--no-create-gitignore",
            "--out-dir",
            str(target),
            ".",
        ],
        environment=environment,
    )
    if command["status"] != "PASS":
        raise RuntimeError("package_build_failed:" + str(command["output_tail"]))
    artifacts = sorted(path for path in target.iterdir() if path.is_file())
    if len(artifacts) != 2 or {path.suffix for path in artifacts} != {".whl", ".gz"}:
        raise RuntimeError("package_artifact_set_invalid")
    sdist = next(path for path in artifacts if path.suffix == ".gz")
    _normalize_sdist(sdist, epoch=int(environment["SOURCE_DATE_EPOCH"]))
    return {"command": command, "artifacts": artifacts}


def _install_smoke(wheel: Path, environment: dict[str, str], temporary: Path) -> dict[str, object]:
    environment = {
        **environment,
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.pop("PYTHONPATH", None)
    venv.EnvBuilder(with_pip=True, clear=False).create(temporary)
    python = temporary / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = _command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        environment=environment,
    )
    if install["status"] != "PASS":
        raise RuntimeError("clean_install_failed")
    smoke = _command(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib.metadata as m, research_workspace as r;"
                f"assert r.__version__ == m.version('local-research-workspace') == {PROJECT_VERSION!r};"
                "print(r.__version__)"
            ),
        ],
        environment=environment,
    )
    if smoke["status"] != "PASS":
        raise RuntimeError("clean_import_smoke_failed")
    script = temporary / ("Scripts/laplace.exe" if os.name == "nt" else "bin/laplace")
    if not script.is_file():
        raise RuntimeError("console_entrypoint_missing")
    version = _command(
        [str(script), "--version"],
        environment=environment,
    )
    if version["status"] != "PASS" or f"laplace {PROJECT_VERSION} (" not in str(
        version["output_tail"]
    ):
        raise RuntimeError("console_entrypoint_smoke_failed")
    maintenance = temporary / (
        "Scripts/laplace-maintenance.exe" if os.name == "nt" else "bin/laplace-maintenance"
    )
    if not maintenance.is_file():
        raise RuntimeError("maintenance_console_entrypoint_missing")
    maintenance_status = _command(
        [
            str(maintenance),
            "--state-root",
            str(temporary / "maintenance-state"),
            "status",
        ],
        environment=environment,
    )
    if maintenance_status["status"] != "PASS" or '"mode": "SHADOW"' not in str(
        maintenance_status["output_tail"]
    ):
        raise RuntimeError("maintenance_console_entrypoint_smoke_failed")
    return {
        "install": install,
        "import_smoke": smoke,
        "console_entrypoint_smoke": version,
        "maintenance_console_entrypoint_smoke": maintenance_status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        {
            "SOURCE_DATE_EPOCH": _source_date_epoch(),
            "PYTHONHASHSEED": "0",
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="laplace-v8-package-", dir=output
        ) as temporary_name:
            temporary = Path(temporary_name)
            environment["UV_CACHE_DIR"] = str(temporary / "uv-cache")
            revision = subprocess.run(  # nosec B603 B607 - fixed read-only Git query
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if revision.returncode != 0 or len(revision.stdout.strip()) != 40:
                raise RuntimeError("package_revision_unavailable")
            environment["LAPLACE_BUILD_REVISION"] = revision.stdout.strip().lower()
            first = _build(temporary / "first", environment)
            second = _build(temporary / "second", environment)
            first_artifacts = {
                path.name: path for path in first["artifacts"]
            }
            second_artifacts = {
                path.name: path for path in second["artifacts"]
            }
            if first_artifacts.keys() != second_artifacts.keys():
                raise RuntimeError("package_filenames_not_reproducible")
            comparisons = {
                name: _sha256(path) == _sha256(second_artifacts[name])
                for name, path in first_artifacts.items()
            }
            if not all(comparisons.values()):
                raise RuntimeError("package_bytes_not_reproducible")
            copied: list[dict[str, object]] = []
            for name, source in sorted(first_artifacts.items()):
                target = output / name
                shutil.copy2(source, target)
                copied.append(_inspect_distribution(target))
            wheel = next(path for path in output.iterdir() if path.suffix == ".whl")
            install = _install_smoke(wheel, environment, temporary / "install")
        result = {
            "schema_version": 1,
            "status": "PASS",
            "source_date_epoch": environment["SOURCE_DATE_EPOCH"],
            "build_commands": [first["command"], second["command"]],
            "byte_reproducible": comparisons,
            "artifacts": copied,
            "clean_environment": install,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        result = {
            "schema_version": 1,
            "status": "FAIL",
            "category": str(exc).split(":", 1)[0],
            "detail": str(exc)[-4_000:],
        }
        if os.environ.get("GITHUB_ACTIONS") == "true":
            annotation = str(exc)[-4_000:].replace("%", "%25")
            annotation = annotation.replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error title=Laplace package audit failed::{annotation}")
    (output / "package_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
