"""Semantic application version and sanitized Git build identity."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TypedDict

from . import __version__

_REVISION = re.compile(r"^[a-f0-9]{40}$")
_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


class VersionRecord(TypedDict):
    application_version: str
    git_revision: str
    build_identity: str


def git_revision(repository_root: Path) -> str:
    configured = os.environ.get("LAPLACE_BUILD_REVISION")
    if configured is not None:
        normalized = configured.strip().lower()
        return normalized if _REVISION.fullmatch(normalized) else "unavailable"
    try:
        completed = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(repository_root.resolve()), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    revision = completed.stdout.strip().lower()
    return revision if completed.returncode == 0 and _REVISION.fullmatch(revision) else "unavailable"


def version_record(repository_root: Path) -> VersionRecord:
    if not _SEMVER.fullmatch(__version__):
        raise RuntimeError("application version is not semantic")
    revision = git_revision(repository_root)
    short = revision[:12] if revision != "unavailable" else revision
    return {
        "application_version": __version__,
        "git_revision": revision,
        "build_identity": f"{__version__}+git.{short}",
    }


def version_line(repository_root: Path) -> str:
    record = version_record(repository_root)
    return f"laplace {record['application_version']} ({record['git_revision']})"
