"""Offline dependency and license inventories for release evidence."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from email.message import Message
from pathlib import Path
from typing import cast

_LOCK = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)$")
_FORBIDDEN_GPU = re.compile(r"(?i)(?:^|[-_.])(cuda|nvidia|torch|triton|rocm)(?:$|[-_.])")


def _normalized(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def locked_dependencies(path: Path) -> tuple[tuple[str, str], ...]:
    dependencies: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK.fullmatch(line)
        if match is None:
            raise ValueError(f"unlocked dependency at line {line_number}")
        name, version = match.groups()
        if _FORBIDDEN_GPU.search(name):
            raise ValueError(f"GPU runtime dependency forbidden in CPU lock: {name}")
        dependencies.append((name, version))
    if not dependencies:
        raise ValueError("dependency lock is empty")
    return tuple(dependencies)


def dependency_inventory(path: Path) -> dict[str, object]:
    locked = locked_dependencies(path)
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        metadata = cast(Message, distribution.metadata)
        name = metadata.get("Name")
        if name:
            installed[_normalized(name)] = distribution.version
    entries = [
        {
            "name": name,
            "locked_version": version,
            "installed_version": installed.get(_normalized(name)),
            "installed_matches": installed.get(_normalized(name)) == version,
        }
        for name, version in locked
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "schema_version": 1,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "lock_format": "exact",
        "dependency_count": len(entries),
        "gpu_runtime_dependencies": [],
        "pip_check_returncode": completed.returncode,
        "pip_check": (completed.stdout + completed.stderr).strip()[:4000],
        "entries": entries,
        "network_vulnerability_database_queried": False,
        "limitation": "Offline inventory does not query a current vulnerability database.",
    }


def license_inventory(path: Path) -> dict[str, object]:
    locked = locked_dependencies(path)
    distributions = {
        _normalized(cast(Message, distribution.metadata).get("Name", "")): distribution
        for distribution in importlib.metadata.distributions()
    }
    entries: list[dict[str, object]] = []
    unknown = 0
    for name, version in locked:
        distribution = distributions.get(_normalized(name))
        expression = ""
        if distribution is not None:
            metadata = cast(Message, distribution.metadata)
            expression = (
                metadata.get("License-Expression")
                or metadata.get("License")
                or ""
            ).strip()
            if not expression:
                classifiers = metadata.get_all("Classifier") or []
                expression = "; ".join(
                    classifier.removeprefix("License :: ").strip()
                    for classifier in classifiers
                    if classifier.startswith("License :: ")
                )
        if not expression:
            expression = "UNKNOWN_REVIEW_REQUIRED"
            unknown += 1
        entries.append({"name": name, "version": version, "license": expression[:1000]})
    return {
        "schema_version": 1,
        "status": "PASS",
        "dependency_count": len(entries),
        "unknown_license_count": unknown,
        "entries": entries,
        "policy": "Inventory only; deployment operator reviews UNKNOWN_REVIEW_REQUIRED entries.",
    }
