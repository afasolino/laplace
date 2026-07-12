from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
DEFAULT_FORMALSCIENCE = Path(r"C:\Users\andre\OneDrive\Desktop\dottorato\FormalScience")
COLLECTIONS = ("MyWorks", "MyTopics", "OtherTopics", "Documentations", "ExpRes")


class ProjectError(ValueError):
    """Unsafe or invalid FormalScience project operation."""


@dataclass(frozen=True)
class ProjectPaths:
    name: str
    root: Path
    config: Path
    data: Path
    outputs: Path


def formalscience_root() -> Path:
    return Path(os.getenv("FORMALSCIENCE_ROOT", str(DEFAULT_FORMALSCIENCE))).expanduser()


def validate_project_name(name: str) -> str:
    if not PROJECT_NAME.fullmatch(name) or name in {".", "..", "_ProjectTemplate"}:
        raise ProjectError(
            "Project names must be 1-80 ASCII letters, digits, '.', '_' or '-' and cannot be reserved"
        )
    return name


def project_paths(name: str, root: Path | None = None) -> ProjectPaths:
    validate_project_name(name)
    if root is not None:
        candidate = Path(root).expanduser().resolve()
        laplace_config = candidate / ".laplace" / "project.yaml"
        if laplace_config.is_file():
            if candidate.name != name:
                raise ProjectError("Laplace project name does not match its directory")
            return ProjectPaths(name, candidate, laplace_config, candidate / "Data", candidate / "Outputs")
    formal = (root or formalscience_root()).resolve()
    workspace = (formal / "Workspace").resolve()
    target = (workspace / name).resolve()
    if workspace not in target.parents:
        raise ProjectError("Project path escapes FormalScience/Workspace")
    return ProjectPaths(name, target, target / "project.yaml", target / "Data", target / "Outputs")


def _tree(paths: ProjectPaths) -> dict[str, Path]:
    values = {
        "Config": paths.root / "Config",
        "Parsed": paths.data / "Parsed",
        "Metadata": paths.data / "Metadata",
        "VectorStore": paths.data / "VectorStore",
        "Cache": paths.data / "Cache",
        "Logs": paths.data / "Logs",
        "Quarantine": paths.data / "Quarantine",
        "Downloads": paths.data / "Downloads",
        "OpenAccess": paths.data / "Downloads" / "OpenAccess",
        "IEEE": paths.data / "Downloads" / "IEEE",
        "Pending": paths.data / "Downloads" / "IEEE" / "Pending",
        "Downloaded": paths.data / "Downloads" / "IEEE" / "Downloaded",
        "Failed": paths.data / "Downloads" / "IEEE" / "Failed",
        "Drafts": paths.outputs / "Drafts",
        "EvidencePackets": paths.outputs / "EvidencePackets",
        "Extractions": paths.outputs / "Extractions",
        "Comparisons": paths.outputs / "Comparisons",
        "Reports": paths.outputs / "Reports",
    }
    return values


def init_project(name: str, *, root: Path | None = None, update: bool = False) -> ProjectPaths:
    paths = project_paths(name, root)
    if paths.root.exists() and not update:
        raise ProjectError(
            f"Project already exists: {paths.root}; pass update=True for a non-destructive update"
        )
    for directory in _tree(paths).values():
        directory.mkdir(parents=True, exist_ok=True)
    if not paths.config.exists() or update:
        config = {
            "project": {
                "name": name,
                "created_at": datetime.now(UTC).isoformat(),
                "shared_library_root": str((root or formalscience_root()) / "Library"),
            },
            "library": {
                "collections": list(COLLECTIONS),
                "selected": ["MyWorks"],
                "reference_only": True,
            },
            "paths": {"data": str(paths.data), "outputs": str(paths.outputs)},
            "models": {"main_text": "qwen3:4b", "embedding": "qwen3-embedding:0.6b"},
            "retrieval": {
                "chunk_tokens": 650,
                "overlap_tokens": 100,
                "final_evidence_chunks": 6,
                "require_page_grounding": True,
            },
            "online": {
                "enabled": True,
                "providers": ["crossref", "openalex", "arxiv", "ieee"],
                "result_limit": 10,
                "timeout_seconds": 20,
                "offline": False,
            },
            "ieee_acquisition": {
                "api_key_env": "IEEE_XPLORE_API_KEY",
                "default_batch_size": 1,
                "max_batch_size": 3,
                "approval_required": True,
                "browser_profile": str(Path.home() / "AppData" / "Local" / "FormalScienceBrowser"),
            },
            "writing": {"style_profile": "formal IEEE-style English"},
            "security": {
                "localhost_only": True,
                "no_credential_storage": True,
                "no_automatic_subscribed_download": True,
            },
        }
        paths.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return paths


def load_project(name: str, *, root: Path | None = None) -> tuple[ProjectPaths, dict[str, Any]]:
    paths = project_paths(name, root)
    if not paths.config.is_file():
        raise ProjectError(f"Project configuration not found: {paths.config}")
    try:
        config = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProjectError(f"Invalid project.yaml: {exc}") from exc
    if not isinstance(config, dict) or config.get("project", {}).get("name") != name:
        raise ProjectError("project.yaml name does not match the project directory")
    return paths, config


def list_projects(*, root: Path | None = None) -> list[str]:
    workspace = (root or formalscience_root()) / "Workspace"
    if not workspace.is_dir():
        return []
    return sorted(
        path.name
        for path in workspace.iterdir()
        if path.is_dir() and path.name != "_ProjectTemplate" and (path / "project.yaml").is_file()
    )


def validate_project(name: str, *, root: Path | None = None) -> dict[str, Any]:
    paths, config = load_project(name, root=root)
    missing = [str(path) for path in _tree(paths).values() if not path.is_dir()]
    return {
        "valid": not missing,
        "project": name,
        "root": str(paths.root),
        "missing_directories": missing,
        "config_keys": sorted(config.keys()),
    }


def project_summary(name: str, *, root: Path | None = None) -> dict[str, Any]:
    paths, config = load_project(name, root=root)
    return {
        "project": name,
        "root": str(paths.root),
        "config": config,
        "validation": validate_project(name, root=root),
    }
