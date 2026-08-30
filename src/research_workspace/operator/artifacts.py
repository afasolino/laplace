"""Artifact-path authorization used by Operator transport routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def safe_artifact_path(
    relative_path: str,
    *,
    state_root: Path,
    repository_root: Path,
    allow_sensitive: bool,
) -> Path:
    """Resolve a bounded, public artifact path or raise an HTTP error."""

    if "\x00" in relative_path or Path(relative_path).is_absolute():
        raise HTTPException(status_code=400, detail="invalid_artifact_path")
    if relative_path.startswith("outputs/"):
        candidate = (repository_root / relative_path).resolve()
    else:
        candidate = (state_root / relative_path).resolve()
    roots = (state_root.resolve(), (repository_root / "outputs").resolve())
    if not any(_is_within(candidate, root) for root in roots) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="artifact_not_found")
    lowered = {part.lower() for part in candidate.parts}
    forbidden = {"held_out", "held-out", "secrets", "credentials", "prompts"}
    if not allow_sensitive and lowered.intersection(forbidden):
        raise HTTPException(status_code=403, detail="artifact_access_forbidden")
    allowed_suffixes = {
        ".json", ".jsonl", ".md", ".html", ".txt", ".log", ".sv", ".v",
        ".zip", ".gz", ".tar", ".bib",
    }
    if candidate.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=403, detail="artifact_type_forbidden")
    if candidate.stat().st_size > 256_000_000:
        raise HTTPException(status_code=413, detail="artifact_too_large")
    return candidate


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
