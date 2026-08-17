"""Fail-closed production model profile selection and Qwen3.6 rollback."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError(f"invalid_selector:{path}")
    return raw


def qwen38_manifest(repository_root: Path) -> dict[str, object]:
    """Reject a Qwen3.8 start unless immutable provenance and certification pass."""

    manifest_path = (
        repository_root.resolve()
        / "configs/model_manifests/qwen38_27b_a6000.json"
    )
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("promotion_allowed") is not True
        or manifest.get("certification_status") != "PASSED"
        or not manifest.get("artifact_sha256")
        or not manifest.get("base_revision")
        or not manifest.get("artifact_revision")
        or not manifest.get("tokenizer_revision")
    ):
        raise RuntimeError("qwen38_promotion_not_certified")
    return manifest


def assert_qwen38_promotable(repository_root: Path) -> None:
    qwen38_manifest(repository_root)


def select(profile: str, repository_root: Path, output: Path | None = None) -> Path:
    """Atomically select certified Qwen3.8 or the retained Qwen3.6 rollback."""

    root = repository_root.resolve()
    target = (
        output.resolve()
        if output is not None
        else root / "configs/selected_serving_profiles.json"
    )
    if profile == "qwen38":
        manifest = qwen38_manifest(root)
        mtp = manifest.get("mtp")
        mtp_passed = isinstance(mtp, dict) and mtp.get("status") == "PASSED"
        source = root / (
            "configs/selected_serving_profiles.qwen38-mtp.json"
            if mtp_passed
            else "configs/selected_serving_profiles.qwen38.json"
        )
    elif profile == "qwen36":
        source = root / "configs/selected_serving_profiles.qwen36.rollback.json"
    else:
        raise RuntimeError(f"unknown_profile:{profile}")
    selected = _load(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(selected, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
