"""Fail-closed production model profile selection and Qwen3.6 rollback."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError(f"invalid_selector:{path}")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _approved_model_roots(repository_root: Path) -> tuple[Path, ...]:
    """Return canonical operator-approved roots for immutable model artifacts."""

    roots: list[Path] = [repository_root.resolve()]
    raw = os.environ.get("LAPLACE_APPROVED_MODEL_ROOTS", "")
    for value in raw.split(os.pathsep):
        value = value.strip()
        if not value:
            continue

        configured = Path(value).expanduser()
        if not configured.is_absolute():
            raise RuntimeError("qwen38_approved_artifact_root_not_absolute")
        if configured.is_symlink():
            raise RuntimeError("qwen38_approved_artifact_root_symlink")

        try:
            resolved = configured.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("qwen38_approved_artifact_root_unavailable") from exc

        if not resolved.is_dir():
            raise RuntimeError("qwen38_approved_artifact_root_not_directory")
        if resolved != configured.absolute():
            raise RuntimeError("qwen38_approved_artifact_root_symlink_parent")

        if resolved not in roots:
            roots.append(resolved)

    return tuple(roots)


def verify_qwen38_artifact(repository_root: Path) -> dict[str, object]:
    """Verify every packed artifact byte against the immutable local manifest."""

    root = repository_root.resolve()
    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 3:
        raise RuntimeError("qwen38_manifest_invalid")
    artifact_path_value = raw.get("artifact_path")
    artifact_manifest_value = raw.get("artifact_manifest")
    if not isinstance(artifact_path_value, str) or not isinstance(artifact_manifest_value, str):
        raise RuntimeError("qwen38_artifact_provenance_missing")
    configured_artifact_root = Path(artifact_path_value)
    if configured_artifact_root.is_symlink():
        raise RuntimeError("qwen38_artifact_root_symlink")
    artifact_root = configured_artifact_root.resolve()

    approved_roots = _approved_model_roots(root)
    if not any(_path_is_within(artifact_root, approved) for approved in approved_roots):
        raise RuntimeError("qwen38_artifact_path_outside_repository")

    artifact_manifest_relative = Path(artifact_manifest_value)
    if artifact_manifest_relative.is_absolute() or ".." in artifact_manifest_relative.parts:
        raise RuntimeError("qwen38_artifact_manifest_path_mismatch")

    artifact_manifest_path = (root / artifact_manifest_relative).resolve()
    if artifact_manifest_path.parent != artifact_root:
        if (
            artifact_manifest_relative.name != "laplace_artifact_manifest.json"
            or artifact_manifest_relative.parent.name != artifact_root.name
        ):
            raise RuntimeError("qwen38_artifact_manifest_path_mismatch")

        candidate = artifact_root / artifact_manifest_relative.name
        if candidate.is_symlink():
            raise RuntimeError("qwen38_artifact_manifest_path_mismatch")

        artifact_manifest_path = candidate.resolve()
        if (
            artifact_manifest_path.parent != artifact_root
            or artifact_manifest_path != candidate.absolute()
        ):
            raise RuntimeError("qwen38_artifact_manifest_path_mismatch")
    artifact_raw: object = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(artifact_raw, dict) or artifact_raw.get("schema_version") != 1:
        raise RuntimeError("qwen38_artifact_manifest_invalid")
    files = artifact_raw.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("qwen38_artifact_files_missing")

    expected_paths: set[str] = set()
    verified_files: list[dict[str, object]] = []
    verified_size = 0
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("qwen38_artifact_file_record_invalid")
        name = item.get("path")
        expected_size = item.get("size")
        expected_digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            raise RuntimeError("qwen38_artifact_file_record_invalid")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in expected_paths:
            raise RuntimeError("qwen38_artifact_file_path_invalid")
        path = artifact_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"qwen38_artifact_file_missing:{name}")
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(artifact_root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"qwen38_artifact_file_escape:{name}") from exc
        if resolved_path != path.absolute():
            raise RuntimeError(f"qwen38_artifact_file_symlink_parent:{name}")
        if path.stat().st_size != expected_size or _sha256(path) != expected_digest:
            raise RuntimeError(f"qwen38_artifact_file_hash_mismatch:{name}")
        expected_paths.add(name)
        verified_size += expected_size
        verified_files.append({"path": name, "size": expected_size, "sha256": expected_digest})

    observed_paths = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
        and path.relative_to(artifact_root).parts[0] != ".cache"
        and path.name != "laplace_artifact_manifest.json"
    }
    if observed_paths != expected_paths:
        raise RuntimeError("qwen38_artifact_file_set_mismatch")
    canonical = json.dumps(verified_files, sort_keys=True, separators=(",", ":")).encode()
    artifact_digest = hashlib.sha256(canonical).hexdigest()
    recipe = raw.get("source_provenance")
    if not isinstance(recipe, dict):
        raise RuntimeError("qwen38_recipe_provenance_missing")
    recipe_path_value = recipe.get("recipe_path")
    recipe_digest = recipe.get("recipe_sha256")
    if not isinstance(recipe_path_value, str) or not isinstance(recipe_digest, str):
        raise RuntimeError("qwen38_recipe_provenance_missing")
    recipe_path = (root / recipe_path_value).resolve()
    try:
        recipe_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("qwen38_recipe_path_outside_repository") from exc
    if (
        artifact_digest != artifact_raw.get("artifact_sha256")
        or artifact_digest != raw.get("artifact_sha256")
        or artifact_raw.get("base_revision") != raw.get("base_revision")
        or artifact_raw.get("recipe_sha256") != recipe_digest
        or _sha256(recipe_path) != recipe_digest
        or verified_size != raw.get("artifact_size_bytes")
    ):
        raise RuntimeError("qwen38_artifact_provenance_mismatch")
    return raw


def qwen38_manifest(repository_root: Path) -> dict[str, object]:
    """Reject a Qwen3.8 start unless immutable provenance and certification pass."""

    root = repository_root.resolve()
    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 3
        or manifest.get("promotion_allowed") is not True
        or manifest.get("certification_status") != "PASSED"
        or not manifest.get("artifact_sha256")
        or not manifest.get("base_revision")
        or not manifest.get("artifact_revision")
        or not manifest.get("tokenizer_revision")
        or not isinstance(manifest.get("certification_evidence"), dict)
        or not isinstance(manifest.get("quantization"), dict)
        or manifest["quantization"].get("status") != "ARTIFACT_VERIFIED"
    ):
        raise RuntimeError("qwen38_promotion_not_certified")
    return verify_qwen38_artifact(root)


def assert_qwen38_promotable(repository_root: Path) -> None:
    qwen38_manifest(repository_root)


def select(
    profile: str,
    repository_root: Path,
    output: Path | None = None,
    *,
    selection_evidence: str | None = None,
) -> Path:
    """Atomically select certified Qwen3.8 or the retained Qwen3.6 rollback."""

    root = repository_root.resolve()
    target = (
        output.resolve() if output is not None else root / "configs/selected_serving_profiles.json"
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
    if selection_evidence is not None:
        if profile != "qwen38" or not selection_evidence.strip():
            raise RuntimeError("invalid_selection_evidence")
        selected["selection_evidence"] = selection_evidence
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
