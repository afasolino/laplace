"""Fail-closed Qwen3.8 P8 production selection and artifact verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

PRODUCTION_PROFILE_ID = "P8_qwen38_w4a16_mtp"
PRODUCTION_MODEL_ID = "laplace-quality-qwen38-mtp8"
PRODUCTION_PROFILE_SOURCE_COMMIT = "54fa762ff9bf7273c320e04183fdd69db391688b"
PRODUCTION_MODEL_RELATIVE = ".models/Qwen3.8-27B-AWQ-4bit-e6b4b8b025f8"
PRODUCTION_SELECTION_EVIDENCE = (
    f"git:{PRODUCTION_PROFILE_SOURCE_COMMIT}#{PRODUCTION_PROFILE_ID}"
)
PRODUCTION_MTP_TOKENS = 8

_EXPECTED_PROFILE: dict[str, object] = {
    "profile_id": PRODUCTION_PROFILE_ID,
    "model_route": "quality",
    "model_path": PRODUCTION_MODEL_RELATIVE,
    "served_model_name": PRODUCTION_MODEL_ID,
    "port": 8207,
    "max_model_len": 131072,
    "max_num_seqs": 2,
    "max_num_batched_tokens": 8192,
    "kv_cache_dtype": "fp8",
    "kv_cache_memory_bytes": 5905580032,
    "enable_prefix_caching": True,
    "prefix_hash_algorithm": "sha256",
    "enable_chunked_prefill": True,
    "scheduling_policy": "fcfs",
    "cpu_offload_gb": 0.0,
    "cpu_offload_params": [],
    "offload_backend": "auto",
    "offload_group_size": 0,
    "offload_num_in_group": 1,
    "offload_prefetch_step": 1,
    "kv_offloading_size": None,
    "kv_offloading_backend": "native",
    "gpu_memory_utilization": 0.755,
    "startup_timeout": 1200,
    "request_timeout": 300,
    "extra_args": [
        "--language-model-only",
        "--reasoning-parser=qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_xml",
        "--mamba-ssm-cache-dtype=bfloat16",
        '--speculative-config={"method":"mtp","num_speculative_tokens":8}',
    ],
}

_EXPECTED_SELECTOR: dict[str, object] = {
    "schema_version": 1,
    "selection_evidence": PRODUCTION_SELECTION_EVIDENCE,
    "default_profile_id": PRODUCTION_PROFILE_ID,
    "high_context_profile_id": PRODUCTION_PROFILE_ID,
    "quality_reserved_slots": 1,
    "standard_capacity": 2,
    "economy_capacity": 4,
    "routes": {
        "quality": {
            "model_id": PRODUCTION_MODEL_ID,
            "endpoint": "http://127.0.0.1:8207",
            "priority": 0,
            "context_limit": 131072,
            "output_limit": 4096,
        },
        "standard": {
            "model_id": PRODUCTION_MODEL_ID,
            "endpoint": "http://127.0.0.1:8207",
            "priority": 10,
            "context_limit": 131072,
            "output_limit": 2048,
        },
        "economy": {
            "model_id": "laplace-codev-r1-rl-qwen-7b-w4a16",
            "endpoint": "http://127.0.0.1:8103",
            "priority": 20,
            "context_limit": 8192,
            "output_limit": 2048,
        },
    },
}


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


def _approved_external_model_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
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


def _artifact_root(repository_root: Path, configured_value: str) -> Path:
    configured = Path(configured_value).expanduser()
    if configured.is_absolute():
        if configured.is_symlink():
            raise RuntimeError("qwen38_artifact_root_symlink")
        resolved = configured.resolve()
        approved = _approved_external_model_roots()
        if not approved or not any(_path_is_within(resolved, root) for root in approved):
            raise RuntimeError("qwen38_artifact_path_outside_repository")
        return resolved
    if not configured.parts or configured == Path(".") or ".." in configured.parts:
        raise RuntimeError("qwen38_artifact_relative_path_invalid")
    candidate = repository_root / configured
    if candidate.is_symlink():
        raise RuntimeError("qwen38_artifact_root_symlink")
    resolved = candidate.resolve()
    if not _path_is_within(resolved, repository_root):
        raise RuntimeError("qwen38_artifact_path_outside_repository")
    if resolved != candidate.absolute():
        raise RuntimeError("qwen38_artifact_root_symlink_parent")
    return resolved


def verify_qwen38_artifact(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve()
    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 3:
        raise RuntimeError("qwen38_manifest_invalid")
    artifact_path_value = raw.get("artifact_path")
    artifact_manifest_value = raw.get("artifact_manifest")
    if not isinstance(artifact_path_value, str) or not isinstance(artifact_manifest_value, str):
        raise RuntimeError("qwen38_artifact_provenance_missing")
    artifact_root = _artifact_root(root, artifact_path_value)

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


def _validate_production_configuration(root: Path, manifest: dict[str, object]) -> None:
    expected_production = {
        "profile_id": PRODUCTION_PROFILE_ID,
        "served_model_name": PRODUCTION_MODEL_ID,
        "profile_source_commit": PRODUCTION_PROFILE_SOURCE_COMMIT,
        "requested_speculative_tokens": PRODUCTION_MTP_TOKENS,
        "status": "FROZEN_OPTIMIZED_STEP_B",
    }
    if manifest.get("production_profile") != expected_production:
        raise RuntimeError("qwen38_p8_production_manifest_invalid")

    profile_root = root / "configs/serving_profiles"
    active_files = sorted(path.name for path in profile_root.glob("*.json"))
    if active_files != [f"{PRODUCTION_PROFILE_ID}.json"]:
        raise RuntimeError("qwen38_active_profile_set_invalid")
    active: object = json.loads(
        (profile_root / f"{PRODUCTION_PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    if active != _EXPECTED_PROFILE:
        raise RuntimeError("qwen38_p8_profile_drift")

    selector_files = sorted(
        path.name for path in (root / "configs").glob("selected_serving_profiles*.json")
    )
    if selector_files != ["selected_serving_profiles.json"]:
        raise RuntimeError("qwen38_legacy_selector_present")
    selected = _load(root / "configs/selected_serving_profiles.json")
    expected_routes = _EXPECTED_SELECTOR["routes"]
    selected_routes = selected.get("routes")
    if not isinstance(expected_routes, dict) or not isinstance(selected_routes, dict):
        raise RuntimeError("qwen38_p8_selector_drift")

    for key in (
        "schema_version",
        "selection_evidence",
        "default_profile_id",
        "high_context_profile_id",
        "quality_reserved_slots",
        "standard_capacity",
        "economy_capacity",
    ):
        if selected.get(key) != _EXPECTED_SELECTOR.get(key):
            raise RuntimeError("qwen38_p8_selector_drift")

    for lane in ("quality", "standard"):
        if selected_routes.get(lane) != expected_routes.get(lane):
            raise RuntimeError("qwen38_p8_selector_drift")

    candidate_files = sorted(
        path.name for path in (root / "configs/serving_profile_candidates").glob("*.json")
    )
    if candidate_files != [f"{PRODUCTION_PROFILE_ID}.json"]:
        raise RuntimeError("qwen38_legacy_candidate_present")


def qwen38_manifest(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve()
    manifest = verify_qwen38_artifact(root)
    quantization = manifest.get("quantization")
    if (
        not manifest.get("artifact_sha256")
        or not manifest.get("base_revision")
        or not manifest.get("artifact_revision")
        or not manifest.get("tokenizer_revision")
        or not isinstance(quantization, dict)
        or quantization.get("status") != "ARTIFACT_VERIFIED"
    ):
        raise RuntimeError("qwen38_promotion_not_certified")
    _validate_production_configuration(root, manifest)
    return manifest


def assert_qwen38_promotable(repository_root: Path) -> None:
    qwen38_manifest(repository_root)


def select(
    profile: str,
    repository_root: Path,
    output: Path | None = None,
    *,
    selection_evidence: str | None = None,
) -> Path:
    root = repository_root.resolve()
    if profile != "qwen38":
        raise RuntimeError(f"unknown_profile:{profile}")
    qwen38_manifest(root)
    source = root / "configs/selected_serving_profiles.json"
    selected = _load(source)
    if selection_evidence is not None and selection_evidence != PRODUCTION_SELECTION_EVIDENCE:
        raise RuntimeError("invalid_selection_evidence")
    if output is None:
        return source

    target = output.resolve()
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
