#!/usr/bin/env python3
"""Pin, materialize, hash, and stage an existing Qwen3.8 quantized checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "configs/model_manifests/qwen38_prequantized_candidates.json"
MANIFEST = ROOT / "configs/model_manifests/qwen38_27b_a6000.json"
SOURCE_DESCRIPTOR = ROOT / "configs/model_manifests/qwen38_prequantized_source.json"
PROFILE_IDS = ("P6_qwen38_w4a16", "P7_qwen38_w4a16_mtp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--candidate", choices=("primary", "fallback"), default="primary")
    parser.add_argument("--revision", default=None, help="Optional HF reference to resolve to an immutable revision.")
    parser.add_argument(
        "--fallback-justification",
        default=None,
        help="Required when selecting the fallback checkpoint; record the demonstrated primary incompatibility.",
    )
    parser.add_argument("--replace", action="store_true", help="Replace an existing materialized target.")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"json_object_required:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_snapshot(
    snapshot: Path,
    target: Path,
    candidate: dict[str, Any],
    *,
    replace: bool,
) -> dict[str, object]:
    """Stage and validate a checkpoint completely before replacing any existing target."""

    models_root = target.parent.resolve()
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(models_root)
    except ValueError as exc:
        raise RuntimeError("target_outside_models_root") from exc
    if target.exists() and not replace:
        raise RuntimeError("prequantized_target_exists_use_replace")

    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    moved_old = False
    try:
        for source in snapshot.rglob("*"):
            relative = source.relative_to(snapshot)
            if ".cache" in relative.parts:
                continue
            destination = staging / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not source.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=True)

        checkpoint_validation = _validate_snapshot(staging, candidate)

        if target.exists():
            os.replace(target, backup)
            moved_old = True
        try:
            os.replace(staging, target)
        except Exception:
            if moved_old and not target.exists() and backup.exists():
                os.replace(backup, target)
                moved_old = False
            raise
        if moved_old:
            shutil.rmtree(backup)
            moved_old = False
        return checkpoint_validation
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if moved_old and backup.exists() and not target.exists():
            os.replace(backup, target)
        elif backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _safetensor_header_tensors(path: Path) -> dict[str, str]:
    """Read tensor dtypes from a safetensors header without loading payload bytes."""

    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise RuntimeError("prequantized_safetensors_header_invalid")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2 or header_size > 128 * 1024 * 1024 or header_size > size - 8:
                raise RuntimeError("prequantized_safetensors_header_invalid")
            raw = handle.read(header_size)
    except OSError as exc:
        raise RuntimeError("prequantized_safetensors_header_unreadable") from exc
    try:
        header: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("prequantized_safetensors_header_invalid") from exc
    if not isinstance(header, dict):
        raise RuntimeError("prequantized_safetensors_header_invalid")
    tensors: dict[str, str] = {}
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
        if (
            not isinstance(key, str)
            or not isinstance(metadata, dict)
            or not isinstance(metadata.get("dtype"), str)
        ):
            raise RuntimeError("prequantized_safetensors_header_invalid")
        tensors[key] = str(metadata["dtype"])
    return tensors


def _checkpoint_tensor_metadata(target: Path) -> tuple[dict[str, str], str]:
    """Read tensor dtypes from the main index plus any unindexed safetensors assets."""

    tensors: dict[str, str] = {}
    indexed_shards: set[str] = set()
    sources: list[str] = []
    index_path = target / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items())
        ):
            raise RuntimeError("prequantized_safetensors_index_invalid")
        indexed_shards.update(str(value) for value in weight_map.values())
        for shard_name in sorted(indexed_shards):
            shard_rel = Path(shard_name)
            shard_path = target / shard_rel
            if (
                shard_rel.is_absolute()
                or ".." in shard_rel.parts
                or shard_rel.suffix != ".safetensors"
                or not shard_path.is_file()
            ):
                raise RuntimeError("prequantized_safetensors_index_shard_missing")
            actual = _safetensor_header_tensors(shard_path)
            actual_keys = set(actual)
            expected_keys = {
                key for key, mapped_shard in weight_map.items() if mapped_shard == shard_name
            }
            if not expected_keys <= actual_keys:
                raise RuntimeError("prequantized_safetensors_index_tensor_missing")
            tensors.update(actual)
            sources.append(f"index+header:{shard_name}")

    shards = sorted(target.rglob("*.safetensors"))
    if not shards and not tensors:
        raise RuntimeError("prequantized_safetensors_missing")
    for shard in shards:
        relative = shard.relative_to(target).as_posix()
        if relative in indexed_shards:
            continue
        tensors.update(_safetensor_header_tensors(shard))
        sources.append(f"header:{relative}")
    if not tensors:
        raise RuntimeError("prequantized_tensor_metadata_missing")
    return tensors, "+".join(sources) if sources else "safetensors_headers"


def _linear_quantization_groups(quantization: dict[str, Any]) -> list[dict[str, Any]]:
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict):
        raise RuntimeError("prequantized_config_groups_missing")
    matched: list[dict[str, Any]] = []
    for raw_group in groups.values():
        if not isinstance(raw_group, dict):
            continue
        targets = raw_group.get("targets")
        if isinstance(targets, list) and any(str(target).casefold() == "linear" for target in targets):
            matched.append(raw_group)
    if not matched:
        raise RuntimeError("prequantized_linear_quantization_group_missing")
    return matched


def _validate_snapshot(target: Path, candidate: dict[str, Any]) -> dict[str, object]:
    config_path = target / "config.json"
    if not config_path.is_file():
        raise RuntimeError("prequantized_config_missing")
    config = _read_json(config_path)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen3_5ForConditionalGeneration" not in architectures:
        raise RuntimeError("prequantized_architecture_mismatch")

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise RuntimeError("prequantized_quantization_config_missing")
    expected_format = str(candidate.get("format", "")).casefold()
    if expected_format != "compressed-tensors":
        raise RuntimeError("prequantized_candidate_format_unsupported")
    if str(quantization.get("quant_method", "")).casefold() != "compressed-tensors":
        raise RuntimeError("prequantized_quant_method_mismatch")
    if str(quantization.get("format", "")).casefold() != "pack-quantized":
        raise RuntimeError("prequantized_storage_format_mismatch")

    expected_bits = candidate.get("weights_bits")
    expected_group_size = candidate.get("group_size")
    expected_symmetric = candidate.get("symmetric")
    expected_activation_bits = candidate.get("activations_bits")
    group_records: list[dict[str, object]] = []
    for group in _linear_quantization_groups(quantization):
        weights = group.get("weights")
        if not isinstance(weights, dict):
            raise RuntimeError("prequantized_linear_weight_config_missing")
        if weights.get("num_bits") != expected_bits:
            raise RuntimeError("prequantized_weight_bits_mismatch")
        if weights.get("group_size") != expected_group_size:
            raise RuntimeError("prequantized_group_size_mismatch")
        if str(weights.get("type", "")).casefold() != "int":
            raise RuntimeError("prequantized_weight_type_mismatch")
        if str(weights.get("strategy", "")).casefold() != "group":
            raise RuntimeError("prequantized_weight_strategy_mismatch")
        if isinstance(expected_symmetric, bool) and weights.get("symmetric") is not expected_symmetric:
            raise RuntimeError("prequantized_symmetric_mismatch")
        if expected_activation_bits == 16 and (
            group.get("input_activations") is not None or group.get("output_activations") is not None
        ):
            raise RuntimeError("prequantized_activation_quantization_mismatch")
        group_records.append(
            {
                "targets": list(group.get("targets", [])) if isinstance(group.get("targets"), list) else [],
                "weight_bits": weights.get("num_bits"),
                "group_size": weights.get("group_size"),
                "symmetric": weights.get("symmetric"),
                "strategy": weights.get("strategy"),
                "type": weights.get("type"),
            }
        )

    tensor_metadata, tensor_source = _checkpoint_tensor_metadata(target)
    tensor_keys = set(tensor_metadata)
    text_config = config.get("text_config")
    mtp_layers = (
        text_config.get("mtp_num_hidden_layers")
        if isinstance(text_config, dict)
        else config.get("mtp_num_hidden_layers")
    )
    mtp_configured = (
        isinstance(mtp_layers, int)
        and not isinstance(mtp_layers, bool)
        and mtp_layers > 0
    )

    def is_mtp_tensor(key: str) -> bool:
        parts = key.casefold().split(".")
        return any(part == "mtp" or part.startswith("mtp_") for part in parts)

    mtp_tensor_keys = sorted(key for key in tensor_keys if is_mtp_tensor(key))
    mtp_tensors_present = bool(mtp_tensor_keys)
    if candidate.get("preserves_mtp") and not (mtp_configured and mtp_tensors_present):
        raise RuntimeError("prequantized_mtp_metadata_missing")

    protected_prefixes = candidate.get("protected_bf16_prefixes", [])
    if (
        not isinstance(protected_prefixes, list)
        or not all(isinstance(item, str) and item for item in protected_prefixes)
    ):
        raise RuntimeError("prequantized_protected_bf16_prefixes_invalid")
    ignored = quantization.get("ignore")
    if protected_prefixes and (
        not isinstance(ignored, list)
        or not all(isinstance(item, str) and item for item in ignored)
    ):
        raise RuntimeError("prequantized_quantization_ignore_missing")
    protected_counts: dict[str, int] = {}
    for prefix in protected_prefixes:
        matched = {
            key: dtype for key, dtype in tensor_metadata.items() if key.startswith(prefix)
        }
        if not matched:
            raise RuntimeError(f"prequantized_protected_bf16_tensor_missing:{prefix}")
        if any(dtype != "BF16" for dtype in matched.values()):
            raise RuntimeError(f"prequantized_protected_bf16_dtype_mismatch:{prefix}")
        root = prefix.rstrip(".")
        if not any(
            target == root or target.startswith(root + ".") or root.startswith(target + ".")
            for target in ignored
        ):
            raise RuntimeError(f"prequantized_protected_bf16_not_ignored:{prefix}")
        protected_counts[prefix] = len(matched)

    dtype_counts: dict[str, int] = {}
    for dtype in tensor_metadata.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return {
        "quant_method": quantization.get("quant_method"),
        "storage_format": quantization.get("format"),
        "linear_quantization_groups": group_records,
        "tensor_metadata_source": tensor_source,
        "tensor_key_count": len(tensor_keys),
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "protected_bf16_prefix_counts": protected_counts,
        "protected_bf16_verified": bool(protected_prefixes),
        "mtp_configured_layers": mtp_layers if mtp_configured else 0,
        "mtp_tensor_count": len(mtp_tensor_keys),
        "mtp_tensors_present": mtp_tensors_present,
    }


def _artifact_records(target: Path) -> tuple[list[dict[str, object]], int, str]:
    records: list[dict[str, object]] = []
    total = 0
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.name == "laplace_artifact_manifest.json":
            continue
        relative = path.relative_to(target).as_posix()
        size = path.stat().st_size
        records.append({"path": relative, "size": size, "sha256": _sha256(path)})
        total += size
    if not records:
        raise RuntimeError("prequantized_snapshot_empty")
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return records, total, hashlib.sha256(canonical).hexdigest()


def _update_profile(root: Path, profile_id: str, model_path: Path) -> str:
    path = root / "configs/serving_profile_candidates" / f"{profile_id}.json"
    profile = _read_json(path)
    profile["model_path"] = str(model_path)
    profile["max_model_len"] = 131072
    _write_json(path, profile)
    return _sha256(path)


def _selection_justification(candidate: str, fallback_justification: object) -> str | None:
    if candidate != "fallback":
        return None
    if not isinstance(fallback_justification, str) or not fallback_justification.strip():
        raise RuntimeError("fallback_requires_demonstrated_primary_incompatibility")
    value = fallback_justification.strip()
    if len(value) > 2_000:
        raise RuntimeError("fallback_justification_too_long")
    return value


def main() -> int:
    args = _parser().parse_args()
    root = args.repository_root.resolve()
    candidates = _read_json(root / CANDIDATES.relative_to(ROOT))
    candidate = candidates.get(args.candidate)
    if not isinstance(candidate, dict):
        raise RuntimeError("candidate_missing")
    fallback_justification = _selection_justification(
        args.candidate, args.fallback_justification
    )
    repo_id = candidate.get("repository")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("candidate_repository_invalid")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub_required: install it in the model-preparation environment"
        ) from exc

    requested_revision = args.revision or "main"
    info = HfApi().model_info(repo_id, revision=requested_revision)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or len(revision) < 12:
        raise RuntimeError("immutable_hf_revision_unavailable")
    snapshot = Path(snapshot_download(repo_id=repo_id, revision=revision)).resolve()

    target_name = f"{repo_id.split('/')[-1]}-{revision[:12]}"
    target = root / ".models" / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_validation = _copy_snapshot(
        snapshot,
        target,
        candidate,
        replace=args.replace,
    )

    descriptor = root / SOURCE_DESCRIPTOR.relative_to(ROOT)
    descriptor_sha = _sha256(descriptor)
    records, size_bytes, artifact_sha = _artifact_records(target)

    current = _read_json(root / MANIFEST.relative_to(ROOT))
    base_revision = current.get("base_revision")
    if not isinstance(base_revision, str) or not base_revision:
        raise RuntimeError("base_revision_missing")

    artifact_manifest = {
        "schema_version": 1,
        "repository": repo_id,
        "revision": revision,
        "base_revision": base_revision,
        "recipe_sha256": descriptor_sha,
        "artifact_sha256": artifact_sha,
        "files": records,
    }
    artifact_manifest_path = target / "laplace_artifact_manifest.json"
    _write_json(artifact_manifest_path, artifact_manifest)

    profile_hashes = {
        profile_id: _update_profile(root, profile_id, target) for profile_id in PROFILE_IDS
    }

    current["artifact_path"] = str(target)
    current["artifact_revision"] = f"hf:{repo_id}@{revision}"
    current["artifact_sha256"] = artifact_sha
    current["artifact_size_bytes"] = size_bytes
    current["artifact_manifest"] = artifact_manifest_path.relative_to(root).as_posix()
    current["tokenizer_revision"] = revision
    current["artifact_present"] = True
    current["availability_status"] = "AVAILABLE_PREQUANTIZED_HUGGINGFACE"
    prepared_at = datetime.now(UTC).isoformat()
    current["availability_checked_at_utc"] = prepared_at
    current["artifact_policy"] = {
        "selected_repository": repo_id,
        "selected_revision": revision,
        "selection": args.candidate,
        "fallback_justification": fallback_justification,
        "local_quantization_role": "final_fallback_only",
        "prepared_at_utc": prepared_at,
    }
    upstream = current.get("upstream_inspection")
    if isinstance(upstream, dict):
        upstream["production_context_tokens"] = 131072
    current["source_provenance"] = {
        "verified_file_count": len(records),
        "source_tree_sha256": artifact_sha,
        "recipe_path": SOURCE_DESCRIPTOR.relative_to(ROOT).as_posix(),
        "recipe_sha256": descriptor_sha,
        "source_repository": repo_id,
        "source_revision": revision,
        "checkpoint_validation": checkpoint_validation,
    }
    current["quantization"] = {
        "status": "ARTIFACT_VERIFIED",
        "format": candidate.get("format"),
        "algorithm": candidate.get("algorithm"),
        "weights_bits": candidate.get("weights_bits", 4),
        "activations_bits": candidate.get("activations_bits", 16),
        "group_size": candidate.get("group_size", 128),
        "symmetric": candidate.get("symmetric"),
        "kernel_target": candidate.get("kernel_target", "ampere_marlin_w4a16"),
        "mtp_copy": "PRESENT_UNCERTIFIED" if checkpoint_validation["mtp_tensors_present"] else "UNKNOWN",
    }
    current["architecture_probe"] = {
        "status": "NOT_RUN_PREQUANTIZED_ARTIFACT",
        "scope": "Previous dummy/local-artifact probes are not certification for this checkpoint.",
    }
    current["serving_candidates"] = {
        "P6_qwen38_w4a16": {
            "profile_sha256": profile_hashes["P6_qwen38_w4a16"],
            "max_model_len": 131072,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_xml",
            "live_status": "NOT_RUN_ARTIFACT_CHANGED",
        },
        "P7_qwen38_w4a16_mtp": {
            "profile_sha256": profile_hashes["P7_qwen38_w4a16_mtp"],
            "max_model_len": 131072,
            "requested_speculative_tokens": 3,
            "live_status": "NOT_RUN_ARTIFACT_CHANGED",
        },
    }
    current["mtp"] = {
        "requested_speculative_tokens": 3,
        "asset_status": "PRESENT_UNCERTIFIED" if checkpoint_validation["mtp_tensors_present"] else "UNKNOWN",
        "runtime_status": "NOT_RUN_ARTIFACT_CHANGED",
        "status": "NOT_CERTIFIED",
    }
    current["external_blocker"] = None
    current["certification_status"] = "NOT_RUN_ARTIFACT_CHANGED"
    current["promotion_allowed"] = False
    _write_json(root / MANIFEST.relative_to(ROOT), current)

    print(
        json.dumps(
            {
                "status": "PREPARED_FAIL_CLOSED",
                "repository": repo_id,
                "revision": revision,
                "artifact_path": str(target),
                "artifact_sha256": artifact_sha,
                "artifact_size_bytes": size_bytes,
                "production_context_tokens": 131072,
                "promotion_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
