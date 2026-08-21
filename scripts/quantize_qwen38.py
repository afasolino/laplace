#!/usr/bin/env python3
"""Reproducibly quantize the pinned official Qwen3.8 checkpoint for Ampere."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


CONFIG = Path("configs/quantization/qwen38_27b_w4a16_awq.json")
MANIFEST_NAME = "laplace_artifact_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recipe(repository_root: Path) -> dict[str, Any]:
    raw: object = json.loads((repository_root / CONFIG).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError("invalid_qwen38_quantization_recipe")
    return raw


def verify_source(recipe: dict[str, Any]) -> dict[str, str]:
    source = Path(str(recipe["source_path"])).resolve()
    expected = recipe.get("source_files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("qwen38_source_hashes_missing")
    observed: dict[str, str] = {}
    for name, expected_digest in sorted(expected.items()):
        if not isinstance(name, str) or not isinstance(expected_digest, str):
            raise RuntimeError("qwen38_source_hashes_invalid")
        path = source / name
        if not path.is_file():
            raise RuntimeError(f"qwen38_source_file_missing:{name}")
        digest = sha256_file(path)
        if digest != expected_digest:
            raise RuntimeError(f"qwen38_source_hash_mismatch:{name}")
        observed[name] = digest
    return observed


def tree_manifest(output: Path) -> tuple[list[dict[str, object]], str]:
    files: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if (
            not path.is_file()
            or relative.parts[0] == ".cache"
            or relative.as_posix() == MANIFEST_NAME
        ):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return files, hashlib.sha256(canonical).hexdigest()


def verify_artifact(repository_root: Path, recipe: dict[str, Any]) -> dict[str, object]:
    """Verify every output byte plus the lossless native MTP payload."""

    import torch
    from safetensors import safe_open

    output = Path(str(recipe["output_path"])).resolve()
    manifest_path = output / MANIFEST_NAME
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError("qwen38_artifact_manifest_invalid")
    expected_files = raw.get("files")
    observed_files, observed_digest = tree_manifest(output)
    if expected_files != observed_files or raw.get("artifact_sha256") != observed_digest:
        raise RuntimeError("qwen38_artifact_hash_mismatch")
    if raw.get("base_revision") != recipe.get("base_revision"):
        raise RuntimeError("qwen38_artifact_base_revision_mismatch")
    if raw.get("recipe_sha256") != sha256_file(repository_root / CONFIG):
        raise RuntimeError("qwen38_artifact_recipe_mismatch")

    config: object = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") if isinstance(config, dict) else None
    groups = quantization.get("config_groups") if isinstance(quantization, dict) else None
    group = groups.get("group_0") if isinstance(groups, dict) else None
    weights = group.get("weights") if isinstance(group, dict) else None
    if (
        not isinstance(quantization, dict)
        or quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("format") != "pack-quantized"
        or not isinstance(weights, dict)
        or weights.get("num_bits") != 4
        or weights.get("group_size") != 128
        or weights.get("symmetric") is not True
    ):
        raise RuntimeError("qwen38_artifact_quantization_schema_mismatch")

    source = Path(str(recipe["source_path"])).resolve()
    source_index: object = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    source_map = source_index.get("weight_map") if isinstance(source_index, dict) else None
    mtp_path = output / "model_mtp.safetensors"
    if not isinstance(source_map, dict) or not mtp_path.is_file():
        raise RuntimeError("qwen38_artifact_mtp_missing")
    mtp_records: list[dict[str, object]] = []
    with safe_open(mtp_path, framework="pt", device="cpu") as destination:
        for key in sorted(destination.keys()):
            source_shard = source_map.get(key)
            if not isinstance(source_shard, str):
                raise RuntimeError(f"qwen38_artifact_mtp_source_missing:{key}")
            with safe_open(source / source_shard, framework="pt", device="cpu") as origin:
                source_bytes = (
                    origin.get_tensor(key).contiguous().view(torch.uint8).numpy().tobytes()
                )
            artifact_bytes = (
                destination.get_tensor(key).contiguous().view(torch.uint8).numpy().tobytes()
            )
            source_digest = hashlib.sha256(source_bytes).hexdigest()
            artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
            if source_digest != artifact_digest:
                raise RuntimeError(f"qwen38_artifact_mtp_mismatch:{key}")
            mtp_records.append({"key": key, "bytes": len(source_bytes), "sha256": source_digest})
    if len(mtp_records) != 15:
        raise RuntimeError("qwen38_artifact_mtp_tensor_count_mismatch")
    return {
        "status": "ARTIFACT_VERIFIED",
        "artifact_sha256": observed_digest,
        "file_count": len(observed_files),
        "mtp_tensor_count": len(mtp_records),
        "mtp_total_bytes": sum(int(record["bytes"]) for record in mtp_records),
        "mtp_shard_sha256": sha256_file(mtp_path),
    }


def _version(name: str) -> str:
    return importlib.metadata.version(name)


def _validate_environment(recipe: dict[str, Any]) -> dict[str, str]:
    runtime = recipe.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("qwen38_quantization_runtime_missing")
    package_keys = {
        "llmcompressor": "llmcompressor",
        "compressed_tensors": "compressed-tensors",
        "transformers": "transformers",
        "torch": "torch",
        "numpy": "numpy",
        "datasets": "datasets",
        "huggingface_hub": "huggingface-hub",
    }
    import torch

    observed = {
        "python": platform.python_version(),
        **{key: _version(package) for key, package in package_keys.items()},
        "quantization_torch_cuda_runtime": str(torch.version.cuda),
    }
    mismatches = {
        key: {"expected": runtime.get(key), "observed": value}
        for key, value in observed.items()
        if runtime.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "qwen38_quantization_environment_mismatch:" + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def _prepare_dataset(recipe: dict[str, Any], tokenizer: Any) -> Any:
    from datasets import load_dataset

    calibration = recipe["calibration"]
    samples = int(calibration["samples"])
    dataset = load_dataset(
        calibration["repository"],
        revision=calibration["revision"],
        split=f"{calibration['split']}[:{samples}]",
    ).shuffle(seed=int(calibration["shuffle_seed"]))
    template_kwargs = dict(calibration["chat_template_kwargs"])

    def preprocess(example: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                **template_kwargs,
            )
        }

    return dataset.map(preprocess)


def _quantize(recipe: dict[str, Any]) -> None:
    import numpy
    import torch
    from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    source = Path(str(recipe["source_path"])).resolve()
    output = Path(str(recipe["output_path"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_nonempty_artifact:{output}")
    output.mkdir(parents=True, exist_ok=True)
    offload = source.parent.parent / "offload" / "qwen38_quantization"
    offload.mkdir(parents=True, exist_ok=True)
    runtime = recipe["runtime"]
    calibration = recipe["calibration"]
    seed = int(calibration["python_numpy_torch_seed"])
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gpu_budget = int(runtime["quantization_gpu_weight_budget_gib"])
    cpu_budget = int(runtime["quantization_cpu_weight_budget_gib"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(  # nosec B615
        source,
        dtype="auto",
        device_map="auto",
        max_memory={0: f"{gpu_budget}GiB", "cpu": f"{cpu_budget}GiB"},
        offload_folder=offload,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)  # nosec B615
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)  # nosec B615
    dataset = _prepare_dataset(recipe, tokenizer)
    quantization = recipe["quantization"]
    recipe_modifiers = [
        AWQModifier(
            offload_device=torch.device("cpu"),
            duo_scaling=bool(quantization["awq_duo_scaling"]),
            n_grid=int(quantization["awq_grid_points"]),
        ),
        QuantizationModifier(
            scheme=str(quantization["scheme"]),
            targets=list(quantization["targets"]),
            ignore=list(quantization["ignore"]),
        ),
    ]
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe_modifiers,
        max_seq_length=int(calibration["sequence_length"]),
        num_calibration_samples=int(calibration["samples"]),
        shuffle_calibration_samples=bool(calibration["llmcompressor_secondary_shuffle"]),
    )
    model.save_pretrained(
        output,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output)
    processor.save_pretrained(output)
    for name in ("LICENSE", "README.md", "chat_template.jinja", "generation_config.json"):
        shutil.copy2(source / name, output / name)
    save_mtp_tensors_to_checkpoint(source_model=str(source), dest_dir=str(output))


def _write_manifest(
    repository_root: Path,
    recipe: dict[str, Any],
    source_hashes: dict[str, str],
    environment: dict[str, str],
) -> dict[str, object]:
    output = Path(str(recipe["output_path"])).resolve()
    files, artifact_digest = tree_manifest(output)
    recipe_path = repository_root / CONFIG
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": recipe["artifact_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_model": recipe["base_model"],
        "base_revision": recipe["base_revision"],
        "source_files": source_hashes,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "recipe_path": CONFIG.as_posix(),
        "recipe_sha256": sha256_file(recipe_path),
        "calibration": recipe["calibration"],
        "quantization": recipe["quantization"],
        "environment": environment,
        "files": files,
        "artifact_sha256": artifact_digest,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acknowledge-local-quantization",
        action="store_true",
        help="acknowledge the long-running, local-only quantization job",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the pinned source and environment without quantizing",
    )
    parser.add_argument(
        "--verify-artifact",
        action="store_true",
        help="verify source, recipe, packed artifact, and lossless native MTP tensors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    recipe = load_recipe(root)
    source_hashes = verify_source(recipe)
    environment = _validate_environment(recipe)
    if arguments.verify_artifact:
        print(json.dumps(verify_artifact(root, recipe), sort_keys=True))
        return 0
    if arguments.verify_only:
        print(
            json.dumps(
                {
                    "status": "VERIFIED",
                    "base_revision": recipe["base_revision"],
                    "source_file_count": len(source_hashes),
                    "environment": environment,
                },
                sort_keys=True,
            )
        )
        return 0
    if not arguments.acknowledge_local_quantization:
        raise RuntimeError("acknowledgement_required")
    _quantize(recipe)
    manifest = _write_manifest(root, recipe, source_hashes, environment)
    print(json.dumps({"status": "CREATED", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
