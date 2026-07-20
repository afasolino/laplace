#!/usr/bin/env python3
"""Reproducible pinned W4A16 preparation for the two Phase-2 models.

This script is intentionally not part of experiment execution. It requires an
explicit acknowledgement because the exact artifacts have not yet been
quantized and served on this A6000.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404
from pathlib import Path
from typing import Any


EXPERIMENT = Path("codex_a6000/experiments/multilanguage_dual_model_ablation_v1")
SHARED_EXPERT_GATE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight$"
)
PACKED_SHARED_EXPERT_GATE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.shared_expert_gate\."
    r"(?:weight_packed|weight_scale|weight_shape)$"
)


def _ensure_single_shard_index(output: Path) -> None:
    index_path = output / "model.safetensors.index.json"
    model_path = output / "model.safetensors"
    if index_path.is_file() or not model_path.is_file():
        return
    from safetensors import safe_open

    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        weight_map = {name: model_path.name for name in sorted(checkpoint.keys())}
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": model_path.stat().st_size},
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _restore_unquantized_shared_expert_gates(source: Path, output: Path) -> None:
    """Restore router gates that vLLM deliberately loads without quantization."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    model_path = output / "model.safetensors"
    index_path = output / "model.safetensors.index.json"
    if not model_path.is_file() or not index_path.is_file():
        raise RuntimeError("Phase-2 main artifact is missing its model checkpoint or index")

    source_gates: dict[str, Any] = {}
    for checkpoint_path in sorted(source.glob("*.safetensors")):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if SHARED_EXPERT_GATE.fullmatch(name):
                    source_gates[name] = checkpoint.get_tensor(name).clone()

    source_config: dict[str, Any] = json.loads(
        (source / "config.json").read_text(encoding="utf-8")
    )
    text_config = source_config.get("text_config", source_config)
    if not isinstance(text_config, dict):
        raise RuntimeError("source model has a malformed text_config")
    expected_gates = int(text_config["num_hidden_layers"])
    if len(source_gates) != expected_gates:
        raise RuntimeError(
            "source checkpoint has an unexpected shared-expert gate count: "
            f"expected {expected_gates}, found {len(source_gates)}"
        )

    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_names = set(checkpoint.keys())
        metadata = checkpoint.metadata()
        existing_gates = {name for name in checkpoint_names if SHARED_EXPERT_GATE.fullmatch(name)}
        packed_gates = {
            name for name in checkpoint_names if PACKED_SHARED_EXPERT_GATE.fullmatch(name)
        }
        if existing_gates == set(source_gates) and not packed_gates:
            return
        expected_packed = expected_gates * 3
        if existing_gates or len(packed_gates) != expected_packed:
            raise RuntimeError(
                "refusing to repair an unexpected shared-expert gate layout: "
                f"unquantized={len(existing_gates)}, packed={len(packed_gates)}"
            )
        tensors = {
            name: checkpoint.get_tensor(name)
            for name in checkpoint.keys()
            if name not in packed_gates
        }

    tensors.update(source_gates)
    temporary_path = output / "model.safetensors.repaired"
    save_file(tensors, temporary_path, metadata=metadata)
    temporary_path.replace(model_path)

    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("saved artifact has a malformed checkpoint index")
    for name in packed_gates:
        weight_map.pop(name, None)
    weight_map.update({name: model_path.name for name in source_gates})
    index["weight_map"] = dict(sorted(weight_map.items()))
    index["metadata"] = {"total_size": model_path.stat().st_size}
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ensure_unquantized_phase2_components_are_ignored(output: Path) -> None:
    from safetensors import safe_open

    vision_tensors: list[tuple[str, str]] = []
    for checkpoint_path in sorted(output.glob("*.safetensors")):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            vision_tensors.extend(
                (name, str(checkpoint.get_slice(name).get_dtype()))
                for name in checkpoint.keys()
                if ".visual." in name
            )
    if not vision_tensors:
        return
    floating_dtypes = {"F16", "BF16", "F32", "F64"}
    compressed = [name for name, dtype in vision_tensors if dtype not in floating_dtypes]
    if compressed:
        raise RuntimeError(
            "refusing to mark compressed vision tensors as unquantized: "
            + ", ".join(compressed[:5])
        )
    config_path = output / "config.json"
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise RuntimeError("saved artifact has no quantization_config")
    ignore = quantization.get("ignore")
    if not isinstance(ignore, list) or not all(isinstance(item, str) for item in ignore):
        raise RuntimeError("saved artifact has a malformed quantization ignore list")
    unquantized_patterns = (
        "re:.*visual.*",
        "re:.*linear_attn.in_proj_a$",
        "re:.*linear_attn.in_proj_b$",
        "re:.*mlp.gate$",
        "re:.*shared_expert_gate$",
        "re:.*lm_head$",
    )
    changed = False
    for pattern in unquantized_patterns:
        if pattern not in ignore:
            ignore.append(pattern)
            changed = True
    if changed:
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_artifact_manifest(root: Path, artifact: str, parser: argparse.ArgumentParser) -> None:
    control_python = root / ".venv" / "bin" / "python"
    manifest_command = [
        str(control_python),
        str(root / "scripts" / "manage_multilanguage_models.py"),
        "manifest",
        "--artifact",
        artifact,
    ]
    completed = subprocess.run(  # nosec B603
        manifest_command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode != 0:
        parser.error(
            "quantized files were saved, but artifact manifest generation failed: "
            + completed.stderr.strip()
        )
    print(completed.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=("phase2_main", "phase2_rtl_worker"), required=True)
    parser.add_argument("--acknowledge-unverified-procedure", action="store_true")
    parser.add_argument("--finalize-saved-artifact", action="store_true")
    arguments = parser.parse_args()
    if not arguments.acknowledge_unverified_procedure:
        parser.error(
            "add --acknowledge-unverified-procedure after reviewing the pinned recipe; "
            "no local Phase-2 artifact has been validated yet"
        )
    root = Path(__file__).resolve().parents[1]
    experiment = root / EXPERIMENT
    metadata: dict[str, Any] = json.loads(
        (experiment / "model_artifacts.json").read_text(encoding="utf-8")
    )
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list):
        parser.error("model artifact configuration has no artifact list")
    record = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("artifact_id") == arguments.artifact
        ),
        None,
    )
    if record is None:
        parser.error(f"model artifact configuration has no {arguments.artifact} profile")
    source = Path(str(record["source_path"])).expanduser().resolve()
    output = Path(str(record["output_path"])).expanduser().resolve()
    if not (source / "config.json").is_file():
        parser.error(f"pinned source model is missing: {source}")
    if arguments.finalize_saved_artifact:
        if not (output / "config.json").is_file():
            parser.error(f"saved artifact is missing: {output}")
        _ensure_single_shard_index(output)
        if arguments.artifact == "phase2_main":
            _restore_unquantized_shared_expert_gates(source, output)
            _ensure_unquantized_phase2_components_are_ignored(output)
        _write_artifact_manifest(root, arguments.artifact, parser)
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    calibration_profile = metadata["calibration"]
    quantization = record["quantization"]
    sample_count = int(quantization["calibration_samples"])
    sequence_length = int(quantization["calibration_sequence_length"])
    dataset = load_dataset(
        calibration_profile["repository"],
        revision=calibration_profile["revision"],
        split=f"{calibration_profile['split']}[:{sample_count}]",
    ).shuffle(seed=int(calibration_profile["shuffle_seed"]))

    if arguments.artifact == "phase2_main":
        from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(  # nosec B615
            source, dtype="auto", device_map="auto", local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(source, local_files_only=True)  # nosec B615
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)  # nosec B615
        ignore = [
            "re:.*lm_head",
            "re:model.visual.*",
            "re:.*mlp.gate$",
            "re:.*embed_tokens$",
            "re:.*shared_expert_gate$",
            "re:.*linear_attn.*",
        ]
        moe = True
    else:
        offload_folder = root / ".models" / "offload" / arguments.artifact
        offload_folder.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            source,
            dtype="auto",
            device_map="auto",
            max_memory={0: "10GiB", "cpu": "192GiB"},
            offload_folder=offload_folder,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)  # nosec B615
        processor = tokenizer
        ignore = ["lm_head"]
        moe = False

    def preprocess(example: dict[str, Any]) -> dict[str, str]:
        return {
            "text": processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(preprocess)
    recipe = [
        AWQModifier(),
        QuantizationModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=ignore,
        ),
    ]
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=sequence_length,
        num_calibration_samples=sample_count,
        moe_calibrate_all_experts=moe,
    )
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    processor.save_pretrained(output)
    if arguments.artifact == "phase2_main":
        save_mtp_tensors_to_checkpoint(source_model=str(source), dest_dir=str(output))
        _ensure_single_shard_index(output)
        _restore_unquantized_shared_expert_gates(source, output)
        _ensure_unquantized_phase2_components_are_ignored(output)
    _ensure_single_shard_index(output)
    _write_artifact_manifest(root, arguments.artifact, parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
