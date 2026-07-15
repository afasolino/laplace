#!/usr/bin/env python3
"""Reproducible pinned W4A16 preparation for the two Phase-2 models.

This script is intentionally not part of experiment execution. It requires an
explicit acknowledgement because the exact artifacts have not yet been
quantized and served on this A6000.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_workspace.model_artifacts import load_model_artifacts, write_artifact_manifest


EXPERIMENT = Path("codex_a6000/experiments/multilanguage_dual_model_ablation_v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=("phase2_main", "phase2_rtl_worker"), required=True)
    parser.add_argument("--acknowledge-unverified-procedure", action="store_true")
    arguments = parser.parse_args()
    if not arguments.acknowledge_unverified_procedure:
        parser.error(
            "add --acknowledge-unverified-procedure after reviewing the pinned recipe; "
            "no local Phase-2 artifact has been validated yet"
        )
    root = Path(__file__).resolve().parents[1]
    experiment = root / EXPERIMENT
    records = load_model_artifacts(experiment / "model_artifacts.json")
    record = records[arguments.artifact]
    source = Path(str(record["source_path"])).expanduser().resolve()
    output = Path(str(record["output_path"])).expanduser().resolve()
    if not (source / "config.json").is_file():
        parser.error(f"pinned source model is missing: {source}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata: dict[str, Any] = json.loads(
        (experiment / "model_artifacts.json").read_text(encoding="utf-8")
    )
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
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            source, dtype="auto", device_map="auto", local_files_only=True
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
    manifest = write_artifact_manifest(experiment, arguments.artifact)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
