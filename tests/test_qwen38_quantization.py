from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _script() -> ModuleType:
    path = ROOT / "scripts/quantize_qwen38.py"
    spec = importlib.util.spec_from_file_location("quantize_qwen38", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen38_quantization_recipe_is_pinned_and_ampere_appropriate() -> None:
    recipe = json.loads(
        (ROOT / "configs/quantization/qwen38_27b_w4a16_awq.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "configs/model_manifests/qwen38_27b_a6000.json").read_text(encoding="utf-8")
    )
    assert recipe["base_model"] == manifest["requested_base_model"]
    assert len(recipe["base_revision"]) == 40
    assert len(recipe["source_files"]) == 32
    assert all(len(value) == 64 for value in recipe["source_files"].values())
    quantization = recipe["quantization"]
    calibration = recipe["calibration"]
    assert calibration["samples"] == 128
    assert calibration["sequence_length"] == 2048
    assert calibration["shuffle_seed"] == 42
    assert calibration["python_numpy_torch_seed"] == 42
    assert calibration["llmcompressor_secondary_shuffle"] is False
    assert quantization["scheme"] == "W4A16"
    assert quantization["weights_bits"] == 4
    assert quantization["activations_bits"] == 16
    assert quantization["group_size"] == 128
    assert quantization["symmetric"] is True
    assert "marlin" in quantization["kernel_target"]
    assert "re:^mtp.*" in quantization["ignore"]
    runtime = recipe["runtime"]
    assert runtime["python"] == "3.11.15"
    assert runtime["quantization_torch_cuda_runtime"] == "12.6"
    assert runtime["quantization_gpu_weight_budget_gib"] == 20
    assert runtime["quantization_cpu_weight_budget_gib"] == 205
    assert [item["gpu_weight_budget_gib"] for item in runtime["placement_trials"]] == [
        38,
        30,
    ]
    assert all(item["result"].startswith("OOM_") for item in runtime["placement_trials"])


def test_qwen38_artifact_tree_hash_is_deterministic(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / "b.bin").write_bytes(b"b")
    (tmp_path / "a.bin").write_bytes(b"a")
    first_files, first_digest = script.tree_manifest(tmp_path)
    second_files, second_digest = script.tree_manifest(tmp_path)
    assert first_files == second_files
    assert first_digest == second_digest
    assert [item["path"] for item in first_files] == ["a.bin", "b.bin"]


def test_qwen38_artifact_tree_hash_excludes_cache_and_manifest(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/ignored").write_text("ignored", encoding="utf-8")
    (tmp_path / script.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    files, _digest = script.tree_manifest(tmp_path)
    assert [item["path"] for item in files] == ["model.safetensors"]
