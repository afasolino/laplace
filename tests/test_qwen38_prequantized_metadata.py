from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "prepare_qwen38_prequantized", ROOT / "scripts/prepare_qwen38_prequantized.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def _write_safetensor_header(path: Path, keys: list[str], *, dtype: str = "BF16") -> None:
    header = {
        key: {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]} for key in keys
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00\x00")


def _candidate() -> dict[str, object]:
    return {
        "format": "compressed-tensors",
        "weights_bits": 4,
        "activations_bits": 16,
        "group_size": 128,
        "symmetric": False,
        "protected_bf16_prefixes": ["model.language_model.mtp."],
        "preserves_mtp": True,
    }


def _snapshot(root: Path, *, include_mtp: bool, weight_bits: int = 4) -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "targets": ["Linear"],
                            "input_activations": None,
                            "output_activations": None,
                            "weights": {
                                "num_bits": weight_bits,
                                "group_size": 128,
                                "type": "int",
                                "strategy": "group",
                                "symmetric": False,
                            },
                        }
                    },
                    "ignore": ["model.language_model.mtp.layers.0.mlp.up_proj"],
                },
                "text_config": {"mtp_num_hidden_layers": 1},
            }
        ),
        encoding="utf-8",
    )
    weight_map = {
        "model.language_model.layers.0.mlp.up_proj.weight": "model-00001-of-00003.safetensors"
    }
    if include_mtp:
        # MTP can live in an ordinary shard and under a nested module prefix.
        weight_map[
            "model.language_model.mtp.layers.0.mlp.up_proj.weight"
        ] = "model-00003-of-00003.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    for shard_name in sorted(set(weight_map.values())):
        _write_safetensor_header(
            root / shard_name,
            [key for key, mapped in weight_map.items() if mapped == shard_name],
        )


def test_mtp_validation_uses_config_and_tensor_metadata_not_filename(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=True)
    result = module._validate_snapshot(tmp_path, _candidate())
    assert result["mtp_tensors_present"] is True
    assert result["mtp_tensor_count"] == 1
    assert result["protected_bf16_verified"] is True
    assert result["protected_bf16_prefix_counts"] == {
        "model.language_model.mtp.": 1
    }
    assert result["tensor_dtype_counts"]["BF16"] == 2
    assert "index+header:model-00001-of-00003.safetensors" in result["tensor_metadata_source"]
    assert result["storage_format"] == "pack-quantized"
    assert result["linear_quantization_groups"][0]["weight_bits"] == 4


def test_mtp_validation_fails_closed_when_tensor_metadata_is_missing(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=False)
    with pytest.raises(RuntimeError, match="prequantized_mtp_metadata_missing"):
        module._validate_snapshot(tmp_path, _candidate())


def test_quantization_group_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=True, weight_bits=8)
    with pytest.raises(RuntimeError, match="prequantized_weight_bits_mismatch"):
        module._validate_snapshot(tmp_path, _candidate())


def test_protected_component_must_remain_bf16(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=True)
    _write_safetensor_header(
        tmp_path / "model-00003-of-00003.safetensors",
        ["model.language_model.mtp.layers.0.mlp.up_proj.weight"],
        dtype="F16",
    )
    with pytest.raises(RuntimeError, match="prequantized_protected_bf16_dtype_mismatch"):
        module._validate_snapshot(tmp_path, _candidate())


def test_mtp_validation_combines_index_and_separate_unindexed_safetensor(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=False)
    _write_safetensor_header(
        tmp_path / "model-mtp.safetensors",
        ["model.language_model.mtp.layers.0.mlp.up_proj.weight"],
    )
    result = module._validate_snapshot(tmp_path, _candidate())
    assert result["mtp_tensors_present"] is True
    assert result["mtp_tensor_count"] == 1
    assert "header:model-mtp.safetensors" in result["tensor_metadata_source"]


def test_fallback_selection_requires_recorded_primary_incompatibility() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="fallback_requires_demonstrated_primary_incompatibility"):
        module._selection_justification("fallback", None)
    reason = "Primary fails deterministic compressed-tensors loader probe on installed serving runtime."
    assert module._selection_justification("fallback", reason) == reason
    assert module._selection_justification("primary", None) is None


def test_indexed_snapshot_rejects_missing_weight_shard(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=True)
    (tmp_path / "model-00003-of-00003.safetensors").unlink()
    with pytest.raises(RuntimeError, match="prequantized_safetensors_index_shard_missing"):
        module._validate_snapshot(tmp_path, _candidate())


def test_snapshot_replace_preserves_existing_target_if_copy_fails(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "file.bin").write_bytes(b"new")
    target = tmp_path / "models" / "checkpoint"
    target.mkdir(parents=True)
    (target / "old.bin").write_bytes(b"old")

    original_copy2 = module.shutil.copy2

    def fail_copy(source, destination, *, follow_symlinks=True):
        if Path(source).name == "file.bin":
            raise OSError("copy failed")
        return original_copy2(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(module.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        module._copy_snapshot(snapshot, target, _candidate(), replace=True)
    assert (target / "old.bin").read_bytes() == b"old"

def test_indexed_snapshot_rejects_tensor_missing_from_mapped_shard(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, include_mtp=True)
    _write_safetensor_header(
        tmp_path / "model-00003-of-00003.safetensors",
        ["unrelated.weight"],
    )
    with pytest.raises(RuntimeError, match="prequantized_safetensors_index_tensor_missing"):
        module._validate_snapshot(tmp_path, _candidate())


def test_snapshot_replace_validates_staging_before_existing_target_moves(
    tmp_path: Path,
) -> None:
    module = _module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    # Invalid snapshot: no config/model metadata.
    (snapshot / "file.bin").write_bytes(b"invalid")
    target = tmp_path / "models" / "checkpoint"
    target.mkdir(parents=True)
    (target / "old.bin").write_bytes(b"old")

    with pytest.raises(RuntimeError, match="prequantized_config_missing"):
        module._copy_snapshot(snapshot, target, _candidate(), replace=True)

    assert (target / "old.bin").read_bytes() == b"old"
