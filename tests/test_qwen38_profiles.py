from __future__ import annotations

import json
from pathlib import Path


def _load(name: str) -> dict[str, object]:
    return json.loads(
        (Path("configs/serving_profile_candidates") / name).read_text(encoding="utf-8")
    )


def test_qwen38_profiles_are_loopback_ampere_oriented_and_distinct() -> None:
    base = _load("P6_qwen38_w4a16.json")
    mtp = _load("P7_qwen38_w4a16_mtp.json")

    for profile in (base, mtp):
        assert int(profile["port"]) >= 1024
        assert profile["max_model_len"] == 131072
        assert profile["cpu_offload_gb"] == 0
        assert "Qwen3.8-27B" in str(profile["model_path"])
        assert profile["kv_cache_dtype"] == "auto"
        args = profile["extra_args"]
        assert "--reasoning-parser=qwen3" in args
        assert "--enable-auto-tool-choice" in args
        assert "--tool-call-parser=qwen3_xml" in args
    assert base["gpu_memory_utilization"] == 0.755
    assert mtp["gpu_memory_utilization"] == 0.77

    assert base["port"] != mtp["port"]
    assert not any(str(arg).startswith("--speculative-config") for arg in base["extra_args"])
    assert any(str(arg).startswith("--speculative-config=") for arg in mtp["extra_args"])


def test_qwen38_p8_freezes_measured_mtp8_memory_configuration() -> None:
    p8 = _load("P8_qwen38_w4a16_mtp.json")

    assert p8["profile_id"] == "P8_qwen38_w4a16_mtp"
    assert p8["max_model_len"] == 131072
    assert p8["max_num_seqs"] == 2
    assert p8["kv_cache_dtype"] == "fp8"
    assert p8["kv_cache_memory_bytes"] == 5905580032
    assert p8["gpu_memory_utilization"] == 0.755
    assert p8["enable_prefix_caching"] is True
    assert p8["enable_chunked_prefill"] is True
    assert p8["cpu_offload_gb"] == 0
    assert p8["kv_offloading_size"] is None

    args = p8["extra_args"]
    assert "--mamba-ssm-cache-dtype=bfloat16" in args
    assert (
        '--speculative-config={"method":"mtp","num_speculative_tokens":8}'
        in args
    )


def test_qwen38_candidates_do_not_pollute_certified_profile_set() -> None:
    certified = {path.name for path in Path("configs/serving_profiles").glob("*.json")}
    assert "P6_qwen38_w4a16.json" not in certified
    assert "P7_qwen38_w4a16_mtp.json" not in certified
    assert "P8_qwen38_w4a16_mtp.json" not in certified
