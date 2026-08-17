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
        assert profile["max_model_len"] == 32768
        assert profile["cpu_offload_gb"] == 0
        assert "Qwen3.8-27B" in str(profile["model_path"])
        assert profile["kv_cache_dtype"] == "auto"
        args = profile["extra_args"]
        assert "--reasoning-parser=qwen3" in args
        assert "--enable-auto-tool-choice" in args
        assert "--tool-call-parser=qwen3_coder" in args

    assert base["port"] != mtp["port"]
    assert not any(str(arg).startswith("--speculative-config") for arg in base["extra_args"])
    assert any(str(arg).startswith("--speculative-config=") for arg in mtp["extra_args"])


def test_qwen38_candidates_do_not_pollute_certified_profile_set() -> None:
    certified = {path.name for path in Path("configs/serving_profiles").glob("*.json")}
    assert "P6_qwen38_w4a16.json" not in certified
    assert "P7_qwen38_w4a16_mtp.json" not in certified
