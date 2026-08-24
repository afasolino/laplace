from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_prequantized_policy_prefers_existing_checkpoint_and_128k() -> None:
    candidates = _json(ROOT / "configs/model_manifests/qwen38_prequantized_candidates.json")
    source = _json(ROOT / "configs/model_manifests/qwen38_prequantized_source.json")
    assert candidates["primary"]["repository"] == "barrydeen/Qwen3.8-27B-AWQ-4bit"
    assert candidates["primary"]["symmetric"] is False
    assert candidates["primary"]["protected_bf16_prefixes"] == [
        "lm_head.",
        "model.visual.",
        "mtp.",
    ]
    assert candidates["fallback"]["repository"] == "soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ"
    assert candidates["fallback"]["symmetric"] is True
    assert source["local_quantization"] == "final_fallback_only"
    assert source["production_context_tokens"] == 131072


def test_qwen38_candidate_profiles_target_128k_and_qwen3_xml_tools() -> None:
    for name in ("P6_qwen38_w4a16.json", "P7_qwen38_w4a16_mtp.json"):
        profile = _json(ROOT / "configs/serving_profile_candidates" / name)
        assert profile["max_model_len"] == 131072
        assert "--tool-call-parser=qwen3_xml" in profile["extra_args"]
