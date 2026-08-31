from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _script() -> ModuleType:
    path = ROOT / "scripts/benchmark_qwen38_mtp_sweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_qwen38_mtp_sweep_step_b", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_profile(script: ModuleType):
    return script._profile(
        ROOT / "configs/serving_profile_candidates/P7_qwen38_w4a16_mtp.json"
    )


def test_step_b_overrides_preserve_mtp_and_expose_cache_controls() -> None:
    script = _script()
    profile = script._apply_profile_overrides(
        _base_profile(script),
        gpu_memory_utilization=0.755,
        kv_cache_dtype="fp8_per_token_head",
        kv_cache_memory_bytes=6_442_450_944,
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        calculate_kv_scales=None,
        max_num_seqs=2,
        max_num_batched_tokens=8192,
    )
    assert profile.kv_cache_dtype == "fp8_per_token_head"
    assert profile.kv_cache_memory_bytes == 6_442_450_944
    assert profile.gpu_memory_utilization == 0.755
    assert "--mamba-cache-dtype=bfloat16" in profile.extra_args
    assert "--mamba-ssm-cache-dtype=float32" in profile.extra_args
    assert any(
        arg.startswith("--speculative-config=") and '"num_speculative_tokens":3' in arg
        for arg in profile.extra_args
    )


def test_step_b_dynamic_fp8_scale_override_is_explicit() -> None:
    script = _script()
    base = _base_profile(script)
    dynamic = script._apply_profile_overrides(
        base,
        gpu_memory_utilization=None,
        kv_cache_dtype="fp8",
        kv_cache_memory_bytes=None,
        mamba_cache_dtype=None,
        mamba_ssm_cache_dtype=None,
        calculate_kv_scales=True,
        max_num_seqs=None,
        max_num_batched_tokens=None,
    )
    assert "--calculate-kv-scales" in dynamic.extra_args
    assert "--no-calculate-kv-scales" not in dynamic.extra_args

    disabled = script._apply_profile_overrides(
        dynamic,
        gpu_memory_utilization=None,
        kv_cache_dtype="fp8",
        kv_cache_memory_bytes=None,
        mamba_cache_dtype=None,
        mamba_ssm_cache_dtype=None,
        calculate_kv_scales=False,
        max_num_seqs=None,
        max_num_batched_tokens=None,
    )
    assert "--calculate-kv-scales" not in disabled.extra_args
    assert "--no-calculate-kv-scales" in disabled.extra_args


def test_dynamic_scale_flag_is_rejected_for_per_token_head_fp8() -> None:
    script = _script()
    with pytest.raises(RuntimeError, match="calculate_kv_scales_requires_fp8"):
        script._apply_profile_overrides(
            _base_profile(script),
            gpu_memory_utilization=None,
            kv_cache_dtype="fp8_per_token_head",
            kv_cache_memory_bytes=None,
            mamba_cache_dtype=None,
            mamba_ssm_cache_dtype=None,
            calculate_kv_scales=True,
            max_num_seqs=None,
            max_num_batched_tokens=None,
        )


def test_exact_kv_bytes_must_be_positive() -> None:
    script = _script()
    with pytest.raises(ValueError, match="kv_cache_memory_bytes must be positive"):
        script._apply_profile_overrides(
            _base_profile(script),
            gpu_memory_utilization=None,
            kv_cache_dtype=None,
            kv_cache_memory_bytes=0,
            mamba_cache_dtype=None,
            mamba_ssm_cache_dtype=None,
            calculate_kv_scales=None,
            max_num_seqs=None,
            max_num_batched_tokens=None,
        )


def test_concurrent_round_aggregates_all_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script()

    def fake_request(client, endpoint, model, *, max_tokens):
        del client, endpoint, model
        return {
            "status": "PASS",
            "completion_tokens": max_tokens,
            "output_tok_s": 10.0,
            "ttft_seconds": 0.1,
        }

    monkeypatch.setattr(script, "_request", fake_request)
    result = script._concurrent_round(
        [object(), object()],
        "http://127.0.0.1:1",
        "model",
        max_tokens=4,
    )
    assert result["status"] == "PASS"
    assert result["concurrency"] == 2
    assert result["aggregate_completion_tokens"] == 8
    assert isinstance(result["aggregate_completion_tok_s"], float)
    assert len(result["requests"]) == 2


def test_partial_benchmark_status_returns_nonzero() -> None:
    script = _script()
    assert script._status_return_code("PASS") == 0
    assert script._status_return_code("PARTIAL") == 2
    assert script._status_return_code("FAILED") == 2
