from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.serving_profiles import (
    InstalledServingCapabilities, ServingProfile, ServingProfileError,
    load_profiles, resolve_profile,
)

ROOT = Path(__file__).resolve().parents[1]
P8 = "P8_qwen38_w4a16_mtp"


def _profile(path: Path) -> ServingProfile:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return ServingProfile.from_mapping(raw)


def _mtp_tokens(profile: ServingProfile) -> int:
    prefix = "--speculative-config="
    values = [arg[len(prefix):] for arg in profile.extra_args if arg.startswith(prefix)]
    assert len(values) == 1
    raw: object = json.loads(values[0])
    assert isinstance(raw, dict)
    value = raw.get("num_speculative_tokens")
    assert isinstance(value, int)
    return value


def _caps() -> InstalledServingCapabilities:
    return InstalledServingCapabilities(
        version="0.25.0+cu129",
        flags=frozenset({
            "--served-model-name", "--host", "--port", "--max-model-len",
            "--max-num-seqs", "--max-num-batched-tokens", "--kv-cache-dtype",
            "--gpu-memory-utilization", "--scheduling-policy", "--generation-config",
            "--kv-cache-memory-bytes", "--enable-prefix-caching",
            "--prefix-caching-hash-algo", "--enable-chunked-prefill",
            "--language-model-only", "--reasoning-parser", "--enable-auto-tool-choice",
            "--tool-call-parser", "--mamba-ssm-cache-dtype", "--speculative-config",
        }),
        help_sha256="synthetic",
    )


def test_only_p8_is_active_and_legacy_candidates_are_absent() -> None:
    active_paths = sorted((ROOT / "configs/serving_profiles").glob("*.json"))
    candidate_paths = sorted((ROOT / "configs/serving_profile_candidates").glob("*.json"))
    assert [path.name for path in active_paths] == [f"{P8}.json"]
    assert [path.name for path in candidate_paths] == [f"{P8}.json"]
    active = _profile(active_paths[0])
    frozen = _profile(candidate_paths[0])
    assert active.profile_id == frozen.profile_id == P8
    assert active.served_model_name == frozen.served_model_name == "laplace-quality-qwen38-mtp8"
    assert active.model_path == ".models/Qwen3.8-27B-AWQ-4bit-e6b4b8b025f8"
    assert _mtp_tokens(active) == _mtp_tokens(frozen) == 8
    for field in (
        "port", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
        "kv_cache_dtype", "kv_cache_memory_bytes", "gpu_memory_utilization", "extra_args",
    ):
        assert getattr(active, field) == getattr(frozen, field)


def test_repo_relative_p8_resolves_from_explicit_checkout_root() -> None:
    active = load_profiles(ROOT / "configs/serving_profiles")[0]
    resolved = resolve_profile(
        active, _caps(), executable=Path("/usr/bin/vllm"),
        require_model=False, repository_root=ROOT,
    )
    assert resolved.command[2] == str((ROOT / active.model_path).resolve())
    assert resolved.command[resolved.command.index("--served-model-name") + 1] == (
        "laplace-quality-qwen38-mtp8"
    )


def test_relative_profile_requires_explicit_repository_root() -> None:
    active = load_profiles(ROOT / "configs/serving_profiles")[0]
    with pytest.raises(ServingProfileError, match="relative_model_path_requires_repository_root"):
        resolve_profile(
            active, _caps(), executable=Path("/usr/bin/vllm"), require_model=False
        )
