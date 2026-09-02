from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_workspace.production_model import (
    PRODUCTION_MODEL_ID, PRODUCTION_MODEL_RELATIVE, PRODUCTION_PROFILE_ID,
    PRODUCTION_PROFILE_SOURCE_COMMIT, PRODUCTION_SELECTION_EVIDENCE,
    qwen38_manifest, select, verify_qwen38_artifact,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_digest(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _profile() -> dict[str, object]:
    return {
        "profile_id": PRODUCTION_PROFILE_ID, "model_route": "quality",
        "model_path": PRODUCTION_MODEL_RELATIVE, "served_model_name": PRODUCTION_MODEL_ID,
        "port": 8207, "max_model_len": 131072, "max_num_seqs": 2,
        "max_num_batched_tokens": 8192, "kv_cache_dtype": "fp8",
        "kv_cache_memory_bytes": 5905580032, "enable_prefix_caching": True,
        "prefix_hash_algorithm": "sha256", "enable_chunked_prefill": True,
        "scheduling_policy": "fcfs", "cpu_offload_gb": 0.0, "cpu_offload_params": [],
        "offload_backend": "auto", "offload_group_size": 0, "offload_num_in_group": 1,
        "offload_prefetch_step": 1, "kv_offloading_size": None,
        "kv_offloading_backend": "native", "gpu_memory_utilization": 0.755,
        "startup_timeout": 1200, "request_timeout": 300,
        "extra_args": [
            "--language-model-only", "--reasoning-parser=qwen3",
            "--enable-auto-tool-choice", "--tool-call-parser=qwen3_xml",
            "--mamba-ssm-cache-dtype=bfloat16",
            '--speculative-config={"method":"mtp","num_speculative_tokens":8}',
        ],
    }


def _selector() -> dict[str, object]:
    return {
        "schema_version": 1, "selection_evidence": PRODUCTION_SELECTION_EVIDENCE,
        "default_profile_id": PRODUCTION_PROFILE_ID,
        "high_context_profile_id": PRODUCTION_PROFILE_ID,
        "quality_reserved_slots": 1, "standard_capacity": 2, "economy_capacity": 4,
        "routes": {
            "quality": {"model_id": PRODUCTION_MODEL_ID, "endpoint": "http://127.0.0.1:8207",
                        "priority": 0, "context_limit": 131072, "output_limit": 4096},
            "standard": {"model_id": PRODUCTION_MODEL_ID, "endpoint": "http://127.0.0.1:8207",
                         "priority": 10, "context_limit": 131072, "output_limit": 2048},
            "economy": {"model_id": "laplace-codev-r1-rl-qwen-7b-w4a16",
                        "endpoint": "http://127.0.0.1:8103", "priority": 20,
                        "context_limit": 8192, "output_limit": 2048},
        },
    }


def _repo(root: Path) -> Path:
    model = root / PRODUCTION_MODEL_RELATIVE
    model.mkdir(parents=True)
    payload = model / "payload.bin"
    payload.write_bytes(b"portable-qwen")
    record = {"path": "payload.bin", "size": payload.stat().st_size, "sha256": _sha(payload)}
    artifact_digest = _artifact_digest([record])
    recipe = root / "configs/quantization/qwen38_w4a16_recipe.json"
    recipe.parent.mkdir(parents=True)
    recipe.write_text('{"recipe":"test"}\n', encoding="utf-8")
    recipe_sha = _sha(recipe)
    (model / "laplace_artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifact_sha256": artifact_digest,
                    "base_revision": "base-revision", "recipe_sha256": recipe_sha,
                    "files": [record]}) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 3, "requested_base_model": "Qwen/Qwen3.8-27B",
        "base_revision": "base-revision", "artifact_path": PRODUCTION_MODEL_RELATIVE,
        "artifact_revision": "hf:test/revision", "artifact_sha256": artifact_digest,
        "artifact_size_bytes": payload.stat().st_size,
        "artifact_manifest": f"{PRODUCTION_MODEL_RELATIVE}/laplace_artifact_manifest.json",
        "tokenizer_revision": "tokenizer-revision",
        "quantization": {"status": "ARTIFACT_VERIFIED"},
        "source_provenance": {"recipe_path": "configs/quantization/qwen38_w4a16_recipe.json",
                              "recipe_sha256": recipe_sha},
        "production_profile": {
            "profile_id": PRODUCTION_PROFILE_ID, "served_model_name": PRODUCTION_MODEL_ID,
            "profile_source_commit": PRODUCTION_PROFILE_SOURCE_COMMIT,
            "requested_speculative_tokens": 8, "status": "FROZEN_OPTIMIZED_STEP_B",
        },
    }
    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    profiles = root / "configs/serving_profiles"
    profiles.mkdir(parents=True)
    (profiles / f"{PRODUCTION_PROFILE_ID}.json").write_text(
        json.dumps(_profile()) + "\n", encoding="utf-8"
    )
    candidates = root / "configs/serving_profile_candidates"
    candidates.mkdir(parents=True)
    (candidates / f"{PRODUCTION_PROFILE_ID}.json").write_text("{}\n", encoding="utf-8")
    (root / "configs/selected_serving_profiles.json").write_text(
        json.dumps(_selector()) + "\n", encoding="utf-8"
    )
    return root


def test_repository_local_artifact_ignores_stale_external_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "checkout")
    monkeypatch.setenv("LAPLACE_APPROVED_MODEL_ROOTS", str(tmp_path / "old-missing-checkout"))
    assert verify_qwen38_artifact(root)["artifact_path"] == PRODUCTION_MODEL_RELATIVE


def test_repository_local_artifact_survives_checkout_relocation(tmp_path: Path) -> None:
    original = _repo(tmp_path / "laplace-v3-refactor")
    moved = tmp_path / "future-laplace-location"
    original.rename(moved)
    assert qwen38_manifest(moved)["production_profile"]["profile_id"] == PRODUCTION_PROFILE_ID


def test_qwen38_manifest_does_not_bind_independent_economy_route(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path / "checkout")
    selector_path = root / "configs/selected_serving_profiles.json"

    selected = json.loads(selector_path.read_text(encoding="utf-8"))
    selected["routes"]["economy"] = {
        "model_id": "laplace-rtl-siliconmind-36k",
        "endpoint": "http://127.0.0.1:8211",
        "priority": 20,
        "context_limit": 16384,
        "output_limit": 16384,
    }
    selector_path.write_text(
        json.dumps(selected) + "\n",
        encoding="utf-8",
    )

    result = qwen38_manifest(root)
    assert result["production_profile"]["profile_id"] == PRODUCTION_PROFILE_ID


def test_qwen38_manifest_fails_if_any_legacy_active_profile_reappears(tmp_path: Path) -> None:
    root = _repo(tmp_path / "checkout")
    (root / "configs/serving_profiles/P7_qwen38_w4a16_mtp.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="qwen38_active_profile_set_invalid"):
        qwen38_manifest(root)


def test_qwen38_manifest_fails_if_legacy_selector_reappears(tmp_path: Path) -> None:
    root = _repo(tmp_path / "checkout")
    (root / "configs/selected_serving_profiles.qwen36.rollback.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="qwen38_legacy_selector_present"):
        qwen38_manifest(root)


def test_qwen38_approved_root_does_not_allow_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "checkout")
    external = tmp_path / "external"
    external.mkdir()
    link = tmp_path / "model-link"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_path"] = str(link)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    monkeypatch.setenv("LAPLACE_APPROVED_MODEL_ROOTS", str(tmp_path))
    with pytest.raises(RuntimeError, match="qwen38_artifact_root_symlink"):
        verify_qwen38_artifact(root)


def test_only_qwen38_selector_is_supported(tmp_path: Path) -> None:
    root = _repo(tmp_path / "checkout")
    assert select("qwen38", root) == root / "configs/selected_serving_profiles.json"
    with pytest.raises(RuntimeError, match="unknown_profile:qwen36"):
        select("qwen36", root)
