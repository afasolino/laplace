from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str = "certify_qwen38_profile") -> ModuleType:
    path = ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_sum_accepts_prometheus_counter_suffix_and_labels() -> None:
    script = _script()
    metrics = """
# HELP vllm:spec_decode_num_draft_tokens Number of draft tokens.
vllm:spec_decode_num_draft_tokens_total{model_name="a",engine="0"} 12
vllm:spec_decode_num_draft_tokens_total{model_name="a",engine="1"} 3
vllm:spec_decode_num_drafts_total{model_name="a",engine="0"} 7
"""
    assert script._metric_sum(metrics, "vllm:spec_decode_num_draft_tokens") == 15
    assert script._metric_sum(metrics, "vllm:spec_decode_num_drafts") == 7


def test_quantized_kernel_gate_requires_format_and_runtime_evidence(
    tmp_path: Path,
) -> None:
    script = _script()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                }
            }
        ),
        encoding="utf-8",
    )
    log = tmp_path / "server.log"
    log.write_text("Using CompressedTensorsW4A16 for Marlin execution\n", encoding="utf-8")
    evidence = script._quantized_kernel(model, log)
    assert evidence["status"] == "PASS"
    assert evidence["runtime_log_markers"] == ["compressedtensorsw4a16", "marlin"]


def test_p6_mandatory_gates_cover_requested_live_surface() -> None:
    script = _script()
    assert set(script.MANDATORY_GATES) == {
        "model_identity",
        "normal_inference",
        "streaming",
        "reasoning",
        "tool_calling",
        "multi_turn",
        "cancellation",
        "context_window",
        "runtime_stability",
        "quantized_kernel",
        "gpu_headroom",
    }


def _profile_certification(
    profile_id: str,
    *,
    profile_sha: str,
    artifact_sha: str,
    revision: str,
    status: str = "PASSED",
) -> dict[str, object]:
    script = _script()
    names = [*script.MANDATORY_GATES]
    if profile_id.endswith("_mtp"):
        names.append("mtp")
    return {
        "schema_version": 1,
        "status": status,
        "profile_id": profile_id,
        "profile_sha256": profile_sha,
        "artifact_sha256": artifact_sha,
        "repository_revision": revision,
        "gates": {name: {"status": "PASS"} for name in names},
        "release": {"status": "RELEASED_OWNED_PROFILE"},
        "endpoint_down_after_release": True,
        "unrelated_processes_signalled": False,
    }


def _production_gate(
    profile_id: str,
    *,
    profile_sha: str,
    artifact_sha: str,
    revision: str,
    certification_sha: str,
    status: str = "PASSED",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "selected_profile_id": profile_id,
        "profile_sha256": profile_sha,
        "artifact_sha256": artifact_sha,
        "repository_revision": revision,
        "profile_certification_sha256": certification_sha,
        "gpu": {"minimum_free_headroom_mib": 4096},
        "release": {
            "quality": {"status": "RELEASED_OWNED_PROFILE"},
            "codev": {"status": "STOPPED_OWNED_CODEV"},
        },
        "target_endpoints_down_after_release": True,
        "unrelated_processes_signalled": False,
    }


def test_production_gate_requires_complete_bound_profile_evidence(tmp_path: Path) -> None:
    script = _script("certify_qwen38_production")
    profile_id = "P6_qwen38_w4a16"
    evidence = _profile_certification(
        profile_id,
        profile_sha="a" * 64,
        artifact_sha="b" * 64,
        revision="c" * 40,
    )
    path = tmp_path / "certification.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    assert (
        script.validate_profile_certification(
            path,
            profile_id=profile_id,
            profile_sha256="a" * 64,
            artifact_sha256="b" * 64,
            repository_revision="c" * 40,
        )["status"]
        == "PASSED"
    )

    evidence["release"] = {"status": "NOT_STARTED"}
    path.write_text(json.dumps(evidence), encoding="utf-8")
    try:
        script.validate_profile_certification(
            path,
            profile_id=profile_id,
            profile_sha256="a" * 64,
            artifact_sha256="b" * 64,
            repository_revision="c" * 40,
        )
    except RuntimeError as exc:
        assert str(exc) == "profile_certification_not_eligible_for_production_gate"
    else:
        raise AssertionError("incomplete certification was accepted")


def test_finalizer_selects_p6_and_enables_p7_only_with_both_gates(
    tmp_path: Path,
) -> None:
    script = _script("finalize_qwen38_certification")
    profiles = tmp_path / "configs/serving_profile_candidates"
    profiles.mkdir(parents=True)
    p6 = profiles / "P6_qwen38_w4a16.json"
    p7 = profiles / "P7_qwen38_w4a16_mtp.json"
    p6.write_text('{"profile":"p6"}\n', encoding="utf-8")
    p7.write_text('{"profile":"p7"}\n', encoding="utf-8")
    p6_sha = script._sha256(p6)
    p7_sha = script._sha256(p7)
    artifact_sha = "d" * 64
    revision = "e" * 40
    p6_cert = tmp_path / "p6-cert.json"
    p6_cert.write_text(
        json.dumps(
            _profile_certification(
                script.P6,
                profile_sha=p6_sha,
                artifact_sha=artifact_sha,
                revision=revision,
            )
        ),
        encoding="utf-8",
    )
    p6_prod = tmp_path / "p6-production.json"
    p6_prod.write_text(
        json.dumps(
            _production_gate(
                script.P6,
                profile_sha=p6_sha,
                artifact_sha=artifact_sha,
                revision=revision,
                certification_sha=script._sha256(p6_cert),
            )
        ),
        encoding="utf-8",
    )
    base = script.evaluate_evidence(
        tmp_path,
        p6_certification=p6_cert,
        p6_production_gate=p6_prod,
        p7_certification=None,
        p7_production_gate=None,
        artifact={"artifact_sha256": artifact_sha},
        repository_revision=revision,
    )
    assert base["selected_profile_id"] == script.P6
    assert base["mtp_enabled"] is False

    p7_cert = tmp_path / "p7-cert.json"
    p7_cert.write_text(
        json.dumps(
            _profile_certification(
                script.P7,
                profile_sha=p7_sha,
                artifact_sha=artifact_sha,
                revision=revision,
            )
        ),
        encoding="utf-8",
    )
    p7_prod = tmp_path / "p7-production.json"
    p7_prod.write_text(
        json.dumps(
            _production_gate(
                script.P7,
                profile_sha=p7_sha,
                artifact_sha=artifact_sha,
                revision=revision,
                certification_sha=script._sha256(p7_cert),
            )
        ),
        encoding="utf-8",
    )
    mtp = script.evaluate_evidence(
        tmp_path,
        p6_certification=p6_cert,
        p6_production_gate=p6_prod,
        p7_certification=p7_cert,
        p7_production_gate=p7_prod,
        artifact={"artifact_sha256": artifact_sha},
        repository_revision=revision,
    )
    assert mtp["selected_profile_id"] == script.P7
    assert mtp["mtp_enabled"] is True

    manifest = {
        "serving_candidates": {script.P6: {}, script.P7: {}},
        "mtp": {"status": "NOT_CERTIFIED"},
        "external_blocker": {"category": "test"},
    }
    promoted = script._promoted_manifest(manifest, mtp, revision)
    assert promoted["promotion_allowed"] is True
    assert promoted["certification_status"] == "PASSED"
    assert promoted["mtp"]["status"] == "PASSED"
    assert "external_blocker" not in promoted
