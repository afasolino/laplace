from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
P8 = "P8_qwen38_w4a16_mtp"


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
vllm:spec_decode_num_draft_tokens_total{model_name="a",engine="0"} 12
vllm:spec_decode_num_draft_tokens_total{model_name="a",engine="1"} 3
vllm:spec_decode_num_drafts_total{model_name="a",engine="0"} 7
"""
    assert script._metric_sum(metrics, "vllm:spec_decode_num_draft_tokens") == 15
    assert script._metric_sum(metrics, "vllm:spec_decode_num_drafts") == 7


def test_mtp_sweep_committed_tokens_are_target_plus_accepted_drafts() -> None:
    script = _script("benchmark_qwen38_mtp_sweep")
    before = {"drafts": 2.0, "draft_tokens": 6.0, "accepted_tokens": 5.0}
    after = {"drafts": 12.0, "draft_tokens": 36.0, "accepted_tokens": 30.0}
    delta = script._counter_delta(after, before)
    assert delta == {"drafts": 10.0, "draft_tokens": 30.0, "accepted_tokens": 25.0}
    assert 1 + delta["accepted_tokens"] / delta["drafts"] == 3.5


def test_mtp_certification_reads_the_current_p8_workpoint() -> None:
    script = _script()
    p8 = script._profile(ROOT / f"configs/serving_profiles/{P8}.json")
    assert script._mtp_tokens(p8) == 8


def test_certification_parsers_accept_only_p8(tmp_path: Path) -> None:
    profile = _script()
    parsed = profile._parser().parse_args([P8, "--output-root", str(tmp_path / "out")])
    assert parsed.profile_id == P8
    with pytest.raises(SystemExit):
        profile._parser().parse_args(
            ["P7_qwen38_w4a16_mtp", "--output-root", str(tmp_path / "old")]
        )
    production = _script("certify_qwen38_production")
    assert production.PROFILE_IDS == (P8,)


def test_production_gate_does_not_bind_independent_economy_worker() -> None:
    production = _script("certify_qwen38_production")

    profile = production._profile(
        ROOT / f"configs/serving_profiles/{P8}.json"
    )
    selected = json.loads(
        (ROOT / "configs/selected_serving_profiles.json").read_text(
            encoding="utf-8"
        )
    )

    selected["routes"]["economy"] = {
        "model_id": "laplace-rtl-siliconmind-36k",
        "endpoint": "http://127.0.0.1:8211",
        "priority": 20,
        "context_limit": 16384,
        "output_limit": 16384,
    }

    routes = production._validate_staged_routes(selected, profile)

    assert routes["economy"]["model_id"] == "laplace-rtl-siliconmind-36k"


def test_quantized_kernel_gate_requires_format_and_runtime_evidence(tmp_path: Path) -> None:
    script = _script()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"quantization_config": {
            "quant_method": "compressed-tensors", "format": "pack-quantized"
        }}), encoding="utf-8"
    )
    log = tmp_path / "server.log"
    log.write_text("Using CompressedTensorsW4A16 for Marlin execution\n", encoding="utf-8")
    evidence = script._quantized_kernel(model, log)
    assert evidence["status"] == "PASS"
    assert evidence["runtime_log_markers"] == ["compressedtensorsw4a16", "marlin"]


def test_production_gate_requires_complete_bound_p8_evidence(tmp_path: Path) -> None:
    script = _script("certify_qwen38_production")
    names = [*script.MANDATORY_PROFILE_GATES, "mtp"]
    evidence = {
        "schema_version": 1, "status": "PASSED", "profile_id": P8,
        "profile_sha256": "a" * 64, "artifact_sha256": "b" * 64,
        "repository_revision": "c" * 40,
        "gates": {name: {"status": "PASS"} for name in names},
        "release": {"status": "RELEASED_OWNED_PROFILE"},
        "endpoint_down_after_release": True,
        "unrelated_processes_signalled": False,
    }
    path = tmp_path / "certification.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    assert script.validate_profile_certification(
        path, profile_id=P8, profile_sha256="a" * 64,
        artifact_sha256="b" * 64, repository_revision="c" * 40,
    )["status"] == "PASSED"
    evidence["release"] = {"status": "NOT_STARTED"}
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="profile_certification_not_eligible_for_production_gate"
    ):
        script.validate_profile_certification(
            path, profile_id=P8, profile_sha256="a" * 64,
            artifact_sha256="b" * 64, repository_revision="c" * 40,
        )
