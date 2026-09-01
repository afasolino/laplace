from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.rtl_specialist_qualification import (
    DEFAULT_CONFIG,
    ROOT,
    SpecialistQualificationError,
    StepC2Configuration,
    build_direct_configuration,
    build_specialist_candidate,
    build_specialist_profile,
    candidate_model_path,
    load_step_c2_configuration,
    parse_ls_remote_head,
    select_candidate_reports,
)


def _report(
    configuration: StepC2Configuration,
    candidate_id: str,
    *,
    deterministic: int,
    held_out: int,
    peak: int,
    rate: float,
) -> dict[str, object]:
    repository = {
        "siliconmind_qwen3_4b_t_2507_36k": "AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507",
        "siliconmind_qwen3_4b_t_2507_76k": "AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507-76k",
        "siliconmind_v12_qwen3_4b_t_2507": (
            "AS-SiliconMind/SiliconMind-V1.2-Qwen3-4B-T-2507"
        ),
    }[candidate_id]
    revision = {
        "siliconmind_qwen3_4b_t_2507_36k": "1" * 40,
        "siliconmind_qwen3_4b_t_2507_76k": "2" * 40,
        "siliconmind_v12_qwen3_4b_t_2507": "4" * 40,
    }[candidate_id]
    return {
        "status": "QUALIFICATION_COMPLETE",
        "candidate_id": candidate_id,
        "repository": repository,
        "revision": revision,
        "model_path": str(
            candidate_model_path(
                configuration,
                candidate_id,
                revision,
            )
        ),
        "base_revision": "3" * 40,
        "configuration_sha256": configuration.config_sha256,
        "clean_owned_server_release": True,
        "metrics": {
            "deterministic_task_pass_count": deterministic,
            "held_out_task_pass_count": held_out,
            "peak_gpu_memory_mib": peak,
            "median_output_tokens_per_second": rate,
            "infrastructure_failure_count": 0,
        },
    }


def test_step_c2_configuration_preserves_c1_and_p8() -> None:
    configuration = load_step_c2_configuration(ROOT, DEFAULT_CONFIG)
    assert configuration.preparation_base_commit == "31d43c9961123690fb8f169188bdf9e79093897f"
    assert len(configuration.worker_task_ids) == 11
    assert configuration.coexistence_task_ids == (
        "v_ready_valid_buffer",
        "sv_ready_valid_buffer",
    )
    assert configuration.max_model_len == 16384
    assert configuration.max_output_tokens == 8192
    assert configuration.gpu_memory_utilization == 0.18
    assert configuration.minimum_free_headroom_mib == 2048
    assert [item.candidate_id for item in configuration.candidates] == [
        "siliconmind_qwen3_4b_t_2507_36k",
        "siliconmind_qwen3_4b_t_2507_76k",
        "siliconmind_v12_qwen3_4b_t_2507",
    ]


def test_huggingface_head_resolution_requires_one_exact_commit() -> None:
    assert parse_ls_remote_head(f"{'a' * 40}\tHEAD\n") == "a" * 40
    with pytest.raises(SpecialistQualificationError):
        parse_ls_remote_head("main\tHEAD\n")
    with pytest.raises(SpecialistQualificationError):
        parse_ls_remote_head(f"{'a' * 40}\tHEAD\n{'b' * 40}\tHEAD\n")


def test_specialist_profile_is_bounded_and_native_reasoning_candidate(tmp_path: Path) -> None:
    configuration = load_step_c2_configuration(ROOT, DEFAULT_CONFIG)
    spec = configuration.candidates[0]
    model_path = tmp_path / "model"
    model_path.mkdir()
    profile = build_specialist_profile(configuration, spec, model_path)
    candidate = build_specialist_candidate(configuration, spec, profile, "a" * 40)
    dual = build_direct_configuration(candidate)
    assert profile.kv_cache_memory_bytes == 1073741824
    assert profile.gpu_memory_utilization == 0.18
    assert profile.extra_args == ("--dtype=bfloat16",)
    assert candidate.context_tokens == 16384
    assert candidate.max_output_tokens == 8192
    assert candidate.temperature == 1.0
    assert dual.main == candidate
    assert dual.rtl_worker == candidate
    assert dual.fallback_to_main is False
    assert dual.worker_reasoning_mode == "model_default"


def test_candidate_model_path_is_repo_local_runtime_storage() -> None:
    configuration = load_step_c2_configuration(ROOT, DEFAULT_CONFIG)
    target = candidate_model_path(
        configuration,
        "siliconmind_qwen3_4b_t_2507_36k",
        "a" * 40,
    )
    assert target.is_relative_to(configuration.output_root)
    assert ".runtime/v3-a6000-completion/step-c2/models" in target.as_posix()


def test_selection_is_correctness_first_then_memory_then_throughput() -> None:
    configuration = load_step_c2_configuration(ROOT, DEFAULT_CONFIG)
    reports = [
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_36k",
            deterministic=10,
            held_out=10,
            peak=9000,
            rate=50.0,
        ),
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_76k",
            deterministic=10,
            held_out=9,
            peak=10000,
            rate=40.0,
        ),
        _report(
            configuration,
            "siliconmind_v12_qwen3_4b_t_2507",
            deterministic=11,
            held_out=9,
            peak=10500,
            rate=35.0,
        ),
    ]
    selection = select_candidate_reports(configuration, "3" * 40, reports)
    selected = selection["selected_candidate"]
    assert isinstance(selected, dict)
    assert selected["candidate_id"] == "siliconmind_v12_qwen3_4b_t_2507"

    reports = [
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_36k",
            deterministic=11,
            held_out=10,
            peak=9000,
            rate=30.0,
        ),
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_76k",
            deterministic=11,
            held_out=10,
            peak=9500,
            rate=80.0,
        ),
        _report(
            configuration,
            "siliconmind_v12_qwen3_4b_t_2507",
            deterministic=10,
            held_out=10,
            peak=8500,
            rate=90.0,
        ),
    ]
    selection = select_candidate_reports(configuration, "3" * 40, reports)
    selected = selection["selected_candidate"]
    assert isinstance(selected, dict)
    assert selected["candidate_id"] == "siliconmind_qwen3_4b_t_2507_36k"


def test_exact_tie_fails_closed_without_promotion() -> None:
    configuration = load_step_c2_configuration(ROOT, DEFAULT_CONFIG)
    reports = [
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_36k",
            deterministic=11,
            held_out=11,
            peak=9000,
            rate=50.0,
        ),
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_76k",
            deterministic=11,
            held_out=11,
            peak=9000,
            rate=50.0,
        ),
        _report(
            configuration,
            "siliconmind_v12_qwen3_4b_t_2507",
            deterministic=11,
            held_out=11,
            peak=9000,
            rate=50.0,
        ),
    ]
    selection = select_candidate_reports(configuration, "3" * 40, reports)
    assert selection["status"] == "NO_PROMOTION_TIE"
    assert selection["selected_candidate"] is None


def test_selection_rejects_infrastructure_failure_evidence() -> None:
    configuration = load_step_c2_configuration(ROOT, DEFAULT_CONFIG)
    reports = [
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_36k",
            deterministic=11,
            held_out=11,
            peak=9000,
            rate=50.0,
        ),
        _report(
            configuration,
            "siliconmind_qwen3_4b_t_2507_76k",
            deterministic=11,
            held_out=11,
            peak=9000,
            rate=50.0,
        ),
        _report(
            configuration,
            "siliconmind_v12_qwen3_4b_t_2507",
            deterministic=11,
            held_out=11,
            peak=9000,
            rate=50.0,
        ),
    ]
    metrics = reports[1]["metrics"]
    assert isinstance(metrics, dict)
    metrics["infrastructure_failure_count"] = 1
    with pytest.raises(SpecialistQualificationError):
        select_candidate_reports(configuration, "3" * 40, reports)
