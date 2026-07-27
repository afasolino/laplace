from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research_workspace.model_servers import (
    EmpiricalMemoryPolicy,
    ModelServerAdmission,
    ModelServerController,
    ModelServerSafetyError,
    ModelServerSpec,
    derive_empirical_memory_policy,
    load_server_specs,
)


ROOT = Path(__file__).resolve().parents[1]


def _specs() -> tuple[ModelServerSpec, ...]:
    return (
        ModelServerSpec(
            profile="phase2_main",
            endpoint="http://127.0.0.1:8102",
            port=8102,
            expected_model_id="main",
            model_path="/models/main",
        ),
        ModelServerSpec(
            profile="phase2_rtl_worker",
            endpoint="http://127.0.0.1:8103",
            port=8103,
            expected_model_id="worker",
            model_path="/models/worker",
        ),
    )


def _policy(required: int = 47_897) -> EmpiricalMemoryPolicy:
    return EmpiricalMemoryPolicy(
        maximum_observed_used_mib=43_801,
        required_residual_free_mib=4_096,
        required_pre_start_free_mib=required,
        evidence_paths=("evidence.json",),
        evidence_sha256=("a" * 64,),
    )


def _gpu(free_mib: int = 49_000) -> dict[str, object]:
    return {
        "status": "OBSERVED",
        "gpu": {
            "name": "NVIDIA RTX A6000",
            "memory_used_mib": 100,
            "memory_free_mib": free_mib,
            "memory_total_mib": 49_140,
            "utilization_percent": 0,
            "power_draw_watts": 20.0,
        },
        "compute_processes": [],
    }


def test_repository_specs_and_empirical_threshold_are_exact() -> None:
    specs = load_server_specs(ROOT)
    assert [(item.profile, item.port, item.expected_model_id) for item in specs] == [
        ("phase2_main", 8102, "laplace-qwen3.6-35b-a3b-w4a16"),
        ("phase2_rtl_worker", 8103, "laplace-codev-r1-rl-qwen-7b-w4a16"),
    ]
    policy = derive_empirical_memory_policy(ROOT)
    assert policy.maximum_observed_used_mib == 43_801
    assert policy.required_residual_free_mib == 4_096
    assert policy.required_pre_start_free_mib == 47_897
    assert len(policy.evidence_paths) == 6
    assert all(len(digest) == 64 for digest in policy.evidence_sha256)


def test_preflight_reuses_only_healthy_exact_models(tmp_path: Path) -> None:
    admission = ModelServerAdmission(
        tmp_path,
        gpu_observer=lambda: _gpu(1),
        endpoint_probe=lambda spec: {
            "status": "HEALTHY_EXACT_MODEL",
            "expected_model_id": spec.expected_model_id,
            "served_model_ids": [spec.expected_model_id],
        },
        port_observer=lambda _port: [{"pid": 123}],
    )
    output = tmp_path / "model_server_preflight.json"
    result = admission.preflight(
        output_path=output,
        specs=_specs(),
        policy=_policy(),
        startup_timeout_seconds=60,
    )
    assert result["decision"] == "REUSED_HEALTHY_SERVERS"
    assert result["failure_category"] is None
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_preflight_refuses_low_free_memory(tmp_path: Path) -> None:
    admission = ModelServerAdmission(
        tmp_path,
        gpu_observer=lambda: _gpu(47_896),
        endpoint_probe=lambda _spec: {"status": "UNAVAILABLE"},
        port_observer=lambda _port: [],
    )
    result = admission.preflight(
        output_path=tmp_path / "preflight.json",
        specs=_specs(),
        policy=_policy(),
        startup_timeout_seconds=60,
    )
    assert result["decision"] == "REFUSED"
    assert result["failure_category"] == "resource_admission_failure"


@pytest.mark.parametrize(
    ("endpoint_status", "owners"),
    [
        ("MODEL_IDENTITY_MISMATCH", []),
        ("UNAVAILABLE", [{"pid": 999, "process_name": "other"}]),
    ],
)
def test_preflight_refuses_identity_and_port_conflicts(
    tmp_path: Path,
    endpoint_status: str,
    owners: list[dict[str, object]],
) -> None:
    admission = ModelServerAdmission(
        tmp_path,
        gpu_observer=_gpu,
        endpoint_probe=lambda _spec: {"status": endpoint_status},
        port_observer=lambda _port: owners,
    )
    result = admission.preflight(
        output_path=tmp_path / "preflight.json",
        specs=_specs(),
        policy=_policy(),
        startup_timeout_seconds=60,
    )
    assert result["decision"] == "REFUSED"
    assert result["failure_category"] == "model_server_port_conflict"


class _AdmissionStub:
    def __init__(self, decision: str, failure: str | None = None) -> None:
        self.decision = decision
        self.failure = failure

    def preflight(self, **kwargs: object) -> dict[str, object]:
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        value = {"decision": self.decision, "failure_category": self.failure}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return value


def test_controller_does_not_start_when_servers_are_reused(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    controller = ModelServerController(
        ROOT,
        tmp_path,
        admission=_AdmissionStub("REUSED_HEALTHY_SERVERS"),  # type: ignore[arg-type]
        command_runner=runner,
    )
    result = controller.start(startup_timeout_seconds=60)
    assert result["status"] == "REUSED_HEALTHY_SERVERS"
    assert calls == []


def test_controller_surfaces_preflight_failure_without_start(tmp_path: Path) -> None:
    controller = ModelServerController(
        ROOT,
        tmp_path,
        admission=_AdmissionStub(  # type: ignore[arg-type]
            "REFUSED", "model_server_port_conflict"
        ),
        command_runner=lambda *_args, **_kwargs: pytest.fail("must not start"),
    )
    with pytest.raises(ModelServerSafetyError) as caught:
        controller.start(startup_timeout_seconds=60)
    assert caught.value.category == "model_server_port_conflict"


def test_release_does_not_signal_without_owned_pid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ModelServerController(
        ROOT,
        tmp_path,
        admission=_AdmissionStub("REUSED_HEALTHY_SERVERS"),  # type: ignore[arg-type]
        command_runner=lambda *_args, **_kwargs: pytest.fail("must not signal"),
    )
    monkeypatch.setattr(controller, "_owned_profiles", lambda: [])
    result = controller.release_owned()
    assert result["status"] == "NO_LAPLACE_OWNED_SERVERS"
    assert result["signalled_pids"] == []


def test_release_preserves_unrelated_gpu_process_and_verifies_endpoints_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_observations = iter(
        [
            {
                "status": "OBSERVED",
                "compute_processes": [
                    {"pid": 101, "process_name": "vllm"},
                    {"pid": 202, "process_name": "unrelated"},
                ],
            },
            {
                "status": "OBSERVED",
                "compute_processes": [
                    {"pid": 202, "process_name": "unrelated"}
                ],
            },
        ]
    )
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "stopped", "")

    controller = ModelServerController(
        ROOT,
        tmp_path,
        admission=_AdmissionStub("REUSED_HEALTHY_SERVERS"),  # type: ignore[arg-type]
        command_runner=runner,
        gpu_observer=lambda: next(gpu_observations),
        endpoint_probe=lambda _spec: {"status": "UNAVAILABLE"},
    )
    owned_states = iter(
        [
            [{"profile": "phase2_main", "pid": 101}],
            [],
        ]
    )
    monkeypatch.setattr(controller, "_owned_profiles", lambda: next(owned_states))
    result = controller.release_owned()
    assert result["status"] == "RELEASED_LAPLACE_OWNED_SERVERS"
    assert result["unrelated_compute_processes_preserved"] is True
    assert result["endpoints_down"] is True
    assert commands[0][-1] == "stop-phase3"
