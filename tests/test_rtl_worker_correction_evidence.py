from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

from research_workspace import engineering
from research_workspace.engineering import AgentTask, LocalToolRunner, ToolResult
from research_workspace.inference import ServingCandidate
from research_workspace.llm import GenerationResult
from research_workspace.model_routing import (
    AuditedModelCaller,
    DualModelConfiguration,
    RoleRouter,
    RoutingTaskMetadata,
    WorkerFallbackDisabled,
)
from research_workspace.multilanguage_ablation import (
    _exception_failure_category,
    _failure_outcome_kind,
)
from research_workspace.repair_protocol import file_sha256
from research_workspace.rtl_contract import (
    RtlWorkerContract,
    build_rtl_worker_contract,
    rtl_worker_prompt,
)
from research_workspace.team_runner import LocalTeamRunner


def _candidate() -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint="http://127.0.0.1:8103",
        model="laplace-codev-r1-rl-qwen-7b-w4a16",
        model_path="/models/codev",
        revision="0" * 40,
        quantization="test",
        kernel="test",
        prefix_caching=True,
        chunked_prefill=True,
        cuda_graph_mode="test",
        scheduler="continuous_batching",
        context_tokens=16384,
        max_output_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        request_timeout_seconds=60,
        context_safety_margin_tokens=512,
        minimum_completion_tokens=256,
        reviewer_max_output_tokens=768,
        structured_serialization_max_output_tokens=4096,
        structured_serialization_temperature=0.0,
        structured_serialization_top_p=1.0,
        structured_serialization_top_k=None,
        structured_serialization_presence_penalty=0.0,
    )


def _metadata() -> RoutingTaskMetadata:
    return RoutingTaskMetadata(
        task_id="elastic",
        experiment_arm="C",
        domain="systemverilog",
        task_kind="repair",
        rtl_scope="bounded_module",
        worker_eligible=True,
        editable_sources=("rtl/elastic.sv",),
        module_count=1,
        synthesizable=True,
        explicit_ports=True,
        cycle_behavior_specified=True,
        deterministic_verification=True,
    )


def _contract(tmp_path: Path) -> RtlWorkerContract:
    source = tmp_path / "rtl/elastic.sv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "module elastic(input logic clk,input logic rst_n,input logic in_valid,"
        "output logic in_ready,input logic [7:0] in_data,output logic out_valid,"
        "input logic out_ready,output logic [7:0] out_data); endmodule\n",
        encoding="utf-8",
    )
    return build_rtl_worker_contract(
        root=tmp_path,
        task_id="elastic",
        task_specification={
            "functional_requirements": [
                "full throughput when unstalled",
                "no loss on simultaneous events",
            ],
            "interfaces": [
                {
                    "name": "stream",
                    "protocol": "ready_valid",
                    "ordering": "preserve order",
                    "backpressure": "hold payload while stalled",
                    "signals": ["use exact ports"],
                }
            ],
            "clock_reset": {"clock_domains": ["clk"]},
            "coding_constraints": ["Synthesizable portable subset only."],
            "error_and_corner_behavior": ["Handle simultaneous pop and push."],
            "verification": {
                "commands": ["iverilog", "vvp", "verilator", "yosys"],
                "acceptance_criteria": ["all checks pass"],
            },
        },
        current_source={
            "path": "rtl/elastic.sv",
            "sha256": file_sha256(source),
            "content": source.read_text(encoding="utf-8"),
        },
        editable_path="rtl/elastic.sv",
        language="systemverilog",
        defect_report={
            "observed_result": "full simultaneous replacement was lost",
            "violated_requirement": ["no loss on simultaneous events"],
        },
    )


def test_repair_prompt_frontloads_behavior_and_rejects_unchanged_source(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    prompt = rtl_worker_prompt(
        contract,
        retry_index=1,
        prior_error="full buffer did not accept replacement during pop",
    )

    guidance_index = prompt.index("MANDATORY BEHAVIORAL REQUIREMENTS")
    contract_index = prompt.index("RTL contract:")
    assert guidance_index < contract_index
    assert "assert in_ready" in prompt
    assert "keep occupancy full" in prompt
    assert "Returning the current source unchanged is invalid" in prompt
    assert contract.current_source_sha256 in prompt


class _NoOpBackend:
    def token_count(self, prompt: str) -> int:
        return len(prompt.split())

    def generate(self, _prompt: str, **_kwargs: object) -> GenerationResult:
        return GenerationResult(
            text="unchanged",
            model="laplace-codev-r1-rl-qwen-7b-w4a16",
            ttft_seconds=0.01,
            output_tokens_per_second=50.0,
            status="measured",
            prompt_tokens=100,
            completion_tokens=20,
            finish_reason="stop",
        )

    def health(self) -> dict[str, str]:
        return {"status": "AVAILABLE"}

    def model_identity(self) -> dict[str, str]:
        return {"model": "test"}


def test_no_source_change_has_dedicated_failure_category(tmp_path: Path) -> None:
    candidate = _candidate()
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate, rtl_worker=candidate)),
        tmp_path / "audits",
        backend_factory=lambda _: _NoOpBackend(),
    )

    def reject_no_op(_: str) -> None:
        raise ValueError("Replacement makes no source change")

    call = caller.generate(
        "return corrected source",
        role="bounded_rtl_repair",
        metadata=_metadata(),
        validator=reject_no_op,
        enable_thinking=False,
    )

    assert call.response_valid is False
    assert call.failure_category == "no_effect_correction"


def test_disabled_fallback_category_is_model_quality_failure() -> None:
    exc = WorkerFallbackDisabled(
        "worker returned unchanged source", category="no_effect_correction"
    )

    assert _exception_failure_category(exc) == "no_effect_correction"
    assert _failure_outcome_kind("no_effect_correction") == "model_quality_failure"
    assert _failure_outcome_kind("endpoint_unavailable") == "infrastructure_failure"


def test_standalone_lane_prepares_missing_governed_corpus(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from research_workspace import multilanguage_ablation as ablation
    from research_workspace.multilanguage_ablation import ExperimentConfiguration

    configuration = cast(
        ExperimentConfiguration,
        SimpleNamespace(
            base_reference_root=tmp_path / "authoritative",
            overlay_root=tmp_path / "overlay",
        ),
    )
    validations = iter(
        [
            {"status": "FAILED", "domains": []},
            {
                "status": "VERIFIED_NON_EMPTY",
                "domains": [
                    {
                        "domain": "systemverilog",
                        "passed": True,
                        "retrieved_chunk_count": 3,
                    }
                ],
            },
        ]
    )
    prepared: list[tuple[Path, Path, Path]] = []

    monkeypatch.setattr(
        ablation,
        "validate_corpus_retrieval",
        lambda _overlay: next(validations),
    )
    monkeypatch.setattr(
        ablation,
        "prepare_corpus_overlay",
        lambda repository_root, base_root, overlay_root: prepared.append(
            (repository_root, base_root, overlay_root)
        )
        or {"status": "CORPUS_OVERLAY_READY"},
    )

    result = ablation._ensure_lane_corpus_ready(tmp_path, configuration)

    assert result["status"] == "PREPARED_VERIFIED_OVERLAY"
    assert prepared == [
        (tmp_path, configuration.base_reference_root, configuration.overlay_root)
    ]
    assert result["validation"]["status"] == "VERIFIED_NON_EMPTY"


def test_standalone_lane_reuses_verified_governed_corpus(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from research_workspace import multilanguage_ablation as ablation
    from research_workspace.multilanguage_ablation import ExperimentConfiguration

    configuration = cast(
        ExperimentConfiguration,
        SimpleNamespace(
            base_reference_root=tmp_path / "authoritative",
            overlay_root=tmp_path / "overlay",
        ),
    )
    monkeypatch.setattr(
        ablation,
        "validate_corpus_retrieval",
        lambda _overlay: {
            "status": "VERIFIED_NON_EMPTY",
            "domains": [
                {
                    "domain": "systemverilog",
                    "passed": True,
                    "retrieved_chunk_count": 2,
                }
            ],
        },
    )

    def unexpected_prepare(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("verified corpus must not be rebuilt")

    monkeypatch.setattr(ablation, "prepare_corpus_overlay", unexpected_prepare)

    result = ablation._ensure_lane_corpus_ready(tmp_path, configuration)

    assert result["status"] == "REUSED_VERIFIED_OVERLAY"
    assert result["preparation"] is None


def test_verilator_simulation_is_required_only_when_declared() -> None:
    without_simulation = cast(
        AgentTask,
        SimpleNamespace(
            specification={
                "quality_contract": {
                    "required_gates": [
                        "verilator_lint",
                        "iverilog_compile",
                        "vvp_simulation",
                        "yosys_synthesis",
                    ]
                }
            }
        ),
    )
    with_simulation = cast(
        AgentTask,
        SimpleNamespace(
            specification={
                "quality_contract": {
                    "required_gates": [
                        "verilator_lint",
                        "verilator_simulation",
                    ]
                }
            }
        ),
    )

    assert LocalTeamRunner._requires_verilator_simulation(without_simulation) is False
    assert LocalTeamRunner._requires_verilator_simulation(with_simulation) is True


def test_reviewer_stale_source_quote_is_rejected() -> None:
    source = {
        "current_worktree_sources": [
            {
                "path": "rtl/elastic.sv",
                "sha256": "a" * 64,
                "content": (
                    "assign in_ready = (count_q < 2) || out_ready;\n"
                    "data_q[0] <= data_q[1];\n"
                    "data_q[1] <= in_data;\n"
                ),
            }
        ],
        "source_state_fingerprint": "b" * 64,
    }
    verdict = {
        "schema_version": 1,
        "verdict": "block",
        "reason": (
            "The stale expression "
            "`data_q[0]<=count_q==1?in_data:data_q[1];` still loses data."
        ),
        "missing_evidence": [],
    }

    error = LocalTeamRunner._stale_reviewer_evidence(
        verdict, source, {"passed": True}
    )

    assert error is not None
    assert error.startswith("reviewer_stale_evidence:")


def test_reviewer_current_source_quote_is_not_rejected() -> None:
    source = {
        "current_worktree_sources": [
            {
                "path": "rtl/elastic.sv",
                "sha256": "a" * 64,
                "content": "assign in_ready = (count_q < 2) || out_ready;\n",
            }
        ]
    }
    verdict = {
        "schema_version": 1,
        "verdict": "request_changes",
        "reason": "Inspect `assign in_ready = (count_q < 2) || out_ready;`.",
        "missing_evidence": [],
    }

    assert (
        LocalTeamRunner._stale_reviewer_evidence(
            verdict, source, {"passed": True}
        )
        is None
    )


def test_review_prompt_frontloads_authoritative_source() -> None:
    task = cast(
        AgentTask,
        SimpleNamespace(
            domain="systemverilog",
            specification={
                "functional_requirements": ["no loss on simultaneous events"]
            }
        ),
    )
    source = {
        "current_worktree_sources": [
            {
                "path": "rtl/elastic.sv",
                "sha256": "a" * 64,
                "content": "module elastic; endmodule\n",
            }
        ]
    }

    prompt = LocalTeamRunner._review_prompt(
        task,
        {"governed_references": [{"reference_id": "example"}]},
        {"passed": True},
        source,
        prior_reviewer_error="reviewer_stale_evidence: obsolete code",
    )

    assert prompt.index("Authoritative current source:") < prompt.index(
        "Reference evidence:"
    )
    assert "a" * 64 in prompt
    assert "prior verdict was rejected" in prompt


def test_systemverilog_normalization_requires_executable_verilator(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        engineering,
        "validate_task_spec",
        lambda *_args, **_kwargs: None,
    )
    raw = {
        "task_id": "rtl_gate_contract",
        "functional_requirements": ["deterministic simulation"],
    }

    systemverilog = engineering.normalize_task_spec(tmp_path, "systemverilog", raw)
    verilog = engineering.normalize_task_spec(tmp_path, "verilog", raw)

    systemverilog_gates = systemverilog["quality_contract"]["required_gates"]
    verilog_gates = verilog["quality_contract"]["required_gates"]
    assert "verilator_simulation" in systemverilog_gates
    assert "verilator_simulation" not in verilog_gates


def _fake_eda_run(
    *, create_verilator_binary: bool
) -> Callable[..., ToolResult]:
    def run(
        tool: str,
        command: list[str],
        *,
        timeout_seconds: int = 300,
    ) -> ToolResult:
        del timeout_seconds
        if tool == "verilator" and "--binary" in command and create_verilator_binary:
            mdir = Path(command[command.index("--Mdir") + 1])
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / "simv").write_text("simulator", encoding="utf-8")
        return ToolResult(
            tool=tool,
            command=tuple(command),
            returncode=0,
            elapsed_seconds=0.01,
            status="PASS",
            stdout="PASS",
            stderr="",
            log_path="/tmp/fake-tool-log.json",
        )

    return run


def test_required_verilator_simulation_executes_generated_binary(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "dut.sv"
    testbench = tmp_path / "tb_public.sv"
    source.write_text("module dut; endmodule\n", encoding="utf-8")
    testbench.write_text(
        "module tb_public; initial begin $display(\"PASS\"); $finish; end endmodule\n",
        encoding="utf-8",
    )
    runner = LocalToolRunner(tmp_path, tmp_path / "tool_logs")
    monkeypatch.setattr(
        engineering.shutil, "which", lambda tool: f"/usr/bin/{tool}"
    )
    monkeypatch.setattr(engineering, "verilator_simulation_available", lambda: True)
    monkeypatch.setattr(
        runner,
        "run",
        _fake_eda_run(create_verilator_binary=True),
    )

    report = runner.run_eda_flow(
        ["dut.sv"],
        top_module="dut",
        testbench="tb_public.sv",
        language="systemverilog",
        require_verilator_simulation=True,
        required_tools=(
            "iverilog",
            "vvp",
            "verilator",
            "verilator_simulation",
            "yosys",
        ),
    )

    assert report["passed"] is True
    assert report["verilator_simulation_executed"] is True
    assert report["missing_required_results"] == []
    assert "verilator_simulation" in report["executed_tools"]


def test_required_verilator_simulation_cannot_be_silently_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "dut.sv"
    testbench = tmp_path / "tb_public.sv"
    source.write_text("module dut; endmodule\n", encoding="utf-8")
    testbench.write_text(
        "module tb_public; initial begin $display(\"PASS\"); $finish; end endmodule\n",
        encoding="utf-8",
    )
    runner = LocalToolRunner(tmp_path, tmp_path / "tool_logs")
    monkeypatch.setattr(
        engineering.shutil, "which", lambda tool: f"/usr/bin/{tool}"
    )
    monkeypatch.setattr(engineering, "verilator_simulation_available", lambda: True)
    monkeypatch.setattr(
        runner,
        "run",
        _fake_eda_run(create_verilator_binary=False),
    )

    report = runner.run_eda_flow(
        ["dut.sv"],
        top_module="dut",
        testbench="tb_public.sv",
        language="systemverilog",
        require_verilator_simulation=True,
        required_tools=(
            "iverilog",
            "vvp",
            "verilator",
            "verilator_simulation",
            "yosys",
        ),
    )

    assert report["passed"] is False
    assert report["verilator_simulation_executed"] is False
    assert report["missing_required_results"] == ["verilator_simulation"]
