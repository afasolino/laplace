from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.inference import ServingCandidate
from research_workspace.llm import GenerationResult, ModelRequired
from research_workspace.model_routing import (
    AuditedModelCaller,
    DualModelConfiguration,
    RoleRouter,
    RoutingTaskMetadata,
)
from research_workspace.repair_protocol import StructuredOutputError, file_sha256
from research_workspace.rtl_contract import (
    build_rtl_worker_contract,
    codev_replacement_plan,
    parse_codev_rtl_answer,
)


def _candidate(*, model: str, endpoint: str, context_tokens: int = 32768) -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint=endpoint,
        model=model,
        model_path="/models/test",
        revision="0" * 40,
        quantization="test",
        kernel="test",
        prefix_caching=True,
        chunked_prefill=True,
        cuda_graph_mode="test",
        scheduler="continuous_batching",
        context_tokens=context_tokens,
        max_output_tokens=6144,
        temperature=0.6,
        top_p=0.95,
        seed=0,
        request_timeout_seconds=60,
        context_safety_margin_tokens=512,
        minimum_completion_tokens=256,
        reviewer_max_output_tokens=2048,
        structured_serialization_max_output_tokens=6144,
        structured_serialization_temperature=0.7,
        structured_serialization_top_p=0.8,
        structured_serialization_top_k=20,
        structured_serialization_presence_penalty=1.5,
    )


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def token_count(self, prompt: str) -> int:
        return len(prompt.split())

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        self.calls.append({"prompt": prompt, **kwargs})
        return GenerationResult(
            text="usable response",
            model="laplace-qwen3.6-35b-a3b-w4a16",
            ttft_seconds=0.01,
            output_tokens_per_second=100.0,
            status="measured",
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
        )

    def health(self) -> dict[str, str]:
        return {"status": "AVAILABLE"}

    def model_identity(self) -> dict[str, str]:
        return {"model": "test"}


def test_non_thinking_machine_role_uses_qwen_sampling_and_larger_cap(tmp_path: Path) -> None:
    candidate = _candidate(
        model="laplace-qwen3.6-35b-a3b-w4a16",
        endpoint="http://127.0.0.1:8102",
    )
    backend = RecordingBackend()
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate)),
        tmp_path,
        backend_factory=lambda _: backend,
    )
    call = caller.generate(
        "interpret the retrieved evidence",
        role="retrieval_interpretation",
        metadata=RoutingTaskMetadata.single_model(
            task_id="task",
            experiment_arm="B",
            domain="c",
        ),
        enable_thinking=False,
    )
    assert call.response_valid
    assert call.budget["requested_completion_tokens"] == 4096
    request = backend.calls[-1]
    assert request["enable_thinking"] is False
    assert request["temperature"] == pytest.approx(0.7)
    assert request["top_p"] == pytest.approx(0.8)
    assert request["top_k"] == 20
    assert request["presence_penalty"] == pytest.approx(1.5)


def test_deterministic_contract_extracts_public_ansi_interface(tmp_path: Path) -> None:
    relative = Path("rtl/elastic.sv")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text(
        "module elastic #(parameter int WIDTH=8)("
        "input logic clk,input logic rst_n,input logic in_valid,output logic in_ready,"
        "input logic [WIDTH-1:0] in_data,output logic out_valid,input logic out_ready,"
        "output logic [WIDTH-1:0] out_data); endmodule\n",
        encoding="utf-8",
    )
    record = {
        "path": relative.as_posix(),
        "language": "systemverilog",
        "kind": "source",
        "sha256": file_sha256(source),
        "content": source.read_text(encoding="utf-8"),
    }
    contract = build_rtl_worker_contract(
        root=tmp_path,
        task_id="elastic",
        task_specification={
            "functional_requirements": ["hold output while stalled"],
            "interfaces": [
                {
                    "name": "stream",
                    "protocol": "ready_valid",
                    "ordering": "preserve order",
                    "backpressure": "hold valid and data until accepted",
                    "signals": ["use exact module ports"],
                }
            ],
            "clock_reset": {"clock_domains": ["clk"]},
            "coding_constraints": ["Synthesizable portable subset only."],
            "error_and_corner_behavior": ["Handle simultaneous dequeue and enqueue."],
            "verification": {
                "commands": [
                    "iverilog compile",
                    "vvp simulation",
                    "verilator lint",
                    "yosys synthesis",
                ],
                "acceptance_criteria": ["all deterministic checks pass"],
            },
        },
        current_source=record,
        editable_path=relative.as_posix(),
        language="systemverilog",
        defect_report=None,
    )
    value = contract.to_json()
    assert value["module_name"] == "elastic"
    assert value["current_source"] == {
        "sha256": file_sha256(source),
        "content": source.read_text(encoding="utf-8"),
    }
    assert [item["name"] for item in value["parameters"]] == ["WIDTH"]
    assert [item["name"] for item in value["ports"]] == [
        "clk",
        "rst_n",
        "in_valid",
        "in_ready",
        "in_data",
        "out_valid",
        "out_ready",
        "out_data",
    ]
    assert value["clock_reset"]["reset"]["active_level"] == "low"

    worker_response = (
        "<think>Implement a one-entry elastic buffer.</think>"
        "<answer>```systemverilog\n"
        "module elastic #(parameter int WIDTH=8)("
        "input logic clk,input logic rst_n,input logic in_valid,output logic in_ready,"
        "input logic [WIDTH-1:0] in_data,output logic out_valid,input logic out_ready,"
        "output logic [WIDTH-1:0] out_data);\n"
        "assign in_ready = out_ready || !out_valid;\n"
        "always_ff @(posedge clk) begin\n"
        "  if (!rst_n) begin out_valid <= 1'b0; out_data <= '0; end\n"
        "  else if (in_ready) begin out_valid <= in_valid; out_data <= in_data; end\n"
        "end\nendmodule\n```</answer>"
    )
    parsed_source = parse_codev_rtl_answer(worker_response, contract=contract)
    assert parsed_source.startswith("module elastic")
    plan = codev_replacement_plan(worker_response, contract=contract)
    assert '"path":"rtl/elastic.sv"' in plan
    assert file_sha256(source) in plan


def test_codev_answer_rejects_extra_modules(tmp_path: Path) -> None:
    relative = Path("rtl/unit.v")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("module unit(input clk); endmodule\n", encoding="utf-8")
    contract = build_rtl_worker_contract(
        root=tmp_path,
        task_id="unit",
        task_specification={
            "functional_requirements": ["preserve the interface"],
            "verification": {
                "commands": ["iverilog compile", "vvp simulation", "yosys synthesis"],
                "acceptance_criteria": ["all checks pass"],
            },
        },
        current_source={
            "path": relative.as_posix(),
            "sha256": file_sha256(source),
            "content": source.read_text(encoding="utf-8"),
        },
        editable_path=relative.as_posix(),
        language="verilog",
        defect_report=None,
    )
    with pytest.raises(StructuredOutputError, match="exactly the contracted module"):
        parse_codev_rtl_answer(
            "<answer>```verilog\nmodule unit(input clk); endmodule\n"
            "module extra; endmodule\n```</answer>",
            contract=contract,
        )


def test_specialist_route_cannot_silently_fallback() -> None:
    main = _candidate(
        model="laplace-qwen3.6-35b-a3b-w4a16",
        endpoint="http://127.0.0.1:8102",
    )
    worker = _candidate(
        model="laplace-codev-r1-rl-qwen-7b-w4a16",
        endpoint="http://127.0.0.1:8103",
        context_tokens=16384,
    )
    router = RoleRouter(
        DualModelConfiguration(
            main=main,
            rtl_worker=worker,
            worker_contract_retries=0,
            worker_response_retries=1,
            fallback_to_main=False,
        )
    )
    metadata = RoutingTaskMetadata(
        task_id="elastic",
        experiment_arm="C",
        domain="systemverilog",
        task_kind="implementation",
        rtl_scope="bounded_module",
        worker_eligible=True,
        editable_sources=("rtl/elastic.sv",),
        module_count=1,
        synthesizable=True,
        explicit_ports=True,
        cycle_behavior_specified=True,
        deterministic_verification=True,
    )
    assert router.select("bounded_rtl_implementation", metadata).selected == "rtl_worker"
    with pytest.raises(ModelRequired, match="fallback is disabled"):
        router.fallback(
            "bounded_rtl_implementation",
            metadata,
            failed_reason="worker response rejected",
        )
