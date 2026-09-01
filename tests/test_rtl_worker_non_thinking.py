from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.inference import ServingCandidate
from research_workspace.llm import GenerationResult, ModelRequired
from research_workspace.model_routing import (
    AuditedModelCaller,
    DualModelConfiguration,
    RoleRouter,
    RoutingTaskMetadata,
    load_dual_model_configuration,
    supports_qwen3_structured_serialization,
)
from research_workspace.repair_protocol import StructuredOutputError, file_sha256
from research_workspace.rtl_contract import (
    RtlWorkerContract,
    build_rtl_worker_contract,
    parse_codev_rtl_answer,
    rtl_worker_prompt,
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


def _contract(tmp_path: Path, requirements: list[str]) -> RtlWorkerContract:
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
    return build_rtl_worker_contract(
        root=tmp_path,
        task_id="elastic",
        task_specification={
            "functional_requirements": requirements,
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
        current_source={
            "path": relative.as_posix(),
            "sha256": file_sha256(source),
            "content": source.read_text(encoding="utf-8"),
        },
        editable_path=relative.as_posix(),
        language="systemverilog",
        defect_report=None,
    )


def test_worker_prompt_requests_one_code_block_without_reasoning(tmp_path: Path) -> None:
    prompt = rtl_worker_prompt(_contract(tmp_path, ["hold output while stalled"]))

    assert "exactly one fenced systemverilog code block" in prompt
    assert "Return the implementation immediately" in prompt
    assert "<think>" not in prompt
    assert "<answer>" not in prompt
    assert "Do not emit reasoning" in prompt


def test_parser_accepts_bare_fence_and_legacy_answer_wrapper(tmp_path: Path) -> None:
    contract = _contract(tmp_path, ["hold output while stalled"])
    source = "module elastic(input logic clk); endmodule"

    assert parse_codev_rtl_answer(
        f"```systemverilog\n{source}\n```", contract=contract
    ).startswith("module elastic")
    assert parse_codev_rtl_answer(
        f"<think>legacy</think><answer>```systemverilog\n{source}\n```</answer>",
        contract=contract,
    ).startswith("module elastic")


def test_worker_prompt_and_parser_accept_model_native_reasoning(tmp_path: Path) -> None:
    contract = _contract(tmp_path, ["hold output while stalled"])
    prompt = rtl_worker_prompt(contract, allow_model_reasoning=True)
    source = "module elastic(input logic clk); endmodule"

    assert "Model-native internal reasoning is permitted" in prompt
    assert "<answer>" in prompt
    assert "Do not emit reasoning" not in prompt
    assert parse_codev_rtl_answer(
        f"<think>check behavior</think>\n```systemverilog\n{source}\n```",
        contract=contract,
    ).startswith("module elastic")
    assert parse_codev_rtl_answer(
        f"<think>check behavior</think><answer>```systemverilog\n{source}\n```</answer>",
        contract=contract,
    ).startswith("module elastic")
    assert parse_codev_rtl_answer(
        f"check behavior</think><answer>```verilog\n{source}\n```</answer>",
        contract=contract,
    ).startswith("module elastic")


def test_parser_rejects_unclosed_or_non_leading_siliconmind_tags(tmp_path: Path) -> None:
    contract = _contract(tmp_path, ["hold output while stalled"])
    source = "module elastic(input logic clk); endmodule"

    with pytest.raises(StructuredOutputError, match="malformed reasoning/answer tags"):
        parse_codev_rtl_answer(
            f"reasoning <think>without close<answer>```verilog\n{source}\n```</answer>",
            contract=contract,
        )
    with pytest.raises(StructuredOutputError, match="contains no source"):
        parse_codev_rtl_answer(
            f"```verilog\n{source}\n``` trailing prose</think>",
            contract=contract,
        )


def test_qwen38_p8_name_is_eligible_for_qwen3_structured_serialization() -> None:
    p8 = _candidate(
        model="laplace-quality-qwen38-mtp8", endpoint="http://127.0.0.1:8207"
    )
    legacy = _candidate(
        model="laplace-qwen3.6-35b-a3b-w4a16", endpoint="http://127.0.0.1:8102"
    )
    non_qwen3 = _candidate(
        model="laplace-codev-r1-rl-qwen-7b-w4a16", endpoint="http://127.0.0.1:8103"
    )

    assert supports_qwen3_structured_serialization(p8)
    assert supports_qwen3_structured_serialization(legacy)
    assert not supports_qwen3_structured_serialization(non_qwen3)


def _candidate_configuration(model: str, endpoint: str) -> dict[str, object]:
    return {
        "engine": "vllm",
        "endpoint": endpoint,
        "model": model,
        "revision": "0" * 40,
        "quantization": "test",
        "kernel": "test",
        "prefix_caching": True,
        "chunked_prefill": True,
        "cuda_graph_mode": "test",
        "scheduler": "continuous_batching",
    }


def _dual_configuration_payload(*, schema_version: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "main": _candidate_configuration(
            "laplace-quality-qwen38-mtp8", "http://127.0.0.1:8207"
        ),
        "rtl_worker": _candidate_configuration(
            "siliconmind-qwen3-4b", "http://127.0.0.1:8208"
        ),
        "worker_contract_retries": 1,
        "worker_response_retries": 1,
        "fallback_to_main": False,
    }
    if schema_version == 2:
        payload["worker_reasoning_mode"] = "model_default"
    return payload


def test_dual_model_schema_v1_remains_non_thinking_by_default(tmp_path: Path) -> None:
    path = tmp_path / "models-v1.json"
    path.write_text(json.dumps(_dual_configuration_payload(schema_version=1)), encoding="utf-8")

    assert load_dual_model_configuration(path).worker_reasoning_mode == "disabled"


def test_dual_model_schema_v2_can_select_model_default_reasoning(tmp_path: Path) -> None:
    path = tmp_path / "models-v2.json"
    path.write_text(json.dumps(_dual_configuration_payload(schema_version=2)), encoding="utf-8")

    assert load_dual_model_configuration(path).worker_reasoning_mode == "model_default"


def test_dual_model_schema_v2_rejects_unknown_reasoning_mode(tmp_path: Path) -> None:
    path = tmp_path / "models-v2-invalid.json"
    payload = _dual_configuration_payload(schema_version=2)
    payload["worker_reasoning_mode"] = "forced"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="worker_reasoning_mode"):
        load_dual_model_configuration(path)


def test_fifo_contract_makes_full_simultaneous_transfer_explicit(tmp_path: Path) -> None:
    value = _contract(
        tmp_path,
        ["full throughput when unstalled", "no loss on simultaneous events"],
    ).to_json()

    cycle = " ".join(value["cycle_requirements"])
    handshake = " ".join(value["handshake_and_events"])
    assert "storage is full" in cycle
    assert "keep occupancy full" in cycle
    assert "same-cycle accepted output transfer" in handshake


def test_no_fallback_preserves_specialist_failure_category() -> None:
    main = _candidate(
        model="laplace-qwen3.6-35b-a3b-w4a16", endpoint="http://127.0.0.1:8102"
    )
    worker = _candidate(
        model="laplace-codev-r1-rl-qwen-7b-w4a16",
        endpoint="http://127.0.0.1:8103",
        context_tokens=16384,
    )
    router = RoleRouter(
        DualModelConfiguration(main=main, rtl_worker=worker, fallback_to_main=False)
    )

    with pytest.raises(ModelRequired) as caught:
        router.fallback(
            "bounded_rtl_implementation",
            _metadata(),
            failed_reason="finish_reason='length'",
            failed_category="truncated_response",
        )

    assert getattr(caught.value, "category", None) == "truncated_response"
    assert "truncated_response" in str(caught.value)
    assert "finish_reason='length'" in str(caught.value)


class _TruncatingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def token_count(self, prompt: str) -> int:
        return len(prompt.split())

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        self.calls.append({"prompt": prompt, **kwargs})
        return GenerationResult(
            text="partial source",
            model="laplace-codev-r1-rl-qwen-7b-w4a16",
            ttft_seconds=0.01,
            output_tokens_per_second=50.0,
            status="measured",
            prompt_tokens=100,
            completion_tokens=4096,
            finish_reason="length",
        )

    def health(self) -> dict[str, str]:
        return {"status": "AVAILABLE"}

    def model_identity(self) -> dict[str, str]:
        return {"model": "test"}


def test_routed_call_exposes_truncation_and_disables_thinking(tmp_path: Path) -> None:
    main = _candidate(
        model="laplace-qwen3.6-35b-a3b-w4a16", endpoint="http://127.0.0.1:8102"
    )
    worker = _candidate(
        model="laplace-codev-r1-rl-qwen-7b-w4a16",
        endpoint="http://127.0.0.1:8103",
        context_tokens=16384,
    )
    backend = _TruncatingBackend()
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=main, rtl_worker=worker)),
        tmp_path / "audits",
        backend_factory=lambda _: backend,
    )

    call = caller.generate(
        "return one RTL module",
        role="bounded_rtl_implementation",
        metadata=_metadata(),
        enable_thinking=False,
    )

    assert call.response_valid is False
    assert call.failure_category == "truncated_response"
    assert call.validation_error is not None
    assert "finish_reason='length'" in call.validation_error
    assert backend.calls[0]["enable_thinking"] is False
