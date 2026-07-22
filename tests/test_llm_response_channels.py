from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest

from research_workspace.engineering import AgentTask, EngineeringError
from research_workspace.inference import ServingCandidate
from research_workspace.llm import GenerationResult, OpenAICompatibleProvider, VllmProvider
from research_workspace.model_routing import (
    AuditedModelCaller,
    DualModelConfiguration,
    RoleRouter,
    RoutingTaskMetadata,
    StructuredSerializationCapacityError,
)
from research_workspace.multilanguage_ablation import _model_response_failure_category
from research_workspace.repair_protocol import (
    estimate_replacement_plan_tokens,
    parse_replacement_plan,
    replacement_plan_json_schema,
)
from research_workspace.team_runner import LocalTeamRunner, Worktree


class _StreamingResponse:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.lines = [f"data: {json.dumps(event)}\n".encode() for event in events]
        self.lines.append(b"data: [DONE]\n")

    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)


class _JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _usage(prompt: int = 7, completion: int = 5) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def test_streaming_null_content_preserves_separate_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(
        [
            {
                "choices": [
                    {
                        "delta": {"content": None, "reasoning_content": "private reasoning"},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": _usage()},
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: response)
    result = VllmProvider("http://127.0.0.1:8102", "qwen").generate("prompt")
    assert result.text == ""
    assert result.reasoning_text == "private reasoning"
    assert result.finish_reason == "length"


def test_streaming_reasoning_then_valid_final_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(
        [
            {
                "choices": [
                    {"delta": {"content": None, "reasoning": "reasoning "}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": '{"status":"READY"}', "reasoning": None},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": _usage()},
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: response)
    result = VllmProvider("http://127.0.0.1:8102", "qwen").generate("prompt")
    assert result.reasoning_text == "reasoning "
    assert result.text == '{"status":"READY"}'
    assert result.finish_reason == "stop"


def test_non_streaming_reasoning_and_content_are_not_concatenated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _JsonResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"status":"READY"}',
                        "reasoning_content": "private reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                **_usage(),
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: response)
    result = OpenAICompatibleProvider("http://127.0.0.1:8102", "qwen").generate("prompt")
    assert result.text == '{"status":"READY"}'
    assert result.reasoning_text == "private reasoning"
    assert result.reasoning_tokens == 3


def test_schema_constrained_json_request_is_valid_and_disables_only_that_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    def urlopen(request: urllib.request.Request, **kwargs: object) -> _JsonResponse:
        requests.append(json.loads(request.data or b"{}"))
        return _JsonResponse(
            {
                "choices": [
                    {
                        "message": {"content": '{"status":"READY"}', "reasoning": None},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(),
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"const": "READY"}},
    }
    result = OpenAICompatibleProvider("http://127.0.0.1:8102", "qwen").generate(
        "prompt",
        response_schema=schema,
        schema_name="smoke",
        enable_thinking=False,
    )
    assert json.loads(result.text) == {"status": "READY"}
    assert requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "smoke", "schema": schema, "strict": True},
    }
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


class _ResultBackend:
    def __init__(self, results: list[GenerationResult]) -> None:
        self.results = results
        self.options: list[dict[str, object]] = []

    def token_count(self, prompt: str) -> int:
        return 10

    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
        response_schema: dict[str, object] | None = None,
        schema_name: str | None = None,
        enable_thinking: bool | None = None,
    ) -> GenerationResult:
        self.options.append(
            {
                "prompt": prompt,
                "context_tokens": context_tokens,
                "max_tokens": max_tokens,
                "response_schema": response_schema,
                "schema_name": schema_name,
                "enable_thinking": enable_thinking,
            }
        )
        return self.results.pop(0)

    def health(self) -> dict[str, str]:
        return {"status": "AVAILABLE"}

    def model_identity(self) -> dict[str, str]:
        return {"backend": "fixture", "model": "qwen"}


def _candidate(
    *,
    model: str = "laplace-qwen3.6-35b-a3b-w4a16",
    structured_cap: int | None = 8192,
) -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint="http://127.0.0.1:8102",
        model=model,
        revision="fixture",
        quantization="fixture",
        kernel="fixture",
        prefix_caching=True,
        chunked_prefill=True,
        cuda_graph_mode="fixture",
        scheduler="fixture",
        context_tokens=32768,
        max_output_tokens=4096,
        structured_serialization_max_output_tokens=structured_cap,
    )


def _metadata() -> RoutingTaskMetadata:
    return RoutingTaskMetadata.single_model(task_id="fixture", experiment_arm="B", domain="python")


def _source_context(source: Path, *, domain: str = "python") -> dict[str, object]:
    return {
        "source_state_fingerprint": "fixture",
        "current_worktree_sources": [
            {
                "path": source.name,
                "language": domain,
                "kind": "source",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "content": source.read_text(encoding="utf-8"),
            }
        ],
    }


def test_empty_final_content_after_reasoning_is_not_accepted(tmp_path: Path) -> None:
    backend = _ResultBackend(
        [
            GenerationResult(
                "",
                "qwen",
                None,
                None,
                "measured",
                10,
                4096,
                None,
                "reasoning only",
                "length",
            )
        ]
    )
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate())),
        tmp_path,
        backend_factory=lambda candidate: backend,
    )
    call = caller.generate("prompt", role="general_implementation", metadata=_metadata())
    assert call.response_valid is False
    assert "truncated" in str(call.validation_error)
    audit = json.loads(Path(call.audit_path).read_text())
    assert audit["failure_category"] == "truncated_response"
    assert audit["content"] == ""
    assert audit["reasoning_characters"] == len("reasoning only")
    assert "reasoning_text" not in audit


def test_malformed_replacement_json_has_precise_rejection(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("value = 1\n", encoding="utf-8")
    backend = _ResultBackend(
        [GenerationResult('{"schema_version":1', "qwen", None, None, "measured", 10, 4)]
    )
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate())),
        tmp_path / "audits",
        backend_factory=lambda candidate: backend,
    )
    call = caller.generate(
        "prompt",
        role="general_implementation",
        metadata=_metadata(),
        validator=lambda text: parse_replacement_plan(
            text, root=tmp_path, allowed_paths=["candidate.py"], domain="python"
        ),
    )
    assert call.response_valid is False
    audit = json.loads(Path(call.audit_path).read_text())
    assert audit["failure_category"] == "malformed_json"
    assert "not valid JSON" in audit["schema_validation_error"]


def test_finish_reason_length_overrides_accidentally_valid_json(tmp_path: Path) -> None:
    backend = _ResultBackend(
        [
            GenerationResult(
                '{"status":"READY"}', "qwen", None, None, "measured", 10, 5, None, "", "length"
            )
        ]
    )
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate())),
        tmp_path,
        backend_factory=lambda candidate: backend,
    )
    call = caller.generate(
        "prompt",
        role="general_implementation",
        metadata=_metadata(),
        validator=lambda text: json.loads(text),
    )
    assert call.response_valid is False
    audit = json.loads(Path(call.audit_path).read_text())
    assert audit["failure_category"] == "truncated_response"


def test_structured_repair_retry_eventually_succeeds(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("value = 1\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    valid = json.dumps(
        {
            "schema_version": 1,
            "replacements": [
                {
                    "path": "candidate.py",
                    "language": "python",
                    "kind": "source",
                    "expected_sha256": digest,
                    "content": "value = 2\n",
                }
            ],
        }
    )
    backend = _ResultBackend(
        [
            GenerationResult(
                "", "qwen", None, None, "measured", 10, 4096, None, "thinking", "length"
            ),
            GenerationResult(
                "",
                "qwen",
                None,
                None,
                "measured",
                10,
                4096,
                None,
                "thinking again",
                "length",
            ),
            GenerationResult(valid, "qwen", None, None, "measured", 10, 100, None, "", "stop"),
        ]
    )
    candidate = _candidate()
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate)),
        tmp_path / "audits",
        backend_factory=lambda selected: backend,
    )
    runner = LocalTeamRunner(tmp_path, tmp_path, candidate)
    task = AgentTask(
        "fixture",
        "python",
        "implementation",
        {"allowed_paths": ["candidate.py"], "functional_requirements": []},
    )
    worktree = Worktree(tmp_path, "a" * 40, "fixture")
    current_sources = {
        "source_state_fingerprint": "fixture",
        "current_worktree_sources": [
            {
                "path": "candidate.py",
                "language": "python",
                "kind": "source",
                "sha256": digest,
                "content": source.read_text(encoding="utf-8"),
            }
        ],
    }
    first = runner._generate_implementation_call(
        caller, task, _metadata(), worktree, {}, current_sources, None, 0
    )
    second = runner._generate_implementation_call(
        caller,
        task,
        _metadata(),
        worktree,
        {},
        current_sources,
        {"observed_result": first.validation_error},
        1,
    )
    third = runner._generate_implementation_call(
        caller,
        task,
        _metadata(),
        worktree,
        {},
        current_sources,
        {"observed_result": second.validation_error},
        2,
        "structured_replacement_serialization",
    )
    assert first.response_valid is False
    assert second.response_valid is False
    assert third.response_valid is True
    assert backend.options[0]["response_schema"] is None
    assert backend.options[0]["enable_thinking"] is None
    assert backend.options[1]["response_schema"] is None
    assert backend.options[1]["enable_thinking"] is None
    assert backend.options[0]["max_tokens"] == 4096
    assert backend.options[1]["max_tokens"] == 4096
    assert backend.options[2]["response_schema"] == replacement_plan_json_schema(
        allowed_paths=["candidate.py"], domain="python"
    )
    assert backend.options[2]["enable_thinking"] is False
    assert backend.options[2]["max_tokens"] == 8192


def test_primary_and_ordinary_qwen_retry_remain_capped_at_4096(tmp_path: Path) -> None:
    backend = _ResultBackend(
        [
            GenerationResult("primary", "qwen", None, None, "measured", 10, 10),
            GenerationResult("retry", "qwen", None, None, "measured", 10, 10),
        ]
    )
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate())),
        tmp_path,
        backend_factory=lambda candidate: backend,
    )
    caller.generate("primary prompt", role="general_implementation", metadata=_metadata())
    caller.generate(
        "ordinary retry prompt",
        role="general_implementation",
        metadata=_metadata(),
        retry_index=1,
    )
    assert [item["max_tokens"] for item in backend.options] == [4096, 4096]


def test_structured_override_uses_dedicated_cap_without_raising_normal_cap(
    tmp_path: Path,
) -> None:
    backend = _ResultBackend(
        [GenerationResult('{"schema_version":1}', "qwen", None, None, "measured", 10, 20)]
    )
    candidate = _candidate()
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate)),
        tmp_path,
        backend_factory=lambda selected: backend,
    )
    call = caller.generate(
        "serialize",
        role="general_implementation",
        metadata=_metadata(),
        requested_completion_tokens=7000,
        response_schema=replacement_plan_json_schema(
            allowed_paths=["candidate.py"], domain="python"
        ),
        schema_name="laplace_replacement_plan",
        enable_thinking=False,
        call_policy="structured_replacement_serialization",
        estimated_serialization_tokens=100,
        serialization_safety_margin_tokens=4096,
    )
    assert candidate.max_output_tokens == 4096
    assert backend.options[0]["max_tokens"] == 7000
    assert call.budget["normal_output_cap"] == 4096
    assert call.budget["structured_serialization_max_output_tokens"] == 8192
    assert call.budget["selected_output_cap"] == 7000


def test_non_qwen_final_attempt_remains_standard(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("value = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    valid = json.dumps(
        {
            "schema_version": 1,
            "replacements": [
                {
                    "path": "candidate.py",
                    "language": "python",
                    "kind": "source",
                    "expected_sha256": digest,
                    "content": "value = 2\n",
                }
            ],
        }
    )
    backend = _ResultBackend(
        [GenerationResult(valid, "plain", None, None, "measured", 10, 100)]
    )
    candidate = _candidate(model="plain-openai-model")
    runner = LocalTeamRunner(tmp_path, tmp_path, candidate)
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate)),
        tmp_path / "audits",
        backend_factory=lambda selected: backend,
    )
    call = runner._generate_implementation_call(
        caller,
        AgentTask(
            "fixture",
            "python",
            "implementation",
            {"allowed_paths": ["candidate.py"], "functional_requirements": []},
        ),
        _metadata(),
        Worktree(tmp_path, "a" * 40, "fixture"),
        {},
        _source_context(source),
        {"observed_result": "prior rejection"},
        2,
        "structured_replacement_serialization",
    )
    assert call.response_valid is True
    assert backend.options[0]["max_tokens"] == 4096
    assert backend.options[0]["response_schema"] is None
    assert backend.options[0]["enable_thinking"] is None


def test_phase3_worker_route_cannot_inherit_qwen_serializer_policy(tmp_path: Path) -> None:
    main = _candidate()
    worker = ServingCandidate(
        **{
            **_candidate(model="laplace-codev-r1-rl-qwen-7b-w4a16").__dict__,
            "endpoint": "http://127.0.0.1:8103",
            "structured_serialization_max_output_tokens": 4096,
        }
    )
    worker_backend = _ResultBackend(
        [GenerationResult("worker", "worker", None, None, "measured", 10, 10)]
    )
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=main, rtl_worker=worker)),
        tmp_path,
        backend_factory=lambda selected: worker_backend,
    )
    metadata = RoutingTaskMetadata(
        task_id="rtl",
        experiment_arm="C",
        domain="verilog",
        task_kind="implementation",
        rtl_scope="bounded_module",
        worker_eligible=True,
        editable_sources=("candidate.v",),
        module_count=1,
        synthesizable=True,
        explicit_ports=True,
        cycle_behavior_specified=True,
        deterministic_verification=True,
    )
    normal = caller.generate(
        "worker prompt",
        role="bounded_rtl_implementation",
        metadata=metadata,
    )
    assert normal.decision.selected == "rtl_worker"
    assert worker_backend.options[0]["max_tokens"] == 4096
    with pytest.raises(EngineeringError, match="routed Qwen3.6 main model"):
        caller.generate(
            "incorrect serializer routing",
            role="bounded_rtl_implementation",
            metadata=metadata,
            response_schema=replacement_plan_json_schema(
                allowed_paths=["candidate.v"], domain="verilog"
            ),
            schema_name="laplace_replacement_plan",
            enable_thinking=False,
            call_policy="structured_replacement_serialization",
            estimated_serialization_tokens=100,
            serialization_safety_margin_tokens=4096,
        )
    assert len(worker_backend.options) == 1


def test_replacement_larger_than_4096_uses_bounded_8192_cap_and_validates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("".join(f"value_{index:04d} = 0\n" for index in range(900)), encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    replacement = "".join(f"value_{index:04d} = 1\n" for index in range(900))
    valid = json.dumps(
        {
            "schema_version": 1,
            "replacements": [
                {
                    "path": "candidate.py",
                    "language": "python",
                    "kind": "source",
                    "expected_sha256": digest,
                    "content": replacement,
                }
            ],
        }
    )
    source_context = _source_context(source)
    estimated = estimate_replacement_plan_tokens(source_context["current_worktree_sources"])
    assert 4096 < estimated < 8192
    backend = _ResultBackend(
        [GenerationResult(valid, "qwen", None, None, "measured", 10, 5000, None, "", "stop")]
    )
    candidate = _candidate()
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate)),
        tmp_path / "audits",
        backend_factory=lambda selected: backend,
    )
    runner = LocalTeamRunner(tmp_path, tmp_path, candidate)
    call = runner._generate_implementation_call(
        caller,
        AgentTask(
            "fixture",
            "python",
            "implementation",
            {"allowed_paths": ["candidate.py"], "functional_requirements": ["update values"]},
        ),
        _metadata(),
        Worktree(tmp_path, "a" * 40, "fixture"),
        {"verbose_evidence_marker": "must be omitted"},
        source_context,
        {"observed_result": "truncated at the old cap", "source_state": source_context},
        2,
        "structured_replacement_serialization",
    )
    assert call.response_valid is True
    assert backend.options[0]["max_tokens"] == 8192
    assert "verbose_evidence_marker" not in str(backend.options[0]["prompt"])
    assert "truncated at the old cap" in str(backend.options[0]["prompt"])
    assert str(backend.options[0]["prompt"]).count(digest) == 1


def test_estimate_above_8192_fails_before_generation_with_precise_category(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("".join(f"value_{index:05d} = 0\n" for index in range(5000)), encoding="utf-8")
    source_context = _source_context(source)
    assert estimate_replacement_plan_tokens(source_context["current_worktree_sources"]) > 8192
    backend = _ResultBackend([])
    candidate = _candidate()
    audit_root = tmp_path / "audits"
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=candidate)),
        audit_root,
        backend_factory=lambda selected: backend,
    )
    runner = LocalTeamRunner(tmp_path, tmp_path, candidate)
    with pytest.raises(StructuredSerializationCapacityError) as raised:
        runner._generate_implementation_call(
            caller,
            AgentTask(
                "fixture",
                "python",
                "implementation",
                {"allowed_paths": ["candidate.py"], "functional_requirements": []},
            ),
            _metadata(),
            Worktree(tmp_path, "a" * 40, "fixture"),
            {},
            source_context,
            {"observed_result": "truncated"},
            2,
            "structured_replacement_serialization",
        )
    assert raised.value.category == "structured_serialization_capacity_exceeded"
    assert backend.options == []
    audit_files = list(audit_root.glob("*.json"))
    assert len(audit_files) == 1
    audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
    assert audit["failure_category"] == "structured_serialization_capacity_exceeded"
    assert audit["status"] == "STRUCTURED_SERIALIZATION_CAPACITY_REJECTED"


def test_retry_budget_exhaustion_uses_last_precise_failure_category() -> None:
    model_calls = {
        "calls": [
            {"response_valid": False, "failure_category": "empty_content"},
            {"response_valid": False, "failure_category": "malformed_json"},
            {"response_valid": False, "failure_category": "truncated_response"},
        ]
    }
    assert _model_response_failure_category(model_calls) == "truncated_response"


def test_ordinary_non_reasoning_provider_payload_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []

    def urlopen(request: urllib.request.Request, **kwargs: object) -> _JsonResponse:
        payloads.append(json.loads(request.data or b"{}"))
        return _JsonResponse(
            {
                "choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}],
                "usage": _usage(),
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    result = OpenAICompatibleProvider("http://127.0.0.1:8101", "plain-model").generate("prompt")
    assert result.text == "plain"
    assert "response_format" not in payloads[0]
    assert "chat_template_kwargs" not in payloads[0]
