from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest

from research_workspace.engineering import AgentTask
from research_workspace.inference import ServingCandidate
from research_workspace.llm import GenerationResult, OpenAICompatibleProvider, VllmProvider
from research_workspace.model_routing import (
    AuditedModelCaller,
    DualModelConfiguration,
    RoleRouter,
    RoutingTaskMetadata,
)
from research_workspace.multilanguage_ablation import _model_response_failure_category
from research_workspace.repair_protocol import (
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


def _candidate() -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint="http://127.0.0.1:8102",
        model="laplace-qwen3.6-35b-a3b-w4a16",
        revision="fixture",
        quantization="fixture",
        kernel="fixture",
        prefix_caching=True,
        chunked_prefill=True,
        cuda_graph_mode="fixture",
        scheduler="fixture",
        context_tokens=8192,
        max_output_tokens=4096,
    )


def _metadata() -> RoutingTaskMetadata:
    return RoutingTaskMetadata.single_model(task_id="fixture", experiment_arm="B", domain="python")


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
        "current_worktree_sources": [],
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
    )
    assert first.response_valid is False
    assert second.response_valid is False
    assert third.response_valid is True
    assert backend.options[0]["response_schema"] is None
    assert backend.options[0]["enable_thinking"] is None
    assert backend.options[1]["response_schema"] is None
    assert backend.options[1]["enable_thinking"] is None
    assert backend.options[2]["response_schema"] == replacement_plan_json_schema(
        allowed_paths=["candidate.py"], domain="python"
    )
    assert backend.options[2]["enable_thinking"] is False


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
