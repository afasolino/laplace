from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.service_tiers import ServiceTierError
from research_workspace.zetsu_mcp import (
    ZetsuError,
    ZetsuMcpDispatcher,
    ZetsuService,
    tool_definitions,
)


class _Agent:
    def __init__(self, *, error: str | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise ServiceTierError(self.error)
        return {"status": "SUCCESS", "content": "ok", "telemetry": {"qwen_calls": 1}}


def _service(agent: _Agent) -> ZetsuService:
    service = object.__new__(ZetsuService)
    service.repository_root = Path(".").resolve()
    service.corpus = object()
    service.tiered = object()
    service._agent_coordinator = agent
    service.available_tools = lambda _user_id: tuple(  # type: ignore[method-assign]
        item for item in tool_definitions() if item["name"] == "agent_task"
    )
    service._agent_service = lambda: agent  # type: ignore[method-assign]
    return service


def test_agent_task_schema_is_bounded_and_closed() -> None:
    definition = next(item for item in tool_definitions() if item["name"] == "agent_task")
    schema = definition["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"repo_id", "instruction"}
    assert schema["properties"]["lane"]["enum"] == ["quality", "standard"]
    assert schema["properties"]["max_steps"]["maximum"] == 32
    session = schema["properties"]["session_id"]
    assert session["minLength"] == 1
    assert session["pattern"] == "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
    verifier = schema["properties"]["verification_argv"]
    assert verifier["maxItems"] == 64
    assert verifier["items"]["maxLength"] == 1_000


def test_agent_task_rejects_unexpected_arguments() -> None:
    service = _service(_Agent())
    with pytest.raises(ZetsuError, match="unexpected_tool_arguments"):
        service.call(
            "user-a",
            "agent_task",
            {"repo_id": "repo", "instruction": "x", "shell": "rm -rf /"},
        )


def test_agent_task_rejects_empty_or_malformed_session_id() -> None:
    service = _service(_Agent())
    for value in ("", "bad/session", " space"):
        with pytest.raises(ZetsuError, match="invalid_session_id"):
            service.call(
                "user-a",
                "agent_task",
                {"repo_id": "repo", "instruction": "x", "session_id": value},
            )


def test_agent_task_service_error_is_mcp_tool_error() -> None:
    dispatcher = ZetsuMcpDispatcher(_service(_Agent(error="agent_session_cancelled")))
    result = dispatcher.dispatch(
        "user-a",
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "agent_task",
                "arguments": {"repo_id": "repo", "instruction": "x"},
            },
        },
    )
    assert result.status_code == 200
    assert result.payload is not None
    tool_result = result.payload["result"]
    assert tool_result["isError"] is True
    assert tool_result["structuredContent"] == {"error": "agent_session_cancelled"}


def test_malformed_tools_call_returns_structured_tool_error() -> None:
    dispatcher = ZetsuMcpDispatcher(_service(_Agent()))
    result = dispatcher.dispatch(
        "user-a",
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": "invalid"},
    )
    assert result.status_code == 200
    assert result.payload is not None
    tool_result = result.payload["result"]
    assert tool_result["isError"] is True
    assert tool_result["structuredContent"]["error"] == "invalid_tool_call"


def test_rtl_and_verify_session_schemas_match_runtime_identifier_rules() -> None:
    definitions = {item["name"]: item for item in tool_definitions()}
    for name in ("rtl_task", "verify"):
        session = definitions[name]["inputSchema"]["properties"]["session_id"]
        assert session["pattern"] == "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
    rtl = definitions["rtl_task"]["inputSchema"]
    assert rtl["properties"]["module_count"]["maximum"] == 1
    assert rtl["properties"]["editable_sources"]["maxItems"] == 1


def test_get_evidence_rejects_unhashable_chunk_values_without_typeerror() -> None:
    service = object.__new__(ZetsuService)
    service.repository_root = Path(".").resolve()
    service.corpus = object()
    service.tiered = object()
    service.available_tools = lambda _user_id: tuple(  # type: ignore[method-assign]
        item for item in tool_definitions() if item["name"] == "get_evidence"
    )
    with pytest.raises(ZetsuError, match="invalid_chunk_ids"):
        service.call("user-a", "get_evidence", {"chunk_ids": [{"bad": "value"}]})


def test_agent_task_validates_telemetry_before_execution() -> None:
    agent = _Agent()
    service = _service(agent)
    with pytest.raises(ZetsuError, match="invalid_telemetry"):
        service.call(
            "user-a",
            "agent_task",
            {"repo_id": "repo", "instruction": "x", "telemetry": "yes"},
        )
    assert agent.calls == []

def test_agent_task_validates_verification_argv_before_execution() -> None:
    agent = _Agent()
    service = _service(agent)
    with pytest.raises(ZetsuError, match="invalid_verification_argv"):
        service.call(
            "user-a",
            "agent_task",
            {"repo_id": "repo", "instruction": "x", "verification_argv": []},
        )
    assert agent.calls == []

    result = service.call(
        "user-a",
        "agent_task",
        {
            "repo_id": "repo",
            "instruction": "x",
            "verification_argv": ["pytest", "tests/test_x.py", "-q"],
        },
    )
    assert result["status"] == "SUCCESS"
    assert agent.calls[-1]["verification_argv"] == ["pytest", "tests/test_x.py", "-q"]
