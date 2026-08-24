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
from research_workspace.zetsu_results import ZetsuResultError, ZetsuResultStore


class _Agent:
    def __init__(
        self, *, error: str | None = None, result: dict[str, object] | None = None
    ) -> None:
        self.error = error
        self.result = result
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise ServiceTierError(self.error)
        return self.result or {
            "status": "SUCCESS",
            "content": "ok",
            "telemetry": {"qwen_calls": 1},
        }


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
    assert schema["properties"]["apply_to_repository"] == {"type": "boolean"}
    verify = next(item for item in tool_definitions() if item["name"] == "verify")
    assert verify["inputSchema"]["properties"]["include_patch"] == {"type": "boolean"}


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
    for name in ("rtl_task", "verify", "get_result"):
        session = definitions[name]["inputSchema"]["properties"]["session_id"]
        assert session["pattern"] == "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
    rtl = definitions["rtl_task"]["inputSchema"]
    assert rtl["properties"]["module_count"]["maximum"] == 1
    assert rtl["properties"]["editable_sources"]["maxItems"] == 1


def test_get_result_pages_exact_content_with_owner_repo_session_isolation(
    tmp_path: Path,
) -> None:
    store = ZetsuResultStore(tmp_path / "results")
    delivery = store.persist(
        user_id="user-a",
        repo_id="repo",
        session_id="session-a",
        status="SUCCESS",
        summary="large",
        artifacts={"result.json": b"0123456789" * 200},
    )
    agent = _Agent()
    agent.results = store  # type: ignore[attr-defined]
    service = _service(agent)
    service.available_tools = lambda _user_id: tuple(  # type: ignore[method-assign]
        item for item in tool_definitions() if item["name"] == "get_result"
    )
    page = service.call(
        "user-a",
        "get_result",
        {
            "result_id": delivery["result_id"],
            "repo_id": "repo",
            "session_id": "session-a",
            "artifact": "result.json",
            "offset": 17,
            "max_bytes": 513,
        },
    )
    assert page["status"] == "SUCCESS"
    assert page["offset"] == 17
    with pytest.raises(ZetsuResultError, match="zetsu_result_not_found"):
        service.call(
            "user-b",
            "get_result",
            {
                "result_id": delivery["result_id"],
                "repo_id": "repo",
                "session_id": "session-a",
                "artifact": "result.json",
            },
        )


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


def test_agent_task_returns_compact_authoritative_handoff() -> None:
    agent = _Agent(
        result={
            "status": "SUCCESS",
            "session_id": "session-a",
            "repo_id": "repo",
            "model_id": "qwen",
            "effective_lane": "quality",
            "content": "verified change",
            "changed_paths": ["src/a.py"],
            "verification": {
                "argv": ["pytest", "tests/test_a.py", "-q"],
                "returncode": 0,
                "passed": True,
                "stdout_tail": "one passed",
            },
            "validation_history": [{"passed": False}, {"passed": True}],
            "unresolved_failures": [],
            "evidence_refs": [],
            "checkpoint_path": "/evidence/checkpoint.json",
            "handoff": {
                "patch": "large exact patch",
                "patch_inline": True,
                "patch_chars": 17,
                "patch_sha256": "a" * 64,
                "patch_path": "/evidence/handoff.patch",
            },
            "promotion": {"requested": True, "applied": True},
            "elapsed_seconds": 12.5,
            "telemetry": {"qwen_calls": 4, "qwen_input_tokens": 2000},
        }
    )
    result = _service(agent).call(
        "user-a",
        "agent_task",
        {
            "repo_id": "repo",
            "instruction": "implement bounded change",
            "verification_argv": ["pytest", "tests/test_a.py", "-q"],
            "apply_to_repository": True,
            "telemetry": True,
        },
    )

    assert agent.calls[-1]["apply_to_repository"] is True
    assert result["promotion"] == {"requested": True, "applied": True}
    assert result["handoff"] == {
        "patch_chars": 17,
        "patch_sha256": "a" * 64,
        "patch_path": "/evidence/handoff.patch",
    }
    assert result["verification"] == {
        "argv": ["pytest", "tests/test_a.py", "-q"],
        "command_id": None,
        "returncode": 0,
        "passed": True,
        "aborted_category": None,
        "qualifies_for_mutation": None,
        "mutation_epoch": None,
        "worktree_mutated": None,
    }
    assert "validation_history" not in result
    assert "patch" not in result["handoff"]
