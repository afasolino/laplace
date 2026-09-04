from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_workspace.prime_agent_harness import repository_agent_backend
from research_workspace.user_capabilities import Capability
from research_workspace.zetsu_mcp import (
    ZetsuError,
    ZetsuService,
    _repository_id,
    tool_definitions,
)


def _agent_task_definition() -> dict[str, object]:
    return next(item for item in tool_definitions() if item.get("name") == "agent_task")


def test_agent_task_schema_exposes_prime_backend_and_logical_repo_id() -> None:
    definition = _agent_task_definition()
    schema = definition["inputSchema"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    repo_id = properties["repo_id"]
    backend = properties["agent_backend"]
    assert isinstance(repo_id, dict)
    assert repo_id["pattern"] == r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
    assert isinstance(backend, dict)
    assert backend["enum"] == ["native", "prime"]


def test_repository_id_rejects_filesystem_paths() -> None:
    assert _repository_id("laplace") == "laplace"
    with pytest.raises(ZetsuError, match="invalid_repo_id"):
        _repository_id("/home/example/repository")
    with pytest.raises(ZetsuError, match="invalid_repo_id"):
        _repository_id("relative/repository")


def test_backend_selection_is_explicit_and_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("LAPLACE_ZETSU_AGENT_BACKEND", raising=False)
    assert repository_agent_backend() == "native"
    assert repository_agent_backend("PRIME") == "prime"
    with pytest.raises(Exception, match="prime_agent_backend_invalid"):
        repository_agent_backend("other")


def test_rtl_task_is_not_advertised_when_codev_is_disabled() -> None:
    service = object.__new__(ZetsuService)
    service.tiered = SimpleNamespace(
        effective_capabilities=lambda _user_id: frozenset({Capability.AGENT}),
        lane_policy=SimpleNamespace(codev_enabled=False),
    )
    names = {str(item["name"]) for item in service.available_tools("owner")}
    assert "agent_task" in names
    assert "rtl_task" not in names

def test_rtl_task_visibility_is_backward_compatible_without_lane_policy() -> None:
    service = object.__new__(ZetsuService)
    service.tiered = SimpleNamespace(
        effective_capabilities=lambda _user_id: frozenset({Capability.AGENT}),
    )
    names = {str(item["name"]) for item in service.available_tools("owner")}
    assert "agent_task" in names
    assert "rtl_task" in names

