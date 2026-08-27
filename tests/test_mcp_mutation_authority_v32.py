from pathlib import Path

import pytest

from research_workspace.zetsu_mcp import ZetsuError, ZetsuService, tool_definitions


class Agent:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "SUCCESS", "content": "ok"}


def service(agent: Agent) -> ZetsuService:
    value = object.__new__(ZetsuService)
    value.repository_root = Path.cwd()
    value._agent_coordinator = agent
    value.available_tools = lambda _user_id: tuple(
        item for item in tool_definitions() if item["name"] == "agent_task"
    )
    value._agent_service = lambda: agent
    return value


def test_agent_task_schema_has_allow_mutation() -> None:
    tool = next(item for item in tool_definitions() if item["name"] == "agent_task")
    assert tool["inputSchema"]["properties"]["allow_mutation"] == {"type": "boolean"}


def test_agent_task_defaults_read_only() -> None:
    agent = Agent()
    service(agent).call(
        "user-a", "agent_task", {"repo_id": "repo", "instruction": "inspect"}
    )
    assert agent.calls[-1]["allow_mutation"] is False


def test_agent_task_write_requires_verifier() -> None:
    agent = Agent()
    with pytest.raises(ZetsuError, match="agent_mutation_requires_verifier"):
        service(agent).call(
            "user-a", "agent_task",
            {"repo_id": "repo", "instruction": "edit", "allow_mutation": True},
        )
    assert not agent.calls
