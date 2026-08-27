from dataclasses import fields

from research_workspace.operator_api import AgentAsyncRunRequest, _agent_async_request_sha256
from research_workspace.zetsu_agent import AgentRunContext


def test_mutation_authority_is_transient_not_checkpoint_identity() -> None:
    names = {item.name for item in fields(AgentRunContext)}
    assert "allow_mutation" in names


def test_operator_async_identity_binds_mutation_authority() -> None:
    base = {
        "lane": "quality",
        "instruction": "inspect",
        "domain": "software_engineering",
        "turn_id": "turn-v32-0001",
    }
    read = AgentAsyncRunRequest(**base, allow_mutation=False)
    write = AgentAsyncRunRequest(**base, allow_mutation=True)
    assert _agent_async_request_sha256(read) != _agent_async_request_sha256(write)
