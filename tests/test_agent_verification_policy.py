from __future__ import annotations

from research_workspace.operator_api import AgentSessionRequest
from research_workspace.zetsu_agent import ZetsuAgentCoordinator


def test_operator_agent_session_defaults_to_run_tests() -> None:
    request = AgentSessionRequest(repo_id="repo", session_id="session-a")

    assert request.allowed_tools == ["read_file", "apply_patch", "run_tests"]


def test_verification_policy_accepts_canonical_and_legacy_alias() -> None:
    assert ZetsuAgentCoordinator._verification_tool_allowed(("run_tests",)) is True
    assert ZetsuAgentCoordinator._verification_tool_allowed(("run_validation",)) is True


def test_verification_policy_fails_closed_without_verifier_capability() -> None:
    assert (
        ZetsuAgentCoordinator._verification_tool_allowed(("read_file", "apply_patch"))
        is False
    )
