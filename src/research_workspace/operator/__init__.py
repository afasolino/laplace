"""Operator HTTP transport components.

The legacy :mod:`research_workspace.operator_api` module remains the public
compatibility facade while responsibilities are extracted here.
"""

from .auth import AuthCredential, AuthPrincipal, OperatorAuth
from .agent_requests import AgentAsyncRunRequest, AgentRunRequest, AgentTaskComplexityRequest
from .settings import OperatorApiSettings

__all__ = [
    "AgentAsyncRunRequest",
    "AgentRunRequest",
    "AgentTaskComplexityRequest",
    "AuthCredential",
    "AuthPrincipal",
    "OperatorApiSettings",
    "OperatorAuth",
]
