"""Neutral repository-agent service boundary shared by standalone and Zetsu adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from .service_tiers import ModelLane

JsonObject: TypeAlias = dict[str, object]


class RepositoryAgentService(Protocol):
    """Owner-scoped repository-agent operations exposed to Laplace Core."""

    def run(
        self,
        *,
        user_id: str,
        repo_id: str,
        instruction: str,
        lane: ModelLane,
        session_id: str | None,
        max_steps: int,
        max_chars: int,
        verification_argv: Sequence[str] | None,
        apply_to_repository: bool,
        wait_timeout_seconds: float,
        task_label: str | None = None,
        allow_mutation: bool = True,
    ) -> JsonObject:
        """Run one bounded repository task through the selected adapter."""
        ...

    def scheduler_status(self, *, user_id: str) -> JsonObject:
        """Return owner-scoped admission state."""
        ...

    def task_status(self, *, user_id: str, session_id: str) -> JsonObject:
        """Return owner-scoped repository-task state."""
        ...

    def cancel_queued(self, *, user_id: str, session_id: str) -> JsonObject:
        """Cancel an owner-scoped queued repository task."""
        ...

    def handoff_evidence(self, session_id: str, *, max_chars: int) -> JsonObject:
        """Return bounded exact handoff evidence for an authorized adapter caller."""
        ...

    def result_page(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        result_id: str,
        artifact: str,
        offset: int,
        max_bytes: int,
    ) -> JsonObject:
        """Read one owner/repository-bound durable result page."""
        ...


class RepositoryAgentConversationService(RepositoryAgentService, Protocol):
    """Optional neutral extension for persistent multi-turn repository sessions."""

    def run_turn(
        self,
        *,
        user_id: str,
        repo_id: str,
        instruction: str,
        lane: ModelLane,
        session_id: str,
        max_steps: int,
        max_chars: int,
        verification_argv: Sequence[str] | None,
        wait_timeout_seconds: float,
        task_label: str | None = None,
        allow_mutation: bool = False,
    ) -> JsonObject:
        """Run one bounded turn without terminalizing the owned worktree."""
        ...
