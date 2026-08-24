"""Shared standalone Laplace services.

The Operator API and Zetsu are adapters over this composition root.  The core
does not import an MCP transport and can therefore be used by the standalone
application when Zetsu is disabled.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from .model_routing import RoutingTaskMetadata, assess_rtl_worker_eligibility
from .memory import MemoryService
from .personal_corpus import PersonalCorpusStore
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .user_capabilities import Capability
from .verification_gates import VerificationGateRegistry
from .zetsu_agent import ZetsuAgentCoordinator

JsonObject: TypeAlias = dict[str, object]
VerificationDomain = Literal["python", "c", "verilog", "systemverilog"]
VerificationScope = Literal["public", "adversarial", "final"]


class LaplaceCoreError(RuntimeError):
    """A shared-core operation failed closed."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class LaplaceCore:
    """Composition root for capabilities shared by standalone Laplace and Zetsu.

    Authorization and model-lane policy remain in their existing services.  This
    class deliberately delegates instead of reimplementing those policies so an
    adapter cannot silently acquire a second routing or isolation path.
    """

    def __init__(
        self,
        repository_root: Path,
        corpus: PersonalCorpusStore,
        tiered: TieredServingService,
        *,
        agent_coordinator: ZetsuAgentCoordinator | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.corpus = corpus
        self.tiered = tiered
        self.memory = memory
        self._agent_coordinator = agent_coordinator
        self._agent_coordinator_lock = threading.Lock()

    @property
    def agent_coordinator(self) -> ZetsuAgentCoordinator:
        """Return the one repository-agent coordinator shared by all adapters."""

        with self._agent_coordinator_lock:
            if self._agent_coordinator is None:
                self._agent_coordinator = ZetsuAgentCoordinator(self.tiered, self.corpus)
            return self._agent_coordinator

    def _require(self, user_id: str, capability: Capability) -> None:
        capabilities = self.tiered.effective_capabilities(user_id)
        if capability not in capabilities:
            raise LaplaceCoreError(
                "capability_required",
                {"user_id": user_id, "capability": capability.value},
            )

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        corpus_id: str | None = None,
        limit: int = 8,
    ) -> JsonObject:
        """Retrieve owner-authorized personal evidence through the shared store."""

        self._require(user_id, Capability.PERSONAL_CORPUS)
        return self.corpus.search(user_id, query, corpus_id=corpus_id, limit=limit)

    def evidence(self, user_id: str, chunk_ids: Sequence[str]) -> JsonObject:
        """Expand owner-authorized evidence through the shared store."""

        self._require(user_id, Capability.PERSONAL_CORPUS)
        return self.corpus.evidence(user_id, tuple(chunk_ids))

    def chat(
        self,
        *,
        user_id: str,
        lane: ModelLane,
        messages: Sequence[Mapping[str, str]],
        domain: str = "general",
        session_id: str | None = None,
    ) -> JsonObject:
        """Route a local chat request through the canonical tiered service."""

        return self.tiered.chat(
            user_id=user_id,
            lane=lane,
            messages=messages,
            domain=domain,
            session_id=session_id,
        )

    def agent_session(
        self,
        *,
        user_id: str,
        session_id: str,
        lane: ModelLane,
        instruction: str,
        domain: str,
    ) -> JsonObject:
        """Run a standalone bounded agent session through the canonical service."""

        return self.tiered.agent(
            user_id=user_id,
            session_id=session_id,
            lane=lane,
            instruction=instruction,
            domain=domain,
        )

    def repository_agent(
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
        wait_timeout_seconds: int,
    ) -> JsonObject:
        """Run the bounded Qwen repository agent without an MCP dependency."""

        return self.agent_coordinator.run(
            user_id=user_id,
            repo_id=repo_id,
            instruction=instruction,
            lane=lane,
            session_id=session_id,
            max_steps=max_steps,
            max_chars=max_chars,
            verification_argv=list(verification_argv) if verification_argv is not None else None,
            apply_to_repository=apply_to_repository,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    def rtl_task(
        self,
        *,
        user_id: str,
        session_id: str,
        instruction: str,
        task_kind: Literal["implementation", "repair"],
        editable_sources: Sequence[str],
        module_count: int,
    ) -> JsonObject:
        """Run only an eligible bounded RTL task on the economy specialist route."""

        metadata = RoutingTaskMetadata(
            task_id=session_id,
            experiment_arm="laplace-core",
            domain="systemverilog",
            task_kind=task_kind,
            rtl_scope="bounded_module",
            worker_eligible=True,
            editable_sources=tuple(editable_sources),
            module_count=module_count,
            synthesizable=True,
            explicit_ports=True,
            cycle_behavior_specified=True,
            deterministic_verification=True,
            unresolved_architecture=False,
        )
        eligibility = assess_rtl_worker_eligibility(metadata)
        if not eligibility.eligible:
            raise LaplaceCoreError(
                "rtl_task_not_eligible",
                {"reason": eligibility.reason},
            )
        try:
            return self.tiered.agent(
                user_id=user_id,
                session_id=session_id,
                lane=ModelLane.ECONOMY,
                instruction=instruction,
                domain="systemverilog",
            )
        except ServiceTierError:
            raise

    def task_status(self, *, user_id: str, session_id: str) -> JsonObject:
        """Read owner-scoped task state from the canonical sandbox service."""

        return self.tiered.agent_session_status(user_id=user_id, session_id=session_id)

    def cancel_task(self, *, user_id: str, session_id: str) -> JsonObject:
        """Cancel owner-scoped standalone task state through the canonical service."""

        return self.tiered.cancel_agent_session(user_id=user_id, session_id=session_id)

    def scheduler_status(self, *, user_id: str) -> JsonObject:
        """Return the bounded repository-agent scheduler state."""

        return self.agent_coordinator.scheduler_status(user_id=user_id)

    @staticmethod
    def deterministic_verification(
        domain: VerificationDomain,
        gate_results: Mapping[str, JsonObject],
        *,
        scope: VerificationScope = "final",
        available_tools: Mapping[str, bool] | None = None,
    ) -> JsonObject:
        """Project deterministic gate results using the single registry."""

        return VerificationGateRegistry.evaluate(
            domain,
            gate_results,
            scope=scope,
            available_tools=available_tools,
        ).to_json()

    @staticmethod
    def validate_verification_command(worktree: Path, argv: Sequence[str]) -> list[str]:
        """Apply the repository-agent verifier policy without executing a command."""

        try:
            return ZetsuAgentCoordinator._verify_argv(worktree.resolve(), argv)
        except ServiceTierError as exc:
            raise LaplaceCoreError(exc.category, exc.evidence) from exc
