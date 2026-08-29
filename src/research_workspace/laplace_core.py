"""Shared standalone Laplace services.

The Operator API and Zetsu are adapters over this composition root.  The core
does not import an MCP transport and can therefore be used by the standalone
application when Zetsu is disabled.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

from .model_routing import RoutingTaskMetadata, assess_rtl_worker_eligibility
from .memory import MemoryService
from .context_planner import ContextPlanner
from .hooks import HookReport, HookService, HookStage
from .idle_consolidation import ConsolidationReport, IdleConsolidator
from .logical_subagents import (
    GpuAwareSubagentScheduler,
    LogicalSubagentOutcome,
    LogicalSubagentTask,
    SubagentExecutor,
)
from .manager_control import (
    ManagerControl,
    ManagerAdmission,
    ManagerProvider,
    ManagerUnavailableError,
    TaskComplexity,
)
from .personal_corpus import PersonalCorpusStore
from .repository_agent_service import (
    RepositoryAgentConversationService,
    RepositoryAgentService,
)
from .repository_context import (
    RepoMap,
    RepositoryContextService,
    RepositoryEdge,
    RepositorySymbol,
)
from .rules import ContextItem, ContextPacket, RuleService
from .skills import SkillRecord, SkillRegistry
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .trajectory import (
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryIdentity,
    TrajectoryProvenance,
    TrajectoryReplay,
    TrajectoryService,
)
from .user_capabilities import Capability
from .verification_gates import VerificationGateRegistry
from .result_store import ResultStore
from .verification_policy import validate_verification_argv

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
        repository_agent_service: RepositoryAgentService | None = None,
        agent_coordinator: RepositoryAgentService | None = None,
        memory: MemoryService | None = None,
        rules: RuleService | None = None,
        repository_context: RepositoryContextService | None = None,
        trajectory: TrajectoryService | None = None,
        context_planner: ContextPlanner | None = None,
        skill_registry: SkillRegistry | None = None,
        hooks: HookService | None = None,
        consolidation: IdleConsolidator | None = None,
        logical_subagents: GpuAwareSubagentScheduler | None = None,
        manager_provider: ManagerProvider | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.corpus = corpus
        self.tiered = tiered
        self.memory = memory
        self.rules = rules
        self.trajectory = trajectory or TrajectoryService(
            self.repository_root / ".laplace-state" / "trajectory"
        )
        self.context_planner = context_planner or ContextPlanner()
        self._skill_registry = skill_registry
        self._hooks = hooks
        self._consolidation = consolidation
        self._logical_subagents = logical_subagents
        self._repository_context = repository_context
        if repository_agent_service is not None and agent_coordinator is not None:
            raise LaplaceCoreError("repository_agent_service_conflict")
        self._repository_agent_service = repository_agent_service or agent_coordinator
        self._manager_control = ManagerControl(manager_provider)
        self._repository_agent_lock = threading.Lock()
        self._repository_context_lock = threading.Lock()
        self._skill_registry_lock = threading.Lock()
        self._hooks_lock = threading.Lock()
        self._consolidation_lock = threading.Lock()
        self._logical_subagents_lock = threading.Lock()

    @property
    def skill_registry(self) -> SkillRegistry:
        """Return the shared local procedural-skill registry."""

        with self._skill_registry_lock:
            if self._skill_registry is None:
                self._skill_registry = SkillRegistry(
                    self.repository_root / ".laplace-state" / "skills"
                )
            return self._skill_registry

    @property
    def hooks(self) -> HookService:
        """Return the shared typed lifecycle hook service."""

        with self._hooks_lock:
            if self._hooks is None:
                self._hooks = HookService(self.repository_root / ".laplace-state" / "hooks")
            return self._hooks

    @property
    def consolidation(self) -> IdleConsolidator:
        """Return the shared shadow-only idle consolidation service."""

        with self._consolidation_lock:
            if self._consolidation is None:
                self._consolidation = IdleConsolidator(
                    self.repository_root / ".laplace-state" / "consolidation"
                )
            return self._consolidation

    @property
    def logical_subagents(self) -> GpuAwareSubagentScheduler:
        """Return the conservative, single-GPU logical-subagent scheduler."""

        with self._logical_subagents_lock:
            if self._logical_subagents is None:
                self._logical_subagents = GpuAwareSubagentScheduler(
                    result_store=ResultStore(
                        self.repository_root / ".laplace-state" / "logical-subagent-results"
                    )
                )
            return self._logical_subagents

    @property
    def repository_context(self) -> RepositoryContextService:
        """Return the standalone repository intelligence service."""

        with self._repository_context_lock:
            if self._repository_context is None:
                self._repository_context = RepositoryContextService(self.repository_root)
            return self._repository_context

    @property
    def repository_agent_service(self) -> RepositoryAgentService | None:
        """Return the adapter supplied for bounded repository-agent work."""

        with self._repository_agent_lock:
            return self._repository_agent_service

    def bind_repository_agent(self, service: RepositoryAgentService) -> None:
        """Bind one repository-agent adapter without importing its implementation."""

        if not all(
            callable(getattr(service, name, None))
            for name in (
                "run",
                "scheduler_status",
                "task_status",
                "cancel_queued",
                "handoff_evidence",
            )
        ):
            raise LaplaceCoreError("repository_agent_service_invalid")
        with self._repository_agent_lock:
            if self._repository_agent_service is not None and self._repository_agent_service is not service:
                raise LaplaceCoreError("repository_agent_service_already_bound")
            self._repository_agent_service = service

    def _require_repository_agent(self) -> RepositoryAgentService:
        service = self.repository_agent_service
        if service is None:
            raise LaplaceCoreError("repository_agent_unavailable")
        return service

    def _admit_repository_agent(
        self,
        *,
        repo_id: str,
        instruction: str,
        task_complexity: TaskComplexity | None,
        task_label: str | None,
    ) -> ManagerAdmission:
        """Use the sole advisory manager seam before either repository entry point."""

        try:
            return self._manager_control.admit(
                repo_id=repo_id,
                instruction=instruction,
                complexity=task_complexity,
                task_label=task_label,
            )
        except ManagerUnavailableError as exc:
            raise LaplaceCoreError("manager_provider_unavailable") from exc
        except ValueError as exc:
            raise LaplaceCoreError("manager_plan_invalid") from exc

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
        task_complexity: TaskComplexity | None = None,
        task_label: str | None = None,
        allow_mutation: bool = True,
    ) -> JsonObject:
        """Run the bounded repository agent, optionally through one manager plan."""

        worker = self._require_repository_agent()
        admission = self._admit_repository_agent(
            repo_id=repo_id,
            instruction=instruction,
            task_complexity=task_complexity,
            task_label=task_label,
        )
        result = worker.run(
            user_id=user_id,
            repo_id=repo_id,
            instruction=admission.instruction_for_worker(instruction),
            lane=lane,
            session_id=session_id,
            max_steps=max_steps,
            max_chars=max_chars,
            verification_argv=list(verification_argv) if verification_argv is not None else None,
            apply_to_repository=apply_to_repository,
            wait_timeout_seconds=wait_timeout_seconds,
            task_label=task_label,
            allow_mutation=allow_mutation,
        )
        return admission.annotate(result)

    def repository_agent_turn(
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
        task_complexity: TaskComplexity | None = None,
    ) -> JsonObject:
        """Run one bounded turn in an explicitly persistent repository session."""

        service = self._require_repository_agent()
        if not callable(getattr(service, "run_turn", None)):
            raise LaplaceCoreError("repository_agent_conversation_unavailable")
        conversation = cast(RepositoryAgentConversationService, service)
        admission = self._admit_repository_agent(
            repo_id=repo_id,
            instruction=instruction,
            task_complexity=task_complexity,
            task_label=task_label,
        )
        result = conversation.run_turn(
            user_id=user_id,
            repo_id=repo_id,
            instruction=admission.instruction_for_worker(instruction),
            lane=lane,
            session_id=session_id,
            max_steps=max_steps,
            max_chars=max_chars,
            verification_argv=verification_argv,
            wait_timeout_seconds=wait_timeout_seconds,
            task_label=task_label,
            allow_mutation=allow_mutation,
        )
        return admission.annotate(result)

    def repository_result_page(
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
        """Return one bounded durable result page through the neutral service."""

        return self._require_repository_agent().result_page(
            user_id=user_id,
            repo_id=repo_id,
            session_id=session_id,
            result_id=result_id,
            artifact=artifact,
            offset=offset,
            max_bytes=max_bytes,
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

        return self._require_repository_agent().scheduler_status(user_id=user_id)

    def select_skills(
        self,
        *,
        owner_id: str,
        project_id: str,
        query: str,
        available_tools: Sequence[str] = (),
        available_verifiers: Sequence[str] = (),
        enabled: bool = True,
    ) -> tuple[SkillRecord, ...]:
        """Select owner-scoped active procedures without granting authority."""

        return self.skill_registry.select(
            query,
            owner_id=owner_id,
            project_id=project_id,
            available_tools=available_tools,
            available_verifiers=available_verifiers,
            enabled=enabled,
        )

    def skill_activation_packet(
        self,
        *,
        owner_id: str,
        project_id: str,
        query: str,
        available_tools: Sequence[str] = (),
        available_verifiers: Sequence[str] = (),
        enabled: bool = True,
    ) -> JsonObject:
        """Build an advisory-only skill packet for a local caller."""

        return self.skill_registry.activation_packet(
            query,
            owner_id=owner_id,
            project_id=project_id,
            available_tools=available_tools,
            available_verifiers=available_verifiers,
            enabled=enabled,
        )

    def emit_hook(
        self,
        stage: HookStage,
        *,
        owner_id: str,
        project_id: str,
        session_id: str,
        task_id: str,
        idempotency_key: str,
        payload: Mapping[str, object] | None = None,
    ) -> HookReport:
        """Emit one owner-bound lifecycle event through the shared hook service."""

        return self.hooks.emit(
            stage,
            owner_id=owner_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            idempotency_key=idempotency_key,
            payload=payload,
        )

    def run_idle_consolidation(
        self,
        *,
        owner_id: str,
        project_id: str,
        session_id: str,
        cycle_id: str,
        trajectory_events: Sequence[object] = (),
        memories: Sequence[object] = (),
        window_id: str = "default",
        now_utc: str | None = None,
    ) -> ConsolidationReport:
        """Run one explicitly requested idle window in shadow mode.

        Lifecycle hooks are advisory control points; the consolidator itself
        remains deterministic and cannot mutate authoritative state.
        """

        self.emit_hook(
            HookStage.IDLE_START,
            owner_id=owner_id,
            project_id=project_id,
            session_id=session_id,
            task_id=cycle_id,
            idempotency_key=f"idle-start-{cycle_id}",
            payload={"cycle_id": cycle_id, "window_id": window_id, "mode": "SHADOW"},
        )
        try:
            report = self.consolidation.run_cycle(
                cycle_id,
                owner_id=owner_id,
                project_id=project_id,
                session_id=session_id,
                trajectory_events=trajectory_events,
                memories=memories,
                window_id=window_id,
                now_utc=now_utc,
            )
        except Exception:
            self.emit_hook(
                HookStage.IDLE_END,
                owner_id=owner_id,
                project_id=project_id,
                session_id=session_id,
                task_id=cycle_id,
                idempotency_key=f"idle-end-{cycle_id}",
                payload={"cycle_id": cycle_id, "window_id": window_id, "status": "FAILURE", "mode": "SHADOW"},
            )
            raise
        self.emit_hook(
            HookStage.IDLE_END,
            owner_id=owner_id,
            project_id=project_id,
            session_id=session_id,
            task_id=cycle_id,
            idempotency_key=f"idle-end-{cycle_id}",
            payload={
                "cycle_id": cycle_id,
                "window_id": window_id,
                "status": report.cycle.status,
                "proposal_count": len(report.proposals),
                "mode": "SHADOW",
            },
        )
        return report

    def run_logical_subagents(
        self,
        tasks: Sequence[LogicalSubagentTask],
        executor: SubagentExecutor,
    ) -> tuple[LogicalSubagentOutcome, ...]:
        """Run owner-scoped logical children through the GPU-aware queue."""

        return self.logical_subagents.run_batch(tasks, executor)

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

    def assemble_context(
        self,
        *,
        user_id: str,
        project_id: str,
        paths: Sequence[str] = (),
        memory: Sequence[ContextItem] = (),
        retrieval: Sequence[ContextItem] = (),
        objective: str = "",
    ) -> ContextPacket:
        """Assemble authoritative rules before advisory memory and retrieval."""

        if self.rules is None:
            raise LaplaceCoreError("rules_unavailable")
        return self.rules.assemble_context(
            user_id=user_id,
            project_id=project_id,
            paths=paths,
            memory=memory,
            retrieval=retrieval,
            objective=objective,
        )

    def repo_map(
        self,
        *,
        query: str = "",
        focus_paths: Sequence[str] = (),
        token_budget: int = 1_000,
    ) -> RepoMap:
        """Return advisory structural context; callers must read real files for changes."""

        return self.repository_context.build_repo_map(
            query=query,
            focus_paths=focus_paths,
            token_budget=token_budget,
        )

    def find_symbol(self, name: str) -> tuple[RepositorySymbol, ...]:
        """Find exact structural definitions without invoking a model."""

        return self.repository_context.find_symbol(name)

    def find_references(self, name: str) -> tuple[RepositoryEdge, ...]:
        """Find reliable structural references without replacing file reads."""

        return self.repository_context.find_references(name)

    def append_trajectory_event(
        self,
        identity: TrajectoryIdentity,
        *,
        event_type: TrajectoryEventType,
        idempotency_key: str,
        payload: JsonObject,
        state_before: JsonObject,
        state_after: JsonObject,
        provenance: TrajectoryProvenance,
    ) -> TrajectoryEvent:
        """Append an owner-bound authoritative event through standalone Core."""

        return self.trajectory.append(
            identity,
            event_type=event_type,
            idempotency_key=idempotency_key,
            payload=payload,
            state_before=state_before,
            state_after=state_after,
            provenance=provenance,
        )

    def replay_trajectory(self, identity: TrajectoryIdentity) -> TrajectoryReplay:
        """Reconstruct exact task state from events, using checkpoints only as acceleration."""

        return self.trajectory.replay(identity)

    def checkpoint_trajectory(self, identity: TrajectoryIdentity) -> Path:
        """Write a derived trajectory checkpoint after authoritative event append."""

        return self.trajectory.checkpoint(identity)

    @staticmethod
    def validate_verification_command(worktree: Path, argv: Sequence[str]) -> list[str]:
        """Apply the repository-agent verifier policy without executing a command."""

        try:
            return validate_verification_argv(worktree.resolve(), argv)
        except ServiceTierError as exc:
            raise LaplaceCoreError(exc.category, exc.evidence) from exc
