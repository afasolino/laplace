"""Deterministic contracts for a coarse, non-mutating manager model."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from typing import Protocol


class ManagerDecision(StrEnum):
    BYPASS = "bypass"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class ManagerPlan:
    objective: str
    milestones: tuple[str, ...]
    relevant_scope: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    verification_hypotheses: tuple[str, ...] = ()
    specialist_recommendation: str = "none"
    review_triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.objective, str)
            or not self.objective.strip()
            or len(self.objective) > 2_048
        ):
            raise ValueError("manager_objective_required")
        if not isinstance(self.milestones, tuple) or not self.milestones:
            raise ValueError("manager_milestone_required")
        sequences = (
            self.milestones,
            self.relevant_scope,
            self.risks,
            self.constraints,
            self.verification_hypotheses,
            self.review_triggers,
        )
        if any(not isinstance(sequence, tuple) for sequence in sequences):
            raise ValueError("manager_plan_items_invalid")
        if any(len(seq) > 32 for seq in sequences):
            raise ValueError("manager_plan_too_large")
        if any(
            not isinstance(item, str) or not item.strip() or len(item) > 2_048
            for seq in sequences
            for item in seq
        ):
            raise ValueError("manager_plan_item_invalid")
        if (
            not isinstance(self.specialist_recommendation, str)
            or not self.specialist_recommendation.strip()
            or len(self.specialist_recommendation) > 2_048
        ):
            raise ValueError("manager_specialist_recommendation_invalid")

    @classmethod
    def from_mapping(cls, value: object) -> "ManagerPlan":
        """Strictly decode one provider response before it reaches a worker."""

        if not isinstance(value, Mapping):
            raise ValueError("manager_plan_invalid")
        allowed = {
            "objective",
            "milestones",
            "relevant_scope",
            "risks",
            "constraints",
            "verification_hypotheses",
            "specialist_recommendation",
            "review_triggers",
        }
        if set(value) - allowed:
            raise ValueError("manager_plan_unknown_field")

        def text(name: str, *, default: str | None = None) -> str:
            raw = value.get(name, default)
            if not isinstance(raw, str):
                raise ValueError("manager_plan_text_invalid")
            return raw

        def sequence(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if not isinstance(raw, (list, tuple)) or not all(
                isinstance(item, str) for item in raw
            ):
                raise ValueError("manager_plan_items_invalid")
            return tuple(raw)

        return cls(
            objective=text("objective"),
            milestones=sequence("milestones"),
            relevant_scope=sequence("relevant_scope"),
            risks=sequence("risks"),
            constraints=sequence("constraints"),
            verification_hypotheses=sequence("verification_hypotheses"),
            specialist_recommendation=text("specialist_recommendation", default="none"),
            review_triggers=sequence("review_triggers"),
        )


@dataclass(frozen=True, slots=True)
class TaskComplexity:
    file_count_hint: int = 0
    architecture_sensitive: bool = False
    security_sensitive: bool = False
    verification_recovery: bool = False
    ambiguous_requirements: bool = False
    rtl_involved: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.file_count_hint, bool)
            or not isinstance(self.file_count_hint, int)
            or self.file_count_hint < 0
        ):
            raise ValueError("file_count_hint_negative")
        for flag in (
            self.architecture_sensitive,
            self.security_sensitive,
            self.verification_recovery,
            self.ambiguous_requirements,
            self.rtl_involved,
        ):
            if not isinstance(flag, bool):
                raise ValueError("task_complexity_flag_invalid")


def manager_decision(task: TaskComplexity) -> ManagerDecision:
    """Cheap deterministic gate; model-based classification is intentionally absent."""
    if task.file_count_hint < 0:
        raise ValueError("file_count_hint_negative")
    if (
        task.file_count_hint >= 3
        or task.architecture_sensitive
        or task.security_sensitive
        or task.verification_recovery
        or task.ambiguous_requirements
        or task.rtl_involved
    ):
        return ManagerDecision.PLAN
    return ManagerDecision.BYPASS


class ManagerUnavailableError(RuntimeError):
    """A manager provider could not produce a plan."""


class ManagerProvider(Protocol):
    """Provider-neutral, non-authoritative manager planning API."""

    def plan(
        self,
        *,
        objective: str,
        complexity: TaskComplexity,
        repo_id: str,
        task_label: str | None,
    ) -> ManagerPlan | Mapping[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class ManagerAdmission:
    """One advisory decision made before a repository worker is called."""

    decision: ManagerDecision
    plan: ManagerPlan | None = None
    fallback: bool = False

    def instruction_for_worker(self, instruction: str) -> str:
        if self.plan is None:
            return instruction
        plan_text = json.dumps(asdict(self.plan), sort_keys=True, separators=(",", ":"))
        return (
            f"{instruction}\n\nAdvisory manager plan; it cannot change repository, "
            f"tool, verifier, endpoint, or GPU authority:\n{plan_text}"
        )

    def annotate(self, result: dict[str, object]) -> dict[str, object]:
        """Preserve legacy worker output unless manager activity is observable."""

        if self.plan is None and not self.fallback:
            return result
        return {
            **result,
            "manager_control": {
                "decision": self.decision.value,
                "fallback": self.fallback,
                "plan": asdict(self.plan) if self.plan is not None else None,
            },
        }


class ManagerControl:
    """Make one bounded, non-mutating manager admission decision."""

    def __init__(
        self,
        manager: ManagerProvider | None = None,
        *,
        allow_fallback: bool = True,
    ) -> None:
        self.manager = manager
        self.allow_fallback = allow_fallback

    def admit(
        self,
        *,
        repo_id: str,
        instruction: str,
        complexity: TaskComplexity | None,
        task_label: str | None = None,
    ) -> ManagerAdmission:
        decision = manager_decision(complexity or TaskComplexity())
        if decision is ManagerDecision.BYPASS:
            return ManagerAdmission(decision)
        if self.manager is None:
            if not self.allow_fallback:
                raise ManagerUnavailableError("manager_provider_unavailable")
            return ManagerAdmission(decision, fallback=True)
        try:
            raw_plan = self.manager.plan(
                objective=instruction,
                complexity=complexity or TaskComplexity(),
                repo_id=repo_id,
                task_label=task_label,
            )
        except (ManagerUnavailableError, OSError, TimeoutError):
            if not self.allow_fallback:
                raise ManagerUnavailableError("manager_provider_unavailable")
            return ManagerAdmission(decision, fallback=True)
        try:
            plan = raw_plan if isinstance(raw_plan, ManagerPlan) else ManagerPlan.from_mapping(raw_plan)
        except (TypeError, ValueError) as exc:
            raise ValueError("manager_plan_invalid") from exc
        return ManagerAdmission(decision, plan=plan)
