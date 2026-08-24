"""Deterministic shared context planning and safe compaction primitives.

The planner owns ordering, bounds and invariant fingerprints.  It never asks a
model to reconstruct policy, authorization, exact execution state or verifier
configuration.  A model-produced summary is an advisory narration field that
can be replaced or discarded without changing the authoritative plan.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias, cast

from .execution_records import canonical_sha256

JsonObject: TypeAlias = dict[str, object]
ContextItem: TypeAlias = Mapping[str, object] | str

DEFAULT_COMPACTION_RATIO = 0.80
MIN_COMPACTION_RATIO = 0.75
MAX_COMPACTION_RATIO = 0.85
_MAX_SECTION_CHARS = 64_000
_MAX_SUMMARY_CHARS = 12_000
_MAX_RECENT_ITEMS = 8
_MAX_RECENT_ITEM_CHARS = 12_000
_MAX_ITEMS = 128


class ContextPlannerError(RuntimeError):
    """A context plan is malformed or cannot preserve its invariants."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


def _text(value: object, *, label: str, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise ContextPlannerError(f"invalid_{label}")
    if not allow_empty and not value.strip():
        raise ContextPlannerError(f"invalid_{label}")
    return value


def _json_value(value: object, *, label: str, maximum: int) -> object:
    def visit(item: object) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ContextPlannerError(f"invalid_{label}")
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ContextPlannerError(f"invalid_{label}")
            for child in item.values():
                visit(child)
            return
        raise ContextPlannerError(f"invalid_{label}")

    visit(value)
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextPlannerError(f"invalid_{label}") from exc
    if len(encoded) > maximum:
        raise ContextPlannerError(f"{label}_too_large")
    return value


def _object(value: object, *, label: str, maximum: int = _MAX_SECTION_CHARS) -> JsonObject:
    checked = _json_value(value, label=label, maximum=maximum)
    if not isinstance(checked, dict):
        raise ContextPlannerError(f"invalid_{label}")
    return cast(JsonObject, checked)


def _items(value: Sequence[ContextItem], *, label: str) -> tuple[str, ...]:
    if len(value) > _MAX_ITEMS:
        raise ContextPlannerError(f"{label}_too_many")
    rendered: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = _text(item, label=label, maximum=_MAX_SECTION_CHARS)
        else:
            text = json.dumps(
                _json_value(dict(item), label=label, maximum=_MAX_SECTION_CHARS),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        rendered.append(text[:_MAX_SECTION_CHARS])
    return tuple(rendered)


def _rough_tokens(value: str) -> int:
    return max(1, (len(value.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True)
class ContextPlan:
    """A deterministic, inspectable LLM-facing context packet."""

    messages: tuple[dict[str, str], dict[str, str]]
    owner_user_id: str
    session_id: str
    objective: str
    exact_state: JsonObject
    policy: JsonObject
    required_verification_argv: tuple[str, ...] | None
    system_prompt: str
    project_rules: tuple[str, ...]
    relevant_memory: tuple[str, ...]
    repository_map: str
    retrieval_evidence: tuple[str, ...]
    exact_state_sha256: str
    policy_sha256: str
    authorization_sha256: str
    verification_sha256: str
    approximate_tokens: int
    compaction_ratio: float
    semantic_summary: str
    recent_trajectory: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "messages": [dict(self.messages[0]), dict(self.messages[1])],
            "owner_user_id": self.owner_user_id,
            "session_id": self.session_id,
            "objective": self.objective,
            "exact_state": self.exact_state,
            "policy": self.policy,
            "required_verification_argv": (
                list(self.required_verification_argv)
                if self.required_verification_argv is not None
                else None
            ),
            "system_prompt": self.system_prompt,
            "project_rules": list(self.project_rules),
            "relevant_memory": list(self.relevant_memory),
            "repository_map": self.repository_map,
            "retrieval_evidence": list(self.retrieval_evidence),
            "exact_state_sha256": self.exact_state_sha256,
            "policy_sha256": self.policy_sha256,
            "authorization_sha256": self.authorization_sha256,
            "verification_sha256": self.verification_sha256,
            "approximate_tokens": self.approximate_tokens,
            "compaction_ratio": self.compaction_ratio,
            "semantic_summary": self.semantic_summary,
            "recent_trajectory": list(self.recent_trajectory),
        }


class ContextPlanner:
    """Shared deterministic planner used by standalone Core and Zetsu."""

    @staticmethod
    def _ratio(value: float) -> float:
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise ContextPlannerError("invalid_compaction_ratio")
        ratio = float(value)
        if not MIN_COMPACTION_RATIO <= ratio <= MAX_COMPACTION_RATIO:
            raise ContextPlannerError("invalid_compaction_ratio")
        return ratio

    @staticmethod
    def _policy(policy: Mapping[str, object]) -> JsonObject:
        return _object(dict(policy), label="policy", maximum=_MAX_SECTION_CHARS)

    @staticmethod
    def _verification(argv: Sequence[str] | None) -> list[str] | None:
        if argv is None:
            return None
        if not 1 <= len(argv) <= 64 or any(
            not isinstance(item, str) or not item or len(item) > 1_000 or "\x00" in item
            for item in argv
        ):
            raise ContextPlannerError("invalid_verification_argv")
        return list(argv)

    @staticmethod
    def _section(label: str, value: object) -> str:
        return f"{label}:\n{value}"

    def plan(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        objective: str,
        exact_state: JsonObject,
        policy: Mapping[str, object],
        required_verification_argv: Sequence[str] | None,
        system_prompt: str,
        project_rules: Sequence[ContextItem] = (),
        relevant_memory: Sequence[ContextItem] = (),
        repository_map: ContextItem | None = None,
        retrieval_evidence: Sequence[ContextItem] = (),
        recent_trajectory: Sequence[str] = (),
        semantic_summary: str = "",
        compaction_ratio: float = DEFAULT_COMPACTION_RATIO,
    ) -> ContextPlan:
        owner = _text(owner_user_id, label="owner_user_id", maximum=256)
        session = _text(session_id, label="session_id", maximum=256)
        objective_value = _text(objective, label="objective", maximum=_MAX_SECTION_CHARS)
        system = _text(system_prompt, label="system_prompt", maximum=_MAX_SECTION_CHARS)
        exact = _object(exact_state, label="exact_state")
        policy_value = self._policy(policy)
        verification = self._verification(required_verification_argv)
        ratio = self._ratio(compaction_ratio)
        rules = _items(project_rules, label="project_rules")
        memory = _items(relevant_memory, label="relevant_memory")
        retrieval = _items(retrieval_evidence, label="retrieval_evidence")
        trajectory = tuple(
            _text(item, label="recent_trajectory", maximum=_MAX_RECENT_ITEM_CHARS)
            for item in recent_trajectory[-_MAX_RECENT_ITEMS:]
        )
        summary = _text(
            semantic_summary,
            label="semantic_summary",
            maximum=_MAX_SUMMARY_CHARS,
            allow_empty=True,
        )
        repo_text = "NONE"
        if repository_map is not None:
            if isinstance(repository_map, str):
                repo_text = _text(repository_map, label="repository_map", maximum=_MAX_SECTION_CHARS)
            else:
                repo_text = json.dumps(
                    _object(dict(repository_map), label="repository_map"),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        exact_text = json.dumps(exact, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        policy_text = json.dumps(policy_value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        verifier_text = json.dumps(verification, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        authorization_sha = canonical_sha256(
            {
                "owner_user_id": owner,
                "session_id": session,
                "policy": policy_value,
                "required_verification_argv": verification,
            }
        )
        exact_sha = canonical_sha256(exact)
        policy_sha = canonical_sha256(policy_value)
        verification_sha = canonical_sha256(verification)
        recent_text = "\n\n".join(trajectory) if trajectory else "NONE"
        user_message = "\n\n".join(
            (
                self._section("OBJECTIVE", objective_value),
                self._section("AUTHORIZATION_FINGERPRINT", authorization_sha),
                self._section("POLICY (AUTHORITATIVE)", policy_text),
                self._section("REQUIRED_VERIFICATION (AUTHORITATIVE)", verifier_text),
                self._section("EXACT TASK STATE (AUTHORITATIVE; NEVER SUMMARIZE)", exact_text),
                self._section("PROJECT RULES (AUTHORITATIVE)", "\n".join(rules) or "NONE"),
                self._section("RELEVANT MEMORY (ADVISORY)", "\n".join(memory) or "NONE"),
                self._section("REPOSITORY MAP (ADVISORY)", repo_text),
                self._section("RETRIEVAL EVIDENCE (ADVISORY)", "\n".join(retrieval) or "NONE"),
                self._section("SEMANTIC SUMMARY (ADVISORY; MAY BE WRONG)", summary or "NONE"),
                self._section("RECENT TRAJECTORY (ADVISORY)", recent_text),
                "Choose the next action using only the allowed local interface.",
            )
        )
        messages = (
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        )
        return ContextPlan(
            messages=messages,
            owner_user_id=owner,
            session_id=session,
            objective=objective_value,
            exact_state=exact,
            policy=policy_value,
            required_verification_argv=tuple(verification) if verification is not None else None,
            system_prompt=system,
            project_rules=rules,
            relevant_memory=memory,
            repository_map=repo_text,
            retrieval_evidence=retrieval,
            exact_state_sha256=exact_sha,
            policy_sha256=policy_sha,
            authorization_sha256=authorization_sha,
            verification_sha256=verification_sha,
            approximate_tokens=_rough_tokens(json.dumps(messages, sort_keys=True, ensure_ascii=False)),
            compaction_ratio=ratio,
            semantic_summary=summary,
            recent_trajectory=trajectory,
        )

    @staticmethod
    def should_compact(*, approximate_tokens: int, context_limit: int, ratio: float = DEFAULT_COMPACTION_RATIO) -> bool:
        if isinstance(approximate_tokens, bool) or approximate_tokens < 0:
            raise ContextPlannerError("invalid_approximate_tokens")
        if isinstance(context_limit, bool) or context_limit < 2_048:
            raise ContextPlannerError("invalid_context_limit")
        if not MIN_COMPACTION_RATIO <= ratio <= MAX_COMPACTION_RATIO:
            raise ContextPlannerError("invalid_compaction_ratio")
        return approximate_tokens >= int(context_limit * ratio)

    def compaction_messages(
        self,
        *,
        objective: str,
        exact_state: JsonObject,
        prior_summary: str,
        recent_trajectory: Sequence[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build the model request for semantic narration only."""

        exact = json.dumps(_object(exact_state, label="exact_state"), sort_keys=True, ensure_ascii=False)
        summary = _text(prior_summary, label="semantic_summary", maximum=_MAX_SUMMARY_CHARS, allow_empty=True)
        recent = "\n\n".join(
            _text(item, label="recent_trajectory", maximum=_MAX_RECENT_ITEM_CHARS)
            for item in recent_trajectory[-_MAX_RECENT_ITEMS:]
        ) or "NONE"
        return (
            {
                "role": "system",
                "content": (
                    "Return exactly one JSON object with string field summary. Summarize semantic "
                    "history only. Do not rewrite the exact state, policy, authorization or verifier."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    (
                        f"OBJECTIVE:\n{_text(objective, label='objective', maximum=_MAX_SECTION_CHARS)}",
                        f"EXACT STATE (AUTHORITATIVE):\n{exact}",
                        f"PRIOR SUMMARY (ADVISORY):\n{summary or 'NONE'}",
                        f"RECENT HISTORY (ADVISORY):\n{recent}",
                    )
                ),
            },
        )

    def compact(
        self,
        *,
        plan: ContextPlan,
        semantic_summary: str,
        recent_trajectory: Sequence[str],
    ) -> ContextPlan:
        """Rebuild a plan after semantic condensation without changing exact inputs."""

        exact = plan.exact_state
        if canonical_sha256(exact) != plan.exact_state_sha256:
            raise ContextPlannerError("context_exact_state_changed_during_compaction")
        if canonical_sha256(plan.policy) != plan.policy_sha256:
            raise ContextPlannerError("context_policy_changed_during_compaction")
        required = plan.required_verification_argv
        if canonical_sha256(required) != plan.verification_sha256:
            raise ContextPlannerError("context_verification_changed_during_compaction")
        return self.plan(
            owner_user_id=plan.owner_user_id,
            session_id=plan.session_id,
            objective=plan.objective,
            exact_state=exact,
            policy=plan.policy,
            required_verification_argv=required,
            system_prompt=plan.system_prompt,
            project_rules=plan.project_rules,
            relevant_memory=plan.relevant_memory,
            repository_map=plan.repository_map,
            retrieval_evidence=plan.retrieval_evidence,
            recent_trajectory=recent_trajectory,
            semantic_summary=semantic_summary,
            compaction_ratio=plan.compaction_ratio,
        )

    @staticmethod
    def assert_invariants(before: ContextPlan, after: ContextPlan) -> None:
        if (
            before.objective != after.objective
            or before.exact_state_sha256 != after.exact_state_sha256
            or before.policy_sha256 != after.policy_sha256
            or before.authorization_sha256 != after.authorization_sha256
            or before.verification_sha256 != after.verification_sha256
        ):
            raise ContextPlannerError("context_authoritative_invariant_changed")

    @staticmethod
    def structured_handoff(plan: ContextPlan) -> JsonObject:
        """Return exact fields separately from compactable narration."""

        return {
            "objective": plan.objective,
            "exact_state": plan.exact_state,
            "exact_state_sha256": plan.exact_state_sha256,
            "policy_sha256": plan.policy_sha256,
            "authorization_sha256": plan.authorization_sha256,
            "verification_sha256": plan.verification_sha256,
            "semantic_summary": plan.semantic_summary,
            "recent_trajectory": list(plan.recent_trajectory),
        }
