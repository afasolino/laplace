"""SWE-agent-inspired ACI primitives for Laplace Step 3.4 Phase A.

The shadow delegates authorization, containment, mutation, diff validation,
verification, cancellation and repository identity to BoundedRepositoryACI.
Only the measured observation/edit ergonomics differ.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Literal, TypeAlias, cast

from .bounded_aci import BoundedACIError, BoundedRepositoryACI
from .laplace_core import LaplaceCore
from .personal_corpus import PersonalCorpusStore
from .service_tiers import TieredServingService
from .zetsu_agent import ZetsuAgentCoordinator
from .zetsu_state import AgentExecutionState, AgentRunContext
from .zetsu_mcp import ZetsuService

JsonObject: TypeAlias = dict[str, object]
Primitive = Literal["view", "search", "syntax_preflight", "governance"]

LAPLACE_V34_BASELINE = "5d8e463511d84ab1a7ee87b0f717b665f91d3e0f"
SWE_AGENT_REFERENCE_COMMIT = "3ea751c087f32b16e039a2233dd6eefecef325d5"
SWE_AGENT_VIEW_LINES = 100
PHASE_A_REPETITIONS = 5
_PREFLIGHT_TEXT_LIMIT = 256_000
_PRIMITIVES: tuple[Primitive, ...] = (
    "view",
    "search",
    "syntax_preflight",
    "governance",
)


class SweAgentACIShadow(BoundedRepositoryACI):
    """Evaluation-only ACI retaining the governed Laplace substrate."""

    def read_region(
        self,
        *,
        path: str,
        start_line: int,
        end_line: int,
    ) -> JsonObject:
        if (
            isinstance(start_line, bool)
            or not isinstance(start_line, int)
            or isinstance(end_line, bool)
            or not isinstance(end_line, int)
        ):
            return super().read_region(
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
        capped_end = min(end_line, start_line + SWE_AGENT_VIEW_LINES - 1)
        result = super().read_region(
            path=path,
            start_line=start_line,
            end_line=capped_end,
        )
        return {
            **result,
            "aci_profile": "swe_agent_shadow_v34",
            "requested_end_line": end_line,
            "window_lines": SWE_AGENT_VIEW_LINES,
            "window_truncated": end_line > capped_end,
        }

    def search_text(self, *, query: str, glob: str = "*") -> JsonObject:
        baseline = super().search_text(query=query, glob=glob)
        raw_matches = baseline.get("matches")
        if not isinstance(raw_matches, list):
            raise BoundedACIError("aci_swe_search_result_invalid")

        files: list[str] = []
        seen: set[str] = set()
        for raw_match in raw_matches:
            if not isinstance(raw_match, Mapping):
                raise BoundedACIError("aci_swe_search_result_invalid")
            path_value = raw_match.get("path")
            if not isinstance(path_value, str):
                raise BoundedACIError("aci_swe_search_result_invalid")
            if path_value not in seen:
                seen.add(path_value)
                files.append(path_value)

        response: JsonObject = {
            "owner_user_id": self.owner_user_id,
            "session_id": self.session_id,
            "query": baseline.get("query", query),
            "glob": baseline.get("glob", glob),
            "files": files,
            "file_count": len(files),
            "source_match_count": len(raw_matches),
            "truncated": bool(baseline.get("truncated", False)),
            "aci_profile": "swe_agent_shadow_v34",
        }
        if not files:
            response["message"] = "NO_MATCHES"
        return response

    def edit_region(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
    ) -> JsonObject:
        # Baseline denial/validation must remain first and authoritative.
        if not self.allow_mutation:
            return super().edit_region(
                path=path,
                old_text=old_text,
                new_text=new_text,
            )
        if (
            not isinstance(old_text, str)
            or not isinstance(new_text, str)
            or not old_text
            or len(old_text) > _PREFLIGHT_TEXT_LIMIT
            or len(new_text) > _PREFLIGHT_TEXT_LIMIT
            or "\x00" in old_text
            or "\x00" in new_text
        ):
            return super().edit_region(
                path=path,
                old_text=old_text,
                new_text=new_text,
            )

        target = self._target(path)
        original = self._read_text(target, category="aci_edit_region")
        if original.count(old_text) != 1:
            return super().edit_region(
                path=path,
                old_text=old_text,
                new_text=new_text,
            )
        replacement = original.replace(old_text, new_text, 1)
        if len(replacement) > _PREFLIGHT_TEXT_LIMIT:
            return super().edit_region(
                path=path,
                old_text=old_text,
                new_text=new_text,
            )

        syntax_preflight = "NOT_APPLICABLE"
        if target.suffix == ".py":
            try:
                ast.parse(replacement, filename=target.name)
            except SyntaxError as exc:
                raise BoundedACIError(
                    "aci_swe_python_syntax_invalid",
                    {
                        "path": target.relative_to(self.worktree).as_posix(),
                        "line": exc.lineno,
                        "offset": exc.offset,
                        "message": exc.msg,
                    },
                ) from exc
            syntax_preflight = "PASS"

        result = super().edit_region(
            path=path,
            old_text=old_text,
            new_text=new_text,
        )
        return {
            **result,
            "aci_profile": "swe_agent_shadow_v34",
            "syntax_preflight": syntax_preflight,
        }


class SweAgentABCoordinator(ZetsuAgentCoordinator):
    """Candidate coordinator used only by a separately constructed A/B service."""

    def _typed_aci(
        self,
        ctx: AgentRunContext,
        state: AgentExecutionState,
    ) -> BoundedRepositoryACI:
        del state
        allow_mutation = (
            ctx.allow_mutation
            and "apply_patch" in ctx.binding.tool_policy.allowed_tools
        )
        return SweAgentACIShadow(
            ctx.worktree,
            owner_user_id=ctx.user_id,
            session_id=ctx.session_id,
            allow_mutation=allow_mutation,
            required_verification_argv=ctx.required_verification_argv,
            is_cancelled=lambda: self.tiered.agent_session_status(
                user_id=ctx.user_id,
                session_id=ctx.session_id,
            ).get("status")
            == "CANCELLED",
        )


def build_swe_agent_candidate_service(
    repository_root: Path,
    corpus: PersonalCorpusStore,
    tiered: TieredServingService,
) -> ZetsuService:
    """Compose the candidate through current neutral LaplaceCore injection."""
    coordinator = SweAgentABCoordinator(tiered, corpus)
    core = LaplaceCore(
        repository_root,
        corpus,
        tiered,
        repository_agent_service=coordinator,
    )
    return ZetsuService(repository_root, corpus, tiered, core=core)


@dataclass(frozen=True)
class PrimitiveTask:
    task_id: str
    primitive: Primitive
    operation: str
    payload: JsonObject
    expected_terms: tuple[str, ...]
    expected_paths: tuple[str, ...]


@dataclass(frozen=True)
class PrimitiveTrial:
    task_id: str
    primitive: Primitive
    correctness: float
    context_tokens: float
    wall_seconds: float
    failure: float
    verifier_success: float
    repository_coverage: float
    relevance: float
    security_preserved: bool

    def as_json(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "primitive": self.primitive,
            "correctness": self.correctness,
            "context_tokens": self.context_tokens,
            "wall_seconds": self.wall_seconds,
            "failure": self.failure,
            "verifier_success": self.verifier_success,
            "repository_coverage": self.repository_coverage,
            "relevance": self.relevance,
            "security_preserved": self.security_preserved,
        }


def _string_list_field(
    task: Mapping[str, object],
    label: str,
) -> tuple[str, ...]:
    value = task.get(label, [])
    if not isinstance(value, list):
        raise TypeError(f"v34_phase_a_task_{label}_invalid")
    if not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"v34_phase_a_task_{label}_invalid")
    return tuple(cast(list[str], value))


def load_phase_a_tasks(path: Path) -> tuple[PrimitiveTask, ...]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("v34_phase_a_manifest_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v34_phase_a_manifest_not_object")
    manifest = cast(dict[str, object], raw)
    if manifest.get("schema") != "laplace.v34.swe_aci_phase_a_tasks.v2":
        raise ValueError("v34_phase_a_manifest_schema_mismatch")
    if manifest.get("laplace_revision") != LAPLACE_V34_BASELINE:
        raise ValueError("v34_phase_a_manifest_laplace_revision_mismatch")
    if manifest.get("swe_agent_revision") != SWE_AGENT_REFERENCE_COMMIT:
        raise ValueError("v34_phase_a_manifest_upstream_revision_mismatch")
    if manifest.get("repetitions") != PHASE_A_REPETITIONS:
        raise ValueError("v34_phase_a_manifest_repetition_mismatch")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 12:
        raise ValueError("v34_phase_a_manifest_task_count_invalid")

    tasks: list[PrimitiveTask] = []
    seen: set[str] = set()
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise TypeError("v34_phase_a_task_invalid")
        task = cast(dict[str, object], raw_task)
        task_id = task.get("task_id")
        primitive = task.get("primitive")
        operation = task.get("operation")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen
            or not isinstance(primitive, str)
            or primitive not in _PRIMITIVES
            or not isinstance(operation, str)
            or not operation
        ):
            raise ValueError("v34_phase_a_task_identity_invalid")

        expected_terms = _string_list_field(task, "expected_terms")
        expected_paths = _string_list_field(task, "expected_paths")
        payload = {
            key: value
            for key, value in task.items()
            if key
            not in {
                "task_id",
                "primitive",
                "operation",
                "expected_terms",
                "expected_paths",
            }
        }
        seen.add(task_id)
        tasks.append(
            PrimitiveTask(
                task_id=task_id,
                primitive=primitive,
                operation=operation,
                payload=payload,
                expected_terms=expected_terms,
                expected_paths=expected_paths,
            )
        )

    counts = {
        primitive: sum(task.primitive == primitive for task in tasks)
        for primitive in _PRIMITIVES
    }
    if counts != {
        "view": 3,
        "search": 4,
        "syntax_preflight": 2,
        "governance": 3,
    }:
        raise ValueError("v34_phase_a_primitive_balance_invalid")
    return tuple(tasks)


def aggregate_trials(trials: Sequence[PrimitiveTrial]) -> JsonObject:
    if not trials:
        raise ValueError("v34_phase_a_trials_empty")
    return {
        "samples": len(trials),
        "correctness": fmean(item.correctness for item in trials),
        "context_tokens": fmean(item.context_tokens for item in trials),
        "wall_seconds_median": median(item.wall_seconds for item in trials),
        "failure_rate": fmean(item.failure for item in trials),
        "verifier_success": fmean(
            item.verifier_success for item in trials
        ),
        "repository_coverage": fmean(
            item.repository_coverage for item in trials
        ),
        "relevance": fmean(item.relevance for item in trials),
        "security_preserved": all(
            item.security_preserved for item in trials
        ),
    }


def _number(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"v34_phase_a_metric_invalid:{key}")
    return float(value)


def assess_phase_a(
    baseline: Sequence[PrimitiveTrial],
    candidate: Sequence[PrimitiveTrial],
) -> JsonObject:
    baseline_ids = [item.task_id for item in baseline]
    candidate_ids = [item.task_id for item in candidate]
    if (
        len(baseline_ids) != 12
        or len(candidate_ids) != 12
        or len(set(baseline_ids)) != 12
        or len(set(candidate_ids)) != 12
        or set(baseline_ids) != set(candidate_ids)
    ):
        return {"assessment": "BLOCKED", "reason": "paired_task_matrix_invalid"}

    baseline_by_id = {item.task_id: item for item in baseline}
    candidate_by_id = {item.task_id: item for item in candidate}
    for task_id, baseline_item in baseline_by_id.items():
        if baseline_item.primitive != candidate_by_id[task_id].primitive:
            return {
                "assessment": "BLOCKED",
                "reason": "paired_task_primitive_mismatch",
            }

    baseline_summary = aggregate_trials(baseline)
    candidate_summary = aggregate_trials(candidate)

    pairwise_security = all(
        (
            not baseline_by_id[task_id].security_preserved
            or candidate_by_id[task_id].security_preserved
        )
        for task_id in baseline_by_id
    )
    overall_non_regression = {
        "pairwise_security": pairwise_security,
        "correctness": _number(candidate_summary, "correctness")
        >= _number(baseline_summary, "correctness"),
        "failure": _number(candidate_summary, "failure_rate")
        <= _number(baseline_summary, "failure_rate"),
        "verifier": _number(candidate_summary, "verifier_success")
        >= _number(baseline_summary, "verifier_success"),
        "coverage": _number(candidate_summary, "repository_coverage") + 0.05
        >= _number(baseline_summary, "repository_coverage"),
        "relevance": _number(candidate_summary, "relevance") + 0.05
        >= _number(baseline_summary, "relevance"),
    }

    by_primitive: JsonObject = {}
    promising_primitives: list[str] = []
    for primitive in _PRIMITIVES:
        baseline_group = [
            item for item in baseline if item.primitive == primitive
        ]
        candidate_group = [
            item for item in candidate if item.primitive == primitive
        ]
        b = aggregate_trials(baseline_group)
        c = aggregate_trials(candidate_group)
        safe = bool(c["security_preserved"]) and (
            _number(c, "correctness") >= _number(b, "correctness")
            and _number(c, "failure_rate") <= _number(b, "failure_rate")
            and _number(c, "verifier_success") >= _number(b, "verifier_success")
            and _number(c, "repository_coverage") + 0.05
            >= _number(b, "repository_coverage")
            and _number(c, "relevance") + 0.05
            >= _number(b, "relevance")
        )
        quality_gain = (
            _number(c, "correctness") > _number(b, "correctness")
            or _number(c, "verifier_success") > _number(b, "verifier_success")
        )
        token_gain = (
            _number(b, "context_tokens") > 0.0
            and _number(c, "context_tokens")
            <= 0.95 * _number(b, "context_tokens")
        )
        wall_gain = (
            _number(b, "wall_seconds_median") > 0.0
            and _number(c, "wall_seconds_median")
            <= 0.95 * _number(b, "wall_seconds_median")
        )
        promising = safe and (quality_gain or token_gain or wall_gain)
        if primitive == "governance":
            # Governance is a gate, not an optimization target.
            promising = False
        elif promising:
            promising_primitives.append(primitive)
        by_primitive[primitive] = {
            "baseline": b,
            "candidate": c,
            "safe": safe,
            "quality_gain": quality_gain,
            "token_gain": token_gain,
            "wall_gain": wall_gain,
            "promising": promising,
        }

    governance = cast(
        Mapping[str, object],
        cast(Mapping[str, object], by_primitive["governance"])["candidate"],
    )
    governance_safe = bool(governance["security_preserved"]) and (
        _number(governance, "correctness") == 1.0
        and _number(governance, "failure_rate") == 0.0
    )
    assessment = (
        "PROMISING"
        if governance_safe
        and all(overall_non_regression.values())
        and promising_primitives
        else "NOT_PROMISING"
    )
    return {
        "schema": "laplace.v34.swe_aci_phase_a_decision.v2",
        "phase": "A_PRIMITIVE_ONLY",
        "laplace_revision": LAPLACE_V34_BASELINE,
        "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
        "tasks": 12,
        "repetitions": PHASE_A_REPETITIONS,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "overall_non_regression": overall_non_regression,
        "governance_safe": governance_safe,
        "by_primitive": by_primitive,
        "promising_primitives": promising_primitives,
        "assessment": assessment,
        "final_adoption_decision": None,
        "next_step": (
            "phase_b_agent_ab_for_promising_primitives"
            if assessment == "PROMISING"
            else "document_keep_laplace_aci_result"
        ),
    }
