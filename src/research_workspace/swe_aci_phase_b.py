"""Step 3.4 Phase-B real-agent A/B for SWE-agent-inspired ACI primitives.

This module is evaluation-only. It changes no production selector or default
coordinator. Baseline and candidate use the same Laplace agent engine; only the
typed ACI implementation differs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, TypeAlias, cast

from .bounded_aci import BoundedRepositoryACI
from .swe_aci_ab import (
    SWE_AGENT_REFERENCE_COMMIT,
    SweAgentACIShadow,
)
from .zetsu_agent import ZetsuAgentCoordinator
from .zetsu_state import AgentExecutionState, AgentRunContext

JsonObject: TypeAlias = dict[str, object]
Arm = Literal["baseline", "candidate"]
TaskKind = Literal["view", "search", "mutation"]

LAPLACE_V34_PHASE_B_BASELINE = "5d8e463511d84ab1a7ee87b0f717b665f91d3e0f"
MIN_PHASE_B_PAIRED_TASKS = 10
PHASE_B_TASK_COUNT = 12


@dataclass(frozen=True)
class PhaseBTask:
    task_id: str
    kind: TaskKind
    instruction: str
    mutation_expected: bool
    expected_answer_terms: tuple[str, ...]
    expected_paths: tuple[str, ...]
    verification_argv: tuple[str, ...] | None
    oracle: JsonObject | None


def _string_list(
    value: object,
    *,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"v34_phase_b_{label}_invalid")
    if not allow_empty and not value:
        raise ValueError(f"v34_phase_b_{label}_empty")
    if not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"v34_phase_b_{label}_invalid")
    return tuple(cast(list[str], value))


def load_phase_b_tasks(path: Path) -> tuple[PhaseBTask, ...]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("v34_phase_b_manifest_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v34_phase_b_manifest_not_object")
    manifest = cast(dict[str, object], raw)
    if manifest.get("schema") != "laplace.v34.swe_aci_phase_b_tasks.v1":
        raise ValueError("v34_phase_b_manifest_schema_mismatch")
    if manifest.get("laplace_revision") != LAPLACE_V34_PHASE_B_BASELINE:
        raise ValueError("v34_phase_b_manifest_laplace_revision_mismatch")
    if manifest.get("swe_agent_revision") != SWE_AGENT_REFERENCE_COMMIT:
        raise ValueError("v34_phase_b_manifest_swe_revision_mismatch")
    if manifest.get("minimum_paired_tasks") != MIN_PHASE_B_PAIRED_TASKS:
        raise ValueError("v34_phase_b_manifest_minimum_mismatch")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != PHASE_B_TASK_COUNT:
        raise ValueError("v34_phase_b_manifest_task_count_invalid")

    tasks: list[PhaseBTask] = []
    seen: set[str] = set()
    valid_kinds = {"view", "search", "mutation"}
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise TypeError("v34_phase_b_task_not_object")
        task = cast(dict[str, object], raw_task)
        task_id = task.get("task_id")
        kind = task.get("kind")
        instruction = task.get("instruction")
        mutation_expected = task.get("mutation_expected")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen
            or not isinstance(kind, str)
            or kind not in valid_kinds
            or not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(mutation_expected, bool)
        ):
            raise ValueError("v34_phase_b_task_identity_invalid")

        expected_terms = _string_list(
            task.get("expected_answer_terms"),
            label="expected_answer_terms",
            allow_empty=True,
        )
        expected_paths = _string_list(
            task.get("expected_paths"),
            label="expected_paths",
            allow_empty=True,
        )
        verifier_raw = task.get("verification_argv")
        verifier = (
            None
            if verifier_raw is None
            else _string_list(
                verifier_raw,
                label="verification_argv",
                allow_empty=False,
            )
        )
        oracle_raw = task.get("oracle")
        if oracle_raw is not None and not isinstance(oracle_raw, dict):
            raise TypeError("v34_phase_b_oracle_invalid")
        oracle = (
            cast(JsonObject, dict(oracle_raw))
            if isinstance(oracle_raw, dict)
            else None
        )

        if kind == "mutation":
            if not mutation_expected or verifier is None or oracle is None:
                raise ValueError("v34_phase_b_mutation_contract_invalid")
        elif mutation_expected or verifier is not None or oracle is not None:
            raise ValueError("v34_phase_b_read_contract_invalid")

        for relative in expected_paths:
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("v34_phase_b_expected_path_invalid")

        seen.add(task_id)
        tasks.append(
            PhaseBTask(
                task_id=task_id,
                kind=cast(TaskKind, kind),
                instruction=instruction,
                mutation_expected=mutation_expected,
                expected_answer_terms=expected_terms,
                expected_paths=expected_paths,
                verification_argv=verifier,
                oracle=oracle,
            )
        )

    counts = {
        kind: sum(task.kind == kind for task in tasks)
        for kind in ("view", "search", "mutation")
    }
    if counts != {"view": 4, "search": 4, "mutation": 4}:
        raise ValueError("v34_phase_b_task_balance_invalid")
    return tuple(tasks)


@dataclass
class ACITrace:
    """Records authorized repository paths exposed through selected ACI primitives."""

    paths: list[str]

    def __init__(self) -> None:
        self.paths = []

    def record(self, path: str) -> None:
        candidate = str(path)
        if candidate and candidate not in self.paths:
            self.paths.append(candidate)

    def record_search_result(self, value: Mapping[str, object]) -> None:
        files = value.get("files")
        if isinstance(files, list):
            for item in files:
                if isinstance(item, str):
                    self.record(item)
        matches = value.get("matches")
        if isinstance(matches, list):
            for item in matches:
                if isinstance(item, Mapping):
                    path = item.get("path")
                    if isinstance(path, str):
                        self.record(path)


class RecordingBaselineACI(BoundedRepositoryACI):
    def __init__(self, *args: object, trace: ACITrace, **kwargs: object) -> None:
        self._v34_trace = trace
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def read_region(
        self,
        *,
        path: str,
        start_line: int,
        end_line: int,
    ) -> JsonObject:
        self._v34_trace.record(path)
        return super().read_region(
            path=path,
            start_line=start_line,
            end_line=end_line,
        )

    def search_text(self, *, query: str, glob: str = "*") -> JsonObject:
        result = super().search_text(query=query, glob=glob)
        self._v34_trace.record_search_result(result)
        return result

    def edit_region(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
    ) -> JsonObject:
        self._v34_trace.record(path)
        return super().edit_region(
            path=path,
            old_text=old_text,
            new_text=new_text,
        )


class RecordingCandidateACI(SweAgentACIShadow):
    def __init__(self, *args: object, trace: ACITrace, **kwargs: object) -> None:
        self._v34_trace = trace
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def read_region(
        self,
        *,
        path: str,
        start_line: int,
        end_line: int,
    ) -> JsonObject:
        self._v34_trace.record(path)
        return super().read_region(
            path=path,
            start_line=start_line,
            end_line=end_line,
        )

    def search_text(self, *, query: str, glob: str = "*") -> JsonObject:
        result = super().search_text(query=query, glob=glob)
        self._v34_trace.record_search_result(result)
        return result

    def edit_region(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
    ) -> JsonObject:
        self._v34_trace.record(path)
        return super().edit_region(
            path=path,
            old_text=old_text,
            new_text=new_text,
        )


class RecordingBaselineCoordinator(ZetsuAgentCoordinator):
    def __init__(self, *args: object, trace: ACITrace, **kwargs: object) -> None:
        self.v34_trace = trace
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

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
        return RecordingBaselineACI(
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
            trace=self.v34_trace,
        )


class RecordingCandidateCoordinator(ZetsuAgentCoordinator):
    def __init__(self, *args: object, trace: ACITrace, **kwargs: object) -> None:
        self.v34_trace = trace
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

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
        return RecordingCandidateACI(
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
            trace=self.v34_trace,
        )


@dataclass(frozen=True)
class AgentTaskRecord:
    task_id: str
    kind: TaskKind
    correct: bool
    input_tokens: int
    output_tokens: int
    tool_rounds: int
    wall_seconds: float
    failed: bool
    verifier_passed: bool | None
    repository_coverage: float
    relevance: float
    usage_complete: bool
    security_preserved: bool
    model_id: str
    fixture_sha256: str

    def as_json(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "correct": self.correct,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_rounds": self.tool_rounds,
            "wall_seconds": self.wall_seconds,
            "failed": self.failed,
            "verifier_passed": self.verifier_passed,
            "repository_coverage": self.repository_coverage,
            "relevance": self.relevance,
            "usage_complete": self.usage_complete,
            "security_preserved": self.security_preserved,
            "model_id": self.model_id,
            "fixture_sha256": self.fixture_sha256,
        }


def _aggregate(records: Sequence[AgentTaskRecord]) -> JsonObject:
    if not records:
        raise ValueError("v34_phase_b_records_empty")
    verifier_values = [
        float(item.verifier_passed)
        for item in records
        if item.verifier_passed is not None
    ]
    if len(verifier_values) != 4:
        raise ValueError("v34_phase_b_verifier_sample_count_invalid")
    return {
        "task_count": len(records),
        "correct_rate": fmean(float(item.correct) for item in records),
        "input_tokens_mean": fmean(item.input_tokens for item in records),
        "output_tokens_mean": fmean(item.output_tokens for item in records),
        "tool_rounds_mean": fmean(item.tool_rounds for item in records),
        "wall_seconds_mean": fmean(item.wall_seconds for item in records),
        "failure_rate": fmean(float(item.failed) for item in records),
        "verifier_success_rate": fmean(verifier_values),
        "repository_coverage_mean": fmean(
            item.repository_coverage for item in records
        ),
        "relevance_mean": fmean(item.relevance for item in records),
        "usage_complete": all(item.usage_complete for item in records),
        "security_preserved": all(
            item.security_preserved for item in records
        ),
    }


def _metric(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"v34_phase_b_metric_invalid:{key}")
    return float(value)


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else 1_000_000.0
    return candidate / baseline


def assess_phase_b(
    baseline: Sequence[AgentTaskRecord],
    candidate: Sequence[AgentTaskRecord],
) -> JsonObject:
    if len(baseline) != PHASE_B_TASK_COUNT or len(candidate) != PHASE_B_TASK_COUNT:
        return {"decision": "BLOCKED", "reason": "paired_task_count_invalid"}
    baseline_by_id = {item.task_id: item for item in baseline}
    candidate_by_id = {item.task_id: item for item in candidate}
    if (
        len(baseline_by_id) != PHASE_B_TASK_COUNT
        or len(candidate_by_id) != PHASE_B_TASK_COUNT
        or set(baseline_by_id) != set(candidate_by_id)
    ):
        return {"decision": "BLOCKED", "reason": "paired_task_sets_invalid"}

    ordered = sorted(baseline_by_id)
    for task_id in ordered:
        before = baseline_by_id[task_id]
        after = candidate_by_id[task_id]
        if before.kind != after.kind:
            return {"decision": "BLOCKED", "reason": "task_kind_mismatch"}
        if before.fixture_sha256 != after.fixture_sha256:
            return {"decision": "BLOCKED", "reason": "fixture_mismatch"}
        if not before.usage_complete or not after.usage_complete:
            return {"decision": "BLOCKED", "reason": "model_usage_incomplete"}

    model_ids = {
        item.model_id for item in [*baseline_by_id.values(), *candidate_by_id.values()]
    }
    if len(model_ids) != 1:
        return {"decision": "BLOCKED", "reason": "model_id_mismatch"}

    baseline_ordered = [baseline_by_id[item] for item in ordered]
    candidate_ordered = [candidate_by_id[item] for item in ordered]
    before_summary = _aggregate(baseline_ordered)
    after_summary = _aggregate(candidate_ordered)

    pairwise = {
        task_id: {
            "correct_non_regression": (
                not baseline_by_id[task_id].correct
                or candidate_by_id[task_id].correct
            ),
            "failure_non_regression": (
                baseline_by_id[task_id].failed
                or not candidate_by_id[task_id].failed
            ),
            "verifier_non_regression": (
                baseline_by_id[task_id].verifier_passed is not True
                or candidate_by_id[task_id].verifier_passed is True
            ),
            "security_non_regression": (
                not baseline_by_id[task_id].security_preserved
                or candidate_by_id[task_id].security_preserved
            ),
        }
        for task_id in ordered
    }
    pairwise_safe = all(
        all(checks.values()) for checks in pairwise.values()
    )

    non_regression = {
        "pairwise_safety": pairwise_safe,
        "correctness": _metric(after_summary, "correct_rate")
        >= _metric(before_summary, "correct_rate"),
        "failure": _metric(after_summary, "failure_rate")
        <= _metric(before_summary, "failure_rate"),
        "verifier": _metric(after_summary, "verifier_success_rate")
        >= _metric(before_summary, "verifier_success_rate"),
        "coverage": _metric(after_summary, "repository_coverage_mean") + 0.05
        >= _metric(before_summary, "repository_coverage_mean"),
        "relevance": _metric(after_summary, "relevance_mean") + 0.05
        >= _metric(before_summary, "relevance_mean"),
        "security": bool(after_summary["security_preserved"]),
    }

    efficiency_keys = (
        "input_tokens_mean",
        "output_tokens_mean",
        "tool_rounds_mean",
        "wall_seconds_mean",
    )
    ratios = {
        key: _ratio(
            _metric(after_summary, key),
            _metric(before_summary, key),
        )
        for key in efficiency_keys
    }
    no_material_efficiency_regression = all(
        value <= 1.05 for value in ratios.values()
    )
    efficiency_gain_count = sum(
        value <= 0.95 for value in ratios.values()
    )
    quality_gain = (
        _metric(after_summary, "correct_rate")
        > _metric(before_summary, "correct_rate")
        or _metric(after_summary, "verifier_success_rate")
        > _metric(before_summary, "verifier_success_rate")
    )
    adopt = (
        all(non_regression.values())
        and no_material_efficiency_regression
        and (quality_gain or efficiency_gain_count >= 2)
    )
    return {
        "schema": "laplace.v34.swe_aci_phase_b_decision.v1",
        "phase": "B_REAL_AGENT",
        "laplace_revision": LAPLACE_V34_PHASE_B_BASELINE,
        "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
        "paired_tasks": PHASE_B_TASK_COUNT,
        "model_id": next(iter(model_ids)),
        "baseline": before_summary,
        "candidate": after_summary,
        "pairwise": pairwise,
        "non_regression": non_regression,
        "efficiency_ratios_candidate_over_baseline": ratios,
        "quality_gain": quality_gain,
        "efficiency_gain_count": efficiency_gain_count,
        "decision": "ADOPT" if adopt else "KEEP_LAPLACE",
    }
