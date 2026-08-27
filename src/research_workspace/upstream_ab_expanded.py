"""Expanded Step-3.3 Aider RepoMap screening methodology.

This module deliberately remains benchmark-only.  It broadens the Phase-A
screening experiment across task strata and token budgets without introducing
Aider into the Laplace agent path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Literal, TypeAlias, cast

from .upstream_ab import (
    AIDER_REPOMAP_REVISION,
    V33_BASELINE_REVISION,
    UpstreamABError,
    char4_tokens,
)

JsonObject: TypeAlias = dict[str, object]

FocusMode = Literal["FOCUSED", "NO_FOCUS"]
Category = Literal[
    "direct_symbol",
    "cross_file",
    "orientation",
    "ambiguous",
    "change_impact",
    "governance",
]
ExpandedAssessment = Literal["PROMISING", "NOT_PROMISING"]

EXPANDED_BUDGETS = (256, 512, 1000, 2000)
EXPANDED_REPETITIONS = 5
EXPANDED_TASK_COUNT = 30
_DISCOVERY_CATEGORIES = {
    "cross_file",
    "orientation",
    "ambiguous",
    "change_impact",
}


@dataclass(frozen=True)
class ExpandedPhaseATask:
    task_id: str
    category: Category
    focus_mode: FocusMode
    query: str
    focus_paths: tuple[str, ...]
    expected_paths: tuple[str, ...]
    expected_terms: tuple[str, ...]
    expected_symbols: tuple[str, ...]


def _string_list(
    task: dict[str, object],
    label: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    value = task.get(label)
    if not isinstance(value, list):
        raise UpstreamABError(f"v33_expanded_{label}_invalid")
    if not allow_empty and not value:
        raise UpstreamABError(f"v33_expanded_{label}_invalid")
    if not all(isinstance(item, str) and item for item in value):
        raise UpstreamABError(f"v33_expanded_{label}_invalid")
    return tuple(cast(list[str], value))


def load_expanded_tasks(path: Path) -> tuple[ExpandedPhaseATask, ...]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamABError("v33_expanded_manifest_invalid") from exc
    if not isinstance(raw, dict):
        raise UpstreamABError("v33_expanded_manifest_not_object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema") != "laplace-v33-aider-repomap-expanded-v1":
        raise UpstreamABError("v33_expanded_manifest_schema_mismatch")
    if payload.get("laplace_revision") != V33_BASELINE_REVISION:
        raise UpstreamABError("v33_expanded_manifest_laplace_revision_mismatch")
    if payload.get("aider_revision") != AIDER_REPOMAP_REVISION:
        raise UpstreamABError("v33_expanded_manifest_aider_revision_mismatch")
    if payload.get("budgets") != list(EXPANDED_BUDGETS):
        raise UpstreamABError("v33_expanded_manifest_budget_mismatch")
    if payload.get("timed_repetitions") != EXPANDED_REPETITIONS:
        raise UpstreamABError("v33_expanded_manifest_repetition_mismatch")

    records = payload.get("tasks")
    if not isinstance(records, list) or len(records) != EXPANDED_TASK_COUNT:
        raise UpstreamABError("v33_expanded_manifest_task_count_invalid")

    tasks: list[ExpandedPhaseATask] = []
    seen: set[str] = set()
    valid_categories = {
        "direct_symbol",
        "cross_file",
        "orientation",
        "ambiguous",
        "change_impact",
        "governance",
    }
    valid_focus = {"FOCUSED", "NO_FOCUS"}

    for raw_task in records:
        if not isinstance(raw_task, dict):
            raise UpstreamABError("v33_expanded_task_not_object")
        task = cast(dict[str, object], raw_task)
        expected_keys = {
            "task_id",
            "category",
            "focus_mode",
            "query",
            "focus_paths",
            "expected_paths",
            "expected_terms",
            "expected_symbols",
        }
        if set(task) != expected_keys:
            raise UpstreamABError("v33_expanded_task_schema_mismatch")

        task_id = task["task_id"]
        category = task["category"]
        focus_mode = task["focus_mode"]
        query = task["query"]
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen
            or len(task_id) > 120
            or not isinstance(category, str)
            or category not in valid_categories
            or not isinstance(focus_mode, str)
            or focus_mode not in valid_focus
            or not isinstance(query, str)
            or not query.strip()
            or len(query) > 4000
        ):
            raise UpstreamABError("v33_expanded_task_fields_invalid")

        focus_paths = _string_list(task, "focus_paths", allow_empty=True)
        expected_paths = _string_list(task, "expected_paths", allow_empty=False)
        expected_terms = _string_list(task, "expected_terms", allow_empty=False)
        expected_symbols = _string_list(task, "expected_symbols", allow_empty=True)

        if focus_mode == "FOCUSED" and not focus_paths:
            raise UpstreamABError("v33_expanded_focused_task_without_focus")
        if focus_mode == "NO_FOCUS" and focus_paths:
            raise UpstreamABError("v33_expanded_no_focus_task_has_focus")

        for value in (*focus_paths, *expected_paths):
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise UpstreamABError("v33_expanded_path_invalid")

        seen.add(task_id)
        tasks.append(
            ExpandedPhaseATask(
                task_id=task_id,
                category=cast(Category, category),
                focus_mode=cast(FocusMode, focus_mode),
                query=query,
                focus_paths=focus_paths,
                expected_paths=expected_paths,
                expected_terms=expected_terms,
                expected_symbols=expected_symbols,
            )
        )

    category_counts: dict[str, int] = {}
    focus_counts: dict[str, int] = {}
    for expanded_task in tasks:
        category_counts[expanded_task.category] = (
            category_counts.get(expanded_task.category, 0) + 1
        )
        focus_counts[expanded_task.focus_mode] = (
            focus_counts.get(expanded_task.focus_mode, 0) + 1
        )
    if min(category_counts.values(), default=0) < 4:
        raise UpstreamABError("v33_expanded_category_underrepresented")
    if focus_counts.get("NO_FOCUS", 0) < 12 or focus_counts.get("FOCUSED", 0) < 8:
        raise UpstreamABError("v33_expanded_focus_balance_invalid")
    return tuple(tasks)


def validate_expanded_task_paths(
    snapshot: Path,
    tasks: tuple[ExpandedPhaseATask, ...],
) -> tuple[str, ...]:
    all_paths = tuple(
        sorted(
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and ".git" not in path.relative_to(snapshot).parts
            and not any(part.startswith(".aider.tags.cache") for part in path.parts)
        )
    )
    available = set(all_paths)
    for task in tasks:
        for relative in (*task.focus_paths, *task.expected_paths):
            if relative not in available:
                raise UpstreamABError(
                    f"v33_expanded_expected_path_missing:{task.task_id}:{relative}"
                )
    return all_paths


def _path_mentions(text: str, all_repo_paths: tuple[str, ...]) -> set[str]:
    return {relative for relative in all_repo_paths if relative in text}


def expanded_quality(
    text: str,
    task: ExpandedPhaseATask,
    all_repo_paths: tuple[str, ...],
) -> JsonObject:
    lowered = text.casefold()
    mentioned_paths = _path_mentions(text, all_repo_paths)
    expected_path_set = set(task.expected_paths)
    path_hits = mentioned_paths & expected_path_set
    path_recall = len(path_hits) / len(expected_path_set)
    path_precision = (
        len(path_hits) / len(mentioned_paths)
        if mentioned_paths
        else 0.0
    )

    term_hits = sum(
        1 for term in task.expected_terms if term.casefold() in lowered
    )
    term_recall = term_hits / len(task.expected_terms)

    symbol_recall: float | None
    symbol_hits = 0
    if task.expected_symbols:
        symbol_hits = sum(
            1 for symbol in task.expected_symbols if symbol in text
        )
        symbol_recall = symbol_hits / len(task.expected_symbols)
    else:
        symbol_recall = None

    components = [path_recall, path_precision, term_recall]
    if symbol_recall is not None:
        components.append(symbol_recall)
    relevance = fmean(components)
    tokens = char4_tokens(text)
    return {
        "chars": len(text),
        "context_tokens": tokens,
        "path_recall": path_recall,
        "path_precision": path_precision,
        "term_recall": term_recall,
        "symbol_recall": symbol_recall,
        "expected_paths_hit": len(path_hits),
        "expected_paths_total": len(expected_path_set),
        "mentioned_repo_paths": len(mentioned_paths),
        "expected_terms_hit": term_hits,
        "expected_terms_total": len(task.expected_terms),
        "expected_symbols_hit": symbol_hits,
        "expected_symbols_total": len(task.expected_symbols),
        "relevance_score": relevance,
        "relevance_per_1k_tokens": relevance * 1000.0 / max(tokens, 1),
    }


@dataclass(frozen=True)
class CandidateRepeatPayload:
    text: str
    wall_times: tuple[float, ...]
    unique_map_hashes: int


def validate_candidate_repeat_payload(raw: object) -> CandidateRepeatPayload:
    if not isinstance(raw, dict):
        raise UpstreamABError("v33_expanded_candidate_repeat_not_object")
    payload = cast(dict[str, object], raw)
    text = payload.get("text")
    repeat_count = payload.get("repeat_count")
    stable = payload.get("within_process_stable")
    unique_hashes = payload.get("unique_map_hashes")
    samples = payload.get("wall_time_seconds_samples")
    if not isinstance(text, str):
        raise UpstreamABError("v33_expanded_candidate_repeat_text_invalid")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count != EXPANDED_REPETITIONS
    ):
        raise UpstreamABError("v33_expanded_candidate_repeat_count_invalid")
    if stable is not True:
        raise UpstreamABError("v33_expanded_candidate_within_process_nondeterministic")
    if (
        isinstance(unique_hashes, bool)
        or not isinstance(unique_hashes, int)
        or unique_hashes != 1
    ):
        raise UpstreamABError("v33_expanded_candidate_hash_count_invalid")
    if not isinstance(samples, list) or len(samples) != EXPANDED_REPETITIONS:
        raise UpstreamABError("v33_expanded_candidate_repeat_samples_invalid")

    wall_times: list[float] = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UpstreamABError("v33_expanded_candidate_repeat_wall_invalid")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise UpstreamABError("v33_expanded_candidate_repeat_wall_invalid")
        wall_times.append(number)
    return CandidateRepeatPayload(
        text=text,
        wall_times=tuple(wall_times),
        unique_map_hashes=unique_hashes,
    )


def timing_summary(values: list[float]) -> JsonObject:
    if len(values) != EXPANDED_REPETITIONS:
        raise UpstreamABError("v33_expanded_timing_repetition_mismatch")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise UpstreamABError("v33_expanded_timing_invalid")
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": list(values),
        "median_seconds": median(values),
        "p95_seconds": ordered[p95_index],
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamABError(f"v33_expanded_{label}_invalid")
    number = float(value)
    if not math.isfinite(number):
        raise UpstreamABError(f"v33_expanded_{label}_invalid")
    return number


def _provider_metrics(row: dict[str, object], provider: str) -> dict[str, object]:
    raw = row.get(provider)
    if not isinstance(raw, dict):
        raise UpstreamABError("v33_expanded_provider_metrics_invalid")
    return cast(dict[str, object], raw)


def _mean_metric(
    rows: list[dict[str, object]],
    provider: str,
    label: str,
) -> float:
    values: list[float] = []
    for row in rows:
        metrics = _provider_metrics(row, provider)
        value = metrics.get(label)
        if value is None and label == "symbol_recall":
            continue
        values.append(_number(value, label))
    if not values:
        return 0.0
    return fmean(values)


def _median_timing(
    rows: list[dict[str, object]],
    provider: str,
) -> float:
    values: list[float] = []
    for row in rows:
        metrics = _provider_metrics(row, provider)
        timing = metrics.get("timing")
        if not isinstance(timing, dict):
            raise UpstreamABError("v33_expanded_timing_missing")
        values.append(_number(timing.get("median_seconds"), "timing_median"))
    return median(values)


def aggregate_expanded_rows(rows: list[dict[str, object]]) -> JsonObject:
    if not rows:
        raise UpstreamABError("v33_expanded_rows_empty")

    def provider(provider: str) -> JsonObject:
        return {
            "context_tokens": _mean_metric(rows, provider, "context_tokens"),
            "path_recall": _mean_metric(rows, provider, "path_recall"),
            "path_precision": _mean_metric(rows, provider, "path_precision"),
            "term_recall": _mean_metric(rows, provider, "term_recall"),
            "symbol_recall": _mean_metric(rows, provider, "symbol_recall"),
            "relevance_score": _mean_metric(rows, provider, "relevance_score"),
            "relevance_per_1k_tokens": _mean_metric(
                rows, provider, "relevance_per_1k_tokens"
            ),
            "map_wall_median_seconds": _median_timing(rows, provider),
        }

    return {
        "samples": len(rows),
        "baseline": provider("baseline"),
        "candidate": provider("candidate"),
    }


def _noninferior_quality(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    margin: float,
) -> bool:
    for label in (
        "path_recall",
        "path_precision",
        "term_recall",
        "symbol_recall",
        "relevance_score",
    ):
        if _number(candidate.get(label), label) + margin < _number(
            baseline.get(label), label
        ):
            return False
    return True


def _material_gain(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    margin: float,
) -> bool:
    before_tokens = _number(baseline.get("context_tokens"), "context_tokens")
    after_tokens = _number(candidate.get("context_tokens"), "context_tokens")
    before_density = _number(
        baseline.get("relevance_per_1k_tokens"), "relevance_per_1k_tokens"
    )
    after_density = _number(
        candidate.get("relevance_per_1k_tokens"), "relevance_per_1k_tokens"
    )
    before_wall = _number(
        baseline.get("map_wall_median_seconds"), "map_wall_median_seconds"
    )
    after_wall = _number(
        candidate.get("map_wall_median_seconds"), "map_wall_median_seconds"
    )
    return (
        (before_tokens > 0.0 and after_tokens <= before_tokens * (1.0 - margin))
        or (
            before_density > 0.0
            and after_density >= before_density * (1.0 + margin)
        )
        or (before_wall > 0.0 and after_wall <= before_wall * (1.0 - margin))
    )


def assess_expanded_phase_a(
    trials: object,
    *,
    margin: float = 0.05,
) -> JsonObject:
    if not isinstance(trials, list):
        raise UpstreamABError("v33_expanded_trials_invalid")
    expected_samples = EXPANDED_TASK_COUNT * len(EXPANDED_BUDGETS)
    if len(trials) != expected_samples:
        raise UpstreamABError("v33_expanded_trial_count_invalid")
    if not math.isfinite(margin) or not 0.0 <= margin < 1.0:
        raise UpstreamABError("v33_expanded_margin_invalid")

    rows: list[dict[str, object]] = []
    pairs: set[tuple[str, int]] = set()
    for raw in trials:
        if not isinstance(raw, dict):
            raise UpstreamABError("v33_expanded_trial_not_object")
        row = cast(dict[str, object], raw)
        task_id = row.get("task_id")
        budget = row.get("token_budget")
        category = row.get("category")
        focus_mode = row.get("focus_mode")
        if (
            not isinstance(task_id, str)
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget not in EXPANDED_BUDGETS
            or not isinstance(category, str)
            or not isinstance(focus_mode, str)
            or row.get("snapshot_match") is not True
        ):
            raise UpstreamABError("v33_expanded_trial_metadata_invalid")
        key = (task_id, budget)
        if key in pairs:
            raise UpstreamABError("v33_expanded_trial_duplicate")
        pairs.add(key)
        rows.append(row)

    overall = aggregate_expanded_rows(rows)
    overall_baseline = cast(dict[str, object], overall["baseline"])
    overall_candidate = cast(dict[str, object], overall["candidate"])
    overall_promising = _noninferior_quality(
        overall_baseline,
        overall_candidate,
        margin=margin,
    ) and _material_gain(
        overall_baseline,
        overall_candidate,
        margin=margin,
    )

    by_budget: JsonObject = {}
    for budget in EXPANDED_BUDGETS:
        subset = [row for row in rows if row["token_budget"] == budget]
        by_budget[str(budget)] = aggregate_expanded_rows(subset)

    by_focus: JsonObject = {}
    for focus_mode in ("FOCUSED", "NO_FOCUS"):
        subset = [row for row in rows if row["focus_mode"] == focus_mode]
        by_focus[focus_mode] = aggregate_expanded_rows(subset)

    categories = sorted({cast(str, row["category"]) for row in rows})
    by_category: JsonObject = {}
    advantage_strata: list[str] = []
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        aggregate = aggregate_expanded_rows(subset)
        by_category[category] = aggregate
        if category not in _DISCOVERY_CATEGORIES or len(subset) < 16:
            continue
        baseline = cast(dict[str, object], aggregate["baseline"])
        candidate = cast(dict[str, object], aggregate["candidate"])
        if _noninferior_quality(
            baseline,
            candidate,
            margin=margin,
        ) and _material_gain(
            baseline,
            candidate,
            margin=margin,
        ):
            advantage_strata.append(f"category:{category}")

    no_focus = cast(dict[str, object], by_focus["NO_FOCUS"])
    no_focus_baseline = cast(dict[str, object], no_focus["baseline"])
    no_focus_candidate = cast(dict[str, object], no_focus["candidate"])
    if _noninferior_quality(
        no_focus_baseline,
        no_focus_candidate,
        margin=margin,
    ) and _material_gain(
        no_focus_baseline,
        no_focus_candidate,
        margin=margin,
    ):
        advantage_strata.append("focus:NO_FOCUS")

    assessment: ExpandedAssessment = (
        "PROMISING"
        if overall_promising or advantage_strata
        else "NOT_PROMISING"
    )
    return {
        "assessment": assessment,
        "phase": "A_EXPANDED_MAP_ONLY",
        "tasks": EXPANDED_TASK_COUNT,
        "budgets": list(EXPANDED_BUDGETS),
        "samples": len(rows),
        "timed_repetitions": EXPANDED_REPETITIONS,
        "margin": margin,
        "overall": overall,
        "by_budget": by_budget,
        "by_focus_mode": by_focus,
        "by_category": by_category,
        "overall_promising": overall_promising,
        "advantage_strata": advantage_strata,
        "final_adoption_decision": None,
        "next_step": (
            "run_paired_agent_phase_b_for_identified_advantage"
            if assessment == "PROMISING"
            else "document_keep_laplace_map_result"
        ),
    }
