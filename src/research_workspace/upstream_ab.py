"""Governed evidence helpers for v3 upstream A/B evaluations.

Step 3.3 Phase A is deliberately map-only.  It may classify Aider RepoMap as
PROMISING or NOT_PROMISING, but it cannot produce the final ADOPT decision.
Final ADOPT/KEEP_LAPLACE requires paired agent-level evidence with the full
roadmap metric set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]

V33_BASELINE_REVISION = "25b2d7264e153e9214ade1cb3e98db81754c8e36"
AIDER_REPOMAP_REVISION = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"

PhaseAAssessment = Literal["PROMISING", "NOT_PROMISING"]
FinalDecision = Literal["ADOPT", "KEEP_LAPLACE"]


class UpstreamABError(RuntimeError):
    """A/B evidence, provenance, or filesystem boundary is invalid."""


def _git(
    repo: Path,
    *args: str,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=text,
        check=False,
        timeout=60,
    )


def exact_head(repo: Path) -> str:
    result = cast(subprocess.CompletedProcess[str], _git(repo, "rev-parse", "HEAD"))
    if result.returncode != 0:
        raise UpstreamABError("v33_git_revision_failed")
    return result.stdout.strip()


def require_v33_baseline(repo: Path) -> str:
    revision = exact_head(repo)
    if revision != V33_BASELINE_REVISION:
        raise UpstreamABError(
            f"v33_baseline_mismatch:expected={V33_BASELINE_REVISION}:actual={revision}"
        )
    return revision


def runtime_executable_path(
    repo: Path,
    requested: Path,
    *,
    runtime_root: Path,
) -> Path:
    """Normalize an executable path without dereferencing venv symlinks."""

    repo = repo.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    candidate = requested.expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise UpstreamABError("v33_runtime_executable_invalid")
    try:
        candidate.relative_to(runtime_root)
    except ValueError as exc:
        raise UpstreamABError("v33_runtime_executable_outside_runtime") from exc
    return candidate


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise UpstreamABError("v33_snapshot_path_invalid")
    return path


def materialize_head_snapshot(
    repo: Path,
    target: Path,
    *,
    expected_revision: str = V33_BASELINE_REVISION,
) -> tuple[str, str]:
    """Materialize regular tracked blobs from an immutable Git commit.

    Working-tree bytes, untracked files, submodules and symlinks are never
    copied.  This makes the baseline and candidate inputs independent of a dirty
    development checkout.
    """

    repo = repo.resolve(strict=True)
    revision = exact_head(repo)
    if revision != expected_revision:
        raise UpstreamABError(
            f"v33_snapshot_revision_mismatch:expected={expected_revision}:actual={revision}"
        )

    listing = cast(
        subprocess.CompletedProcess[bytes],
        _git(repo, "ls-tree", "-rz", "--full-tree", revision, text=False),
    )
    if listing.returncode != 0:
        raise UpstreamABError("v33_git_ls_tree_failed")

    target = target.resolve()
    if target == repo or repo not in target.parents:
        # Phase-A artifacts must remain under the development repository.
        raise UpstreamABError("v33_snapshot_target_outside_repository")
    if ".runtime" not in target.relative_to(repo).parts:
        raise UpstreamABError("v33_snapshot_target_not_runtime")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    digest = hashlib.sha256()
    count = 0
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
            relative_text = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise UpstreamABError("v33_git_tree_record_invalid") from exc
        if object_type != "blob" or mode == "120000":
            continue
        relative = _safe_relative(relative_text)
        blob = cast(
            subprocess.CompletedProcess[bytes],
            _git(repo, "cat-file", "blob", object_id, text=False),
        )
        if blob.returncode != 0:
            raise UpstreamABError("v33_git_blob_read_failed")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob.stdout)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(object_id.encode("ascii"))
        digest.update(b"\0")
        count += 1
    if count == 0:
        raise UpstreamABError("v33_snapshot_empty")
    return revision, digest.hexdigest()


def runtime_output_path(repo: Path, requested: Path | None, *, filename: str) -> Path:
    root = (repo.resolve() / ".runtime" / "v33-aider" / "results").resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / filename if requested is None else (
        requested if requested.is_absolute() else repo.resolve() / requested
    )
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise UpstreamABError("v33_output_outside_runtime_results") from exc
    return candidate


def manifest_sha256(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise UpstreamABError("v33_manifest_unreadable") from exc
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class PhaseATask:
    task_id: str
    query: str
    focus_paths: tuple[str, ...]
    expected_paths: tuple[str, ...]
    expected_terms: tuple[str, ...]
    token_budget: int


def load_phase_a_tasks(path: Path) -> tuple[PhaseATask, ...]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamABError("v33_manifest_invalid") from exc
    if not isinstance(raw, dict):
        raise UpstreamABError("v33_manifest_not_object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema") != "laplace-v33-aider-repomap-tasks-v1":
        raise UpstreamABError("v33_manifest_schema_mismatch")
    if payload.get("laplace_revision") != V33_BASELINE_REVISION:
        raise UpstreamABError("v33_manifest_laplace_revision_mismatch")
    if payload.get("aider_revision") != AIDER_REPOMAP_REVISION:
        raise UpstreamABError("v33_manifest_aider_revision_mismatch")
    records = payload.get("tasks")
    if not isinstance(records, list) or not 10 <= len(records) <= 20:
        raise UpstreamABError("v33_manifest_task_count_invalid")

    tasks: list[PhaseATask] = []
    seen: set[str] = set()
    expected_keys = {
        "task_id",
        "query",
        "focus_paths",
        "expected_paths",
        "expected_terms",
        "token_budget",
    }
    for raw_task in records:
        if not isinstance(raw_task, dict) or set(raw_task) != expected_keys:
            raise UpstreamABError("v33_manifest_task_schema_mismatch")
        task = cast(dict[str, object], raw_task)
        task_id = task["task_id"]
        query = task["query"]
        token_budget = task["token_budget"]
        if (
            not isinstance(task_id, str)
            or not task_id
            or len(task_id) > 120
            or task_id in seen
            or not isinstance(query, str)
            or not query.strip()
            or len(query) > 4000
            or isinstance(token_budget, bool)
            or not isinstance(token_budget, int)
            or not 1 <= token_budget <= 100_000
        ):
            raise UpstreamABError("v33_manifest_task_fields_invalid")

        focus = _string_list_field(task, "focus_paths")
        paths = _string_list_field(task, "expected_paths")
        terms = _string_list_field(task, "expected_terms")
        for item in (*focus, *paths):
            _safe_relative(item)
        seen.add(task_id)
        tasks.append(
            PhaseATask(
                task_id=task_id,
                query=query,
                focus_paths=focus,
                expected_paths=paths,
                expected_terms=terms,
                token_budget=token_budget,
            )
        )
    return tuple(tasks)


def char4_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _string_list_field(task: dict[str, object], label: str) -> tuple[str, ...]:
    value = task[label]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise UpstreamABError(f"v33_manifest_{label}_invalid")
    return tuple(cast(list[str], value))


def map_quality(text: str, task: PhaseATask) -> JsonObject:
    lowered = text.casefold()
    path_hits = sum(1 for item in task.expected_paths if item in text)
    term_hits = sum(1 for item in task.expected_terms if item.casefold() in lowered)
    return {
        "context_tokens": char4_tokens(text),
        "chars": len(text),
        "path_recall": path_hits / len(task.expected_paths),
        "term_recall": term_hits / len(task.expected_terms),
        "expected_paths_hit": path_hits,
        "expected_paths_total": len(task.expected_paths),
        "expected_terms_hit": term_hits,
        "expected_terms_total": len(task.expected_terms),
    }


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamABError(f"ab_{label}_invalid")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise UpstreamABError(f"ab_{label}_invalid")
    return number


def assess_phase_a(trials: object, *, margin: float = 0.05) -> JsonObject:
    """Classify map-only evidence without making an adoption decision."""

    if not isinstance(trials, list) or len(trials) < 10:
        raise UpstreamABError("v33_phase_a_insufficient_tasks")
    margin = _finite_number(margin, "margin")
    baseline_context: list[float] = []
    candidate_context: list[float] = []
    baseline_path: list[float] = []
    candidate_path: list[float] = []
    baseline_term: list[float] = []
    candidate_term: list[float] = []
    baseline_wall: list[float] = []
    candidate_wall: list[float] = []

    for raw in trials:
        if not isinstance(raw, dict):
            raise UpstreamABError("v33_phase_a_trial_invalid")
        row = cast(dict[str, object], raw)
        if row.get("snapshot_match") is not True:
            raise UpstreamABError("v33_phase_a_snapshot_mismatch")
        baseline = row.get("baseline")
        candidate = row.get("candidate")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise UpstreamABError("v33_phase_a_provider_metrics_invalid")
        b = cast(dict[str, object], baseline)
        c = cast(dict[str, object], candidate)
        baseline_context.append(_finite_number(b.get("context_tokens"), "context_tokens"))
        candidate_context.append(_finite_number(c.get("context_tokens"), "context_tokens"))
        baseline_path.append(_finite_number(b.get("path_recall"), "path_recall"))
        candidate_path.append(_finite_number(c.get("path_recall"), "path_recall"))
        baseline_term.append(_finite_number(b.get("term_recall"), "term_recall"))
        candidate_term.append(_finite_number(c.get("term_recall"), "term_recall"))
        baseline_wall.append(_finite_number(b.get("wall_time_seconds"), "wall_time_seconds"))
        candidate_wall.append(_finite_number(c.get("wall_time_seconds"), "wall_time_seconds"))

    aggregates: dict[str, dict[str, float]] = {
        "baseline": {
            "context_tokens": fmean(baseline_context),
            "path_recall": fmean(baseline_path),
            "term_recall": fmean(baseline_term),
            "wall_time_seconds": fmean(baseline_wall),
        },
        "candidate": {
            "context_tokens": fmean(candidate_context),
            "path_recall": fmean(candidate_path),
            "term_recall": fmean(candidate_term),
            "wall_time_seconds": fmean(candidate_wall),
        },
    }
    baseline_metrics = aggregates["baseline"]
    candidate_metrics = aggregates["candidate"]
    quality_ok = (
        candidate_metrics["path_recall"] + margin >= baseline_metrics["path_recall"]
        and candidate_metrics["term_recall"] + margin >= baseline_metrics["term_recall"]
    )
    material_gain = (
        candidate_metrics["context_tokens"]
        <= baseline_metrics["context_tokens"] * (1.0 - margin)
        or candidate_metrics["path_recall"]
        >= baseline_metrics["path_recall"] + margin
        or candidate_metrics["term_recall"]
        >= baseline_metrics["term_recall"] + margin
    )
    assessment: PhaseAAssessment = (
        "PROMISING" if quality_ok and material_gain else "NOT_PROMISING"
    )
    return {
        "assessment": assessment,
        "phase": "A_MAP_ONLY",
        "tasks": len(trials),
        "margin": margin,
        "aggregates": aggregates,
        "final_adoption_decision": None,
        "next_step": (
            "run_paired_agent_phase_b"
            if assessment == "PROMISING"
            else "document_keep_laplace_map_result"
        ),
    }


@dataclass(frozen=True)
class TrialMetrics:
    """Phase-B paired agent evidence for the final roadmap decision."""

    task_id: str
    provider: str
    laplace_revision: str
    candidate_revision: str | None
    task_manifest_sha256: str
    snapshot_sha256: str
    correctness: float
    context_tokens: int
    completion_tokens: int
    tool_rounds: int
    wall_time_seconds: float
    failed: bool
    verifier_success: bool
    repository_coverage: float
    relevance: float

    @classmethod
    def from_json(cls, raw: object) -> TrialMetrics:
        if not isinstance(raw, dict):
            raise UpstreamABError("ab_trial_not_object")
        values = cast(dict[str, object], raw)
        expected = {
            "task_id",
            "provider",
            "laplace_revision",
            "candidate_revision",
            "task_manifest_sha256",
            "snapshot_sha256",
            "correctness",
            "context_tokens",
            "completion_tokens",
            "tool_rounds",
            "wall_time_seconds",
            "failed",
            "verifier_success",
            "repository_coverage",
            "relevance",
        }
        if set(values) != expected:
            raise UpstreamABError("ab_trial_schema_mismatch")

        def nonempty(label: str, maximum: int = 200) -> str:
            value = values[label]
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise UpstreamABError(f"ab_trial_{label}_invalid")
            return value

        def integer(label: str) -> int:
            value = values[label]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UpstreamABError(f"ab_trial_{label}_invalid")
            return value

        candidate_revision = values["candidate_revision"]
        if candidate_revision is not None and (
            not isinstance(candidate_revision, str) or len(candidate_revision) != 40
        ):
            raise UpstreamABError("ab_trial_candidate_revision_invalid")
        failed = values["failed"]
        verifier_success = values["verifier_success"]
        if not isinstance(failed, bool) or not isinstance(verifier_success, bool):
            raise UpstreamABError("ab_trial_boolean_invalid")

        correctness = _finite_number(values["correctness"], "trial_correctness")
        coverage = _finite_number(values["repository_coverage"], "trial_repository_coverage")
        relevance = _finite_number(values["relevance"], "trial_relevance")
        if correctness > 1.0 or coverage > 1.0 or relevance > 1.0:
            raise UpstreamABError("ab_trial_fraction_invalid")

        return cls(
            task_id=nonempty("task_id"),
            provider=nonempty("provider"),
            laplace_revision=nonempty("laplace_revision", 40),
            candidate_revision=candidate_revision,
            task_manifest_sha256=nonempty("task_manifest_sha256", 64),
            snapshot_sha256=nonempty("snapshot_sha256", 64),
            correctness=correctness,
            context_tokens=integer("context_tokens"),
            completion_tokens=integer("completion_tokens"),
            tool_rounds=integer("tool_rounds"),
            wall_time_seconds=_finite_number(
                values["wall_time_seconds"], "trial_wall_time_seconds"
            ),
            failed=failed,
            verifier_success=verifier_success,
            repository_coverage=coverage,
            relevance=relevance,
        )

    def to_json(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "provider": self.provider,
            "laplace_revision": self.laplace_revision,
            "candidate_revision": self.candidate_revision,
            "task_manifest_sha256": self.task_manifest_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "correctness": self.correctness,
            "context_tokens": self.context_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_rounds": self.tool_rounds,
            "wall_time_seconds": self.wall_time_seconds,
            "failed": self.failed,
            "verifier_success": self.verifier_success,
            "repository_coverage": self.repository_coverage,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class AggregateMetrics:
    tasks: int
    correctness: float
    context_tokens: float
    completion_tokens: float
    tool_rounds: float
    wall_time_seconds: float
    failure_rate: float
    verifier_success_rate: float
    repository_coverage: float
    relevance: float

    def to_json(self) -> JsonObject:
        return {
            "tasks": self.tasks,
            "correctness": self.correctness,
            "context_tokens": self.context_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_rounds": self.tool_rounds,
            "wall_time_seconds": self.wall_time_seconds,
            "failure_rate": self.failure_rate,
            "verifier_success_rate": self.verifier_success_rate,
            "repository_coverage": self.repository_coverage,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class ABComparison:
    decision: FinalDecision
    baseline: AggregateMetrics
    candidate: AggregateMetrics
    efficiency_margin: float
    reasons: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "decision": self.decision,
            "baseline": self.baseline.to_json(),
            "candidate": self.candidate.to_json(),
            "efficiency_margin": self.efficiency_margin,
            "reasons": list(self.reasons),
        }


def load_trials(path: Path) -> tuple[TrialMetrics, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UpstreamABError("ab_trial_file_unreadable") from exc
    trials: list[TrialMetrics] = []
    seen: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            trial = TrialMetrics.from_json(json.loads(line))
        except json.JSONDecodeError as exc:
            raise UpstreamABError("ab_trial_json_invalid") from exc
        if trial.task_id in seen:
            raise UpstreamABError("ab_trial_task_duplicate")
        seen.add(trial.task_id)
        trials.append(trial)
    return tuple(trials)


def _aggregate(trials: tuple[TrialMetrics, ...]) -> AggregateMetrics:
    if not trials:
        raise UpstreamABError("ab_trials_empty")
    return AggregateMetrics(
        tasks=len(trials),
        correctness=fmean(item.correctness for item in trials),
        context_tokens=fmean(item.context_tokens for item in trials),
        completion_tokens=fmean(item.completion_tokens for item in trials),
        tool_rounds=fmean(item.tool_rounds for item in trials),
        wall_time_seconds=fmean(item.wall_time_seconds for item in trials),
        failure_rate=fmean(1.0 if item.failed else 0.0 for item in trials),
        verifier_success_rate=fmean(1.0 if item.verifier_success else 0.0 for item in trials),
        repository_coverage=fmean(item.repository_coverage for item in trials),
        relevance=fmean(item.relevance for item in trials),
    )


def compare_trials(
    baseline: tuple[TrialMetrics, ...],
    candidate: tuple[TrialMetrics, ...],
    *,
    minimum_tasks: int = 10,
    efficiency_margin: float = 0.05,
) -> ABComparison:
    """Make the final conservative Phase-B ADOPT/KEEP_LAPLACE decision."""

    if minimum_tasks < 10:
        raise UpstreamABError("ab_minimum_tasks_below_roadmap")
    margin = _finite_number(efficiency_margin, "efficiency_margin")
    if margin >= 1.0:
        raise UpstreamABError("ab_efficiency_margin_invalid")
    if len(baseline) < minimum_tasks or len(candidate) < minimum_tasks:
        raise UpstreamABError("ab_trial_count_insufficient")

    by_base = {item.task_id: item for item in baseline}
    by_candidate = {item.task_id: item for item in candidate}
    if set(by_base) != set(by_candidate):
        raise UpstreamABError("ab_trial_tasks_unpaired")

    for task_id in sorted(by_base):
        left = by_base[task_id]
        right = by_candidate[task_id]
        if left.provider != "laplace" or right.provider != "aider-repomap-v33":
            raise UpstreamABError("ab_trial_provider_mismatch")
        if left.candidate_revision is not None:
            raise UpstreamABError("ab_baseline_candidate_revision_present")
        if right.candidate_revision != AIDER_REPOMAP_REVISION:
            raise UpstreamABError("ab_candidate_revision_mismatch")
        if (
            left.laplace_revision != right.laplace_revision
            or left.task_manifest_sha256 != right.task_manifest_sha256
            or left.snapshot_sha256 != right.snapshot_sha256
        ):
            raise UpstreamABError("ab_trial_provenance_mismatch")

    base = _aggregate(baseline)
    cand = _aggregate(candidate)
    reasons: list[str] = []

    if cand.correctness < base.correctness:
        reasons.append("correctness_regression")
    if cand.verifier_success_rate < base.verifier_success_rate:
        reasons.append("verifier_success_regression")
    if cand.failure_rate > base.failure_rate:
        reasons.append("failure_rate_regression")
    if cand.repository_coverage < base.repository_coverage:
        reasons.append("repository_coverage_regression")
    if cand.relevance < base.relevance:
        reasons.append("relevance_regression")

    efficiency = {
        "context_tokens": (base.context_tokens, cand.context_tokens),
        "completion_tokens": (base.completion_tokens, cand.completion_tokens),
        "tool_rounds": (base.tool_rounds, cand.tool_rounds),
        "wall_time_seconds": (base.wall_time_seconds, cand.wall_time_seconds),
    }
    improved = False
    for label, (before, after) in efficiency.items():
        if before == 0.0:
            if after > 0.0:
                reasons.append(f"{label}_regression")
            continue
        delta = (after - before) / before
        if delta > margin:
            reasons.append(f"{label}_regression")
        if delta <= -margin:
            improved = True
            reasons.append(f"{label}_improved")

    blocking = [item for item in reasons if item.endswith("_regression")]
    decision: FinalDecision = "ADOPT" if not blocking and improved else "KEEP_LAPLACE"
    if not improved:
        reasons.append("no_material_efficiency_improvement")
    return ABComparison(
        decision=decision,
        baseline=base,
        candidate=cand,
        efficiency_margin=margin,
        reasons=tuple(reasons),
    )
