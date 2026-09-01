"""Evidence-gated SiliconMind qualification for Laplace roadmap Step C2.

This module deliberately reuses the existing multilingual RTL task definitions,
held-out evaluator, local-team verifier, serving-profile resolver, and owned
vLLM lifecycle.  It does not mutate the historical CodeV experiment or the
frozen Step-B P8 profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess  # nosec B404 - fixed argv, local model/runtime control only
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Sequence, cast

from .engineering import (
    AgentTaskStore,
    EngineeringError,
    JsonObject,
    normalize_task_spec,
    write_json_atomic,
)
from .inference import ServingCandidate
from .model_routing import DualModelConfiguration
from .multilanguage_ablation import (
    AblationArm,
    AblationTask,
    ExperimentConfiguration,
    _evaluate_held_out,
    _exception_failure_category,
    _failure_outcome_kind,
    _task_spec,
    load_benchmark_manifest,
    validate_held_out_pack,
)
from .serving_profile_runtime import (
    GpuSnapshot,
    ServingProfileRuntime,
    ServingRuntimeError,
    observe_gpu,
)
from .serving_profiles import (
    InstalledServingCapabilities,
    KVCacheDType,
    ServingProfile,
    endpoint_for,
    resolve_profile,
)
from .team_runner import LocalTeamRunner, TeamWorkflowOptions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "rtl_specialist_candidates"
    / "siliconmind_qwen3_4b_step_c2.json"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
CandidateId = Literal[
    "siliconmind_qwen3_4b_t_2507_36k",
    "siliconmind_qwen3_4b_t_2507_76k",
    "siliconmind_v12_qwen3_4b_t_2507",
]
RoleMode = Literal["direct", "five_role"]
EXPECTED_BRANCH = "work/laplace-v3-refactor-certified-20260831"


class SpecialistQualificationError(EngineeringError):
    """Step-C2 configuration, evidence, or runtime state is invalid."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: CandidateId
    repository: str
    profile_id: str
    served_model_name: str


@dataclass(frozen=True)
class StepC2Configuration:
    path: Path
    preparation_base_commit: str
    c1_manifest_path: Path
    c1_manifest_git_blob_sha: str
    benchmark_manifest_path: Path
    p8_profile_path: Path
    p8_profile_git_blob_sha: str
    output_root: Path
    vllm_executable: Path
    vllm_python: Path
    ffmpeg_library_path: Path
    held_out_environment_variable: str
    specialist_port: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: KVCacheDType
    kv_cache_memory_bytes: int
    calculate_kv_scales: bool
    gpu_memory_utilization: float
    startup_timeout: int
    request_timeout: int
    max_output_tokens: int
    temperature: float
    reasoning_parser: str
    thinking_token_budget: int
    use_v2_model_runner: bool
    worker_task_ids: tuple[str, ...]
    coexistence_task_ids: tuple[str, ...]
    minimum_free_headroom_mib: int
    candidates: tuple[CandidateSpec, ...]
    preselected_candidate: CandidateId | None

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def candidate(self, candidate_id: str) -> CandidateSpec:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise SpecialistQualificationError(f"Unknown SiliconMind candidate: {candidate_id}")


def _read_json(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecialistQualificationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecialistQualificationError(f"Expected JSON object in {path}")
    return dict(value)


def _exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SpecialistQualificationError(
            f"{label} keys are invalid; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecialistQualificationError(f"{label} must be non-empty text")
    return value


def _integer(value: object, *, label: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SpecialistQualificationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpecialistQualificationError(f"{label} must be numeric")
    return float(value)


def _repository_path(
    root: Path,
    value: object,
    *,
    label: str,
    follow_symlinks: bool = True,
) -> Path:
    raw = _non_empty_string(value, label=label)
    candidate = root / raw
    path = (
        candidate.resolve()
        if follow_symlinks
        else Path(os.path.abspath(candidate))
    )
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SpecialistQualificationError(f"{label} escapes the repository root") from exc
    return path


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    payload = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    # Git object identity, not a security digest.
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()  # nosec B324


def _require_blob(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise SpecialistQualificationError(f"{label} is missing: {path}")
    actual = _git_blob_sha(path)
    if actual != expected:
        raise SpecialistQualificationError(
            f"{label} drifted: expected Git blob {expected}, found {actual}"
        )


def load_step_c2_configuration(
    repository_root: Path = ROOT,
    path: Path = DEFAULT_CONFIG,
) -> StepC2Configuration:
    root = repository_root.resolve()
    config_path = path.resolve()
    value = _read_json(config_path)
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "roadmap_step",
            "preparation_base_commit",
            "c1_manifest",
            "candidate_scope_extension",
            "benchmark_manifest",
            "runtime",
            "specialist_serving",
            "candidates",
            "qualification",
            "coexistence",
            "promotion_gate",
            "boundaries",
        },
        label="Step C2 configuration",
    )
    if (
        value.get("schema_version") != 1
        or value.get("status") != "live_qualification_prepared"
        or value.get("roadmap_step") != "C2"
    ):
        raise SpecialistQualificationError("Step C2 identity is invalid")
    preparation_base = _non_empty_string(
        value.get("preparation_base_commit"), label="preparation_base_commit"
    )
    if not _COMMIT.fullmatch(preparation_base):
        raise SpecialistQualificationError("preparation_base_commit must be an exact commit")

    c1_raw = value.get("c1_manifest")
    extension_raw = value.get("candidate_scope_extension")
    benchmark_raw = value.get("benchmark_manifest")
    runtime_raw = value.get("runtime")
    serving_raw = value.get("specialist_serving")
    qualification_raw = value.get("qualification")
    coexistence_raw = value.get("coexistence")
    promotion_raw = value.get("promotion_gate")
    boundaries_raw = value.get("boundaries")
    if not all(
        isinstance(item, dict)
        for item in (
            c1_raw,
            extension_raw,
            benchmark_raw,
            runtime_raw,
            serving_raw,
            qualification_raw,
            coexistence_raw,
            promotion_raw,
            boundaries_raw,
        )
    ):
        raise SpecialistQualificationError("Step C2 object sections are malformed")
    c1 = cast(dict[str, object], c1_raw)
    extension = cast(dict[str, object], extension_raw)
    benchmark = cast(dict[str, object], benchmark_raw)
    runtime = cast(dict[str, object], runtime_raw)
    serving = cast(dict[str, object], serving_raw)
    qualification = cast(dict[str, object], qualification_raw)
    coexistence = cast(dict[str, object], coexistence_raw)
    promotion = cast(dict[str, object], promotion_raw)
    boundaries = cast(dict[str, object], boundaries_raw)

    _exact_keys(c1, {"path", "git_blob_sha"}, label="c1_manifest")
    _exact_keys(
        extension,
        {"discovery_date", "reason", "source", "candidate_ids"},
        label="candidate_scope_extension",
    )
    if (
        extension.get("discovery_date") != "2026-09-01"
        or extension.get("reason")
        != "official_upstream_v1_2_discovered_after_c1_freeze"
        or extension.get("source")
        != "AS-SiliconMind official Hugging Face collection"
        or extension.get("candidate_ids")
        != ["siliconmind_v12_qwen3_4b_t_2507"]
    ):
        raise SpecialistQualificationError(
            "C2 candidate-scope extension is invalid"
        )
    _exact_keys(
        benchmark,
        {"path", "worker_task_ids", "preserve_task_order"},
        label="benchmark_manifest",
    )
    _exact_keys(
        runtime,
        {
            "output_root",
            "vllm_executable",
            "vllm_python",
            "ffmpeg_library_path",
            "held_out_environment_variable",
        },
        label="runtime",
    )
    _exact_keys(
        serving,
        {
            "port",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "kv_cache_dtype",
            "kv_cache_memory_bytes",
            "calculate_kv_scales",
            "gpu_memory_utilization",
            "startup_timeout",
            "request_timeout",
            "max_output_tokens",
            "temperature",
            "dtype",
            "prefix_caching",
            "chunked_prefill",
            "reasoning_parser",
            "thinking_token_budget",
            "use_v2_model_runner",
        },
        label="specialist_serving",
    )
    _exact_keys(
        qualification,
        {
            "role_mode",
            "retrieval_mode",
            "fallback_to_main",
            "worker_reasoning_mode",
            "require_clean_gpu_before_start",
            "require_held_out",
            "require_deterministic_verification",
        },
        label="qualification",
    )
    _exact_keys(
        coexistence,
        {
            "p8_profile",
            "start_order",
            "integration_task_ids",
            "minimum_free_headroom_mib",
            "require_main_and_worker_routes",
            "require_held_out",
        },
        label="coexistence",
    )
    _exact_keys(
        promotion,
        {
            "preselected_candidate",
            "primary_metric",
            "secondary_metric",
            "tie_breakers",
            "tie_after_all_metrics",
            "requires_coexistence_certification",
            "requires_clean_owned_server_release",
        },
        label="promotion_gate",
    )
    _exact_keys(
        boundaries,
        {
            "modify_historical_codev_artifacts",
            "preserve_historical_ablation_semantics",
            "modify_p8_profile",
            "auto_promote_without_live_evidence",
            "three_model_lifecycle_is_step_e",
        },
        label="boundaries",
    )

    if benchmark.get("preserve_task_order") is not True:
        raise SpecialistQualificationError("C2 must preserve C1 worker-task order")
    raw_worker_ids = benchmark.get("worker_task_ids")
    if not isinstance(raw_worker_ids, list) or not all(
        isinstance(item, str) and item for item in raw_worker_ids
    ):
        raise SpecialistQualificationError("worker_task_ids must be non-empty strings")
    worker_ids = tuple(cast(list[str], raw_worker_ids))
    if len(worker_ids) != 11 or len(worker_ids) != len(set(worker_ids)):
        raise SpecialistQualificationError("C2 must contain exactly 11 unique worker tasks")

    if (
        qualification.get("role_mode") != "direct"
        or qualification.get("retrieval_mode") != "none"
        or qualification.get("fallback_to_main") is not False
        or qualification.get("worker_reasoning_mode") != "model_default"
        or qualification.get("require_clean_gpu_before_start") is not True
        or qualification.get("require_held_out") is not True
        or qualification.get("require_deterministic_verification") is not True
    ):
        raise SpecialistQualificationError("C2 qualification policy drifted from C1")
    if (
        serving.get("dtype") != "bfloat16"
        or serving.get("kv_cache_dtype") != "fp8"
        or serving.get("calculate_kv_scales") is not True
        or serving.get("prefix_caching") is not True
        or serving.get("chunked_prefill") is not True
        or serving.get("reasoning_parser") != "qwen3"
        or serving.get("use_v2_model_runner") is not False
    ):
        raise SpecialistQualificationError("C2 specialist serving policy is invalid")
    max_output_tokens = _integer(
        serving.get("max_output_tokens"), label="max_output_tokens"
    )
    if max_output_tokens > 8192:
        raise SpecialistQualificationError("C2 may not widen Laplace max_output_tokens")
    thinking_token_budget = _integer(
        serving.get("thinking_token_budget"), label="thinking_token_budget"
    )
    if thinking_token_budget >= max_output_tokens:
        raise SpecialistQualificationError(
            "C2 thinking budget must leave capacity for final RTL source"
        )

    p8_raw = coexistence.get("p8_profile")
    if not isinstance(p8_raw, dict):
        raise SpecialistQualificationError("coexistence.p8_profile must be an object")
    p8 = dict(p8_raw)
    _exact_keys(p8, {"path", "git_blob_sha"}, label="coexistence.p8_profile")
    if coexistence.get("start_order") != ["p8", "specialist"]:
        raise SpecialistQualificationError("C2 coexistence must start P8 before the specialist")
    raw_integration_ids = coexistence.get("integration_task_ids")
    if not isinstance(raw_integration_ids, list) or not all(
        isinstance(item, str) and item for item in raw_integration_ids
    ):
        raise SpecialistQualificationError("integration_task_ids must be strings")
    integration_ids = tuple(cast(list[str], raw_integration_ids))
    if not integration_ids or not set(integration_ids).issubset(worker_ids):
        raise SpecialistQualificationError(
            "Coexistence tasks must be a non-empty worker-task subset"
        )
    if coexistence.get("require_main_and_worker_routes") is not True:
        raise SpecialistQualificationError(
            "Coexistence must exercise both P8 and specialist routes"
        )
    if coexistence.get("require_held_out") is not True:
        raise SpecialistQualificationError("Coexistence must retain held-out evaluation")

    if (
        promotion.get("preselected_candidate")
        != "siliconmind_qwen3_4b_t_2507_36k"
        or promotion.get("primary_metric") != "deterministic_task_pass_count"
        or promotion.get("secondary_metric") != "held_out_task_pass_count"
        or promotion.get("tie_breakers")
        != ["lower_peak_gpu_memory_mib", "higher_output_tokens_per_second"]
        or promotion.get("tie_after_all_metrics") != "no_promotion"
        or promotion.get("requires_coexistence_certification") is not True
        or promotion.get("requires_clean_owned_server_release") is not True
    ):
        raise SpecialistQualificationError("C2 promotion gate drifted from C1")
    if (
        any(
            boundaries.get(key) is not False
            for key in (
                "modify_historical_codev_artifacts",
                "modify_p8_profile",
                "auto_promote_without_live_evidence",
            )
        )
        or boundaries.get("preserve_historical_ablation_semantics") is not True
        or boundaries.get("three_model_lifecycle_is_step_e") is not True
    ):
        raise SpecialistQualificationError("C2 roadmap boundaries are invalid")

    candidates_raw = value.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 3:
        raise SpecialistQualificationError(
            "C2 requires exactly three SiliconMind 4B candidates"
        )
    candidates: list[CandidateSpec] = []
    expected_ids: tuple[CandidateId, ...] = (
        "siliconmind_qwen3_4b_t_2507_36k",
        "siliconmind_qwen3_4b_t_2507_76k",
        "siliconmind_v12_qwen3_4b_t_2507",
    )
    for index, raw_candidate in enumerate(candidates_raw):
        if not isinstance(raw_candidate, dict):
            raise SpecialistQualificationError("Candidate entries must be objects")
        candidate = dict(raw_candidate)
        _exact_keys(
            candidate,
            {"candidate_id", "repository", "profile_id", "served_model_name"},
            label=f"candidate[{index}]",
        )
        candidate_id = candidate.get("candidate_id")
        if candidate_id != expected_ids[index]:
            raise SpecialistQualificationError(
                "Candidate order or identity drifted from C2 scope"
            )
        repository = _non_empty_string(candidate.get("repository"), label="candidate repository")
        if not _REPOSITORY.fullmatch(repository):
            raise SpecialistQualificationError("Candidate repository identifier is unsafe")
        candidates.append(
            CandidateSpec(
                candidate_id=cast(CandidateId, candidate_id),
                repository=repository,
                profile_id=_non_empty_string(candidate.get("profile_id"), label="profile_id"),
                served_model_name=_non_empty_string(
                    candidate.get("served_model_name"), label="served_model_name"
                ),
            )
        )

    c1_path = _repository_path(root, c1.get("path"), label="c1_manifest.path")
    c1_blob = _non_empty_string(c1.get("git_blob_sha"), label="c1_manifest.git_blob_sha")
    p8_path = _repository_path(root, p8.get("path"), label="p8_profile.path")
    p8_blob = _non_empty_string(p8.get("git_blob_sha"), label="p8_profile.git_blob_sha")
    benchmark_path = _repository_path(root, benchmark.get("path"), label="benchmark_manifest.path")
    _require_blob(c1_path, c1_blob, label="C1 manifest")
    _require_blob(p8_path, p8_blob, label="P8 profile")
    if not benchmark_path.is_file():
        raise SpecialistQualificationError("Benchmark manifest is missing")

    c1_value = _read_json(c1_path)
    c1_candidates = c1_value.get("candidates")
    if not isinstance(c1_candidates, list):
        raise SpecialistQualificationError("C1 candidate list is malformed")
    c1_pairs = [
        (item.get("candidate_id"), item.get("repository"))
        for item in c1_candidates
        if isinstance(item, dict)
    ]
    frozen_c1_pairs = [
        (item.candidate_id, item.repository)
        for item in candidates[:2]
    ]
    if c1_pairs != frozen_c1_pairs:
        raise SpecialistQualificationError(
            "C2 does not preserve the frozen C1 candidate prefix"
        )
    if extension.get("candidate_ids") != [candidates[2].candidate_id]:
        raise SpecialistQualificationError(
            "C2 current-upstream extension disagrees with candidate set"
        )
    c1_manifest = c1_value.get("qualification_manifest")
    c1_worker_ids = c1_manifest.get("worker_task_ids") if isinstance(c1_manifest, dict) else None
    if c1_worker_ids != list(worker_ids):
        raise SpecialistQualificationError("C2 worker-task list disagrees with C1")

    manifest_tasks = load_benchmark_manifest(benchmark_path)
    manifest_worker_ids = tuple(
        task.task_id for task in manifest_tasks if task.routing.worker_eligible
    )
    if manifest_worker_ids != worker_ids:
        raise SpecialistQualificationError(
            "C2 worker-task list does not match the authoritative benchmark manifest"
        )

    output_root = _repository_path(root, runtime.get("output_root"), label="runtime.output_root")
    vllm_executable = _repository_path(
        root,
        runtime.get("vllm_executable"),
        label="runtime.vllm_executable",
        follow_symlinks=False,
    )
    vllm_python = _repository_path(
        root,
        runtime.get("vllm_python"),
        label="runtime.vllm_python",
        follow_symlinks=False,
    )
    ffmpeg_library_path = _repository_path(
        root,
        runtime.get("ffmpeg_library_path"),
        label="runtime.ffmpeg_library_path",
        follow_symlinks=False,
    )
    held_env = _non_empty_string(
        runtime.get("held_out_environment_variable"), label="held_out_environment_variable"
    )
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", held_env):
        raise SpecialistQualificationError("held_out_environment_variable is unsafe")

    return StepC2Configuration(
        path=config_path,
        preparation_base_commit=preparation_base,
        c1_manifest_path=c1_path,
        c1_manifest_git_blob_sha=c1_blob,
        benchmark_manifest_path=benchmark_path,
        p8_profile_path=p8_path,
        p8_profile_git_blob_sha=p8_blob,
        output_root=output_root,
        vllm_executable=vllm_executable,
        vllm_python=vllm_python,
        ffmpeg_library_path=ffmpeg_library_path,
        held_out_environment_variable=held_env,
        specialist_port=_integer(serving.get("port"), label="specialist port"),
        max_model_len=_integer(serving.get("max_model_len"), label="max_model_len"),
        max_num_seqs=_integer(serving.get("max_num_seqs"), label="max_num_seqs"),
        max_num_batched_tokens=_integer(
            serving.get("max_num_batched_tokens"), label="max_num_batched_tokens"
        ),
        kv_cache_dtype=cast(KVCacheDType, serving.get("kv_cache_dtype")),
        kv_cache_memory_bytes=_integer(
            serving.get("kv_cache_memory_bytes"), label="kv_cache_memory_bytes"
        ),
        calculate_kv_scales=serving.get("calculate_kv_scales") is True,
        gpu_memory_utilization=_number(
            serving.get("gpu_memory_utilization"), label="gpu_memory_utilization"
        ),
        startup_timeout=_integer(serving.get("startup_timeout"), label="startup_timeout"),
        request_timeout=_integer(serving.get("request_timeout"), label="request_timeout"),
        max_output_tokens=_integer(serving.get("max_output_tokens"), label="max_output_tokens"),
        temperature=_number(serving.get("temperature"), label="temperature"),
        reasoning_parser=_non_empty_string(
            serving.get("reasoning_parser"), label="reasoning_parser"
        ),
        thinking_token_budget=_integer(
            serving.get("thinking_token_budget"), label="thinking_token_budget"
        ),
        use_v2_model_runner=serving.get("use_v2_model_runner") is True,
        worker_task_ids=worker_ids,
        coexistence_task_ids=integration_ids,
        minimum_free_headroom_mib=_integer(
            coexistence.get("minimum_free_headroom_mib"), label="minimum_free_headroom_mib"
        ),
        candidates=tuple(candidates),
        preselected_candidate=cast(
            CandidateId, promotion.get("preselected_candidate")
        ),
    )



def _runtime_output_path(
    configuration: StepC2Configuration,
    path: Path,
    *,
    label: str,
) -> Path:
    target = path.resolve()
    try:
        target.relative_to(configuration.output_root.resolve())
    except ValueError as exc:
        raise SpecialistQualificationError(
            f"{label} must remain under {configuration.output_root}"
        ) from exc
    return target

def _git(repository_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise SpecialistQualificationError(
            f"git {' '.join(arguments)} failed: {(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stdout.strip()


def require_clean_execution_base(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
) -> None:
    if not _COMMIT.fullmatch(base_revision):
        raise SpecialistQualificationError("--base-revision must be an exact 40-character commit")
    branch = _git(repository_root, ["branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise SpecialistQualificationError(
            f"Step C2 live execution requires {EXPECTED_BRANCH}, found {branch}"
        )
    head = _git(repository_root, ["rev-parse", "HEAD"])
    if head != base_revision:
        raise SpecialistQualificationError(
            f"Execution base must equal clean HEAD: HEAD={head}, requested={base_revision}"
        )
    worktree_status = _git(
        repository_root, ["status", "--porcelain", "--untracked-files=normal"]
    )
    disallowed_status = [
        line
        for line in worktree_status.splitlines()
        if line.strip() != "?? .codex-step-c-finalize/"
    ]
    if disallowed_status:
        raise SpecialistQualificationError("Step C2 live execution requires a clean worktree")
    ancestor = subprocess.run(  # nosec B603
        [
            "git",
            "merge-base",
            "--is-ancestor",
            configuration.preparation_base_commit,
            base_revision,
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if ancestor.returncode != 0:
        raise SpecialistQualificationError(
            "Execution HEAD is not descended from the certified C1 preparation base"
        )


def parse_ls_remote_head(output: str) -> str:
    rows = [line.strip().split() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != "HEAD":
        raise SpecialistQualificationError(
            "Hugging Face HEAD resolution returned an unexpected shape"
        )
    revision = rows[0][0]
    if not _COMMIT.fullmatch(revision):
        raise SpecialistQualificationError("Hugging Face HEAD did not resolve to an exact commit")
    return revision


def resolve_candidate_revision(candidate: CandidateSpec) -> str:
    url = f"https://huggingface.co/{candidate.repository}"
    completed = subprocess.run(  # nosec B603
        ["git", "ls-remote", url, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise SpecialistQualificationError(
            f"Cannot resolve {candidate.repository}: {completed.stderr.strip()}"
        )
    return parse_ls_remote_head(completed.stdout)


def write_revision_lock(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
    output_path: Path,
) -> JsonObject:
    require_clean_execution_base(repository_root, configuration, base_revision)
    records = [
        {
            "candidate_id": candidate.candidate_id,
            "repository": candidate.repository,
            "ref": "HEAD",
            "revision": resolve_candidate_revision(candidate),
        }
        for candidate in _active_candidates(configuration)
    ]
    payload: JsonObject = {
        "schema_version": 1,
        "status": "IMMUTABLE_REVISIONS_LOCKED",
        "roadmap_step": "C2",
        "base_revision": base_revision,
        "configuration_sha256": configuration.config_sha256,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "candidates": records,
    }
    output = _runtime_output_path(
        configuration, output_path, label="revision-lock output"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload, readonly=True)
    return payload


def load_revision_lock(
    configuration: StepC2Configuration,
    path: Path,
    *,
    base_revision: str,
) -> JsonObject:
    lock_path = _runtime_output_path(configuration, path, label="revision lock")
    value = _read_json(lock_path)
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "roadmap_step",
            "base_revision",
            "configuration_sha256",
            "captured_at_utc",
            "candidates",
        },
        label="revision lock",
    )
    if (
        value.get("schema_version") != 1
        or value.get("status") != "IMMUTABLE_REVISIONS_LOCKED"
        or value.get("roadmap_step") != "C2"
        or value.get("base_revision") != base_revision
        or value.get("configuration_sha256") != configuration.config_sha256
    ):
        raise SpecialistQualificationError("Revision lock is incompatible with this C2 execution")
    raw_candidates = value.get("candidates")
    active_candidates = _active_candidates(configuration)
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(active_candidates):
        raise SpecialistQualificationError("Revision lock candidate set is malformed")
    expected = [(item.candidate_id, item.repository) for item in active_candidates]
    actual: list[tuple[object, object]] = []
    for item in raw_candidates:
        if not isinstance(item, dict) or item.get("ref") != "HEAD":
            raise SpecialistQualificationError("Revision lock candidate entry is malformed")
        revision = item.get("revision")
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise SpecialistQualificationError("Revision lock contains a non-immutable revision")
        actual.append((item.get("candidate_id"), item.get("repository")))
    if actual != expected:
        raise SpecialistQualificationError("Revision lock candidate identities drifted")
    return value


def _active_candidates(configuration: StepC2Configuration) -> tuple[CandidateSpec, ...]:
    """Return the published Step-C candidate without rewriting C1 provenance."""
    if configuration.preselected_candidate is None:
        return configuration.candidates
    return (configuration.candidate(configuration.preselected_candidate),)


def locked_revision(lock: JsonObject, candidate_id: str) -> str:
    raw = lock.get("candidates")
    if not isinstance(raw, list):
        raise SpecialistQualificationError("Revision lock has no candidates")
    for item in raw:
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
            revision = item.get("revision")
            if isinstance(revision, str) and _COMMIT.fullmatch(revision):
                return revision
    raise SpecialistQualificationError(f"Candidate revision is absent from lock: {candidate_id}")


def candidate_model_path(
    configuration: StepC2Configuration,
    candidate_id: str,
    revision: str,
) -> Path:
    if not _COMMIT.fullmatch(revision):
        raise SpecialistQualificationError("Model path requires an immutable revision")
    configuration.candidate(candidate_id)
    target = (
        configuration.output_root / "models" / candidate_id / revision
    ).resolve()
    try:
        target.relative_to(configuration.output_root.resolve())
    except ValueError as exc:
        raise SpecialistQualificationError(
            "Candidate model path escapes C2 runtime storage"
        ) from exc
    return target


def _snapshot_summary(target: Path) -> JsonObject:
    files = [path for path in target.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    critical: JsonObject = {}
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = target / name
        if path.is_file():
            critical[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file_count": len(files),
        "regular_file_bytes": total,
        "critical_file_sha256": critical,
    }


def _owned_snapshot_matches(
    marker: JsonObject,
    target: Path,
    candidate: CandidateSpec,
    revision: str,
) -> bool:
    """Check immutable identity and critical file content before a metadata rebind."""
    if (
        marker.get("candidate_id") != candidate.candidate_id
        or marker.get("repository") != candidate.repository
        or marker.get("revision") != revision
    ):
        return False
    prior_summary = marker.get("snapshot")
    if not isinstance(prior_summary, dict):
        return False
    prior_hashes = prior_summary.get("critical_file_sha256")
    if not isinstance(prior_hashes, dict):
        return False
    current = _snapshot_summary(target).get("critical_file_sha256")
    return isinstance(current, dict) and current == prior_hashes


def acquire_candidate_snapshot(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
    lock_path: Path,
    candidate_id: str,
) -> JsonObject:
    require_clean_execution_base(repository_root, configuration, base_revision)
    lock = load_revision_lock(configuration, lock_path, base_revision=base_revision)
    candidate = configuration.candidate(candidate_id)
    revision = locked_revision(lock, candidate_id)
    target = candidate_model_path(configuration, candidate_id, revision)
    marker = target / ".laplace-step-c2-snapshot.json"
    if marker.is_file():
        prior = _read_json(marker)
        if (
            prior.get("candidate_id") == candidate_id
            and prior.get("repository") == candidate.repository
            and prior.get("revision") == revision
            and prior.get("configuration_sha256") == configuration.config_sha256
            and prior.get("base_revision") == base_revision
            and (target / "config.json").is_file()
        ):
            return {**prior, "status": "REUSED_IMMUTABLE_SNAPSHOT"}
        if _owned_snapshot_matches(prior, target, candidate, revision):
            rebound: JsonObject = {
                **prior,
                "status": "IMMUTABLE_SNAPSHOT_REBOUND",
                "base_revision": base_revision,
                "configuration_sha256": configuration.config_sha256,
                "rebound_at_utc": datetime.now(UTC).isoformat(),
                "rebound_from": {
                    "base_revision": prior.get("base_revision"),
                    "configuration_sha256": prior.get("configuration_sha256"),
                },
            }
            write_json_atomic(marker, rebound, readonly=True)
            return rebound
        raise SpecialistQualificationError("Existing candidate snapshot marker is incompatible")
    if target.exists() and any(target.iterdir()):
        raise SpecialistQualificationError(
            f"Refusing to reuse unowned/non-empty model directory without marker: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    if not configuration.vllm_python.is_file():
        raise SpecialistQualificationError(
            f"vLLM Python environment is missing: {configuration.vllm_python}"
        )
    hf_home = (configuration.output_root / "hf-home").resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    script = (
        "from huggingface_hub import snapshot_download; import sys; "
        "print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))"
    )
    environment = dict(os.environ)
    environment["HF_HOME"] = str(hf_home)
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    completed = subprocess.run(  # nosec B603
        [
            str(configuration.vllm_python),
            "-c",
            script,
            candidate.repository,
            revision,
            str(target),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=14_400,
    )
    if completed.returncode != 0 or not (target / "config.json").is_file():
        raise SpecialistQualificationError(
            "Immutable snapshot acquisition failed: "
            + (completed.stderr or completed.stdout)[-4_000:]
        )
    payload: JsonObject = {
        "schema_version": 1,
        "status": "IMMUTABLE_SNAPSHOT_ACQUIRED",
        "candidate_id": candidate_id,
        "repository": candidate.repository,
        "revision": revision,
        "base_revision": base_revision,
        "configuration_sha256": configuration.config_sha256,
        "model_path": str(target),
        "hf_home": str(hf_home),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "snapshot": _snapshot_summary(target),
    }
    write_json_atomic(marker, payload, readonly=True)
    return payload


def build_specialist_profile(
    configuration: StepC2Configuration,
    candidate: CandidateSpec,
    model_path: Path,
) -> ServingProfile:
    return ServingProfile(
        profile_id=candidate.profile_id,
        model_route="economy",
        model_path=str(model_path.resolve()),
        served_model_name=candidate.served_model_name,
        port=configuration.specialist_port,
        max_model_len=configuration.max_model_len,
        max_num_seqs=configuration.max_num_seqs,
        max_num_batched_tokens=configuration.max_num_batched_tokens,
        kv_cache_dtype=configuration.kv_cache_dtype,
        kv_cache_memory_bytes=configuration.kv_cache_memory_bytes,
        enable_prefix_caching=True,
        prefix_hash_algorithm="sha256",
        enable_chunked_prefill=True,
        scheduling_policy="fcfs",
        cpu_offload_gb=0.0,
        cpu_offload_params=(),
        offload_backend="auto",
        offload_group_size=0,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        kv_offloading_size=None,
        kv_offloading_backend="native",
        gpu_memory_utilization=configuration.gpu_memory_utilization,
        startup_timeout=configuration.startup_timeout,
        request_timeout=configuration.request_timeout,
        extra_args=(
            "--dtype=bfloat16",
            f"--reasoning-parser={configuration.reasoning_parser}",
            "--calculate-kv-scales",
        ),
    )


def build_specialist_candidate(
    configuration: StepC2Configuration,
    candidate: CandidateSpec,
    profile: ServingProfile,
    revision: str,
) -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint=endpoint_for(profile),
        model=profile.served_model_name,
        revision=revision,
        quantization="none_bfloat16",
        kernel="vllm_native",
        prefix_caching=profile.enable_prefix_caching,
        chunked_prefill=profile.enable_chunked_prefill,
        cuda_graph_mode="vllm_default",
        scheduler=profile.scheduling_policy,
        model_path=profile.model_path,
        context_tokens=configuration.max_model_len,
        max_output_tokens=configuration.max_output_tokens,
        temperature=configuration.temperature,
        top_p=1.0,
        seed=0,
        request_timeout_seconds=configuration.request_timeout,
        context_safety_margin_tokens=512,
        minimum_completion_tokens=256,
        reviewer_max_output_tokens=768,
        thinking_token_budget=configuration.thinking_token_budget,
    )


def build_direct_configuration(candidate: ServingCandidate) -> DualModelConfiguration:
    return DualModelConfiguration(
        main=candidate,
        rtl_worker=candidate,
        worker_contract_retries=1,
        worker_response_retries=1,
        fallback_to_main=False,
        worker_reasoning_mode="model_default",
    )


def _installed_capabilities(
    executable: Path,
    ffmpeg_library_path: Path,
) -> InstalledServingCapabilities:
    if not executable.is_file():
        raise SpecialistQualificationError(f"vLLM executable is missing: {executable}")
    if not ffmpeg_library_path.is_dir():
        raise SpecialistQualificationError(
            f"Step-B FFmpeg compatibility library is missing: {ffmpeg_library_path}"
        )
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(ffmpeg_library_path)
    version = subprocess.run(  # nosec B603
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout.strip()
    help_text = subprocess.run(  # nosec B603
        [str(executable), "serve", "--help=all"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout
    return InstalledServingCapabilities.from_help(version=version, help_text=help_text)


def _specialist_runtime(
    state_root: Path, configuration: StepC2Configuration
) -> ServingProfileRuntime:
    """Use vLLM's native V1 thinking-budget implementation for Step C.

    vLLM 0.25.0 rejects ``thinking_token_budget`` on its V2 model runner.  The
    runtime setting is recorded in owned-process evidence and is not a Laplace
    token-forcing mechanism.
    """
    if configuration.use_v2_model_runner:
        raise SpecialistQualificationError(
            "Step C native thinking budget requires the installed vLLM V1 model runner"
        )
    return ServingProfileRuntime(
        state_root,
        residual_free_mib=configuration.minimum_free_headroom_mib,
        ffmpeg_library_path=configuration.ffmpeg_library_path,
        launch_environment={"VLLM_USE_V2_MODEL_RUNNER": "0"},
    )


class _GpuSampler:
    def __init__(self) -> None:
        self.samples: list[GpuSnapshot] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(observe_gpu())
            except ServingRuntimeError:
                pass
            self._stop.wait(0.25)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _evaluation_configuration(
    configuration: StepC2Configuration,
    base_revision: str,
    output_root: Path,
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        path=configuration.path,
        base_revision=base_revision,
        base_revision_environment_variable="LAPLACE_STEP_C2_BASE_REVISION",
        manifest_path=configuration.benchmark_manifest_path,
        model_artifacts_path=configuration.path,
        arms=(),
        phases=(),
        base_reference_root=configuration.path.parent,
        overlay_root=output_root / "unused_corpus",
        output_root=output_root,
        held_out_environment_variable=configuration.held_out_environment_variable,
        default_timeout_seconds=900,
        correction_budget=2,
        bootstrap_samples=1000,
        bootstrap_seed=20260901,
        confidence_level=0.95,
    )


def _audit_summary(project: Path) -> JsonObject:
    root = project / "Outputs" / "AgentTeam" / "team_logs" / "model_calls"
    worker_rates: list[float] = []
    main_calls = 0
    worker_calls = 0
    invalid_worker_calls = 0
    malformed_audits = 0
    files = sorted(root.glob("*.json")) if root.is_dir() else []
    for path in files:
        try:
            record = _read_json(path)
        except SpecialistQualificationError:
            malformed_audits += 1
            continue
        routing = record.get("routing")
        selected = routing.get("selected") if isinstance(routing, dict) else None
        if selected == "main":
            main_calls += 1
        elif selected == "rtl_worker":
            worker_calls += 1
            if record.get("response_valid") is not True:
                invalid_worker_calls += 1
            rate = record.get("output_tokens_per_second")
            if (
                record.get("response_valid") is True
                and isinstance(rate, (int, float))
                and not isinstance(rate, bool)
                and float(rate) > 0
            ):
                worker_rates.append(float(rate))
    return {
        "audit_file_count": len(files),
        "main_call_count": main_calls,
        "worker_call_count": worker_calls,
        "invalid_worker_call_count": invalid_worker_calls,
        "malformed_audit_count": malformed_audits,
        "worker_output_tokens_per_second": worker_rates,
    }


def _task_map(configuration: StepC2Configuration) -> dict[str, AblationTask]:
    return {
        task.task_id: task
        for task in load_benchmark_manifest(configuration.benchmark_manifest_path)
    }


def _run_task(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
    task: AblationTask,
    dual: DualModelConfiguration,
    project: Path,
    held_out_root: Path,
    *,
    experiment_arm: str,
    role_mode: RoleMode,
) -> JsonObject:
    if project.exists():
        raise SpecialistQualificationError(f"Refusing to reuse task project: {project}")
    project.mkdir(parents=True, exist_ok=False)
    normalized = normalize_task_spec(repository_root, task.language, _task_spec(task))
    normalized["resolved_public_tests"] = list(task.public_tests)
    store = AgentTaskStore(project)
    store.create(task.language, normalized)
    metadata = replace(task.routing, experiment_arm=experiment_arm)
    runner = LocalTeamRunner(
        repository_root,
        project,
        dual.main,
        rtl_worker_candidate=dual.rtl_worker,
        dual_model_configuration=dual,
        routing_metadata=metadata,
        experiment_arm=experiment_arm,
        options=TeamWorkflowOptions(
            base_commit=base_revision,
            retrieval_mode="none",
            role_mode=role_mode,
            adversarial_verification=True,
            reviewer_invariants=role_mode == "five_role",
            required_tools=task.required_tools,
            cuda_probe_python=configuration.vllm_python,
        ),
        shared_reference_root=None,
    )
    started = time.monotonic()
    lane = runner.run(task.task_id, query=task.objective)
    elapsed = time.monotonic() - started
    held_out: JsonObject = {
        "status": "NOT_RUN",
        "reason": "deterministic lane did not complete",
    }
    if lane.get("status") == "COMPLETE":
        worktree_raw = lane.get("worktree")
        if not isinstance(worktree_raw, str):
            raise SpecialistQualificationError("Completed lane did not expose its worktree")
        evaluation_configuration = _evaluation_configuration(
            configuration, base_revision, project / "held_out_evaluation"
        )
        evaluation_arm = AblationArm(
            arm_id="C",
            label=experiment_arm,
            models_path=configuration.path,
            models=dual,
            worker_enabled=True,
        )
        held_out = _evaluate_held_out(
            repository_root,
            evaluation_configuration,
            task,
            evaluation_arm,
            Path(worktree_raw),
            held_out_root,
        )
    lane_status = str(lane.get("status", "UNKNOWN"))
    infrastructure_statuses = {
        "BLOCKED_GPU",
        "BLOCKED_REFERENCE_EMPTY",
        "MODEL_REQUIRED",
    }
    outcome_kind = (
        "infrastructure_failure"
        if lane_status in infrastructure_statuses
        else "candidate_result"
    )
    return {
        "task_id": task.task_id,
        "lane_status": lane_status,
        "deterministic_pass": lane_status == "COMPLETE",
        "held_out": held_out,
        "held_out_pass": held_out.get("status") == "PASS",
        "elapsed_seconds": elapsed,
        "worktree": lane.get("worktree"),
        "outcome_kind": outcome_kind,
        "audit": _audit_summary(project),
        "lane": lane,
    }


def _model_failure_record(task_id: str, exc: Exception) -> JsonObject:
    category = _exception_failure_category(exc)
    outcome = _failure_outcome_kind(category)
    return {
        "task_id": task_id,
        "lane_status": "EXCEPTION",
        "deterministic_pass": False,
        "held_out": {"status": "NOT_RUN"},
        "held_out_pass": False,
        "failure_category": category,
        "outcome_kind": outcome,
        "error": str(exc),
        "audit": {},
    }


def _aggregate_task_metrics(tasks: Sequence[JsonObject]) -> JsonObject:
    deterministic = sum(item.get("deterministic_pass") is True for item in tasks)
    held_out = sum(item.get("held_out_pass") is True for item in tasks)
    rates: list[float] = []
    worker_calls = 0
    invalid_worker_calls = 0
    infrastructure_failures = 0
    for item in tasks:
        if item.get("outcome_kind") == "infrastructure_failure":
            infrastructure_failures += 1
        audit = item.get("audit")
        if not isinstance(audit, dict):
            continue
        raw_rates = audit.get("worker_output_tokens_per_second")
        if isinstance(raw_rates, list):
            rates.extend(
                float(value)
                for value in raw_rates
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > 0
            )
        raw_calls = audit.get("worker_call_count")
        raw_invalid = audit.get("invalid_worker_call_count")
        raw_malformed = audit.get("malformed_audit_count")
        if isinstance(raw_calls, int) and not isinstance(raw_calls, bool):
            worker_calls += raw_calls
        if isinstance(raw_invalid, int) and not isinstance(raw_invalid, bool):
            invalid_worker_calls += raw_invalid
        if (
            isinstance(raw_malformed, int)
            and not isinstance(raw_malformed, bool)
            and raw_malformed > 0
        ):
            infrastructure_failures += 1
    return {
        "deterministic_task_pass_count": deterministic,
        "held_out_task_pass_count": held_out,
        "worker_call_count": worker_calls,
        "invalid_worker_call_count": invalid_worker_calls,
        "median_output_tokens_per_second": statistics.median(rates) if rates else None,
        "infrastructure_failure_count": infrastructure_failures,
    }


def _validate_held_out(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
    held_out_root: Path,
    output_root: Path,
) -> JsonObject:
    evaluation_configuration = _evaluation_configuration(
        configuration, base_revision, output_root / "held_out_validation"
    )
    return validate_held_out_pack(repository_root, evaluation_configuration, held_out_root)


def qualify_candidate(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
    lock_path: Path,
    candidate_id: str,
    held_out_root: Path,
    output_root: Path,
    *,
    task_ids: tuple[str, ...] | None = None,
) -> JsonObject:
    require_clean_execution_base(repository_root, configuration, base_revision)
    lock = load_revision_lock(configuration, lock_path, base_revision=base_revision)
    candidate_spec = configuration.candidate(candidate_id)
    revision = locked_revision(lock, candidate_id)
    model_path = candidate_model_path(configuration, candidate_id, revision)
    marker = model_path / ".laplace-step-c2-snapshot.json"
    if not marker.is_file() or not (model_path / "config.json").is_file():
        raise SpecialistQualificationError(
            f"Candidate snapshot is not acquired and owned by C2: {model_path}"
        )
    marker_value = _read_json(marker)
    if (
        marker_value.get("revision") != revision
        or marker_value.get("repository") != candidate_spec.repository
        or marker_value.get("candidate_id") != candidate_id
        or marker_value.get("base_revision") != base_revision
        or marker_value.get("configuration_sha256") != configuration.config_sha256
    ):
        raise SpecialistQualificationError("Candidate snapshot marker disagrees with revision lock")
    output = _runtime_output_path(
        configuration, output_root, label="qualification output root"
    )
    if output.exists():
        raise SpecialistQualificationError(f"Refusing to reuse qualification output root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    held_out = held_out_root.resolve()
    held_out_validation = _validate_held_out(
        repository_root, configuration, base_revision, held_out, output
    )
    capabilities = _installed_capabilities(
        configuration.vllm_executable, configuration.ffmpeg_library_path
    )
    profile = build_specialist_profile(configuration, candidate_spec, model_path)
    resolved = resolve_profile(
        profile,
        capabilities,
        executable=configuration.vllm_executable.resolve(),
        require_model=True,
    )
    serving_candidate = build_specialist_candidate(
        configuration, candidate_spec, profile, revision
    )
    dual = build_direct_configuration(serving_candidate)
    initial = observe_gpu()
    if initial.compute_pids:
        raise SpecialistQualificationError(
            "Candidate qualification requires a clean GPU; unrelated compute PIDs are present: "
            + ",".join(str(pid) for pid in initial.compute_pids)
        )
    runtime = _specialist_runtime(output / "server", configuration)
    sampler = _GpuSampler()
    sampler.samples.append(initial)
    sampler.start()
    owned: JsonObject | None = None
    readiness: JsonObject | None = None
    release: JsonObject = {"status": "NOT_STARTED"}
    tasks: list[JsonObject] = []
    fatal_error: JsonObject | None = None
    selected_task_ids = task_ids or configuration.worker_task_ids
    if not selected_task_ids or not set(selected_task_ids).issubset(
        configuration.worker_task_ids
    ):
        raise SpecialistQualificationError("Qualification task subset is invalid")
    is_full_qualification = selected_task_ids == configuration.worker_task_ids
    try:
        owned_process = runtime.start(resolved)
        owned = asdict(owned_process)
        readiness = runtime.wait_ready(resolved)
        task_by_id = _task_map(configuration)
        for task_id in selected_task_ids:
            task = task_by_id[task_id]
            project = output / "tasks" / task_id
            try:
                tasks.append(
                    _run_task(
                        repository_root,
                        configuration,
                        base_revision,
                        task,
                        dual,
                        project,
                        held_out,
                        experiment_arm=candidate_id,
                        role_mode="direct",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve terminal experiment evidence
                tasks.append(_model_failure_record(task_id, exc))
    except Exception as exc:  # noqa: BLE001 - preserve terminal runtime evidence
        fatal_error = {
            "type": type(exc).__name__,
            "category": getattr(exc, "category", None),
            "error": str(exc),
        }
    finally:
        sampler.stop()
        if runtime.ownership_path.is_file():
            try:
                release = runtime.release_owned(timeout_seconds=90)
            except ServingRuntimeError as exc:
                release = {
                    "status": "FAIL",
                    "category": exc.category,
                    "evidence": exc.evidence,
                }
    try:
        residual = observe_gpu()
        residual_json: JsonObject = asdict(residual)
    except ServingRuntimeError as exc:
        residual = None
        residual_json = {
            "status": "UNAVAILABLE",
            "category": exc.category,
            "evidence": exc.evidence,
        }
    metrics = _aggregate_task_metrics(tasks)
    peak_used = max((sample.used_mib for sample in sampler.samples), default=initial.used_mib)
    metrics["peak_gpu_memory_mib"] = peak_used
    metrics["peak_gpu_memory_delta_mib"] = max(0, peak_used - initial.used_mib)
    clean_release = (
        release.get("status") == "RELEASED_OWNED_PROFILE"
        and residual is not None
        and not residual.compute_pids
    )
    smoke_evidence_complete = (
        metrics["worker_call_count"] > 0
        and any(item.get("lane_status") == "COMPLETE" for item in tasks)
    )
    success_status = "QUALIFICATION_COMPLETE" if is_full_qualification else "SMOKE_COMPLETE"
    status = (
        success_status
        if fatal_error is None
        and len(tasks) == len(selected_task_ids)
        and metrics["infrastructure_failure_count"] == 0
        and clean_release
        and (is_full_qualification or smoke_evidence_complete)
        else "QUALIFICATION_INVALID_INFRASTRUCTURE"
    )
    report: JsonObject = {
        "schema_version": 1,
        "status": status,
        "roadmap_step": "C2",
        "candidate_id": candidate_id,
        "repository": candidate_spec.repository,
        "revision": revision,
        "model_path": str(model_path),
        "base_revision": base_revision,
        "configuration_sha256": configuration.config_sha256,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "qualification_policy": {
            "task_ids": list(selected_task_ids),
            "full_qualification": is_full_qualification,
            "role_mode": "direct",
            "retrieval_mode": "none",
            "fallback_to_main": False,
            "worker_reasoning_mode": "model_default",
            "held_out_required": True,
        },
        "serving_profile": profile.to_json(),
        "resolved_serving_profile": resolved.to_json(),
        "native_reasoning": {
            "reasoning_parser": configuration.reasoning_parser,
            "thinking_token_budget": configuration.thinking_token_budget,
            "vllm_use_v2_model_runner": configuration.use_v2_model_runner,
        },
        "held_out_validation": held_out_validation,
        "owned_process": owned,
        "readiness": readiness,
        "initial_gpu": asdict(initial),
        "gpu_samples": [asdict(sample) for sample in sampler.samples],
        "residual_gpu": residual_json,
        "release": release,
        "clean_owned_server_release": clean_release,
        "tasks": tasks,
        "metrics": metrics,
        "fatal_error": fatal_error,
    }
    write_json_atomic(output / "qualification.json", report)
    return report


def _validated_candidate_reports(
    configuration: StepC2Configuration,
    base_revision: str,
    reports: Sequence[JsonObject],
) -> dict[str, JsonObject]:
    by_id: dict[str, JsonObject] = {}
    for report in reports:
        candidate_id = report.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in by_id:
            raise SpecialistQualificationError(
                "Candidate reports have missing/duplicate identities"
            )
        candidate = configuration.candidate(candidate_id)
        revision = report.get("revision")
        model_path = report.get("model_path")
        if (
            report.get("repository") != candidate.repository
            or not isinstance(revision, str)
            or not _COMMIT.fullmatch(revision)
            or model_path != str(candidate_model_path(configuration, candidate_id, revision))
        ):
            raise SpecialistQualificationError(
                f"Candidate report identity is invalid: {candidate_id}"
            )
        if (
            report.get("status") != "QUALIFICATION_COMPLETE"
            or report.get("base_revision") != base_revision
            or report.get("configuration_sha256") != configuration.config_sha256
            or report.get("clean_owned_server_release") is not True
        ):
            raise SpecialistQualificationError(
                f"Candidate report is not promotion-eligible: {candidate_id}"
            )
        metrics = report.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("infrastructure_failure_count") != 0:
            raise SpecialistQualificationError(
                f"Candidate report contains infrastructure failures: {candidate_id}"
            )
        deterministic = metrics.get("deterministic_task_pass_count")
        held_out = metrics.get("held_out_task_pass_count")
        peak = metrics.get("peak_gpu_memory_mib")
        rate = metrics.get("median_output_tokens_per_second")
        if (
            not isinstance(deterministic, int)
            or isinstance(deterministic, bool)
            or not 0 <= deterministic <= len(configuration.worker_task_ids)
            or not isinstance(held_out, int)
            or isinstance(held_out, bool)
            or not 0 <= held_out <= deterministic
            or not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak <= 0
            or (
                rate is not None
                and (
                    not isinstance(rate, (int, float))
                    or isinstance(rate, bool)
                    or float(rate) <= 0
                )
            )
        ):
            raise SpecialistQualificationError(
                f"Candidate report metrics are malformed: {candidate_id}"
            )
        by_id[candidate_id] = report
    expected = (
        {configuration.preselected_candidate}
        if configuration.preselected_candidate is not None
        else {item.candidate_id for item in configuration.candidates}
    )
    if set(by_id) != expected:
        raise SpecialistQualificationError(
            "Selection requires exactly all configured candidates"
        )
    return by_id


def select_candidate_reports(
    configuration: StepC2Configuration,
    base_revision: str,
    reports: Sequence[JsonObject],
) -> JsonObject:
    by_id = _validated_candidate_reports(configuration, base_revision, reports)
    pool = list(by_id.values())
    trace: list[JsonObject] = []
    if configuration.preselected_candidate is not None:
        trace.append(
            {
                "metric": "official_upstream_evidence_preselection",
                "best": configuration.preselected_candidate,
                "remaining": [configuration.preselected_candidate],
            }
        )

    def integer_metric(report: JsonObject, name: str) -> int:
        raw = report.get("metrics")
        value = raw.get(name) if isinstance(raw, dict) else None
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpecialistQualificationError(f"Candidate metric {name} must be an integer")
        return value

    def float_metric(report: JsonObject, name: str) -> float | None:
        raw = report.get("metrics")
        value = raw.get(name) if isinstance(raw, dict) else None
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SpecialistQualificationError(f"Candidate metric {name} must be numeric or null")
        return float(value)

    primary = max(integer_metric(item, "deterministic_task_pass_count") for item in pool)
    pool = [
        item
        for item in pool
        if integer_metric(item, "deterministic_task_pass_count") == primary
    ]
    trace.append(
        {
            "metric": "deterministic_task_pass_count",
            "best": primary,
            "remaining": [item["candidate_id"] for item in pool],
        }
    )
    if len(pool) > 1:
        secondary = max(integer_metric(item, "held_out_task_pass_count") for item in pool)
        pool = [
            item
            for item in pool
            if integer_metric(item, "held_out_task_pass_count") == secondary
        ]
        trace.append(
            {
                "metric": "held_out_task_pass_count",
                "best": secondary,
                "remaining": [item["candidate_id"] for item in pool],
            }
        )
    if len(pool) > 1:
        peaks = [integer_metric(item, "peak_gpu_memory_mib") for item in pool]
        best_peak = min(peaks)
        pool = [
            item for item in pool if integer_metric(item, "peak_gpu_memory_mib") == best_peak
        ]
        trace.append(
            {
                "metric": "lower_peak_gpu_memory_mib",
                "best": best_peak,
                "remaining": [item["candidate_id"] for item in pool],
            }
        )
    if len(pool) > 1:
        rates = [float_metric(item, "median_output_tokens_per_second") for item in pool]
        numeric = [value for value in rates if value is not None]
        if numeric:
            best_rate = max(numeric)
            pool = [
                item
                for item in pool
                if float_metric(item, "median_output_tokens_per_second") == best_rate
            ]
            trace.append(
                {
                    "metric": "higher_output_tokens_per_second",
                    "best": best_rate,
                    "remaining": [item["candidate_id"] for item in pool],
                }
            )
        else:
            trace.append(
                {
                    "metric": "higher_output_tokens_per_second",
                    "best": None,
                    "remaining": [item["candidate_id"] for item in pool],
                }
            )

    if len(pool) != 1:
        return {
            "schema_version": 1,
            "status": "NO_PROMOTION_TIE",
            "roadmap_step": "C2",
            "base_revision": base_revision,
            "configuration_sha256": configuration.config_sha256,
            "selection_trace": trace,
            "remaining_candidates": [item["candidate_id"] for item in pool],
            "selected_candidate": None,
        }
    selected = pool[0]
    return {
        "schema_version": 1,
        "status": "CANDIDATE_SELECTED_PENDING_COEXISTENCE",
        "roadmap_step": "C2",
        "base_revision": base_revision,
        "configuration_sha256": configuration.config_sha256,
        "selection_trace": trace,
        "selected_candidate": {
            "candidate_id": selected["candidate_id"],
            "repository": selected["repository"],
            "revision": selected["revision"],
            "model_path": selected["model_path"],
            "metrics": selected["metrics"],
        },
    }


def _load_p8_profile(configuration: StepC2Configuration) -> ServingProfile:
    _require_blob(
        configuration.p8_profile_path,
        configuration.p8_profile_git_blob_sha,
        label="frozen P8 profile",
    )
    value = _read_json(configuration.p8_profile_path)
    return ServingProfile.from_mapping(value)


def _p8_candidate(configuration: StepC2Configuration, profile: ServingProfile) -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint=endpoint_for(profile),
        model=profile.served_model_name,
        revision=f"p8-profile-blob-{configuration.p8_profile_git_blob_sha}",
        quantization="w4a16_awq",
        kernel="vllm_native",
        prefix_caching=profile.enable_prefix_caching,
        chunked_prefill=profile.enable_chunked_prefill,
        cuda_graph_mode="vllm_profile_default",
        scheduler=profile.scheduling_policy,
        model_path=profile.model_path,
        context_tokens=profile.max_model_len,
        max_output_tokens=8192,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        request_timeout_seconds=profile.request_timeout,
        context_safety_margin_tokens=512,
        minimum_completion_tokens=256,
        reviewer_max_output_tokens=2048,
        structured_serialization_max_output_tokens=8192,
        structured_serialization_temperature=0.0,
        structured_serialization_top_p=1.0,
    )


def _selection_value(
    configuration: StepC2Configuration,
    base_revision: str,
    selection_path: Path,
) -> JsonObject:
    selection = _read_json(
        _runtime_output_path(configuration, selection_path, label="selection evidence")
    )
    if (
        selection.get("status") != "CANDIDATE_SELECTED_PENDING_COEXISTENCE"
        or selection.get("base_revision") != base_revision
        or selection.get("configuration_sha256") != configuration.config_sha256
    ):
        raise SpecialistQualificationError(
            "Selection evidence is incompatible with coexistence run"
        )
    selected = selection.get("selected_candidate")
    if not isinstance(selected, dict):
        raise SpecialistQualificationError("Selection contains no candidate")
    configuration.candidate(
        _non_empty_string(selected.get("candidate_id"), label="selected candidate")
    )
    revision = selected.get("revision")
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise SpecialistQualificationError("Selected candidate revision is not immutable")
    model_path = selected.get("model_path")
    candidate_id = _non_empty_string(
        selected.get("candidate_id"), label="selected candidate"
    )
    expected_path = candidate_model_path(configuration, candidate_id, revision)
    if (
        not isinstance(model_path, str)
        or Path(model_path).resolve() != expected_path
        or not expected_path.is_dir()
    ):
        raise SpecialistQualificationError("Selected candidate model path is unavailable")
    return selection


def certify_coexistence(
    repository_root: Path,
    configuration: StepC2Configuration,
    base_revision: str,
    selection_path: Path,
    held_out_root: Path,
    output_root: Path,
) -> JsonObject:
    require_clean_execution_base(repository_root, configuration, base_revision)
    selection = _selection_value(configuration, base_revision, selection_path)
    selected = cast(dict[str, object], selection["selected_candidate"])
    candidate_id = _non_empty_string(selected.get("candidate_id"), label="selected candidate")
    revision = _non_empty_string(selected.get("revision"), label="selected revision")
    model_path = Path(_non_empty_string(selected.get("model_path"), label="selected model path"))
    candidate_spec = configuration.candidate(candidate_id)
    marker = model_path / ".laplace-step-c2-snapshot.json"
    marker_value = _read_json(marker) if marker.is_file() else {}
    if (
        marker_value.get("revision") != revision
        or marker_value.get("candidate_id") != candidate_id
        or marker_value.get("repository") != candidate_spec.repository
        or marker_value.get("base_revision") != base_revision
        or marker_value.get("configuration_sha256") != configuration.config_sha256
    ):
        raise SpecialistQualificationError("Selected specialist snapshot is not owned/immutable")
    output = _runtime_output_path(
        configuration, output_root, label="coexistence output root"
    )
    if output.exists():
        raise SpecialistQualificationError(f"Refusing to reuse coexistence output root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    held_out = held_out_root.resolve()
    _validate_held_out(repository_root, configuration, base_revision, held_out, output)
    capabilities = _installed_capabilities(
        configuration.vllm_executable, configuration.ffmpeg_library_path
    )
    p8_profile = _load_p8_profile(configuration)
    p8_resolved = resolve_profile(
        p8_profile,
        capabilities,
        executable=configuration.vllm_executable.resolve(),
        require_model=True,
    )
    specialist_profile = build_specialist_profile(configuration, candidate_spec, model_path)
    specialist_resolved = resolve_profile(
        specialist_profile,
        capabilities,
        executable=configuration.vllm_executable.resolve(),
        require_model=True,
    )
    p8 = _p8_candidate(configuration, p8_profile)
    specialist = build_specialist_candidate(
        configuration, candidate_spec, specialist_profile, revision
    )
    dual = DualModelConfiguration(
        main=p8,
        rtl_worker=specialist,
        worker_contract_retries=1,
        worker_response_retries=1,
        fallback_to_main=False,
        worker_reasoning_mode="model_default",
    )
    initial = observe_gpu()
    if initial.compute_pids:
        raise SpecialistQualificationError(
            "Coexistence certification requires a clean GPU before owned P8 launch"
        )
    p8_runtime = ServingProfileRuntime(
        output / "p8_server",
        residual_free_mib=configuration.minimum_free_headroom_mib,
        ffmpeg_library_path=configuration.ffmpeg_library_path,
    )
    specialist_runtime = _specialist_runtime(output / "specialist_server", configuration)
    sampler = _GpuSampler()
    sampler.samples.append(initial)
    sampler.start()
    p8_owned: JsonObject | None = None
    specialist_owned: JsonObject | None = None
    p8_readiness: JsonObject | None = None
    specialist_readiness: JsonObject | None = None
    p8_release: JsonObject = {"status": "NOT_STARTED"}
    specialist_release: JsonObject = {"status": "NOT_STARTED"}
    tasks: list[JsonObject] = []
    fatal_error: JsonObject | None = None
    try:
        p8_owned_process = p8_runtime.start(p8_resolved)
        p8_owned = asdict(p8_owned_process)
        p8_readiness = p8_runtime.wait_ready(p8_resolved)
        specialist_owned_process = specialist_runtime.start(specialist_resolved)
        specialist_owned = asdict(specialist_owned_process)
        specialist_readiness = specialist_runtime.wait_ready(specialist_resolved)
        task_by_id = _task_map(configuration)
        for task_id in configuration.coexistence_task_ids:
            try:
                tasks.append(
                    _run_task(
                        repository_root,
                        configuration,
                        base_revision,
                        task_by_id[task_id],
                        dual,
                        output / "tasks" / task_id,
                        held_out,
                        experiment_arm="step_c2_coexistence",
                        role_mode="five_role",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve terminal integration evidence
                tasks.append(_model_failure_record(task_id, exc))
    except Exception as exc:  # noqa: BLE001 - preserve terminal coexistence evidence
        fatal_error = {
            "type": type(exc).__name__,
            "category": getattr(exc, "category", None),
            "error": str(exc),
        }
    finally:
        sampler.stop()
        if specialist_runtime.ownership_path.is_file():
            try:
                specialist_release = specialist_runtime.release_owned(timeout_seconds=90)
            except ServingRuntimeError as exc:
                specialist_release = {
                    "status": "FAIL",
                    "category": exc.category,
                    "evidence": exc.evidence,
                }
        if p8_runtime.ownership_path.is_file():
            try:
                p8_release = p8_runtime.release_owned(timeout_seconds=90)
            except ServingRuntimeError as exc:
                p8_release = {
                    "status": "FAIL",
                    "category": exc.category,
                    "evidence": exc.evidence,
                }
    try:
        residual = observe_gpu()
        residual_json: JsonObject = asdict(residual)
    except ServingRuntimeError as exc:
        residual = None
        residual_json = {
            "status": "UNAVAILABLE",
            "category": exc.category,
            "evidence": exc.evidence,
        }
    peak_used = max((sample.used_mib for sample in sampler.samples), default=initial.used_mib)
    minimum_free = initial.total_mib - peak_used
    main_calls = 0
    worker_calls = 0
    for task in tasks:
        audit = task.get("audit")
        if isinstance(audit, dict):
            main = audit.get("main_call_count")
            worker = audit.get("worker_call_count")
            if isinstance(main, int) and not isinstance(main, bool):
                main_calls += main
            if isinstance(worker, int) and not isinstance(worker, bool):
                worker_calls += worker
    tasks_pass = len(tasks) == len(configuration.coexistence_task_ids) and all(
        task.get("deterministic_pass") is True and task.get("held_out_pass") is True
        for task in tasks
    )
    clean_release = (
        specialist_release.get("status") == "RELEASED_OWNED_PROFILE"
        and p8_release.get("status") == "RELEASED_OWNED_PROFILE"
        and residual is not None
        and not residual.compute_pids
    )
    status = (
        "COEXISTENCE_CERTIFIED"
        if fatal_error is None
        and tasks_pass
        and main_calls > 0
        and worker_calls > 0
        and minimum_free >= configuration.minimum_free_headroom_mib
        and clean_release
        else "COEXISTENCE_FAILED"
    )
    report: JsonObject = {
        "schema_version": 1,
        "status": status,
        "roadmap_step": "C2",
        "base_revision": base_revision,
        "configuration_sha256": configuration.config_sha256,
        "selected_candidate": selected,
        "start_order": ["p8", "specialist"],
        "p8_profile": p8_profile.to_json(),
        "specialist_profile": specialist_profile.to_json(),
        "p8_resolved": p8_resolved.to_json(),
        "specialist_resolved": specialist_resolved.to_json(),
        "p8_owned_process": p8_owned,
        "specialist_owned_process": specialist_owned,
        "p8_readiness": p8_readiness,
        "specialist_readiness": specialist_readiness,
        "initial_gpu": asdict(initial),
        "gpu_samples": [asdict(sample) for sample in sampler.samples],
        "peak_gpu_memory_mib": peak_used,
        "minimum_free_headroom_mib": minimum_free,
        "required_minimum_free_headroom_mib": configuration.minimum_free_headroom_mib,
        "route_evidence": {"main_call_count": main_calls, "worker_call_count": worker_calls},
        "tasks": tasks,
        "specialist_release": specialist_release,
        "p8_release": p8_release,
        "clean_owned_server_release": clean_release,
        "residual_gpu": residual_json,
        "fatal_error": fatal_error,
    }
    write_json_atomic(output / "coexistence.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")

    resolve = sub.add_parser("resolve-revisions")
    resolve.add_argument("--base-revision", required=True)
    resolve.add_argument("--output", type=Path, required=True)

    acquire = sub.add_parser("acquire")
    acquire.add_argument("--base-revision", required=True)
    acquire.add_argument("--lock", type=Path, required=True)
    acquire.add_argument("--candidate", required=True)

    qualify = sub.add_parser("qualify")
    qualify.add_argument("--base-revision", required=True)
    qualify.add_argument("--lock", type=Path, required=True)
    qualify.add_argument("--candidate", required=True)
    qualify.add_argument("--held-out-root", type=Path, required=True)
    qualify.add_argument("--output-root", type=Path, required=True)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--base-revision", required=True)
    smoke.add_argument("--lock", type=Path, required=True)
    smoke.add_argument("--candidate", required=True)
    smoke.add_argument("--held-out-root", type=Path, required=True)
    smoke.add_argument("--output-root", type=Path, required=True)

    select = sub.add_parser("select")
    select.add_argument("--base-revision", required=True)
    select.add_argument("--reports", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)

    coexist = sub.add_parser("coexist")
    coexist.add_argument("--base-revision", required=True)
    coexist.add_argument("--selection", type=Path, required=True)
    coexist.add_argument("--held-out-root", type=Path, required=True)
    coexist.add_argument("--output-root", type=Path, required=True)
    return parser


def _print(value: JsonObject) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    configuration = load_step_c2_configuration(ROOT, arguments.config)
    command = arguments.command
    if command == "validate":
        _print(
            {
                "status": "STEP_C2_CONFIGURATION_VALID",
                "preparation_base_commit": configuration.preparation_base_commit,
                "configuration_sha256": configuration.config_sha256,
                "worker_task_ids": list(configuration.worker_task_ids),
                "candidates": [asdict(item) for item in configuration.candidates],
            }
        )
        return 0
    if command == "resolve-revisions":
        _print(
            write_revision_lock(
                ROOT,
                configuration,
                arguments.base_revision,
                arguments.output,
            )
        )
        return 0
    if command == "acquire":
        _print(
            acquire_candidate_snapshot(
                ROOT,
                configuration,
                arguments.base_revision,
                arguments.lock,
                arguments.candidate,
            )
        )
        return 0
    if command == "qualify":
        report = qualify_candidate(
            ROOT,
            configuration,
            arguments.base_revision,
            arguments.lock,
            arguments.candidate,
            arguments.held_out_root,
            arguments.output_root,
        )
        _print(report)
        return 0 if report["status"] == "QUALIFICATION_COMPLETE" else 2
    if command == "smoke":
        report = qualify_candidate(
            ROOT,
            configuration,
            arguments.base_revision,
            arguments.lock,
            arguments.candidate,
            arguments.held_out_root,
            arguments.output_root,
            task_ids=configuration.coexistence_task_ids,
        )
        _print(report)
        return 0 if report["status"] == "SMOKE_COMPLETE" else 2
    if command == "select":
        reports = [
            _read_json(
                _runtime_output_path(
                    configuration, path, label="qualification report evidence"
                )
            )
            for path in arguments.reports
        ]
        selection = select_candidate_reports(configuration, arguments.base_revision, reports)
        output = _runtime_output_path(
            configuration, arguments.output, label="selection output"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, selection, readonly=True)
        _print(selection)
        return 0 if selection["status"] == "CANDIDATE_SELECTED_PENDING_COEXISTENCE" else 2
    if command == "coexist":
        report = certify_coexistence(
            ROOT,
            configuration,
            arguments.base_revision,
            arguments.selection,
            arguments.held_out_root,
            arguments.output_root,
        )
        _print(report)
        return 0 if report["status"] == "COEXISTENCE_CERTIFIED" else 2
    raise SpecialistQualificationError(f"Unknown Step C2 command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
