"""Configuration, planning, execution, resume, evaluation and reports for one ablation.

Plan-only mode is deliberately side-effect-light: it validates files and local
tool availability but never contacts a model endpoint or runs task verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import statistics
import subprocess  # nosec B404
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import yaml

from .engineering import (
    AgentTaskStore,
    Domain,
    EngineeringError,
    JsonObject,
    LocalToolRunner,
    ReferenceLibrary,
    ReferencePolicyError,
    _inside,
    _safe_relative,
    _write_json_atomic,
    collect_cuda_evidence,
    normalize_task_spec,
    verilator_simulation_available,
)
from .governed_corpus import (
    load_bundled_corpus_manifest,
    load_installed_external_manifest,
    prepare_corpus_overlay,
    validate_corpus_retrieval,
)
from .inference import gpu_memory_snapshot
from .model_artifacts import (
    validate_local_artifacts,
    validate_profile_alignment,
    validate_serving_environments,
)
from .model_routing import (
    ContextBudgetError,
    DualModelConfiguration,
    RtlScope,
    RoutingTaskMetadata,
    assess_rtl_worker_eligibility,
    load_dual_model_configuration,
)
from .llm import ModelInvocationError, ModelRequired
from .team_runner import (
    LocalTeamRunner,
    TeamWorkflowOptions,
    WorktreeManager,
    _run_git,
    apply_validated_patch,
)


EXPERIMENT_ID = "multilanguage_dual_model_ablation_v1"
RESULT_SCHEMA_VERSION = 2
EVALUATION_PROTOCOL_VERSION = "failure-accounting-v2"
_TASK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
Category = Literal["implementation", "repair", "edge_case", "integration"]
PhaseId = Literal["phase1", "phase2", "phase3"]
_PHASE_IDS: tuple[PhaseId, ...] = ("phase1", "phase2", "phase3")
_ARM_PHASE: dict[str, PhaseId] = {"A": "phase1", "B": "phase2", "C": "phase3"}


def _activate_isolated_tools(repository_root: Path) -> None:
    """Prepend the repository-owned profile without discarding the caller's PATH."""
    tool_bin = repository_root.resolve() / ".tools" / "multilanguage" / "bin"
    if not tool_bin.is_dir():
        return
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    value = str(tool_bin)
    if value not in entries:
        os.environ["PATH"] = os.pathsep.join((value, *entries))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineeringError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EngineeringError(f"Expected a JSON object in {path}")
    return dict(value)


def _exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise EngineeringError(
            f"{label} keys are invalid; missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EngineeringError(f"{label} must be non-empty text")
    return value


def _strings(value: object, *, label: str, non_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EngineeringError(f"{label} must be a list of non-empty strings")
    if non_empty and not value:
        raise EngineeringError(f"{label} must not be empty")
    return tuple(value)


@dataclass(frozen=True)
class AblationTask:
    task_id: str
    language: Domain
    category: Category
    regression: bool
    fixture: str
    editable_sources: tuple[str, ...]
    public_tests: tuple[str, ...]
    held_out_id: str
    objective: str
    requirements: tuple[str, ...]
    required_tools: tuple[str, ...]
    routing: RoutingTaskMetadata


@dataclass(frozen=True)
class AblationArm:
    arm_id: Literal["A", "B", "C"]
    label: str
    models_path: Path
    models: DualModelConfiguration
    worker_enabled: bool


@dataclass(frozen=True)
class ExperimentPhase:
    phase_id: PhaseId
    label: str
    arm_ids: tuple[Literal["A", "B", "C"], ...]
    requires_completed_phases: tuple[PhaseId, ...]


@dataclass(frozen=True)
class ExperimentConfiguration:
    path: Path
    base_revision: str | None
    base_revision_environment_variable: str
    manifest_path: Path
    model_artifacts_path: Path
    arms: tuple[AblationArm, ...]
    phases: tuple[ExperimentPhase, ...]
    base_reference_root: Path
    overlay_root: Path
    output_root: Path
    held_out_environment_variable: str
    default_timeout_seconds: int
    correction_budget: int
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float


def _task_kind(category: Category) -> Literal["implementation", "repair", "integration"]:
    if category == "repair":
        return "repair"
    if category == "integration":
        return "integration"
    return "implementation"


def _routing_metadata(
    task_id: str,
    language: Domain,
    category: Category,
    editable_sources: tuple[str, ...],
    raw: object,
    *,
    arm: str = "manifest_validation",
) -> RoutingTaskMetadata:
    if not isinstance(raw, dict):
        raise EngineeringError(f"Task {task_id} routing must be an object")
    routing = dict(raw)
    if language in {"python", "c"}:
        _exact_keys(routing, {"worker_eligible", "rtl_scope"}, label=f"Task {task_id} routing")
        if routing.get("worker_eligible") is not False or routing.get("rtl_scope") != "not_rtl":
            raise EngineeringError(f"Software task {task_id} cannot be worker eligible")
        return RoutingTaskMetadata(
            task_id=task_id,
            experiment_arm=arm,
            domain=language,
            task_kind=_task_kind(category),
            rtl_scope="not_rtl",
            worker_eligible=False,
            editable_sources=editable_sources,
            module_count=0,
            synthesizable=False,
            explicit_ports=False,
            cycle_behavior_specified=False,
            deterministic_verification=True,
        )
    expected = {
        "worker_eligible",
        "rtl_scope",
        "module_count",
        "synthesizable",
        "explicit_ports",
        "cycle_behavior_specified",
        "deterministic_verification",
        "unresolved_architecture",
    }
    _exact_keys(routing, expected, label=f"Task {task_id} routing")
    rtl_scope = routing.get("rtl_scope")
    allowed_scopes = {
        "bounded_module",
        "multi_file_subsystem",
        "protocol_integration",
        "software_rtl_codesign",
        "cdc_architecture",
        "uvm",
        "unresolved_architecture",
    }
    if rtl_scope not in allowed_scopes:
        raise EngineeringError(f"Task {task_id} has an invalid RTL scope")
    booleans = {
        key: routing.get(key)
        for key in (
            "worker_eligible",
            "synthesizable",
            "explicit_ports",
            "cycle_behavior_specified",
            "deterministic_verification",
            "unresolved_architecture",
        )
    }
    if not all(isinstance(value, bool) for value in booleans.values()):
        raise EngineeringError(f"Task {task_id} routing flags must be booleans")
    module_count = routing.get("module_count")
    if not isinstance(module_count, int) or isinstance(module_count, bool) or module_count < 1:
        raise EngineeringError(f"Task {task_id} module_count must be positive")
    metadata = RoutingTaskMetadata(
        task_id=task_id,
        experiment_arm=arm,
        domain=language,
        task_kind=_task_kind(category),
        rtl_scope=cast(RtlScope, rtl_scope),
        worker_eligible=cast(bool, booleans["worker_eligible"]),
        editable_sources=editable_sources,
        module_count=module_count,
        synthesizable=cast(bool, booleans["synthesizable"]),
        explicit_ports=cast(bool, booleans["explicit_ports"]),
        cycle_behavior_specified=cast(bool, booleans["cycle_behavior_specified"]),
        deterministic_verification=cast(bool, booleans["deterministic_verification"]),
        unresolved_architecture=cast(bool, booleans["unresolved_architecture"]),
    )
    eligibility = assess_rtl_worker_eligibility(metadata)
    if metadata.worker_eligible != eligibility.eligible:
        raise EngineeringError(
            f"Task {task_id} worker declaration violates deterministic policy: {eligibility.reason}"
        )
    return metadata


def load_benchmark_manifest(path: Path) -> tuple[AblationTask, ...]:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EngineeringError(f"Cannot read benchmark manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise EngineeringError("Benchmark manifest must be an object")
    value = dict(raw)
    _exact_keys(
        value,
        {
            "schema_version",
            "benchmark_id",
            "task_count",
            "language_counts",
            "regression_task_ids",
            "defaults",
            "tasks",
        },
        label="Benchmark manifest",
    )
    if value.get("schema_version") != 1 or value.get("benchmark_id") != EXPERIMENT_ID:
        raise EngineeringError("Benchmark manifest identity or schema_version is invalid")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 32 or value.get("task_count") != 32:
        raise EngineeringError("Benchmark manifest must contain exactly 32 tasks")
    tasks: list[AblationTask] = []
    seen: set[str] = set()
    held_out_ids: set[str] = set()
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            raise EngineeringError(f"Benchmark task {index} must be an object")
        task = dict(raw_task)
        expected = {
            "task_id",
            "language",
            "category",
            "regression",
            "fixture",
            "editable_sources",
            "public_tests",
            "held_out_id",
            "objective",
            "requirements",
            "required_tools",
            "routing",
        }
        _exact_keys(task, expected, label=f"Benchmark task {index}")
        task_id = _string(task.get("task_id"), label=f"Task {index} id")
        if not _TASK_ID.fullmatch(task_id) or task_id in seen:
            raise EngineeringError(f"Task id is unsafe or duplicated: {task_id}")
        seen.add(task_id)
        language = task.get("language")
        if language not in {"python", "c", "verilog", "systemverilog"}:
            raise EngineeringError(f"Task {task_id} language is invalid")
        domain = cast(Domain, language)
        category_raw = task.get("category")
        if category_raw not in {"implementation", "repair", "edge_case", "integration"}:
            raise EngineeringError(f"Task {task_id} category is invalid")
        category = cast(Category, category_raw)
        editable = _strings(task.get("editable_sources"), label=f"Task {task_id} sources")
        public = _strings(task.get("public_tests"), label=f"Task {task_id} public tests")
        if set(editable) & set(public):
            raise EngineeringError(f"Task {task_id} public tests cannot be editable")
        for raw_path in (*editable, *public):
            _safe_relative(raw_path, label=f"Task {task_id} path")
            lowered = raw_path.lower()
            if "heldout" in lowered or "held_out" in lowered or "solution" in lowered:
                raise EngineeringError(
                    f"Task {task_id} exposes evaluator-only material in a public path"
                )
        if domain == "c" and any(
            Path(path).suffix.lower() in {".cc", ".cpp", ".cxx"} for path in editable
        ):
            raise EngineeringError(f"Task {task_id} must use C rather than C++ sources")
        if domain == "verilog" and any(
            Path(path).suffix.lower() != ".v" for path in (*editable, *public)
        ):
            raise EngineeringError(f"Verilog task {task_id} must use Verilog-compatible .v files")
        if domain == "systemverilog" and any(
            Path(path).suffix.lower() not in {".sv", ".v"} for path in (*editable, *public)
        ):
            raise EngineeringError(f"SystemVerilog task {task_id} has an invalid RTL source suffix")
        routing = _routing_metadata(task_id, domain, category, editable, task.get("routing"))
        if category == "integration" and routing.worker_eligible:
            raise EngineeringError(f"RTL integration task {task_id} must remain main-model-only")
        regression = task.get("regression")
        if not isinstance(regression, bool):
            raise EngineeringError(f"Task {task_id} regression must be boolean")
        held_out_id = _string(task.get("held_out_id"), label=f"Task {task_id} held-out id")
        if held_out_id in held_out_ids:
            raise EngineeringError(f"Held-out identifier is duplicated: {held_out_id}")
        held_out_ids.add(held_out_id)
        required_tools = _strings(
            task.get("required_tools"), label=f"Task {task_id} required tools"
        )
        required_minimum: dict[Domain, set[str]] = {
            "python": {"pytest", "ruff", "mypy"},
            "c": {"gcc"},
            "verilog": {"iverilog", "vvp", "yosys"},
            "systemverilog": {"iverilog", "vvp", "verilator", "yosys"},
        }
        if not required_minimum[domain].issubset(required_tools):
            raise EngineeringError(f"Task {task_id} omits mandatory {domain} deterministic tools")
        tasks.append(
            AblationTask(
                task_id=task_id,
                language=domain,
                category=category,
                regression=regression,
                fixture=_string(task.get("fixture"), label=f"Task {task_id} fixture"),
                editable_sources=editable,
                public_tests=public,
                held_out_id=held_out_id,
                objective=_string(task.get("objective"), label=f"Task {task_id} objective"),
                requirements=_strings(
                    task.get("requirements"), label=f"Task {task_id} requirements"
                ),
                required_tools=required_tools,
                routing=routing,
            )
        )
    counts = {
        domain: sum(item.language == domain for item in tasks)
        for domain in (
            "python",
            "c",
            "verilog",
            "systemverilog",
        )
    }
    if counts != {"python": 8, "c": 8, "verilog": 8, "systemverilog": 8}:
        raise EngineeringError(f"Benchmark language balance is invalid: {counts}")
    required_regressions = {
        "py_safe_async_job",
        "py_fastapi_strict_endpoint",
        "py_sqlite_transaction",
        "py_safe_path_cli",
        "sv_ready_valid_buffer",
        "sv_axi_lite_irq_regs",
    }
    actual_regressions = {task.task_id for task in tasks if task.regression}
    declared = set(_strings(value.get("regression_task_ids"), label="Regression task ids"))
    if actual_regressions != required_regressions or declared != required_regressions:
        raise EngineeringError("The six required regression tasks are not preserved exactly")
    return tuple(tasks)


def load_experiment_configuration(
    repository_root: Path, path: Path, *, base_revision: str | None = None
) -> ExperimentConfiguration:
    root = repository_root.resolve()
    value = _read_json(path.resolve())
    _exact_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "base_revision_environment_variable",
            "manifest",
            "model_artifacts",
            "arms",
            "phases",
            "arm_order_strategy",
            "corpus",
            "held_out",
            "execution",
            "reporting",
        },
        label="Experiment configuration",
    )
    if value.get("schema_version") != 1 or value.get("experiment_id") != EXPERIMENT_ID:
        raise EngineeringError("Experiment identity or schema_version is invalid")
    base_environment = _string(
        value.get("base_revision_environment_variable"),
        label="base_revision_environment_variable",
    )
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", base_environment):
        raise EngineeringError("base_revision_environment_variable is unsafe")
    resolved_base = base_revision or os.getenv(base_environment)
    if resolved_base is not None and not _COMMIT.fullmatch(resolved_base):
        raise EngineeringError(
            f"{base_environment} or --base-revision must be an exact 40-character commit"
        )
    if value.get("arm_order_strategy") != "serialized_phase_then_deterministic_rotation":
        raise EngineeringError("Experiment arm order strategy is invalid")
    manifest_path = _inside(
        root,
        root / _safe_relative(_string(value.get("manifest"), label="manifest"), label="manifest"),
    )
    model_artifacts_path = _inside(
        root,
        root
        / _safe_relative(
            _string(value.get("model_artifacts"), label="model_artifacts"),
            label="model_artifacts",
        ),
    )
    raw_arms = value.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) != 3:
        raise EngineeringError("Experiment must configure exactly three arms")
    arms: list[AblationArm] = []
    for raw_arm in raw_arms:
        if not isinstance(raw_arm, dict):
            raise EngineeringError("Experiment arm must be an object")
        arm_value = dict(raw_arm)
        _exact_keys(arm_value, {"arm_id", "label", "models", "worker_enabled"}, label="Arm")
        arm_id = arm_value.get("arm_id")
        if arm_id not in {"A", "B", "C"}:
            raise EngineeringError("Arm id must be A, B or C")
        models_path = _inside(
            root,
            root / _safe_relative(_string(arm_value.get("models"), label="models"), label="models"),
        )
        try:
            models = load_dual_model_configuration(models_path)
        except ValueError as exc:
            raise EngineeringError(f"Invalid model configuration for arm {arm_id}: {exc}") from exc
        worker_enabled = arm_value.get("worker_enabled")
        if not isinstance(worker_enabled, bool):
            raise EngineeringError("Arm worker_enabled must be boolean")
        if worker_enabled != (models.rtl_worker is not None):
            raise EngineeringError(f"Arm {arm_id} worker flag disagrees with model configuration")
        arms.append(
            AblationArm(
                cast(Literal["A", "B", "C"], arm_id),
                _string(arm_value.get("label"), label="arm label"),
                models_path,
                models,
                worker_enabled,
            )
        )
    if [arm.arm_id for arm in arms] != ["A", "B", "C"]:
        raise EngineeringError("Arms must be listed in A, B, C order")
    if arms[0].models.rtl_worker is not None or arms[1].models.rtl_worker is not None:
        raise EngineeringError("Arms A and B must remain single-model")
    if arms[1].models.main != arms[2].models.main:
        raise EngineeringError("Arms B and C must use the identical main-model profile")
    raw_phases = value.get("phases")
    if not isinstance(raw_phases, list) or len(raw_phases) != 3:
        raise EngineeringError("Experiment must configure exactly three serialized phases")
    phases: list[ExperimentPhase] = []
    for index, raw_phase in enumerate(raw_phases):
        if not isinstance(raw_phase, dict):
            raise EngineeringError(f"Experiment phase {index} must be an object")
        phase_value = dict(raw_phase)
        _exact_keys(
            phase_value,
            {"phase_id", "label", "arms", "requires_completed_phases"},
            label=f"Experiment phase {index}",
        )
        phase_id = phase_value.get("phase_id")
        expected_id = _PHASE_IDS[index]
        if phase_id != expected_id:
            raise EngineeringError(
                "Serialized phases must be listed as phase1, phase2, then phase3"
            )
        phase_arms = _strings(phase_value.get("arms"), label=f"{phase_id} arms")
        required = _strings(
            phase_value.get("requires_completed_phases"),
            label=f"{phase_id} dependencies",
            non_empty=False,
        )
        if index == 0 and (phase_arms != ("A",) or required):
            raise EngineeringError("Phase 1 must contain only Arm A and have no dependency")
        if index == 1 and (phase_arms != ("B",) or required != ("phase1",)):
            raise EngineeringError("Phase 2 must contain only Arm B and require Phase 1")
        if index == 2 and (phase_arms != ("C",) or required != ("phase1", "phase2")):
            raise EngineeringError("Phase 3 must contain only Arm C and require Phases 1 and 2")
        phases.append(
            ExperimentPhase(
                phase_id=cast(PhaseId, phase_id),
                label=_string(phase_value.get("label"), label=f"{phase_id} label"),
                arm_ids=cast(tuple[Literal["A", "B", "C"], ...], phase_arms),
                requires_completed_phases=cast(tuple[PhaseId, ...], required),
            )
        )
    aligned = (
        arms[0].models.main.temperature,
        arms[0].models.main.top_p,
        arms[0].models.main.seed,
        arms[0].models.main.max_output_tokens,
        arms[0].models.main.context_tokens,
    )
    for configured_arm in arms[1:]:
        candidate = configured_arm.models.main
        if (
            candidate.temperature,
            candidate.top_p,
            candidate.seed,
            candidate.max_output_tokens,
            candidate.context_tokens,
        ) != aligned:
            raise EngineeringError(
                "Main-model decoding and context settings must align across arms"
            )
    corpus = value.get("corpus")
    held_out = value.get("held_out")
    execution = value.get("execution")
    reporting = value.get("reporting")
    if not all(isinstance(item, dict) for item in (corpus, held_out, execution, reporting)):
        raise EngineeringError(
            "Experiment corpus, held-out, execution and reporting must be objects"
        )
    corpus_value = cast(dict[str, object], corpus)
    held_value = cast(dict[str, object], held_out)
    execution_value = cast(dict[str, object], execution)
    reporting_value = cast(dict[str, object], reporting)
    _exact_keys(
        corpus_value,
        {"base_reference_root", "overlay_root", "bundled_manifest", "require_non_empty_domains"},
        label="Experiment corpus",
    )
    _exact_keys(
        held_value,
        {
            "root_environment_variable",
            "copy_into_implementation_worktrees",
            "evaluate_only_after_lane_completion",
            "require_manifest_hashes",
        },
        label="Experiment held-out policy",
    )
    _exact_keys(
        execution_value,
        {
            "output_root",
            "resume_completed_pairs",
            "default_timeout_seconds",
            "correction_budget",
            "require_cuda_a6000",
            "require_clean_base_revision",
        },
        label="Experiment execution",
    )
    _exact_keys(
        reporting_value,
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "confidence_level",
            "statistical_generality_claim",
        },
        label="Experiment reporting",
    )
    if corpus_value.get("require_non_empty_domains") != [
        "python",
        "c",
        "verilog",
        "systemverilog",
    ]:
        raise EngineeringError("Corpus validation must require all four language domains")
    bundled_manifest = _inside(
        root,
        root
        / _safe_relative(
            _string(corpus_value.get("bundled_manifest"), label="bundled_manifest"),
            label="bundled_manifest",
        ),
    )
    if bundled_manifest != root / "codex_a6000" / "governed_corpus" / "manifest.json":
        raise EngineeringError("Experiment bundled corpus manifest path is not canonical")
    if held_value.get("copy_into_implementation_worktrees") is not False:
        raise EngineeringError("Held-out tests must never be copied into implementation worktrees")
    if held_value.get("evaluate_only_after_lane_completion") is not True:
        raise EngineeringError("Held-out evaluation must occur only after lane completion")
    if held_value.get("require_manifest_hashes") is not True:
        raise EngineeringError("Held-out evaluator files must require immutable hashes")
    for key in (
        "resume_completed_pairs",
        "require_cuda_a6000",
        "require_clean_base_revision",
    ):
        if execution_value.get(key) is not True:
            raise EngineeringError(f"Experiment execution {key} must be true")
    output_root = _inside(
        root,
        root
        / _safe_relative(
            _string(execution_value.get("output_root"), label="output_root"),
            label="output_root",
        ),
    )
    overlay_root = _inside(
        root,
        root
        / _safe_relative(
            _string(corpus_value.get("overlay_root"), label="overlay_root"),
            label="overlay_root",
        ),
    )
    timeout = execution_value.get("default_timeout_seconds")
    corrections = execution_value.get("correction_budget")
    samples = reporting_value.get("bootstrap_samples")
    seed = reporting_value.get("bootstrap_seed")
    confidence = reporting_value.get("confidence_level")
    if not isinstance(timeout, int) or timeout < 60:
        raise EngineeringError("default_timeout_seconds must be at least 60")
    if corrections != 2:
        raise EngineeringError("The experiment correction budget must equal two")
    if not isinstance(samples, int) or samples < 1000:
        raise EngineeringError("bootstrap_samples must be at least 1000")
    if not isinstance(seed, int):
        raise EngineeringError("bootstrap_seed must be an integer")
    if not isinstance(confidence, (int, float)) or not 0.5 < float(confidence) < 1.0:
        raise EngineeringError("confidence_level must be between 0.5 and 1")
    if reporting_value.get("statistical_generality_claim") is not False:
        raise EngineeringError("This 32-task experiment cannot claim statistical generality")
    return ExperimentConfiguration(
        path=path.resolve(),
        base_revision=resolved_base,
        base_revision_environment_variable=base_environment,
        manifest_path=manifest_path,
        model_artifacts_path=model_artifacts_path,
        arms=tuple(arms),
        phases=tuple(phases),
        base_reference_root=Path(
            _string(corpus_value.get("base_reference_root"), label="base_reference_root")
        ).expanduser(),
        overlay_root=overlay_root,
        output_root=output_root,
        held_out_environment_variable=_string(
            held_value.get("root_environment_variable"),
            label="held-out root environment variable",
        ),
        default_timeout_seconds=timeout,
        correction_budget=corrections,
        bootstrap_samples=samples,
        bootstrap_seed=seed,
        confidence_level=float(confidence),
    )


def _task_spec(task: AblationTask) -> JsonObject:
    if task.language == "python":
        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "repository_root": ".",
            "allowed_paths": list(task.editable_sources),
            "public_interfaces": [
                {
                    "name": Path(task.editable_sources[0]).name,
                    "contract": "Preserve the public fixture interface and documented errors.",
                    "compatibility": "Public and held-out callers use the same interface.",
                }
            ],
            "functional_requirements": list(task.requirements),
            "input_validation": ["Reject malformed and boundary inputs explicitly."],
            "error_behavior": ["Do not mutate durable state after a rejected operation."],
            "concurrency_and_lifecycle": ["Complete cleanup before returning or raising."],
            "security_and_paths": ["Remain inside the declared fixture and editable files."],
            "quality_requirements": {
                "python": ">=3.11",
                "typing": "strict mypy",
                "formatting": "ruff format --check",
                "lint": "ruff check",
                "tests": "public plus evaluator-held pytest",
            },
            "references": [
                {"path_or_id": task.fixture, "purpose": "Repository-local conventions first."}
            ],
            "verification_commands": [f"python -m pytest {path}" for path in task.public_tests],
            "deliverables": ["bounded source replacement", "deterministic verifier evidence"],
            "out_of_scope": ["held-out tests", "network", "files outside editable_sources"],
            "assumptions": ["Held-out tests are inaccessible during implementation."],
        }
    if task.language == "c":
        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "repository_root": ".",
            "allowed_paths": list(task.editable_sources),
            "public_interfaces": [
                {
                    "name": Path(task.editable_sources[0]).name,
                    "contract": "Preserve the fixture public header and ownership contract.",
                    "compatibility": "C11 callers and CMake tests retain ABI compatibility.",
                }
            ],
            "functional_requirements": list(task.requirements),
            "input_validation": ["Validate pointer-length and range relationships before use."],
            "error_behavior": ["Preserve caller-visible state and release resources on failure."],
            "ownership_and_lifetime": ["Every acquisition has one documented release path."],
            "integer_safety": ["Check arithmetic and range before conversion or allocation."],
            "file_and_process_behavior": [
                "Distinguish partial I/O, EOF and errors where applicable."
            ],
            "portability_constraints": ["C11 without compiler extensions."],
            "quality_requirements": {
                "language": "C11",
                "warnings": "-Wall -Wextra -Wpedantic -Werror",
                "build": "CMake/CTest mandatory"
                if {"cmake", "ctest"}.issubset(task.required_tools)
                else "direct GCC build mandatory; CMake/CTest are strengthening gates when available",
                "tests": "Self-checking executable; CTest when declared or available",
                "sanitizers": "Mandatory: "
                + ", ".join(tool for tool in task.required_tools if tool in {"asan", "ubsan"})
                if any(tool in {"asan", "ubsan"} for tool in task.required_tools)
                else "ASan/UBSan strengthening when supported",
                "static_analysis": "Clang diagnostics when available",
            },
            "references": [
                {"path_or_id": task.fixture, "purpose": "Public header and project conventions."}
            ],
            "verification_commands": [
                "gcc -std=c11 -Wall -Wextra -Wpedantic -Werror and execute public test",
                *(
                    ["cmake configure", "cmake build", "ctest"]
                    if "cmake" in task.required_tools
                    else []
                ),
                *[
                    f"required {tool} compile and execution"
                    for tool in task.required_tools
                    if tool in {"asan", "ubsan"}
                ],
            ],
            "deliverables": ["bounded C source replacement", "deterministic verifier evidence"],
            "out_of_scope": ["held-out tests", "network", "ABI changes"],
            "assumptions": ["Held-out tests are inaccessible during implementation."],
        }
    protocol = (
        "ready_valid"
        if any(
            token in task.objective.lower() for token in ("ready", "channel", "fifo", "pipeline")
        )
        else "register_or_event"
    )
    return {
        "task_id": task.task_id,
        "objective": task.objective,
        "target": {
            "class": "portable_rtl",
            "language": "Verilog-2001"
            if task.language == "verilog"
            else "SystemVerilog-2017 subset",
            "toolchain": ["iverilog", "verilator", "yosys"],
            "technology_or_device": None,
            "frequency_mhz": None,
        },
        "parameters": [],
        "interfaces": [
            {
                "name": "public_fixture_ports",
                "protocol": protocol,
                "direction": "bidirectional",
                "signals": ["Use the exact module declaration in the editable source."],
                "ordering": "Preserve the specification's cycle ordering.",
                "backpressure": "Hold protocol state stable until an accepted event.",
            }
        ],
        "clock_reset": {
            "clock_domains": ["clk"],
            "reset_semantics": "Use the exact public fixture reset polarity and timing.",
            "cdc_rdc_assumptions": "Single clock; CDC architecture is out of worker scope.",
        },
        "functional_requirements": list(task.requirements),
        "error_and_corner_behavior": [
            "Cover reset, stalls, simultaneous events, boundaries, overflow and underflow where relevant."
        ],
        "coding_constraints": [
            "Synthesizable portable subset only.",
            "No delays, force/release, DPI, UVM, or tool execution in implementation output.",
        ],
        "files_allowed_to_change": list(task.editable_sources),
        "references": [
            {"kind": "project", "identifier": task.fixture, "purpose": "Public fixture first."}
        ],
        "verification": {
            "self_checking": True,
            "tests": list(task.public_tests),
            "assertions": ["Stable state and payload while stalled where applicable."],
            "commands": ["verilator lint", "iverilog/vvp", "yosys synthesis"],
            "acceptance_criteria": [
                "All public and adversarial simulations pass.",
                "Lint and synthesis pass without a protocol waiver.",
            ],
        },
        "deliverables": ["bounded RTL replacement", "deterministic verifier evidence"],
        "out_of_scope": ["held-out tests", "network", "vendor IP", "unlisted files"],
        "assumptions": ["Held-out tests are inaccessible during implementation."],
        "blocking_questions": [],
    }


def _require_base_revision(configuration: ExperimentConfiguration) -> str:
    revision = configuration.base_revision
    if revision is None:
        raise EngineeringError(
            "The execution base revision is not set; pass --base-revision or set "
            f"{configuration.base_revision_environment_variable} after committing the reviewed implementation"
        )
    return revision


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_worktree_clean(repository_root: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    completed = subprocess.run(  # nosec B603
        [git, "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _git_commit_resolves(repository_root: Path, revision: str) -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    completed = subprocess.run(  # nosec B603
        [git, "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0


def _phase(configuration: ExperimentConfiguration, phase_id: PhaseId) -> ExperimentPhase:
    return next(item for item in configuration.phases if item.phase_id == phase_id)


def _phase_arms(
    configuration: ExperimentConfiguration, phase_id: PhaseId
) -> tuple[AblationArm, ...]:
    identifiers = set(_phase(configuration, phase_id).arm_ids)
    return tuple(arm for arm in configuration.arms if arm.arm_id in identifiers)


def _phase_for_arm(arm_id: str) -> PhaseId:
    try:
        return _ARM_PHASE[arm_id]
    except KeyError as exc:
        raise EngineeringError(f"Unknown experiment arm: {arm_id}") from exc


def _artifact_ids_for_phase(phase_id: PhaseId | None) -> set[str]:
    if phase_id == "phase1":
        return {"phase1_main"}
    if phase_id == "phase2":
        return {"phase2_main"}
    if phase_id == "phase3":
        return {"phase2_main", "phase2_rtl_worker"}
    return {"phase1_main", "phase2_main", "phase2_rtl_worker"}


def _phase_manifest_path(configuration: ExperimentConfiguration, phase_id: PhaseId) -> Path:
    return _result_set_root(configuration) / "phases" / phase_id / "manifest.json"


def _result_set_root(configuration: ExperimentConfiguration) -> Path:
    revision = configuration.base_revision or "UNRESOLVED_BASE"
    return configuration.output_root / "result_sets" / f"{revision}_{EVALUATION_PROTOCOL_VERSION}"


def _evaluation_settings(configuration: ExperimentConfiguration) -> JsonObject:
    return {
        "timeout_seconds": configuration.default_timeout_seconds,
        "correction_budget": configuration.correction_budget,
        "bootstrap_samples": configuration.bootstrap_samples,
        "bootstrap_seed": configuration.bootstrap_seed,
        "confidence_level": configuration.confidence_level,
        "held_out_manifest_hashes_required": True,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
    }


def _configuration_fingerprint(
    configuration: ExperimentConfiguration,
    corpus_hashes: dict[str, str],
    held_out_manifest_sha256: str,
) -> JsonObject:
    base_revision = _require_base_revision(configuration)
    settings = json.dumps(_evaluation_settings(configuration), sort_keys=True).encode()
    return {
        "base_revision": base_revision,
        "benchmark_sha256": _sha256_file(configuration.manifest_path),
        "experiment_sha256": _sha256_file(configuration.path),
        "model_artifacts_sha256": _sha256_file(configuration.model_artifacts_path),
        "model_configuration_hashes": {
            arm.arm_id: _sha256_file(arm.models_path) for arm in configuration.arms
        },
        "corpus_hashes": corpus_hashes or {},
        "held_out_manifest_sha256": held_out_manifest_sha256,
        "evaluation_settings_sha256": hashlib.sha256(settings).hexdigest(),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
    }


def _tool_version(name: str) -> JsonObject:
    def executable(command: str) -> str:
        return shutil.which(command) or command

    commands: dict[str, list[str]] = {
        "pytest": [sys.executable, "-m", "pytest", "--version"],
        "ruff": [sys.executable, "-m", "ruff", "--version"],
        "mypy": [sys.executable, "-m", "mypy", "--version"],
        "bandit": [sys.executable, "-m", "bandit", "--version"],
        "verilator_simulation": [executable("verilator"), "--version"],
        "iverilog": [executable("iverilog"), "-V"],
        "vvp": [executable("vvp"), "-V"],
    }
    sanitizer_compiler = _sanitizer_compiler(name) if name in {"asan", "ubsan"} else None
    command = (
        [sanitizer_compiler or "gcc", "--version"]
        if name in {"asan", "ubsan"}
        else commands.get(name, [executable(name), "--version"])
    )
    if not _tool_available(name):
        return {
            "available": False,
            "command": command,
            "version": None,
            "required_version": _TOOL_REQUIREMENTS.get(name, "task contract"),
            "remediation": _tool_remediation(name),
        }
    try:
        completed = subprocess.run(  # nosec B603
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "command": command,
            "version": None,
            "required_version": _TOOL_REQUIREMENTS.get(name, "task contract"),
            "remediation": _tool_remediation(name),
            "error": str(exc),
        }
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return {
        "available": True,
        "command": command,
        "version": output[0] if output else None,
        "required_version": _TOOL_REQUIREMENTS.get(name, "task contract"),
        "remediation": None,
        "version_command_returncode": completed.returncode,
    }


def _arm_order(index: int) -> tuple[str, ...]:
    base = ("A", "B", "C")
    offset = index % len(base)
    return base[offset:] + base[:offset]


_TOOL_REQUIREMENTS = {
    "pytest": "pytest >=8",
    "ruff": "Ruff >=0.8",
    "mypy": "mypy >=1.13",
    "bandit": "Bandit >=1.7",
    "gcc": "GCC >=8 with C11 support",
    "clang": "Clang 18.1.8 profile",
    "cmake": "CMake >=3.20 (bootstrap pins 3.30.5)",
    "ctest": "CTest matching CMake >=3.20",
    "asan": "functional AddressSanitizer compile and link",
    "ubsan": "functional UndefinedBehaviorSanitizer compile and link",
    "iverilog": "Icarus Verilog >=11",
    "vvp": "vvp matching Icarus Verilog",
    "yosys": "Yosys >=0.40",
    "verilator": "Verilator >=4 for lint",
    "verilator_simulation": "Verilator >=5 with timed --binary simulation",
    "cppcheck": "optional C static analysis",
    "vivado": "optional explicitly configured Vivado",
}


def _tool_remediation(name: str) -> str:
    if name in {
        "clang",
        "cmake",
        "ctest",
        "asan",
        "ubsan",
        "verilator",
        "verilator_simulation",
        "cppcheck",
    }:
        return (
            "scripts/bootstrap_multilanguage_tools.sh install; "
            'export PATH="$PWD/.tools/multilanguage/bin:$PATH"'
        )
    if name == "vivado":
        return (
            "Configure an existing licensed Vivado installation and add its bin directory to PATH"
        )
    if name in {"pytest", "ruff", "mypy", "bandit"}:
        return 'uv pip install --python .venv/bin/python -e ".[dev]"'
    return f"Install {name} in a user-local environment and add it to PATH"


def _sanitizer_compiler(name: str) -> str | None:
    flag = "-fsanitize=address" if name == "asan" else "-fsanitize=undefined"
    for compiler in ("gcc", "clang"):
        executable = shutil.which(compiler)
        if executable is None:
            continue
        # Compile-link a constant probe because some toolchains expose stale
        # linker scripts even when the sanitizer runtime itself is absent.
        with tempfile.TemporaryDirectory(prefix="laplace-sanitizer-probe-") as directory:
            try:
                completed = subprocess.run(  # nosec B603
                    [executable, "-x", "c", "-", flag, "-o", str(Path(directory) / "probe")],
                    input="int main(void) { return 0; }\n",
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if completed.returncode == 0:
                return executable
    return None


def _tool_available(name: str) -> bool:
    if name == "verilator_simulation":
        return verilator_simulation_available()
    if name in {"asan", "ubsan"}:
        return _sanitizer_compiler(name) is not None
    if name in {"pytest", "ruff", "mypy", "bandit"}:
        return importlib.util.find_spec(name) is not None
    return shutil.which(name) is not None


def build_plan(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    *,
    phase_id: PhaseId | None = None,
) -> JsonObject:
    """Describe all work without endpoint calls, model calls, or verification."""
    root = repository_root.resolve()
    _activate_isolated_tools(root)
    tasks = load_benchmark_manifest(configuration.manifest_path)
    selected_arms = (
        _phase_arms(configuration, phase_id) if phase_id is not None else configuration.arms
    )
    tool_availability = {
        name: _tool_available(name)
        for name in sorted({tool for task in tasks for tool in task.required_tools})
    }
    records: list[JsonObject] = []
    global_missing: set[str] = set()
    base_revision = configuration.base_revision
    if base_revision is None:
        global_missing.add(
            f"base_revision_environment:{configuration.base_revision_environment_variable}"
        )
    elif not _git_commit_resolves(root, base_revision):
        global_missing.add(f"base_revision_unresolvable:{base_revision}")
    revision_placeholders = [
        f"arm_{arm.arm_id}:{candidate.model}"
        for arm in selected_arms
        for candidate in (
            [arm.models.main]
            + ([arm.models.rtl_worker] if arm.models.rtl_worker is not None else [])
        )
        if candidate.revision.startswith("SET_EXACT_")
    ]
    if revision_placeholders:
        global_missing.add("exact_model_revisions:" + ",".join(revision_placeholders))
    path_placeholders = [
        f"arm_{arm.arm_id}:{candidate.model}"
        for arm in selected_arms
        for candidate in (
            [arm.models.main]
            + ([arm.models.rtl_worker] if arm.models.rtl_worker is not None else [])
        )
        if candidate.model_path is None or candidate.model_path.startswith("SET_EXACT_")
    ]
    if path_placeholders:
        global_missing.add("exact_model_paths:" + ",".join(path_placeholders))
    for arm in selected_arms:
        for candidate in [arm.models.main] + (
            [arm.models.rtl_worker] if arm.models.rtl_worker is not None else []
        ):
            if (
                candidate.model_path is not None
                and not candidate.model_path.startswith("SET_EXACT_")
                and not Path(candidate.model_path).expanduser().is_dir()
            ):
                global_missing.add(f"model_path:{candidate.model_path}")
    held_root = os.getenv(configuration.held_out_environment_variable)
    if not held_root:
        global_missing.add(
            f"held_out_root_environment:{configuration.held_out_environment_variable}"
        )
    else:
        try:
            validate_held_out_pack(root, configuration, Path(held_root))
        except EngineeringError as exc:
            global_missing.add(f"held_out_pack:{exc}")
    if not configuration.base_reference_root.is_dir():
        global_missing.add(f"base_reference_root:{configuration.base_reference_root}")
    try:
        load_bundled_corpus_manifest(root)
        load_installed_external_manifest(root)
    except ReferencePolicyError as exc:
        global_missing.add(f"governed_corpus:{exc}")
    profile_alignment = validate_profile_alignment(configuration.model_artifacts_path.parent)
    if profile_alignment.get("status") != "VALID_MODEL_PROFILES":
        global_missing.add(
            "model_profiles:" + ",".join(cast(list[str], profile_alignment["errors"]))
        )
    selected_artifact_ids = _artifact_ids_for_phase(phase_id)
    local_artifacts = validate_local_artifacts(
        configuration.model_artifacts_path.parent,
        selected_artifact_ids,
        verify_hashes=False,
    )
    available_artifact_ids: set[str] = set()
    for raw_artifact in cast(list[object], local_artifacts["artifacts"]):
        if not isinstance(raw_artifact, dict):
            continue
        artifact_id = str(raw_artifact.get("artifact_id"))
        if artifact_id not in selected_artifact_ids:
            continue
        if raw_artifact.get("available") is True:
            available_artifact_ids.add(artifact_id)
        else:
            global_missing.add(f"model_artifact:{artifact_id}:{raw_artifact.get('output_path')}")
    if available_artifact_ids:
        serving_environments = validate_serving_environments(
            configuration.model_artifacts_path.parent,
            available_artifact_ids,
            probe_cli=False,
        )
        for raw_environment in cast(list[object], serving_environments["environments"]):
            if not isinstance(raw_environment, dict) or raw_environment.get("available") is True:
                continue
            global_missing.add(
                "serving_environment:"
                f"{raw_environment.get('artifact_id')}:{raw_environment.get('environment_path')}"
            )
    for index, task in enumerate(tasks):
        missing: list[str] = []
        for path in (*task.editable_sources, *task.public_tests):
            if not (root / path).is_file():
                missing.append(f"fixture_path:{path}")
            if base_revision is not None and not _git_object_exists(root, base_revision, path):
                missing.append(f"base_revision_missing:{path}")
        for tool in task.required_tools:
            if not tool_availability[tool]:
                missing.append(f"tool:{tool}")
        global_missing.update(missing)
        arm_by_id = {arm.arm_id: arm for arm in configuration.arms}
        expected_routes: JsonObject = {
            "A": f"main:{arm_by_id['A'].models.main.model}",
            "B": f"main:{arm_by_id['B'].models.main.model}",
            "C": (
                f"rtl_worker:{arm_by_id['C'].models.rtl_worker.model}"
                if task.routing.worker_eligible and arm_by_id["C"].models.rtl_worker is not None
                else f"main:{arm_by_id['C'].models.main.model}"
            ),
        }
        records.append(
            {
                "task_id": task.task_id,
                "language": task.language,
                "category": task.category,
                "regression": task.regression,
                "arm_order": list(_arm_order(index)),
                "phase_arm_order": {
                    "phase1": ["A"],
                    "phase2": ["B"],
                    "phase3": ["C"],
                },
                "worker_eligibility": assess_rtl_worker_eligibility(task.routing).to_json(),
                "expected_model_route": expected_routes,
                "required_tools": list(task.required_tools),
                "missing_prerequisites": sorted(missing),
            }
        )
    return {
        "schema_version": 1,
        "status": "PLAN_READY" if not global_missing else "PLAN_READY_WITH_MISSING_PREREQUISITES",
        "experiment_id": EXPERIMENT_ID,
        "mode": "PLAN_ONLY_NO_MODEL_OR_VERIFICATION",
        "base_revision": base_revision,
        "base_revision_source": (
            "resolved_runtime_argument_or_environment"
            if base_revision is not None
            else f"unset:{configuration.base_revision_environment_variable}"
        ),
        "execution_phases": [
            {
                "phase_id": phase.phase_id,
                "label": phase.label,
                "arms": list(phase.arm_ids),
                "requires_completed_phases": list(phase.requires_completed_phases),
            }
            for phase in configuration.phases
        ],
        "selected_phase": phase_id,
        "task_count": len(records),
        "arm_order_strategy": "serialized_phase_then_deterministic_rotation",
        "endpoint_status": "NOT_PROBED_PLAN_ONLY",
        "tasks": records,
        "missing_prerequisites": sorted(global_missing),
    }


def _git_object_exists(repository_root: Path, revision: str, path: str) -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    completed = subprocess.run(  # nosec B603
        [git, "cat-file", "-e", f"{revision}:{path}"],
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0


def runtime_prerequisites(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    *,
    phase_id: PhaseId | None = None,
) -> JsonObject:
    plan = build_plan(repository_root, configuration, phase_id=phase_id)
    missing = list(cast(list[object], plan["missing_prerequisites"]))
    if configuration.base_revision is None:
        return {"passed": False, "missing": sorted(set(str(item) for item in missing))}
    if not _git_worktree_clean(repository_root):
        missing.append("repository_worktree_not_clean")
    committed_paths = [
        str(configuration.path.relative_to(repository_root)),
        str(configuration.manifest_path.relative_to(repository_root)),
        str(configuration.model_artifacts_path.relative_to(repository_root)),
        *(str(arm.models_path.relative_to(repository_root)) for arm in configuration.arms),
    ]
    for path in committed_paths:
        if not _git_object_exists(repository_root, configuration.base_revision, path):
            missing.append(f"base_revision_missing_experiment_file:{path}")
    for task in load_benchmark_manifest(configuration.manifest_path):
        for path in (*task.editable_sources, *task.public_tests):
            if not _git_object_exists(repository_root, configuration.base_revision, path):
                missing.append(f"base_revision_missing:{path}")
    if phase_id is not None:
        held_value = os.getenv(configuration.held_out_environment_variable)
        held_hash = (
            _sha256_file(Path(held_value).expanduser().resolve() / "manifest.json")
            if held_value
            else "UNRESOLVED"
        )
        corpus_hashes = _corpus_snapshot_hashes(configuration.overlay_root)
        for dependency in _phase(configuration, phase_id).requires_completed_phases:
            dependency_status = phase_status(configuration, dependency)
            if dependency_status.get("status") != "COMPLETE":
                missing.append(f"{dependency}_not_complete_or_incompatible")
                continue
            manifest = dependency_status.get("manifest")
            if not isinstance(manifest, dict):
                missing.append(f"{dependency}_manifest_missing")
                continue
            try:
                _assert_phase_manifest_compatible(
                    configuration,
                    cast(JsonObject, manifest),
                    corpus_hashes,
                    held_hash,
                )
            except EngineeringError as exc:
                missing.append(f"{dependency}_incompatible:{exc}")
    return {"passed": not missing, "missing": sorted(set(str(item) for item in missing))}


def validate_runtime(
    repository_root: Path, configuration: ExperimentConfiguration, phase_id: PhaseId
) -> JsonObject:
    """Probe hardware and exact served identities without generating tokens."""
    prerequisites = runtime_prerequisites(repository_root, configuration, phase_id=phase_id)
    if prerequisites.get("passed") is not True:
        return {
            "status": "FAILED",
            "prerequisites": prerequisites,
            "cuda": {"status": "NOT_PROBED"},
            "endpoints": {},
        }
    held_value = os.getenv(configuration.held_out_environment_variable)
    if not held_value:
        raise EngineeringError("Held-out root environment variable is not configured")
    validate_held_out_pack(repository_root, configuration, Path(held_value))
    cuda = collect_cuda_evidence(
        LocalToolRunner(repository_root, configuration.output_root / "runtime_preflight_logs")
    )
    if cuda.get("status") != "CUDA_A6000_VERIFIED":
        return {"status": "FAILED", "prerequisites": prerequisites, "cuda": cuda, "endpoints": {}}
    from .model_routing import AuditedModelCaller, RoleRouter

    endpoints: JsonObject = {}
    passed = True
    for arm in _phase_arms(configuration, phase_id):
        health = AuditedModelCaller(
            RoleRouter(arm.models), configuration.output_root / "runtime_preflight_model_calls"
        ).health(include_worker=arm.worker_enabled)
        endpoints[arm.arm_id] = health
        passed = passed and all(
            isinstance(record, dict) and record.get("status") == "AVAILABLE"
            for record in health.values()
        )
    return {
        "status": "RUNTIME_READY" if passed else "FAILED",
        "phase_id": phase_id,
        "prerequisites": prerequisites,
        "cuda": cuda,
        "endpoints": endpoints,
    }


def _corpus_snapshot_hashes(overlay_root: Path) -> dict[str, str]:
    return {
        domain: ReferenceLibrary(overlay_root, domain, shared=True).snapshot_hash()
        or "UNINITIALIZED"
        for domain in ("python", "c", "verilog", "systemverilog")
    }


def _pair_complete(
    configuration: ExperimentConfiguration,
    arm_id: str,
    task_id: str,
    *,
    held_out_manifest_sha256: str | None = None,
    configuration_fingerprint: JsonObject | None = None,
) -> bool:
    path = _pair_result_path(configuration, arm_id, task_id)
    if not path.is_file():
        return False
    value = _read_json(path)
    try:
        _validate_terminal_pair_result(value)
    except EngineeringError:
        return False
    expected_phase = _phase_for_arm(arm_id)
    return (
        value.get("schema_version") == RESULT_SCHEMA_VERSION
        and value.get("status") == "COMPLETE_EVALUATED"
        and value.get("terminal") is True
        and value.get("outcome_kind") == "candidate_result"
        and value.get("base_revision") == configuration.base_revision
        and value.get("execution_phase") == expected_phase
        and configuration_fingerprint is not None
        and value.get("configuration_fingerprint") == configuration_fingerprint
        and (
            held_out_manifest_sha256 is None
            or value.get("held_out_manifest_sha256") == held_out_manifest_sha256
        )
    )


def phase_status(configuration: ExperimentConfiguration, phase_id: PhaseId) -> JsonObject:
    tasks = load_benchmark_manifest(configuration.manifest_path)
    phase = _phase(configuration, phase_id)
    expected = [(task.task_id, arm_id) for task in tasks for arm_id in phase.arm_ids]
    manifest_path = _phase_manifest_path(configuration, phase_id)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
    fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
    compatibility_errors: list[str] = []
    if isinstance(fingerprint, dict):
        raw_corpus = fingerprint.get("corpus_hashes")
        raw_held = fingerprint.get("held_out_manifest_sha256")
        if isinstance(raw_corpus, dict) and isinstance(raw_held, str):
            expected_fingerprint = _configuration_fingerprint(
                configuration,
                {str(key): str(value) for key, value in raw_corpus.items()},
                raw_held,
            )
            compatibility_errors = [
                key
                for key in (
                    "base_revision",
                    "benchmark_sha256",
                    "experiment_sha256",
                    "model_artifacts_sha256",
                    "model_configuration_hashes",
                    "evaluation_settings_sha256",
                    "result_schema_version",
                    "evaluation_protocol_version",
                )
                if fingerprint.get(key) != expected_fingerprint[key]
            ]
        else:
            compatibility_errors = ["malformed_fingerprint"]
    held_out_manifest_sha256 = (
        fingerprint.get("held_out_manifest_sha256")
        if isinstance(fingerprint, dict)
        and isinstance(fingerprint.get("held_out_manifest_sha256"), str)
        else None
    )
    compatible_fingerprint = (
        cast(JsonObject, fingerprint)
        if isinstance(fingerprint, dict) and not compatibility_errors
        else None
    )
    completed = [
        {"task_id": task_id, "arm_id": arm_id}
        for task_id, arm_id in expected
        if _pair_complete(
            configuration,
            arm_id,
            task_id,
            held_out_manifest_sha256=held_out_manifest_sha256,
            configuration_fingerprint=compatible_fingerprint,
        )
    ]
    terminal_failures: list[JsonObject] = []
    attempted_terminal: list[JsonObject] = []
    for task_id, arm_id in expected:
        path = _pair_result_path(configuration, arm_id, task_id)
        if not path.is_file():
            continue
        row = _read_json(path)
        compatible = (
            compatible_fingerprint is not None
            and row.get("schema_version") == RESULT_SCHEMA_VERSION
            and row.get("terminal") is True
            and row.get("base_revision") == configuration.base_revision
            and row.get("execution_phase") == phase_id
            and row.get("configuration_fingerprint") == compatible_fingerprint
            and row.get("held_out_manifest_sha256") == held_out_manifest_sha256
        )
        if not compatible:
            continue
        record: JsonObject = {
            "task_id": task_id,
            "arm_id": arm_id,
            "status": row.get("status"),
        }
        attempted_terminal.append(record)
        if row.get("status") == "TERMINAL_FAILURE":
            terminal_failures.append(record)
    pairs_complete = len(completed) == len(expected)
    status = "INCOMPLETE"
    if pairs_complete:
        status = (
            "COMPLETE"
            if manifest is not None and manifest.get("status") == "COMPLETE"
            else "PAIRS_COMPLETE_MANIFEST_PENDING"
        )
    if manifest is not None and manifest.get("status") == "INCOMPATIBLE":
        status = "INCOMPATIBLE"
    if compatibility_errors:
        status = "INCOMPATIBLE"
    return {
        "status": status,
        "phase_id": phase_id,
        "expected_pairs": len(expected),
        "completed_pairs": len(completed),
        "remaining_pairs": len(expected) - len(completed),
        "attempted_terminal_pairs": len(attempted_terminal),
        "terminal_failure_pairs": len(terminal_failures),
        "completed_task_arm_pairs": completed,
        "terminal_failure_task_arm_pairs": terminal_failures,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "compatibility_errors": compatibility_errors,
    }


def _assert_phase_manifest_compatible(
    configuration: ExperimentConfiguration,
    manifest: JsonObject,
    corpus_hashes: dict[str, str],
    held_out_manifest_sha256: str,
) -> None:
    expected = _configuration_fingerprint(configuration, corpus_hashes, held_out_manifest_sha256)
    actual = manifest.get("fingerprint")
    if not isinstance(actual, dict):
        raise EngineeringError("Phase manifest has no compatibility fingerprint")
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise EngineeringError(
            "Phase results are incompatible and cannot be resumed or merged; mismatched: "
            + ", ".join(mismatches)
        )


def _phase_model_record(arm: AblationArm) -> JsonObject:
    return {
        "arm_id": arm.arm_id,
        "models_config_path": str(arm.models_path),
        "models_config_sha256": _sha256_file(arm.models_path),
        "main": arm.models.main.to_json(),
        "rtl_worker": arm.models.rtl_worker.to_json()
        if arm.models.rtl_worker is not None
        else None,
    }


def _write_phase_manifest(
    configuration: ExperimentConfiguration,
    phase_id: PhaseId,
    *,
    status: str,
    started_at: str,
    corpus_hashes: dict[str, str],
    tool_versions: dict[str, JsonObject],
    endpoint_health: JsonObject,
    gpu_observations: JsonObject,
    held_out_manifest_sha256: str,
) -> JsonObject:
    phase = _phase(configuration, phase_id)
    current = phase_status(configuration, phase_id)
    path = _phase_manifest_path(configuration, phase_id)
    prior = _read_json(path) if path.is_file() else {}
    effective_start = str(prior.get("started_at", started_at))
    wall_elapsed_seconds: float | None = None
    try:
        wall_elapsed_seconds = (
            datetime.now(UTC) - datetime.fromisoformat(effective_start)
        ).total_seconds()
    except ValueError:
        pass
    pair_task_seconds = 0.0
    all_terminal_items = cast(list[object], current["completed_task_arm_pairs"]) + cast(
        list[object], current["terminal_failure_task_arm_pairs"]
    )
    for item in all_terminal_items:
        if not isinstance(item, dict):
            continue
        pair_path = _pair_result_path(
            configuration, str(item.get("arm_id")), str(item.get("task_id"))
        )
        pair = _read_json(pair_path)
        elapsed = pair.get("total_task_seconds")
        if isinstance(elapsed, (int, float)):
            pair_task_seconds += float(elapsed)
    record: JsonObject = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase_id": phase_id,
        "phase_label": phase.label,
        "status": status,
        "active_configuration": [
            _phase_model_record(arm) for arm in _phase_arms(configuration, phase_id)
        ],
        "expected_model_identities": [
            candidate.model
            for arm in _phase_arms(configuration, phase_id)
            for candidate in [arm.models.main]
            + ([arm.models.rtl_worker] if arm.models.rtl_worker is not None else [])
        ],
        "fingerprint": _configuration_fingerprint(
            configuration, corpus_hashes, held_out_manifest_sha256
        ),
        "started_at": effective_start,
        "completed_at": _now() if status == "COMPLETE" else None,
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "sum_completed_task_seconds": pair_task_seconds,
        "runtime_definition": "wall elapsed includes interruptions; summed task time is the measured total of completed task-arm pairs",
        "task_arm_pairs_completed": current["completed_task_arm_pairs"],
        "task_arm_pairs_terminal_failures": current["terminal_failure_task_arm_pairs"],
        "task_arm_pairs_attempted_terminal": current["attempted_terminal_pairs"],
        "expected_pairs": current["expected_pairs"],
        "decoding_settings": {
            arm.arm_id: {
                "temperature": arm.models.main.temperature,
                "top_p": arm.models.main.top_p,
                "seed": arm.models.main.seed,
                "max_output_tokens": arm.models.main.max_output_tokens,
                "context_tokens": arm.models.main.context_tokens,
            }
            for arm in _phase_arms(configuration, phase_id)
        },
        "tool_versions": tool_versions,
        "gpu_memory_observations": gpu_observations,
        "endpoint_health": endpoint_health,
        "server_lifecycle": {
            "management_mode": os.getenv("LAPLACE_SERVER_MANAGEMENT_MODE", "external"),
            "orchestration_id": os.getenv("LAPLACE_SERVER_OWNER_TOKEN"),
            "profiles": sorted(_artifact_ids_for_phase(phase_id)),
        },
        "output_paths": {
            "phase_manifest": str(path),
            "result_set_root": str(_result_set_root(configuration)),
            "pair_results": str(_result_set_root(configuration) / "pairs"),
            "reports": str(_result_set_root(configuration) / "reports"),
        },
    }
    _write_json_atomic(path, record)
    return record


def preflight_report(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    *,
    phase_id: PhaseId | None = None,
    probe_runtime: bool = False,
) -> JsonObject:
    _activate_isolated_tools(repository_root)
    tasks = load_benchmark_manifest(configuration.manifest_path)
    tool_names = sorted({tool for task in tasks for tool in task.required_tools})
    tools = {name: _tool_version(name) for name in tool_names}
    for name, record in tools.items():
        record["gate"] = "mandatory"
        record["affected_tasks"] = [task.task_id for task in tasks if name in task.required_tools]
    optional_names = ("clang", "cmake", "ctest", "cppcheck", "vivado")
    optional_tools = {name: _tool_version(name) for name in optional_names if name not in tools}
    for record in optional_tools.values():
        record["gate"] = "optional_strengthening"
        record["affected_tasks"] = []
    affected = [
        {
            "task_id": task.task_id,
            "language": task.language,
            "missing_required_tools": [
                tool for tool in task.required_tools if tools[tool].get("available") is not True
            ],
        }
        for task in tasks
    ]
    affected = [item for item in affected if item["missing_required_tools"]]
    phases: JsonObject = {}
    selected_phases: tuple[PhaseId, ...] = (phase_id,) if phase_id is not None else _PHASE_IDS
    for selected_phase in selected_phases:
        prerequisites = runtime_prerequisites(
            repository_root, configuration, phase_id=selected_phase
        )
        runtime: JsonObject = {"status": "NOT_PROBED"}
        if probe_runtime and prerequisites.get("passed") is True:
            runtime = validate_runtime(repository_root, configuration, selected_phase)
        phases[selected_phase] = {
            "prerequisites": prerequisites,
            "runtime": runtime,
            "can_start": runtime.get("status") == "RUNTIME_READY" if probe_runtime else False,
            "can_start_reason": (
                "runtime_probe_required"
                if not probe_runtime and prerequisites.get("passed") is True
                else "missing_prerequisites"
                if prerequisites.get("passed") is not True
                else None
            ),
        }
    return {
        "status": "PREFLIGHT_REPORTED",
        "model_calls": False,
        "runtime_probed": probe_runtime,
        "selected_phase": phase_id,
        "tools": tools,
        "optional_strengthening_tools": optional_tools,
        "gate_policy": "A missing task-declared required tool blocks that task; missing optional strengthening tools are recorded and never silently treated as passes.",
        "tasks_affected_by_missing_required_tools": affected,
        "phases": phases,
    }


def validate_phase_setup(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    phase_id: PhaseId,
    *,
    probe_runtime: bool = False,
) -> JsonObject:
    """Validate one phase without requiring artifacts or endpoints from other phases."""
    _activate_isolated_tools(repository_root)
    selected_artifact_ids = _artifact_ids_for_phase(phase_id)
    artifact_validation = validate_local_artifacts(
        configuration.model_artifacts_path.parent, selected_artifact_ids
    )
    prerequisites = runtime_prerequisites(repository_root, configuration, phase_id=phase_id)
    held_value = os.getenv(configuration.held_out_environment_variable)
    held_out: JsonObject = {"status": "MISSING"}
    if held_value:
        held_out = validate_held_out_pack(repository_root, configuration, Path(held_value))
    with tempfile.TemporaryDirectory(prefix=f"laplace-{phase_id}-corpus-") as temporary:
        overlay = Path(temporary) / "Library"
        preparation = prepare_corpus_overlay(
            repository_root, configuration.base_reference_root, overlay
        )
        corpus = validate_corpus_retrieval(overlay)
    runtime: JsonObject = {"status": "NOT_PROBED"}
    if probe_runtime and prerequisites.get("passed") is True:
        runtime = validate_runtime(repository_root, configuration, phase_id)
    ready = (
        prerequisites.get("passed") is True
        and artifact_validation.get("status") == "ALL_MODEL_ARTIFACTS_AVAILABLE"
        and held_out.get("status") == "VALID_ISOLATED_HELD_OUT_PACK"
        and corpus.get("status") == "VERIFIED_NON_EMPTY"
        and (not probe_runtime or runtime.get("status") == "RUNTIME_READY")
    )
    return {
        "status": "PHASE_CONFIGURATION_READY" if ready else "FAILED",
        "phase_id": phase_id,
        "model_calls": False,
        "runtime_probed": probe_runtime,
        "prerequisites": prerequisites,
        "artifact_validation": artifact_validation,
        "held_out": held_out,
        "corpus_preparation": preparation,
        "corpus_validation": corpus,
        "runtime": runtime,
    }


def _held_out_manifest(root: Path, tasks: tuple[AblationTask, ...]) -> JsonObject:
    if root.is_symlink():
        raise EngineeringError("Held-out root must not be a symbolic link")
    manifest_path = root / "manifest.json"
    value = _read_json(manifest_path)
    _exact_keys(value, {"schema_version", "experiment_id", "tasks"}, label="Held-out manifest")
    if value.get("schema_version") != 1 or value.get("experiment_id") != EXPERIMENT_ID:
        raise EngineeringError("Held-out manifest identity is invalid")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, dict):
        raise EngineeringError("Held-out manifest tasks must be an object")
    for task in tasks:
        raw_entry = raw_tasks.get(task.task_id)
        if not isinstance(raw_entry, dict):
            raise EngineeringError(f"Held-out manifest is missing {task.task_id}")
        entry = dict(raw_entry)
        _exact_keys(entry, {"directory", "files"}, label=f"Held-out {task.task_id}")
        directory = _inside(
            root,
            root
            / _safe_relative(
                _string(entry.get("directory"), label="held-out directory"),
                label="held-out directory",
            ),
        )
        files = entry.get("files")
        if not isinstance(files, dict) or not files:
            raise EngineeringError(f"Held-out {task.task_id} files are malformed")
        file_names = set(files)
        if task.language == "python" and "test_heldout.py" not in file_names:
            raise EngineeringError(f"Held-out Python pack is incomplete for {task.task_id}")
        if task.language == "c" and (
            "CMakeLists.txt" not in file_names
            or not any(Path(name).suffix == ".c" for name in file_names)
        ):
            raise EngineeringError(f"Held-out C pack is incomplete for {task.task_id}")
        expected_rtl = (
            "tb_heldout.v"
            if task.language == "verilog"
            else "tb_heldout.sv"
            if task.language == "systemverilog"
            else None
        )
        if expected_rtl is not None and expected_rtl not in file_names:
            raise EngineeringError(f"Held-out RTL pack is incomplete for {task.task_id}")
        for raw_path, digest in files.items():
            if not isinstance(raw_path, str) or not isinstance(digest, str):
                raise EngineeringError(f"Held-out {task.task_id} hash record is malformed")
            file_path = _inside(
                directory, directory / _safe_relative(raw_path, label="held-out file")
            )
            if (
                file_path.is_symlink()
                or not file_path.is_file()
                or hashlib.sha256(file_path.read_bytes()).hexdigest() != digest
            ):
                raise EngineeringError(f"Held-out file hash failed: {task.task_id}/{raw_path}")
    return value


def validate_held_out_pack(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    held_out_root: Path,
) -> JsonObject:
    root = held_out_root.expanduser().resolve()
    repository = repository_root.resolve()
    if root == repository or repository in root.parents:
        raise EngineeringError(
            "Evaluator-held tests must be outside the repository and all implementation worktrees"
        )
    tasks = load_benchmark_manifest(configuration.manifest_path)
    manifest = _held_out_manifest(root, tasks)
    return {
        "status": "VALID_ISOLATED_HELD_OUT_PACK",
        "root": str(root),
        "manifest_sha256": _sha256_file(root / "manifest.json"),
        "task_count": len(cast(dict[str, object], manifest["tasks"])),
        "implementation_content_exposure": False,
    }


def _load_artifact(task_value: JsonObject, name: str) -> JsonObject:
    artifacts = task_value.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    path = artifacts.get(name)
    if not isinstance(path, str) or not Path(path).is_file():
        return {}
    try:
        return _read_json(Path(path))
    except EngineeringError as exc:
        return {
            "status": "MALFORMED_ARTIFACT",
            "artifact_name": name,
            "error": str(exc),
        }


def _model_call_totals(project_root: Path) -> JsonObject:
    prompt_tokens = 0
    completion_tokens = 0
    generation_seconds = 0.0
    count = 0
    invalid = 0
    calls: list[JsonObject] = []
    for path in sorted(project_root.glob("Outputs/AgentTeam/team_logs/model_calls/*.json")):
        try:
            call = _read_json(path)
        except EngineeringError as exc:
            call = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "MALFORMED_MODEL_CALL_AUDIT",
                "response_valid": False,
                "failure_category": "malformed_response",
                "error": str(exc),
                "audit_path": str(path),
            }
        count += 1
        raw_prompt_tokens = call.get("prompt_tokens")
        raw_completion_tokens = call.get("completion_tokens")
        raw_generation_seconds = call.get("generation_seconds")
        if isinstance(raw_prompt_tokens, int):
            prompt_tokens += raw_prompt_tokens
        if isinstance(raw_completion_tokens, int):
            completion_tokens += raw_completion_tokens
        if isinstance(raw_generation_seconds, (int, float)):
            generation_seconds += raw_generation_seconds
        if call.get("response_valid") is not True:
            invalid += 1
        calls.append(call)
    return {
        "count": count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_seconds": generation_seconds,
        "invalid_responses": invalid,
        "calls": calls,
    }


def _rejection_metrics(model_calls: JsonObject) -> JsonObject:
    raw_calls = model_calls.get("calls")
    entries = (
        [item for item in raw_calls if isinstance(item, dict)]
        if isinstance(raw_calls, list)
        else []
    )
    rejected_entries = [item for item in entries if item.get("response_valid") is not True]
    errors = [
        str(item.get("validation_error") or item.get("error") or "").lower()
        for item in rejected_entries
    ]
    return {
        "rejected": len(rejected_entries),
        "malformed": sum(
            any(
                token in error
                for token in (
                    "not valid json",
                    "must be one json object",
                    "keys are invalid",
                    "response is empty",
                    "fenced block",
                )
            )
            for error in errors
        ),
        "stale": sum("stale" in error for error in errors),
        "duplicate_path": sum("duplicated" in error for error in errors),
        "no_op": sum("no source change" in error for error in errors),
        "out_of_scope": sum("outside task scope" in error for error in errors),
    }


def _evaluate_held_out(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    task: AblationTask,
    arm: AblationArm,
    implementation_worktree: Path,
    held_out_root: Path,
) -> JsonObject:
    """Apply the stopped lane's patch in a fresh worktree before hidden checks."""
    _, patch, _ = _run_git(implementation_worktree, ["diff", "--no-ext-diff", "--unified=3"])
    evaluation_project = (
        configuration.output_root / "evaluation_projects" / arm.arm_id / task.task_id
    )
    worktree = WorktreeManager(repository_root, evaluation_project).create(
        f"eval-{arm.arm_id}-{task.task_id}", _require_base_revision(configuration)
    )
    if patch.strip():
        apply_validated_patch(
            worktree,
            patch,
            list(task.editable_sources),
            evaluation_project / "patch_logs",
        )
    manifest = _read_json(held_out_root / "manifest.json")
    entries = manifest.get("tasks")
    if not isinstance(entries, dict) or not isinstance(entries.get(task.task_id), dict):
        raise EngineeringError(f"Held-out evaluator entry is missing for {task.task_id}")
    entry = cast(dict[str, object], entries[task.task_id])
    hidden_directory = _inside(
        held_out_root,
        held_out_root
        / _safe_relative(
            _string(entry.get("directory"), label="held-out directory"),
            label="held-out directory",
        ),
    )
    runner = LocalToolRunner(worktree.root, evaluation_project / "tool_logs")
    if task.language == "python":
        hidden_test = hidden_directory / "test_heldout.py"
        target = worktree.root / task.fixture / "test_evaluator_heldout.py"
        shutil.copy2(hidden_test, target)
        result = runner.run(
            "pytest",
            [sys.executable, "-m", "pytest", "-q", str(target.relative_to(worktree.root))],
            timeout_seconds=configuration.default_timeout_seconds,
        ).to_json()
    elif task.language == "c":
        evaluation_fixture = worktree.root / "heldout_c_evaluation" / task.task_id
        evaluation_fixture.mkdir(parents=True, exist_ok=True)
        for source in task.editable_sources:
            shutil.copy2(worktree.root / source, evaluation_fixture / Path(source).name)
        for hidden_file in hidden_directory.iterdir():
            if hidden_file.is_file():
                shutil.copy2(hidden_file, evaluation_fixture / hidden_file.name)
        result = runner.run_c_quality_gates(
            str(evaluation_fixture.relative_to(worktree.root)),
            timeout_seconds=configuration.default_timeout_seconds,
            required_tools=task.required_tools,
        )
    else:
        suffix = ".v" if task.language == "verilog" else ".sv"
        hidden_test = hidden_directory / f"tb_heldout{suffix}"
        target = worktree.root / task.fixture / f"tb_heldout{suffix}"
        shutil.copy2(hidden_test, target)
        result = runner.run_eda_flow(
            list(task.editable_sources),
            top_module="tb_heldout",
            testbench=str(target.relative_to(worktree.root)),
            language="verilog" if task.language == "verilog" else "systemverilog",
            require_verilator_simulation=task.language == "systemverilog",
            required_tools=task.required_tools,
            timeout_seconds=configuration.default_timeout_seconds,
        )
    passed = result.get("status") == "PASS" or result.get("passed") is True
    return {
        "status": "PASS" if passed else "FAILED",
        "score": 100.0 if passed else 0.0,
        "evaluation_worktree": str(worktree.root),
        "held_out_content_exposed_to_lane": False,
        "evidence": result,
    }


def _pair_result_path(configuration: ExperimentConfiguration, arm_id: str, task_id: str) -> Path:
    return _result_set_root(configuration) / "pairs" / arm_id / f"{task_id}.json"


def _validate_terminal_pair_result(value: JsonObject) -> None:
    required = {
        "schema_version",
        "status",
        "terminal",
        "outcome_kind",
        "experiment_id",
        "task_id",
        "arm_id",
        "execution_phase",
        "base_revision",
        "configuration_fingerprint",
        "held_out_manifest_sha256",
        "started_at",
        "ended_at",
        "total_task_seconds",
        "model_calls",
        "held_out",
        "resumability",
        "lane_result",
    }
    missing = sorted(required - set(value))
    if missing:
        raise EngineeringError(f"Terminal pair result is missing required fields: {missing}")
    if value.get("schema_version") != RESULT_SCHEMA_VERSION or value.get("terminal") is not True:
        raise EngineeringError("Terminal pair result schema identity is invalid")
    status = value.get("status")
    outcome = value.get("outcome_kind")
    if status == "COMPLETE_EVALUATED" and outcome != "candidate_result":
        raise EngineeringError("Complete evaluated result must be a candidate result")
    if status == "TERMINAL_FAILURE":
        failure_required = {
            "route",
            "stage",
            "failure_category",
            "error",
            "retry_counts",
            "deterministic_verification",
            "reviewer_status",
        }
        failure_missing = sorted(failure_required - set(value))
        if outcome != "infrastructure_failure" or failure_missing:
            raise EngineeringError(
                "Typed infrastructure failure is malformed; missing: " + ", ".join(failure_missing)
            )
    elif status != "COMPLETE_EVALUATED":
        raise EngineeringError(f"Unknown terminal pair result status: {status}")


def _task_failure_stage(task_value: JsonObject | None) -> str:
    state = task_value.get("state") if isinstance(task_value, dict) else None
    return {
        "request": "task_normalization",
        "requirements": "planning",
        "plan": "planning",
        "retrieval": "retrieval",
        "implementation": "implementation",
        "verification": "verification",
        "review": "reviewer",
        "bounded_correction": "repair",
        "final_report": "finalization",
        "blocked": "orchestration",
    }.get(str(state), "orchestration")


def _exception_failure_category(exc: Exception) -> str:
    if isinstance(exc, ContextBudgetError):
        return "context_overflow"
    if isinstance(exc, ModelInvocationError):
        return exc.category
    if isinstance(exc, ModelRequired):
        return "endpoint_unavailable"
    if isinstance(exc, (subprocess.TimeoutExpired, TimeoutError)):
        return "timeout"
    name = exc.__class__.__name__.lower()
    if "structured" in name or "validation" in name:
        return "malformed_response"
    if "tool" in name:
        return "verification_infrastructure_failure"
    return "orchestration_failure"


def _terminal_lane_failure(
    *,
    task: AblationTask,
    arm: AblationArm,
    stage: str,
    category: str,
    error: str,
    task_value: JsonObject | None = None,
    worktree: str | None = None,
) -> JsonObject:
    return {
        "status": "TERMINAL_FAILURE",
        "terminal": True,
        "outcome_kind": "infrastructure_failure",
        "task_id": task.task_id,
        "arm_id": arm.arm_id,
        "stage": stage,
        "failure_category": category,
        "error": error,
        "task": task_value or {},
        "worktree": worktree,
        "resumability": {
            "classification": "RETRYABLE_TERMINAL_FAILURE",
            "skip_on_resume": False,
        },
    }


def _execute_task_lane(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    task: AblationTask,
    arm: AblationArm,
    project: Path,
) -> JsonObject:
    """Execute one implementation lane in the child process, without held-out access."""
    normalized = normalize_task_spec(repository_root, task.language, _task_spec(task))
    normalized["resolved_public_tests"] = list(task.public_tests)
    store = AgentTaskStore(project)
    store.create(task.language, normalized)
    metadata = _routing_metadata(
        task.task_id,
        task.language,
        task.category,
        task.editable_sources,
        {
            "worker_eligible": task.routing.worker_eligible,
            "rtl_scope": task.routing.rtl_scope,
            **(
                {}
                if task.language in {"python", "c"}
                else {
                    "module_count": task.routing.module_count,
                    "synthesizable": task.routing.synthesizable,
                    "explicit_ports": task.routing.explicit_ports,
                    "cycle_behavior_specified": task.routing.cycle_behavior_specified,
                    "deterministic_verification": task.routing.deterministic_verification,
                    "unresolved_architecture": task.routing.unresolved_architecture,
                }
            ),
        },
        arm=arm.arm_id,
    )
    runner = LocalTeamRunner(
        repository_root,
        project,
        arm.models.main,
        rtl_worker_candidate=arm.models.rtl_worker,
        dual_model_configuration=arm.models,
        routing_metadata=metadata,
        experiment_arm=arm.arm_id,
        options=TeamWorkflowOptions(
            base_commit=_require_base_revision(configuration),
            required_tools=task.required_tools,
        ),
        shared_reference_root=configuration.overlay_root,
    )
    host_runner = LocalToolRunner(repository_root, project / "host_logs")
    gpu_before = gpu_memory_snapshot(host_runner)
    try:
        lane = runner.run(task.task_id, query=task.objective)
    except Exception as exc:
        try:
            failed_task = store.load(task.task_id).to_json()
        except EngineeringError:
            failed_task = {}
        worktrees = sorted((project / "Data" / "AgentTeam" / "worktrees").glob("*"))
        lane = _terminal_lane_failure(
            task=task,
            arm=arm,
            stage=_task_failure_stage(failed_task),
            category=_exception_failure_category(exc),
            error=str(exc),
            task_value=failed_task,
            worktree=str(worktrees[-1]) if worktrees else None,
        )
    gpu_after = gpu_memory_snapshot(host_runner)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": lane.get("status"),
        "task_id": task.task_id,
        "arm_id": arm.arm_id,
        "project": str(project),
        "lane": lane,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def _execute_lane_request(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    request_path: Path,
) -> JsonObject:
    request = _read_json(request_path)
    _exact_keys(
        request,
        {"schema_version", "task_id", "arm_id", "project", "result_path"},
        label="Lane request",
    )
    if request.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise EngineeringError(f"Lane request schema_version must equal {RESULT_SCHEMA_VERSION}")
    task_id = _string(request.get("task_id"), label="lane task_id")
    arm_id = _string(request.get("arm_id"), label="lane arm_id")
    tasks = {task.task_id: task for task in load_benchmark_manifest(configuration.manifest_path)}
    arms = {arm.arm_id: arm for arm in configuration.arms}
    if task_id not in tasks or arm_id not in arms:
        raise EngineeringError("Lane request task or arm is unknown")
    project = _inside(
        configuration.output_root,
        Path(_string(request.get("project"), label="lane project")).resolve(),
    )
    result_path = _inside(
        configuration.output_root,
        Path(_string(request.get("result_path"), label="lane result path")).resolve(),
    )
    try:
        bundle = _execute_task_lane(
            repository_root,
            configuration,
            tasks[task_id],
            arms[arm_id],
            project,
        )
    except Exception as exc:
        task_value: JsonObject = {}
        try:
            task_value = AgentTaskStore(project).load(task_id).to_json()
        except EngineeringError:
            pass
        lane = _terminal_lane_failure(
            task=tasks[task_id],
            arm=arms[arm_id],
            stage=_task_failure_stage(task_value),
            category=_exception_failure_category(exc),
            error=str(exc),
            task_value=task_value,
        )
        bundle = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "TERMINAL_FAILURE",
            "task_id": task_id,
            "arm_id": arm_id,
            "project": str(project),
            "lane": lane,
            "gpu_before": {"status": "UNAVAILABLE"},
            "gpu_after": {"status": "UNAVAILABLE"},
        }
    _write_json_atomic(result_path, bundle)
    return bundle


def _run_lane_subprocess(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    task: AblationTask,
    arm: AblationArm,
) -> JsonObject:
    """Launch a killable task lane using the established paired-benchmark timeout helper."""
    from .paired_benchmark import _await_lane

    attempt = uuid.uuid4().hex
    project = configuration.output_root / "projects" / arm.arm_id / task.task_id / attempt
    request_path = (
        configuration.output_root / "lane_requests" / f"{arm.arm_id}_{task.task_id}_{attempt}.json"
    )
    lane_result_path = (
        configuration.output_root / "lane_results" / f"{arm.arm_id}_{task.task_id}_{attempt}.json"
    )
    command_log = (
        configuration.output_root / "command_logs" / task.task_id / f"{arm.arm_id}_{attempt}.log"
    )
    _write_json_atomic(
        request_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task_id": task.task_id,
            "arm_id": arm.arm_id,
            "project": str(project),
            "result_path": str(lane_result_path),
        },
    )
    command_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "research_workspace.multilanguage_ablation",
        "run-pair",
        "--config",
        str(configuration.path),
        "--base-revision",
        _require_base_revision(configuration),
        "--request",
        str(request_path),
    ]
    started_at = _now()
    started = time.monotonic()
    lane_environment = os.environ.copy()
    for key in tuple(lane_environment):
        if key.startswith("LAPLACE_ABLATION_") or key == "LAPLACE_SERVER_OWNER_TOKEN":
            lane_environment.pop(key, None)
    process_result: JsonObject
    try:
        with command_log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(  # nosec B603
                command,
                cwd=repository_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=lane_environment,
            )
            process_result = _await_lane(
                process, timeout_seconds=configuration.default_timeout_seconds
            )
    except (OSError, subprocess.SubprocessError) as exc:
        process_result = {
            "status": "FAILED_TO_START",
            "returncode": None,
            "error": str(exc),
        }
    bundle: JsonObject = {}
    if lane_result_path.is_file():
        try:
            bundle = _read_json(lane_result_path)
        except EngineeringError as exc:
            bundle = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "TERMINAL_FAILURE",
                "task_id": task.task_id,
                "arm_id": arm.arm_id,
                "project": str(project),
                "lane": _terminal_lane_failure(
                    task=task,
                    arm=arm,
                    stage="subprocess",
                    category="malformed_lane_result",
                    error=str(exc),
                ),
            }
    if not bundle:
        process_status = str(process_result.get("status", "FAILED"))
        category = "timeout" if process_status == "TIMEOUT" else "subprocess_termination"
        bundle = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "TERMINAL_FAILURE",
            "task_id": task.task_id,
            "arm_id": arm.arm_id,
            "project": str(project),
            "lane": _terminal_lane_failure(
                task=task,
                arm=arm,
                stage="subprocess",
                category=category,
                error=(
                    f"Task lane ended with {process_status} and return code "
                    f"{process_result.get('returncode')} before emitting a result: "
                    f"{process_result.get('error', 'no child result')}"
                ),
            ),
        }
    return {
        "status": process_result.get("status"),
        "returncode": process_result.get("returncode"),
        "started_at": started_at,
        "ended_at": _now(),
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": configuration.default_timeout_seconds,
        "command": command,
        "command_log": str(command_log),
        "project": str(project),
        "bundle": bundle,
    }


def _terminal_pair_result(
    *,
    configuration: ExperimentConfiguration,
    configuration_fingerprint: JsonObject,
    held_out_manifest_sha256: str,
    task: AblationTask,
    arm: AblationArm,
    phase_id: PhaseId,
    pair_started_at: str,
    elapsed_seconds: float,
    lane_process: JsonObject,
    lane: JsonObject,
    model_calls: JsonObject,
    verification: JsonObject,
    review: JsonObject,
    expected_route: str,
    actual_routes: list[object],
    stage: str,
    category: str,
    error: str,
) -> JsonObject:
    raw_task = lane.get("task")
    task_value = cast(JsonObject, raw_task) if isinstance(raw_task, dict) else {}
    retry_indexes: list[int] = []
    for item in cast(list[object], model_calls.get("calls", [])):
        if not isinstance(item, dict):
            continue
        retry_index = item.get("retry_index")
        if isinstance(retry_index, int):
            retry_indexes.append(retry_index)
    lane_process_evidence = dict(lane_process)
    lane_process_evidence.pop("bundle", None)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "TERMINAL_FAILURE",
        "terminal": True,
        "outcome_kind": "infrastructure_failure",
        "experiment_id": EXPERIMENT_ID,
        "task_id": task.task_id,
        "language": task.language,
        "category": task.category,
        "arm_id": arm.arm_id,
        "base_revision": _require_base_revision(configuration),
        "execution_phase": phase_id,
        "configuration_fingerprint": configuration_fingerprint,
        "held_out_manifest_sha256": held_out_manifest_sha256,
        "started_at": pair_started_at,
        "ended_at": _now(),
        "total_task_seconds": elapsed_seconds,
        "configured_timeout_seconds": configuration.default_timeout_seconds,
        "route": {"expected": expected_route, "actual": actual_routes},
        "stage": stage,
        "failure_category": category,
        "error": error,
        "retry_counts": {
            "correction_loops": task_value.get("correction_loops", 0),
            "model_call_retries": max(retry_indexes, default=0),
        },
        "model_calls": model_calls,
        "deterministic_verification": {
            "status": "PASS"
            if verification.get("passed") is True
            else "FAILED"
            if verification
            else "NOT_RUN",
            "evidence": verification,
        },
        "held_out": {
            "status": "NOT_RUN_INFRASTRUCTURE_FAILURE",
            "score": None,
            "included_in_model_quality_metrics": False,
        },
        "reviewer_status": {
            "status": review.get("status", "NOT_RUN") if review else "NOT_RUN",
            "evidence": review,
        },
        "resumability": {
            "classification": "RETRYABLE_TERMINAL_FAILURE",
            "skip_on_resume": False,
        },
        "lane_process": lane_process_evidence,
        "command_log": lane_process.get("command_log"),
        "lane_result": lane,
    }


def run_phase(
    repository_root: Path,
    configuration: ExperimentConfiguration,
    phase_id: PhaseId,
) -> JsonObject:
    """Run or resume one serialized phase of the single logical experiment."""
    prerequisites = runtime_prerequisites(repository_root, configuration, phase_id=phase_id)
    if prerequisites.get("passed") is not True:
        raise EngineeringError(
            "Experiment prerequisites are incomplete: "
            + ", ".join(cast(list[str], prerequisites["missing"]))
        )
    tasks = load_benchmark_manifest(configuration.manifest_path)
    held_value = os.getenv(configuration.held_out_environment_variable)
    if not held_value:
        raise EngineeringError("Held-out root environment variable is not configured")
    held_root = Path(held_value).expanduser().resolve()
    held_out_validation = validate_held_out_pack(repository_root, configuration, held_root)
    held_out_manifest_sha256 = _string(
        held_out_validation.get("manifest_sha256"),
        label="held-out manifest sha256",
    )
    configuration.output_root.mkdir(parents=True, exist_ok=True)
    corpus = prepare_corpus_overlay(
        repository_root,
        configuration.base_reference_root,
        configuration.overlay_root,
    )
    corpus_validation = validate_corpus_retrieval(configuration.overlay_root)
    if corpus_validation.get("status") != "VERIFIED_NON_EMPTY":
        raise EngineeringError("Governed corpus validation did not pass")
    host_runner = LocalToolRunner(repository_root, configuration.output_root / "host_logs")
    cuda = collect_cuda_evidence(host_runner)
    if cuda.get("status") != "CUDA_A6000_VERIFIED":
        raise EngineeringError("Real A6000 CUDA is required; CPU inference is not a fallback")
    endpoint_health: JsonObject = {}
    from .model_routing import AuditedModelCaller, RoleRouter

    selected_arms = _phase_arms(configuration, phase_id)
    for arm in selected_arms:
        caller = AuditedModelCaller(
            RoleRouter(arm.models), configuration.output_root / "preflight_model_calls"
        )
        health = caller.health(include_worker=arm.worker_enabled)
        endpoint_health[arm.arm_id] = health
        if any(
            not isinstance(record, dict) or record.get("status") != "AVAILABLE"
            for record in health.values()
        ):
            raise EngineeringError(f"Arm {arm.arm_id} endpoint identity preflight failed")
    corpus_hashes = _corpus_snapshot_hashes(configuration.overlay_root)
    configuration_fingerprint = _configuration_fingerprint(
        configuration, corpus_hashes, held_out_manifest_sha256
    )
    for dependency in _phase(configuration, phase_id).requires_completed_phases:
        dependency_path = _phase_manifest_path(configuration, dependency)
        if not dependency_path.is_file():
            raise EngineeringError(f"{phase_id} requires a completed {dependency} manifest")
        dependency_manifest = _read_json(dependency_path)
        if dependency_manifest.get("status") != "COMPLETE":
            raise EngineeringError(f"{phase_id} requires {dependency} status COMPLETE")
        _assert_phase_manifest_compatible(
            configuration,
            dependency_manifest,
            corpus_hashes,
            held_out_manifest_sha256,
        )
    tool_names = sorted({tool for task in tasks for tool in task.required_tools})
    tool_versions = {name: _tool_version(name) for name in tool_names}
    completed = 0
    incomplete = 0
    resumed = 0
    started_at = _now()
    phase_gpu_start = gpu_memory_snapshot(host_runner)
    phase_manifest_path = _phase_manifest_path(configuration, phase_id)
    if phase_manifest_path.is_file():
        _assert_phase_manifest_compatible(
            configuration,
            _read_json(phase_manifest_path),
            corpus_hashes,
            held_out_manifest_sha256,
        )
    _write_phase_manifest(
        configuration,
        phase_id,
        status="RUNNING",
        started_at=started_at,
        corpus_hashes=corpus_hashes,
        tool_versions=tool_versions,
        endpoint_health=endpoint_health,
        gpu_observations={"cuda_validation": cuda, "phase_start": phase_gpu_start},
        held_out_manifest_sha256=held_out_manifest_sha256,
    )
    for index, task in enumerate(tasks):
        by_id = {arm.arm_id: arm for arm in selected_arms}
        phase_order = [arm_id for arm_id in _arm_order(index) if arm_id in by_id]
        for arm_id in phase_order:
            arm = by_id[arm_id]
            result_path = _pair_result_path(configuration, arm.arm_id, task.task_id)
            if _pair_complete(
                configuration,
                arm.arm_id,
                task.task_id,
                held_out_manifest_sha256=held_out_manifest_sha256,
                configuration_fingerprint=configuration_fingerprint,
            ):
                resumed += 1
                continue
            pair_started = time.monotonic()
            pair_started_at = _now()
            lane_process = _run_lane_subprocess(repository_root, configuration, task, arm)
            raw_bundle = lane_process.get("bundle")
            bundle = cast(JsonObject, raw_bundle) if isinstance(raw_bundle, dict) else {}
            raw_lane = bundle.get("lane")
            lane = (
                cast(JsonObject, raw_lane)
                if isinstance(raw_lane, dict)
                else {
                    "status": lane_process.get("status", "FAILED"),
                    "error": "Task lane did not emit a typed result before it stopped.",
                }
            )
            project = Path(str(bundle.get("project", lane_process["project"]))).resolve()
            raw_gpu_before = bundle.get("gpu_before")
            raw_gpu_after = bundle.get("gpu_after")
            gpu_before = (
                cast(JsonObject, raw_gpu_before) if isinstance(raw_gpu_before, dict) else {}
            )
            gpu_after = cast(JsonObject, raw_gpu_after) if isinstance(raw_gpu_after, dict) else {}
            metadata = _routing_metadata(
                task.task_id,
                task.language,
                task.category,
                task.editable_sources,
                {
                    "worker_eligible": task.routing.worker_eligible,
                    "rtl_scope": task.routing.rtl_scope,
                    **(
                        {}
                        if task.language in {"python", "c"}
                        else {
                            "module_count": task.routing.module_count,
                            "synthesizable": task.routing.synthesizable,
                            "explicit_ports": task.routing.explicit_ports,
                            "cycle_behavior_specified": task.routing.cycle_behavior_specified,
                            "deterministic_verification": task.routing.deterministic_verification,
                            "unresolved_architecture": task.routing.unresolved_architecture,
                        }
                    ),
                },
                arm=arm.arm_id,
            )
            raw_task_value = lane.get("task")
            task_value = (
                cast(JsonObject, raw_task_value) if isinstance(raw_task_value, dict) else {}
            )
            verification = _load_artifact(task_value, "verification_report")
            review = _load_artifact(task_value, "review_report")
            model_calls = _model_call_totals(project)
            rejection_metrics = _rejection_metrics(model_calls)
            raw_calls = model_calls.get("calls")
            call_records = (
                [item for item in raw_calls if isinstance(item, dict)]
                if isinstance(raw_calls, list)
                else []
            )
            actual_routes = [
                cast(dict[str, object], item["routing"]).get("selected")
                for item in call_records
                if isinstance(item.get("routing"), dict)
            ]
            contract_failures = sum(
                isinstance(item.get("routing"), dict)
                and cast(dict[str, object], item["routing"]).get("role")
                == "rtl_contract_generation"
                and item.get("response_valid") is not True
                for item in call_records
            )
            expected_route = (
                "rtl_worker" if arm.arm_id == "C" and metadata.worker_eligible else "main"
            )
            implementation_worktree = lane.get("worktree")
            terminal_stage: str | None = None
            terminal_category: str | None = None
            terminal_error: str | None = None
            lane_status = str(lane.get("status", "TERMINAL_FAILURE"))
            if lane_status == "TERMINAL_FAILURE":
                terminal_stage = str(lane.get("stage", "orchestration"))
                terminal_category = str(lane.get("failure_category", "orchestration_failure"))
                terminal_error = str(lane.get("error", "Task lane failed without an error"))
            elif lane_status in {"MODEL_REQUIRED", "BLOCKED_GPU", "BLOCKED_REFERENCE_EMPTY"}:
                terminal_stage = _task_failure_stage(task_value)
                terminal_category = {
                    "MODEL_REQUIRED": "endpoint_unavailable",
                    "BLOCKED_GPU": "gpu_unavailable",
                    "BLOCKED_REFERENCE_EMPTY": "retrieval_infrastructure_failure",
                }[lane_status]
                terminal_error = str(lane.get("error", lane_status))
            elif not (
                isinstance(implementation_worktree, str) and Path(implementation_worktree).is_dir()
            ):
                terminal_stage = "orchestration"
                terminal_category = "missing_implementation_worktree"
                terminal_error = "Task lane did not return an existing isolated worktree"
            elif verification.get("status") == "MALFORMED_ARTIFACT":
                terminal_stage = "verification"
                terminal_category = "verification_infrastructure_failure"
                terminal_error = str(verification.get("error", "Malformed verifier artifact"))
            elif review.get("status") == "MALFORMED_ARTIFACT":
                terminal_stage = "reviewer"
                terminal_category = "reviewer_failure"
                terminal_error = str(review.get("error", "Malformed reviewer artifact"))
            elif not verification:
                terminal_stage = _task_failure_stage(task_value)
                terminal_category = (
                    "malformed_response"
                    if model_calls.get("invalid_responses")
                    else "verification_infrastructure_failure"
                )
                terminal_error = "Task lane emitted no deterministic verification artifact"
            elif not review:
                terminal_stage = "reviewer"
                terminal_category = "reviewer_failure"
                terminal_error = "Task lane emitted no reviewer artifact"

            held_out: JsonObject = {"status": "NOT_RUN", "score": None}
            if terminal_category is None:
                try:
                    held_out = _evaluate_held_out(
                        repository_root,
                        configuration,
                        task,
                        arm,
                        Path(cast(str, implementation_worktree)),
                        held_root,
                    )
                except Exception as exc:
                    terminal_stage = "evaluator"
                    terminal_category = "evaluator_failure"
                    terminal_error = str(exc)
            verification_history = verification.get("history")
            first_pass = verification.get("passed")
            if (
                isinstance(verification_history, list)
                and verification_history
                and isinstance(verification_history[0], dict)
            ):
                first_pass = verification_history[0].get("passed")
            lane_process_evidence = dict(lane_process)
            lane_process_evidence.pop("bundle", None)
            if terminal_category is not None:
                pair = _terminal_pair_result(
                    configuration=configuration,
                    configuration_fingerprint=configuration_fingerprint,
                    held_out_manifest_sha256=held_out_manifest_sha256,
                    task=task,
                    arm=arm,
                    phase_id=phase_id,
                    pair_started_at=pair_started_at,
                    elapsed_seconds=time.monotonic() - pair_started,
                    lane_process=lane_process,
                    lane=lane,
                    model_calls=model_calls,
                    verification=verification,
                    review=review,
                    expected_route=expected_route,
                    actual_routes=actual_routes,
                    stage=terminal_stage or "orchestration",
                    category=terminal_category,
                    error=terminal_error or terminal_category,
                )
            else:
                pair = {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "COMPLETE_EVALUATED",
                    "terminal": True,
                    "outcome_kind": "candidate_result",
                    "experiment_id": EXPERIMENT_ID,
                    "task_id": task.task_id,
                    "language": task.language,
                    "category": task.category,
                    "arm_id": arm.arm_id,
                    "base_revision": _require_base_revision(configuration),
                    "execution_phase": phase_id,
                    "configuration_fingerprint": configuration_fingerprint,
                    "held_out_manifest_sha256": held_out_manifest_sha256,
                    "started_at": pair_started_at,
                    "ended_at": _now(),
                    "total_task_seconds": time.monotonic() - pair_started,
                    "configured_timeout_seconds": configuration.default_timeout_seconds,
                    "lane_process": lane_process_evidence,
                    "command_log": lane_process.get("command_log"),
                    "first_pass_deterministic_success": first_pass,
                    "final_deterministic_success": verification.get("passed"),
                    "held_out": {
                        **held_out,
                        "included_in_model_quality_metrics": True,
                    },
                    "reviewer_decision": review.get("reviewer_verdict"),
                    "false_approval": review.get("reviewer_approved") is True
                    and held_out.get("status") == "FAILED",
                    "false_rejection": review.get("reviewer_approved") is False
                    and held_out.get("status") == "PASS",
                    "repair_cycles": task_value.get("correction_loops"),
                    "meaningful_repairs": task_value.get("correction_loops"),
                    "response_rejections": rejection_metrics,
                    "worker_eligibility": assess_rtl_worker_eligibility(metadata).to_json(),
                    "expected_route": expected_route,
                    "actual_routes": actual_routes,
                    "worker_fallback": any(
                        item.get("fallback_used") is True for item in call_records
                    ),
                    "contract_failures": contract_failures,
                    "model_calls": model_calls,
                    "gpu_memory": {
                        "status": "OBSERVED_SNAPSHOTS"
                        if gpu_before.get("status") == "OBSERVED"
                        and gpu_after.get("status") == "OBSERVED"
                        else "UNAVAILABLE",
                        "before": gpu_before,
                        "after": gpu_after,
                        "observed_mib": gpu_after.get("memory_used_mib"),
                        "note": "These are observed pre/post snapshots and are not claimed as peak VRAM.",
                    },
                    "lane_result": lane,
                    "resumability": {
                        "classification": "COMPATIBLE_COMPLETE_RESULT",
                        "skip_on_resume": True,
                    },
                }
            _validate_terminal_pair_result(pair)
            _write_json_atomic(result_path, pair)
            if pair["status"] == "COMPLETE_EVALUATED":
                completed += 1
            else:
                incomplete += 1
    phase_state = phase_status(configuration, phase_id)
    final_status = (
        "COMPLETE"
        if phase_state.get("remaining_pairs") == 0
        and phase_state.get("terminal_failure_pairs") == 0
        else "INCOMPLETE"
    )
    phase_manifest = _write_phase_manifest(
        configuration,
        phase_id,
        status=final_status,
        started_at=started_at,
        corpus_hashes=corpus_hashes,
        tool_versions=tool_versions,
        endpoint_health=endpoint_health,
        gpu_observations={
            "cuda_validation": cuda,
            "phase_start": phase_gpu_start,
            "phase_end": gpu_memory_snapshot(host_runner),
        },
        held_out_manifest_sha256=held_out_manifest_sha256,
    )
    return {
        "status": final_status,
        "experiment_id": EXPERIMENT_ID,
        "phase_id": phase_id,
        "started_at": started_at,
        "ended_at": _now(),
        "completed_pairs": completed,
        "incomplete_pairs": incomplete,
        "resumed_pairs": resumed,
        "held_out_validation": held_out_validation,
        "corpus": corpus,
        "corpus_validation": corpus_validation,
        "cuda": cuda,
        "endpoint_health": endpoint_health,
        "phase_manifest": phase_manifest,
    }


def _bootstrap_interval(
    differences: list[float], *, samples: int, seed: int, confidence: float
) -> tuple[float | None, float | None]:
    if not differences:
        return None, None
    # The PRNG is used only for reproducible statistical resampling, not security.
    generator = random.Random(seed)  # nosec B311
    means = [
        statistics.mean(generator.choice(differences) for _ in differences) for _ in range(samples)
    ]
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(means) - 1, int(tail * len(means))))
    high_index = max(0, min(len(means) - 1, int((1.0 - tail) * len(means)) - 1))
    return means[low_index], means[high_index]


def _numeric(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _sum_nested_numeric(rows: list[JsonObject], container: str, metric: str) -> float:
    values: list[float] = []
    for row in rows:
        raw = row.get(container)
        if not isinstance(raw, dict):
            continue
        value = _numeric(raw.get(metric))
        if value is not None:
            values.append(value)
    return sum(values)


def _sum_numeric_field(rows: list[JsonObject], field: str) -> float:
    values = [_numeric(row.get(field)) for row in rows]
    return sum(value for value in values if value is not None)


def merge_phase_results(configuration: ExperimentConfiguration) -> JsonObject:
    """Validate all serialized phases and package their one logical result set."""
    _require_base_revision(configuration)
    corpus_hashes = _corpus_snapshot_hashes(configuration.overlay_root)
    phase_manifests: list[JsonObject] = []
    for phase_id in _PHASE_IDS:
        status = phase_status(configuration, phase_id)
        if status.get("status") != "COMPLETE":
            raise EngineeringError(
                f"Cannot merge: {phase_id} is not complete ({status.get('completed_pairs')}/"
                f"{status.get('expected_pairs')} task-arm pairs)"
            )
        manifest = status.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("status") != "COMPLETE":
            raise EngineeringError(f"Cannot merge: {phase_id} manifest is not COMPLETE")
        raw_fingerprint = manifest.get("fingerprint")
        if not isinstance(raw_fingerprint, dict):
            raise EngineeringError(f"Cannot merge: {phase_id} fingerprint is missing")
        held_hash = _string(
            raw_fingerprint.get("held_out_manifest_sha256"),
            label=f"{phase_id} held-out manifest sha256",
        )
        _assert_phase_manifest_compatible(configuration, dict(manifest), corpus_hashes, held_hash)
        phase_manifests.append(dict(manifest))
    first_fingerprint = phase_manifests[0].get("fingerprint")
    if any(manifest.get("fingerprint") != first_fingerprint for manifest in phase_manifests[1:]):
        raise EngineeringError(
            "Serialized phase fingerprints differ; task definitions, corpus, base revision, "
            "or evaluation settings changed"
        )
    if not isinstance(first_fingerprint, dict):
        raise EngineeringError("Merged phase fingerprint is malformed")
    expected_held_hash = first_fingerprint.get("held_out_manifest_sha256")
    tasks = load_benchmark_manifest(configuration.manifest_path)
    expected = {(task.task_id, arm) for task in tasks for arm in ("A", "B", "C")}
    observed: set[tuple[str, str]] = set()
    result_paths = sorted((_result_set_root(configuration) / "pairs").glob("*/*.json"))
    if len(result_paths) != len(expected):
        raise EngineeringError(
            f"Cannot merge: expected exactly {len(expected)} pair result files, "
            f"found {len(result_paths)}"
        )
    for path in result_paths:
        row = _read_json(path)
        key = (str(row.get("task_id")), str(row.get("arm_id")))
        if key not in expected:
            raise EngineeringError(f"Cannot merge: unexpected task-arm pair {key} in {path}")
        if key in observed:
            raise EngineeringError(f"Cannot merge: duplicate task-arm pair {key}")
        if (
            row.get("schema_version") != RESULT_SCHEMA_VERSION
            or row.get("status") != "COMPLETE_EVALUATED"
            or row.get("terminal") is not True
            or row.get("outcome_kind") != "candidate_result"
        ):
            raise EngineeringError(f"Cannot merge: pair {key} is not COMPLETE_EVALUATED")
        if row.get("base_revision") != configuration.base_revision:
            raise EngineeringError(f"Pair {key} has an incompatible base revision")
        if row.get("held_out_manifest_sha256") != expected_held_hash:
            raise EngineeringError(f"Pair {key} has an incompatible held-out pack")
        if row.get("configuration_fingerprint") != first_fingerprint:
            raise EngineeringError(f"Pair {key} has an incompatible fingerprint")
        expected_phase = _phase_for_arm(key[1])
        if row.get("execution_phase") != expected_phase:
            raise EngineeringError(f"Pair {key} has an invalid serialized phase record")
        observed.add(key)
    missing = sorted(expected - observed)
    if missing:
        raise EngineeringError(f"Cannot merge: {len(missing)} completed task-arm pairs are missing")
    reports = package_results(configuration)
    return {
        "status": "MERGED_COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "base_revision": _require_base_revision(configuration),
        "serialized_phases": phase_manifests,
        "task_arm_pairs": len(observed),
        "reports": reports,
    }


def package_results(configuration: ExperimentConfiguration) -> JsonObject:
    rows: list[JsonObject] = []
    for path in sorted((_result_set_root(configuration) / "pairs").glob("*/*.json")):
        rows.append(_read_json(path))
    infrastructure_failures = [
        row for row in rows if row.get("outcome_kind") == "infrastructure_failure"
    ]
    quality_rows = [row for row in rows if row.get("outcome_kind") == "candidate_result"]
    by_pair = {(str(row.get("task_id")), str(row.get("arm_id"))): row for row in quality_rows}
    aggregates: list[JsonObject] = []
    for arm in ("A", "B", "C"):
        for language in ("all", "python", "c", "verilog", "systemverilog"):
            selected = [
                row
                for row in quality_rows
                if row.get("arm_id") == arm
                and (language == "all" or row.get("language") == language)
            ]
            scores: list[float] = []
            task_seconds: list[float] = []
            for row in selected:
                held_value = row.get("held_out")
                if isinstance(held_value, dict):
                    score = _numeric(held_value.get("score"))
                    if score is not None:
                        scores.append(score)
                elapsed = _numeric(row.get("total_task_seconds"))
                if elapsed is not None:
                    task_seconds.append(elapsed)
            aggregates.append(
                {
                    "arm_id": arm,
                    "language": language,
                    "tasks": len(selected),
                    "mean_held_out_score": statistics.mean(scores) if scores else None,
                    "first_pass_deterministic_pass_rate": statistics.mean(
                        1.0 if row.get("first_pass_deterministic_success") is True else 0.0
                        for row in selected
                    )
                    if selected
                    else None,
                    "final_deterministic_pass_rate": statistics.mean(
                        1.0 if row.get("final_deterministic_success") is True else 0.0
                        for row in selected
                    )
                    if selected
                    else None,
                    "false_approvals": sum(row.get("false_approval") is True for row in selected),
                    "false_rejections": sum(row.get("false_rejection") is True for row in selected),
                    "reviewer_agreement_rate": statistics.mean(
                        0.0
                        if row.get("false_approval") is True or row.get("false_rejection") is True
                        else 1.0
                        for row in selected
                    )
                    if selected
                    else None,
                    "meaningful_repairs": sum(
                        int(value)
                        for row in selected
                        if isinstance((value := row.get("meaningful_repairs")), int)
                    ),
                    "contract_failures": sum(
                        int(value)
                        for row in selected
                        if isinstance((value := row.get("contract_failures")), int)
                    ),
                    "worker_fallback_rate": statistics.mean(
                        1.0 if row.get("worker_fallback") is True else 0.0
                        for row in selected
                        if row.get("expected_route") == "rtl_worker"
                    )
                    if any(row.get("expected_route") == "rtl_worker" for row in selected)
                    else None,
                    "response_rejections": {
                        metric: _sum_nested_numeric(selected, "response_rejections", metric)
                        for metric in ("rejected", "malformed", "stale", "no_op")
                    },
                    "model_calls": _sum_nested_numeric(selected, "model_calls", "count"),
                    "prompt_tokens": _sum_nested_numeric(selected, "model_calls", "prompt_tokens"),
                    "completion_tokens": _sum_nested_numeric(
                        selected, "model_calls", "completion_tokens"
                    ),
                    "generation_seconds": _sum_nested_numeric(
                        selected, "model_calls", "generation_seconds"
                    ),
                    "mean_task_seconds": statistics.mean(task_seconds) if task_seconds else None,
                }
            )
        scoped_groups = {
            "worker_eligible_rtl": [
                row
                for row in quality_rows
                if row.get("arm_id") == arm
                and isinstance(row.get("worker_eligibility"), dict)
                and cast(dict[str, object], row["worker_eligibility"]).get("eligible") is True
            ],
            "rtl_integration": [
                row
                for row in quality_rows
                if row.get("arm_id") == arm
                and row.get("language") in {"verilog", "systemverilog"}
                and row.get("category") == "integration"
            ],
            "implementation": [
                row
                for row in quality_rows
                if row.get("arm_id") == arm and row.get("category") == "implementation"
            ],
            "repair": [
                row
                for row in quality_rows
                if row.get("arm_id") == arm and row.get("category") == "repair"
            ],
        }
        for scope, selected in scoped_groups.items():
            scope_scores: list[float] = []
            for row in selected:
                held_value = row.get("held_out")
                if isinstance(held_value, dict):
                    score = _numeric(held_value.get("score"))
                    if score is not None:
                        scope_scores.append(score)
            aggregates.append(
                {
                    "arm_id": arm,
                    "scope": scope,
                    "tasks": len(selected),
                    "mean_held_out_score": statistics.mean(scope_scores) if scope_scores else None,
                    "final_deterministic_pass_rate": statistics.mean(
                        1.0 if row.get("final_deterministic_success") is True else 0.0
                        for row in selected
                    )
                    if selected
                    else None,
                }
            )
    contrasts: list[JsonObject] = []
    task_ids = sorted({str(row.get("task_id")) for row in quality_rows})
    scopes = (
        "all",
        "python",
        "c",
        "verilog",
        "systemverilog",
        "worker_eligible_rtl",
        "rtl_integration",
        "implementation",
        "repair",
    )

    def in_scope(row: JsonObject, scope: str) -> bool:
        if scope == "all":
            return True
        if scope in {"python", "c", "verilog", "systemverilog"}:
            return row.get("language") == scope
        if scope == "worker_eligible_rtl":
            eligibility = row.get("worker_eligibility")
            return isinstance(eligibility, dict) and eligibility.get("eligible") is True
        if scope == "rtl_integration":
            return (
                row.get("language") in {"verilog", "systemverilog"}
                and row.get("category") == "integration"
            )
        return row.get("category") == scope

    for contrast_index, (left, right, label) in enumerate(
        (("A", "B", "B_minus_A"), ("B", "C", "C_minus_B"))
    ):
        for scope_index, scope in enumerate(scopes):
            differences: list[float] = []
            paired: list[JsonObject] = []
            for task_id in task_ids:
                left_row = by_pair.get((task_id, left))
                right_row = by_pair.get((task_id, right))
                if left_row is None or right_row is None or not in_scope(left_row, scope):
                    continue
                left_held = left_row.get("held_out")
                right_held = right_row.get("held_out")
                if not isinstance(left_held, dict) or not isinstance(right_held, dict):
                    continue
                left_score = _numeric(left_held.get("score"))
                right_score = _numeric(right_held.get("score"))
                if left_score is None or right_score is None:
                    continue
                difference = right_score - left_score
                differences.append(difference)
                paired.append({"task_id": task_id, "difference": difference})
            low, high = _bootstrap_interval(
                differences,
                samples=configuration.bootstrap_samples,
                seed=configuration.bootstrap_seed + contrast_index * 100 + scope_index,
                confidence=configuration.confidence_level,
            )
            contrasts.append(
                {
                    "contrast": label,
                    "scope": scope,
                    "paired_tasks": paired,
                    "mean_difference": statistics.mean(differences) if differences else None,
                    "bootstrap_interval": [low, high],
                    "confidence_level": configuration.confidence_level,
                }
            )
    phase_records = [
        _read_json(path)
        for phase_id in _PHASE_IDS
        if (path := _phase_manifest_path(configuration, phase_id)).is_file()
    ]
    payload: JsonObject = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at": _now(),
        "task_arm_results": rows,
        "infrastructure_failures": infrastructure_failures,
        "model_quality_task_arm_results": len(quality_rows),
        "aggregates": aggregates,
        "contrasts": contrasts,
        "serialized_phases": phase_records,
        "runtime": {
            "phase_specific": [
                {
                    "phase_id": phase.get("phase_id"),
                    "wall_elapsed_seconds": phase.get("wall_elapsed_seconds"),
                    "sum_completed_task_seconds": phase.get("sum_completed_task_seconds"),
                }
                for phase in phase_records
            ],
            "sum_all_task_seconds": _sum_numeric_field(rows, "total_task_seconds"),
        },
        "comparison_context": {
            "arm_phase_mapping": {"A": "phase1", "B": "phase2", "C": "phase3"},
            "B_minus_A": "cross_model_cross_phase_serialized_comparison",
            "C_minus_B": "specialist_routing_comparison_across_serialized_phases",
            "timing_caution": "Every arm ran in a separate serving phase. Cross-phase timing includes server and phase conditions and must not be interpreted as a pure model-quality effect.",
        },
        "statistical_generality_claim": False,
        "warning": "This controlled 32-task benchmark supports paired diagnostic comparisons only; it does not establish statistical generality.",
    }
    report_root = _result_set_root(configuration) / "reports"
    json_path = report_root / "results.json"
    csv_path = report_root / "results.csv"
    markdown_path = report_root / "summary.md"
    _write_json_atomic(json_path, payload)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "language",
        "category",
        "arm_id",
        "execution_phase",
        "base_revision",
        "first_pass_deterministic_success",
        "final_deterministic_success",
        "held_out_score",
        "reviewer_decision",
        "false_approval",
        "false_rejection",
        "worker_eligible",
        "expected_route",
        "actual_routes",
        "worker_fallback",
        "contract_failures",
        "repair_cycles",
        "rejected_responses",
        "prompt_tokens",
        "completion_tokens",
        "generation_seconds",
        "total_task_seconds",
        "gpu_memory_mib",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            held_raw = row.get("held_out")
            reviewer_raw = row.get("reviewer_decision")
            eligibility_raw = row.get("worker_eligibility")
            calls_raw = row.get("model_calls")
            gpu_raw = row.get("gpu_memory")
            held: dict[str, object] = dict(held_raw) if isinstance(held_raw, dict) else {}
            reviewer: dict[str, object] = (
                dict(reviewer_raw) if isinstance(reviewer_raw, dict) else {}
            )
            eligibility: dict[str, object] = (
                dict(eligibility_raw) if isinstance(eligibility_raw, dict) else {}
            )
            calls: dict[str, object] = dict(calls_raw) if isinstance(calls_raw, dict) else {}
            gpu: dict[str, object] = dict(gpu_raw) if isinstance(gpu_raw, dict) else {}
            writer.writerow(
                {
                    "task_id": row.get("task_id"),
                    "language": row.get("language"),
                    "category": row.get("category"),
                    "arm_id": row.get("arm_id"),
                    "execution_phase": row.get("execution_phase"),
                    "base_revision": row.get("base_revision"),
                    "first_pass_deterministic_success": row.get("first_pass_deterministic_success"),
                    "final_deterministic_success": row.get("final_deterministic_success"),
                    "held_out_score": held.get("score"),
                    "reviewer_decision": reviewer.get("verdict"),
                    "false_approval": row.get("false_approval"),
                    "false_rejection": row.get("false_rejection"),
                    "worker_eligible": eligibility.get("eligible"),
                    "expected_route": row.get("expected_route"),
                    "actual_routes": json.dumps(row.get("actual_routes", [])),
                    "worker_fallback": row.get("worker_fallback"),
                    "contract_failures": row.get("contract_failures"),
                    "repair_cycles": row.get("repair_cycles"),
                    "rejected_responses": (
                        cast(dict[str, object], row["response_rejections"]).get("rejected")
                        if isinstance(row.get("response_rejections"), dict)
                        else None
                    ),
                    "prompt_tokens": calls.get("prompt_tokens"),
                    "completion_tokens": calls.get("completion_tokens"),
                    "generation_seconds": calls.get("generation_seconds"),
                    "total_task_seconds": row.get("total_task_seconds"),
                    "gpu_memory_mib": gpu.get("observed_mib"),
                }
            )
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"Packaged task-arm pairs: `{len(rows)}`.",
        "",
        "Arm A uses Qwen2.5-Coder for every role. Arm B uses Qwen3.6 for every role. "
        "Arm C uses the same Qwen3.6 main profile and routes only metadata-eligible bounded RTL implementation or repair to CodeV.",
        "",
        "B minus A isolates the main-model replacement. C minus B isolates the RTL specialist. "
        "All held-out checks occur only after implementation stops in a separate evaluation worktree.",
        "",
        "Arm A is Phase 1, Arm B is Phase 2, and Arm C is Phase 3. "
        "Both B-minus-A and C-minus-B timings are cross-phase and include serving-phase conditions.",
        "",
        "No statistical generality is claimed from 32 tasks. Missing GPU-memory or token measurements remain null rather than estimated.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def _default_config(root: Path) -> Path:
    return root / "codex_a6000" / "experiments" / EXPERIMENT_ID / "experiment.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the local dual-model ablation.")
    parser.add_argument(
        "command",
        choices=(
            "validate-config",
            "validate-complete",
            "validate-phase1",
            "validate-phase2",
            "validate-phase3",
            "validate-manifest",
            "validate-corpus",
            "validate-heldout",
            "preflight",
            "validate-runtime",
            "plan-only",
            "run-phase1",
            "run-phase2",
            "run-phase3",
            "resume-phase1",
            "resume-phase2",
            "resume-phase3",
            "phase-status",
            "run-pair",
            "merge-report",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-revision")
    parser.add_argument("--held-out-root", type=Path)
    parser.add_argument("--phase", choices=_PHASE_IDS)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Make phase-status fail unless every selected phase is compatible and complete.",
    )
    parser.add_argument(
        "--probe-runtime",
        action="store_true",
        help="Probe CUDA and the required phase endpoints without generating tokens.",
    )
    parser.add_argument("--corpus-overlay", type=Path)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    root = Path.cwd().resolve()
    _activate_isolated_tools(root)
    config_path = (arguments.config or _default_config(root)).resolve()
    try:
        configuration = load_experiment_configuration(
            root, config_path, base_revision=arguments.base_revision
        )
        if arguments.held_out_root is not None:
            os.environ[configuration.held_out_environment_variable] = str(
                arguments.held_out_root.resolve()
            )
        if arguments.command == "validate-config":
            result: JsonObject = {
                "status": "VALID_WITH_RUNTIME_PREREQUISITES",
                "experiment_id": EXPERIMENT_ID,
                "base_revision": configuration.base_revision,
                "base_revision_environment_variable": configuration.base_revision_environment_variable,
                "phases": [
                    {
                        "phase_id": phase.phase_id,
                        "arms": list(phase.arm_ids),
                        "requires_completed_phases": list(phase.requires_completed_phases),
                    }
                    for phase in configuration.phases
                ],
                "arms": [
                    {
                        "arm_id": arm.arm_id,
                        "main": arm.models.main.to_json(),
                        "rtl_worker": arm.models.rtl_worker.to_json()
                        if arm.models.rtl_worker is not None
                        else None,
                    }
                    for arm in configuration.arms
                ],
                "runtime_prerequisites": build_plan(root, configuration)["missing_prerequisites"],
            }
        elif arguments.command == "validate-complete":
            plan = build_plan(root, configuration)
            missing = list(cast(list[object], plan["missing_prerequisites"]))
            held_value = os.getenv(configuration.held_out_environment_variable)
            held_out: JsonObject = {"status": "MISSING"}
            if held_value:
                held_out = validate_held_out_pack(root, configuration, Path(held_value))
            else:
                missing.append(
                    f"held_out_root_environment:{configuration.held_out_environment_variable}"
                )
            load_bundled_corpus_manifest(root)
            load_installed_external_manifest(root)
            result = {
                "status": "VALID_COMPLETE" if not missing else "FAILED",
                "experiment_id": EXPERIMENT_ID,
                "plan": plan,
                "held_out": held_out,
                "missing_prerequisites": sorted(set(str(item) for item in missing)),
            }
        elif arguments.command in {"validate-phase1", "validate-phase2", "validate-phase3"}:
            phase_to_validate = cast(PhaseId, arguments.command.removeprefix("validate-"))
            result = validate_phase_setup(
                root,
                configuration,
                phase_to_validate,
                probe_runtime=arguments.probe_runtime,
            )
        elif arguments.command == "validate-manifest":
            tasks = load_benchmark_manifest(configuration.manifest_path)
            result = {
                "status": "VALID",
                "task_count": len(tasks),
                "counts": {
                    domain: sum(task.language == domain for task in tasks)
                    for domain in ("python", "c", "verilog", "systemverilog")
                },
                "worker_eligible_rtl": sum(task.routing.worker_eligible for task in tasks),
                "categories_by_language": {
                    domain: {
                        category: sum(
                            task.language == domain and task.category == category for task in tasks
                        )
                        for category in (
                            "implementation",
                            "repair",
                            "edge_case",
                            "integration",
                        )
                    }
                    for domain in ("python", "c", "verilog", "systemverilog")
                },
                "public_specifications_complete": True,
                "deterministic_public_verification_declared": True,
                "held_out_content_in_manifest_or_public_paths": False,
                "rtl_integration_tasks_main_model_only": all(
                    not task.routing.worker_eligible
                    for task in tasks
                    if task.language in {"verilog", "systemverilog"}
                    and task.category == "integration"
                ),
            }
        elif arguments.command == "validate-corpus":
            load_bundled_corpus_manifest(root)
            if arguments.corpus_overlay is not None:
                overlay = arguments.corpus_overlay.resolve()
                preparation = prepare_corpus_overlay(
                    root, configuration.base_reference_root, overlay
                )
                validation = validate_corpus_retrieval(overlay)
            else:
                with tempfile.TemporaryDirectory(prefix="laplace-corpus-") as temporary:
                    overlay = Path(temporary) / "Library"
                    preparation = prepare_corpus_overlay(
                        root, configuration.base_reference_root, overlay
                    )
                    validation = validate_corpus_retrieval(overlay)
            result = {
                "status": validation["status"],
                "preparation": preparation,
                "validation": validation,
            }
        elif arguments.command == "plan-only":
            selected_phase = cast(PhaseId, arguments.phase) if arguments.phase is not None else None
            result = build_plan(root, configuration, phase_id=selected_phase)
        elif arguments.command == "validate-heldout":
            held_value = os.getenv(configuration.held_out_environment_variable)
            if not held_value:
                raise EngineeringError(
                    "Set --held-out-root or "
                    f"{configuration.held_out_environment_variable} to the evaluator-owned pack"
                )
            result = validate_held_out_pack(root, configuration, Path(held_value))
        elif arguments.command == "preflight":
            selected_phase = cast(PhaseId, arguments.phase) if arguments.phase is not None else None
            result = preflight_report(
                root,
                configuration,
                phase_id=selected_phase,
                probe_runtime=arguments.probe_runtime,
            )
        elif arguments.command == "validate-runtime":
            if arguments.phase is None:
                raise EngineeringError("validate-runtime requires --phase phase1, phase2 or phase3")
            result = validate_runtime(root, configuration, cast(PhaseId, arguments.phase))
        elif arguments.command in {
            "run-phase1",
            "resume-phase1",
            "run-phase2",
            "resume-phase2",
            "run-phase3",
            "resume-phase3",
        }:
            phase_id = cast(PhaseId, arguments.command.rsplit("-", 1)[-1])
            result = run_phase(root, configuration, phase_id)
        elif arguments.command == "phase-status":
            _require_base_revision(configuration)
            selected = (
                (cast(PhaseId, arguments.phase),) if arguments.phase is not None else _PHASE_IDS
            )
            phase_results = {
                selected_phase: phase_status(configuration, selected_phase)
                for selected_phase in selected
            }
            all_complete = all(
                record.get("status") == "COMPLETE" for record in phase_results.values()
            )
            result = {
                "status": (
                    "COMPLETE"
                    if arguments.require_complete and all_complete
                    else "INCOMPLETE"
                    if arguments.require_complete
                    else "PHASE_STATUS"
                ),
                "phases": phase_results,
            }
        elif arguments.command == "run-pair":
            if arguments.request is None:
                raise EngineeringError("run-pair requires an internal --request path")
            result = _execute_lane_request(root, configuration, arguments.request.resolve())
        else:
            result = merge_phase_results(configuration)
    except (EngineeringError, ReferencePolicyError, ValueError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("status") not in {"FAILED", "INCOMPLETE", "INCOMPATIBLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
