"""Frozen, idempotent live-certification wrapper around the native team runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, TypeAlias

from .execution_records import (
    AppendOnlyEventLog,
    LocalTraceRecorder,
    RunIdentity,
    canonical_json_bytes,
    canonical_sha256,
)
from .model_servers import ModelServerController
from .multilanguage_ablation import (
    AblationArm,
    ExperimentConfiguration,
    _activate_isolated_tools,
    _corpus_snapshot_hashes,
    _ensure_lane_corpus_ready,
    _evaluate_held_out,
    _execute_task_lane,
    _load_artifact,
    _model_call_totals,
    load_benchmark_manifest,
    load_experiment_configuration,
    validate_held_out_pack,
)
from .reproducibility import (
    ContextPacketBuilder,
    FrozenSkillRegistry,
    held_out_evaluator_identity,
    local_tool_versions,
    write_reproducibility_locks,
)

JsonObject: TypeAlias = dict[str, object]


class CertificationRunError(RuntimeError):
    """A live measured run does not satisfy its frozen certification contract."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        for value in values:
            handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _compact_model_call(call: Mapping[str, object]) -> JsonObject:
    """Retain bounded call evidence without copying generated source or reasoning."""

    allowed = (
        "call_id",
        "call_policy",
        "completion_tokens",
        "created_at",
        "experiment_arm",
        "failure_category",
        "fallback_used",
        "finish_reason",
        "generation_seconds",
        "prompt_characters",
        "prompt_sha256",
        "prompt_tokens",
        "reasoning_characters",
        "reasoning_present",
        "reasoning_sha256",
        "reasoning_tokens",
        "response_valid",
        "retry_index",
        "routing",
        "schema_validation_error",
        "status",
        "task_id",
        "thinking_mode",
        "validation_error",
    )
    return {key: call.get(key) for key in allowed}


def _project_gate_logs(project_root: Path, verification: JsonObject) -> None:
    results_raw = verification.get("results")
    results = (
        [dict(item) for item in results_raw if isinstance(item, dict)]
        if isinstance(results_raw, list)
        else []
    )
    selectors: dict[str, Callable[[JsonObject], bool]] = {
        "verilator_build_log.json": lambda item: item.get("phase") == "verilator_build",
        "verilator_simulation_log.json": lambda item: (
            item.get("gate") == "verilator_simulation"
        ),
        "iverilog_log.json": lambda item: item.get("gate") == "iverilog_compile",
        "vvp_public_log.json": lambda item: item.get("gate") == "vvp_simulation",
        "yosys_log.json": lambda item: item.get("gate") == "yosys_synthesis",
    }
    for filename, selector in selectors.items():
        record = next((item for item in results if selector(item)), None)
        if record is not None:
            _write_json(project_root / filename, record)
    adversarial = verification.get("adversarial")
    if isinstance(adversarial, dict) and isinstance(adversarial.get("simulation"), dict):
        _write_json(project_root / "vvp_adversarial_log.json", adversarial["simulation"])


def _reviewer_approved(review: Mapping[str, object]) -> bool:
    verdict = review.get("reviewer_verdict")
    return (
        review.get("status") == "APPROVED"
        and review.get("reviewer_approved") is True
        and isinstance(verdict, dict)
        and verdict.get("verdict") == "approve"
    )


class LiveCertificationExecutor:
    """Prepare locks, invoke the native runner, and project immutable evidence."""

    def __init__(
        self,
        repository_root: Path,
        *,
        base_revision: str,
        held_out_root: Path,
        model_servers: ModelServerController,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.base_revision = base_revision
        self.held_out_root = held_out_root.resolve()
        self.model_servers = model_servers

    def _configuration(self, project_root: Path) -> tuple[ExperimentConfiguration, AblationArm]:
        path = (
            self.repository_root
            / "codex_a6000/experiments/multilanguage_dual_model_ablation_v1"
            / "experiment.json"
        )
        loaded = load_experiment_configuration(
            self.repository_root,
            path,
            base_revision=self.base_revision,
        )
        arm = next(item for item in loaded.arms if item.arm_id == "C")
        certification_arm = replace(
            arm,
            models=replace(arm.models, fallback_to_main=False),
        )
        configuration = replace(
            loaded,
            output_root=project_root / "evaluation",
        )
        worker = certification_arm.models.rtl_worker
        if worker is None or worker.max_output_tokens != 4096:
            raise CertificationRunError("CodeV certification profile is not the frozen 4096-token arm")
        return configuration, certification_arm

    def __call__(
        self,
        identity: RunIdentity,
        frozen_operator_configuration: JsonObject,
        project_path: Path,
    ) -> JsonObject:
        project_root = project_path.resolve()
        if (
            identity.task_id != "sv_elastic_buffer2"
            or identity.arm_id != "C"
            or frozen_operator_configuration.get("gpu_required") is not True
            or frozen_operator_configuration.get("model_route") != "main+codev"
            or frozen_operator_configuration.get("smoke_profile") != "codev-live"
        ):
            raise CertificationRunError("Operator request does not match the live CodeV smoke")
        _activate_isolated_tools(self.repository_root)
        configuration, arm = self._configuration(project_root)
        task = next(
            item
            for item in load_benchmark_manifest(configuration.manifest_path)
            if item.task_id == identity.task_id
        )
        event_log = AppendOnlyEventLog(
            project_root / "events.jsonl",
            run_id=identity.run_id,
            task_id=identity.task_id,
            arm_id=identity.arm_id,
        )
        trace = LocalTraceRecorder(project_root / "trace.jsonl")
        model_status = self.model_servers.status()
        _write_json(project_root / "model_server_status_before.json", model_status)
        servers_raw = model_status.get("servers")
        servers = (
            [item for item in servers_raw if isinstance(item, dict)]
            if isinstance(servers_raw, list)
            else []
        )
        if len(servers) != 2 or any(
            not isinstance(item.get("endpoint_observation"), dict)
            or item["endpoint_observation"].get("status") != "HEALTHY_EXACT_MODEL"
            for item in servers
        ):
            raise CertificationRunError("Exact local model endpoints are not healthy")
        event_log.append(
            attempt=0,
            event_type="execution_started",
            from_state=None,
            to_state="lock_freeze",
            source_state_fingerprint=None,
            payload={"model_server_status_sha256": canonical_sha256(model_status)},
        )

        with trace.span("corpus_preflight", attributes={"run_id": identity.run_id}):
            corpus_preflight = _ensure_lane_corpus_ready(
                self.repository_root, configuration
            )
        held_out_validation = validate_held_out_pack(
            self.repository_root, configuration, self.held_out_root
        )
        corpus_hashes = _corpus_snapshot_hashes(configuration.overlay_root)
        corpus_snapshot_sha256 = canonical_sha256(corpus_hashes)
        skill_registry = FrozenSkillRegistry(
            self.repository_root / "codex_a6000" / "skills"
        )
        skills_lock = skill_registry.write_lock(project_root / "skills.lock.json")
        context = ContextPacketBuilder(self.repository_root).build(
            project_root,
            role="bounded_rtl_implementation",
            attempt=0,
            sections={
                "objective": task.objective,
                "requirements": "\n".join(task.requirements),
            },
            source_paths=[
                *(self.repository_root / path for path in task.editable_sources),
                *(self.repository_root / path for path in task.public_tests),
            ],
            skills_lock_sha256=str(skills_lock["skills_lock_sha256"]),
            corpus_snapshot_sha256=corpus_snapshot_sha256,
        )
        models_identity: JsonObject = {
            "main": arm.models.main.to_json(),
            "rtl_worker": (
                arm.models.rtl_worker.to_json()
                if arm.models.rtl_worker is not None
                else None
            ),
            "fallback_to_main": arm.models.fallback_to_main,
        }
        request: JsonObject = {
            "run_id": identity.run_id,
            "task_id": identity.task_id,
            "arm_id": identity.arm_id,
            "operator_request_sha256": identity.request_sha256,
            "fallback_to_main": False,
            "codev_max_output_tokens": 4096,
        }
        run_lock = write_reproducibility_locks(
            project_root,
            skills_lock=skills_lock,
            context_manifests=[Path(context.manifest_path)],
            corpus_identity={
                "source_root": str(configuration.base_reference_root),
                "overlay_root": str(configuration.overlay_root),
                "domain_snapshot_hashes": corpus_hashes,
                "snapshot_sha256": corpus_snapshot_sha256,
            },
            models_identity=models_identity,
            tools_identity=local_tool_versions(
                ("python", "verilator", "iverilog", "vvp", "yosys")
            ),
            base_revision=self.base_revision,
            experiment_configuration={
                "experiment_path": str(configuration.path.relative_to(self.repository_root)),
                "arm": "C",
                "fallback_to_main": False,
                "codev_max_output_tokens": 4096,
            },
            request=request,
            held_out_identity=held_out_evaluator_identity(self.held_out_root),
        )
        if frozen_operator_configuration.get("skills_lock_sha256") != skills_lock.get(
            "skills_lock_sha256"
        ):
            raise CertificationRunError("Operator skill hash does not match frozen skills")
        if frozen_operator_configuration.get("corpus_snapshot_sha256") != corpus_snapshot_sha256:
            raise CertificationRunError("Operator corpus hash does not match frozen corpus")
        event_log.append(
            attempt=0,
            event_type="locks_frozen",
            from_state="lock_freeze",
            to_state="native_execution",
            source_state_fingerprint=context.context_sha256,
            payload={"run_lock_sha256": run_lock["run_lock_sha256"]},
        )

        native_project = project_root / "native_project"
        with trace.span(
            "native_team_runner",
            attributes={"run_id": identity.run_id, "arm_id": identity.arm_id},
        ):
            bundle = _execute_task_lane(
                self.repository_root,
                configuration,
                task,
                arm,
                native_project,
            )
        lane_raw = bundle.get("lane")
        lane = dict(lane_raw) if isinstance(lane_raw, dict) else {}
        task_raw = lane.get("task")
        task_value = dict(task_raw) if isinstance(task_raw, dict) else {}
        verification = _load_artifact(task_value, "verification_report")
        review = _load_artifact(task_value, "review_report")
        rag = _load_artifact(task_value, "evidence_packet")
        _write_json(project_root / "verification_compact.json", verification)
        _write_json(project_root / "review_compact.json", review)
        _write_json(project_root / "rag_compact.json", rag)
        _project_gate_logs(project_root, verification)
        model_calls = _model_call_totals(native_project)
        calls_raw = model_calls.get("calls")
        calls = (
            [_compact_model_call(item) for item in calls_raw if isinstance(item, dict)]
            if isinstance(calls_raw, list)
            else []
        )
        _write_jsonl(project_root / "relevant_model_calls.jsonl", calls)

        worktree_value = lane.get("worktree")
        worktree = Path(worktree_value).resolve() if isinstance(worktree_value, str) else None
        held_out: JsonObject = {"status": "NOT_RUN"}
        if lane.get("status") == "COMPLETE" and worktree is not None and worktree.is_dir():
            with trace.span(
                "held_out_evaluation",
                attributes={"run_id": identity.run_id, "held_out": "isolated"},
            ):
                held_out = _evaluate_held_out(
                    self.repository_root,
                    configuration,
                    task,
                    arm,
                    worktree,
                    self.held_out_root,
                )
            source = worktree / task.editable_sources[0]
            if source.is_file():
                shutil.copy2(source, project_root / "final_source.sv")
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                (project_root / "final_source.sha256").write_text(
                    digest + "\n", encoding="ascii", newline="\n"
                )

        correction_loops = task_value.get("correction_loops")
        accepted = {
            "lane_complete": lane.get("status") == "COMPLETE",
            "correction_loops_zero": correction_loops == 0,
            "verification_passed": verification.get("passed") is True,
            "require_verilator_simulation": (
                verification.get("require_verilator_simulation") is True
            ),
            "verilator_simulation_executed": (
                verification.get("verilator_simulation_executed") is True
            ),
            "missing_tools_empty": verification.get("missing_tools") == [],
            "missing_required_results_empty": (
                verification.get("missing_required_results") == []
            ),
            "reviewer_approved": _reviewer_approved(review),
            "held_out_passed": held_out.get("status") == "PASS",
            "fallback_disabled": arm.models.fallback_to_main is False,
        }
        status = "COMPLETE" if all(accepted.values()) else "TERMINAL_FAILURE"
        result: JsonObject = {
            "schema_version": 1,
            "status": status,
            "failure_category": None if status == "COMPLETE" else "certification_contract_failure",
            "run_id": identity.run_id,
            "task_id": identity.task_id,
            "arm_id": identity.arm_id,
            "trace_id": trace.trace_id,
            "run_lock_sha256": run_lock["run_lock_sha256"],
            "corpus_snapshot_sha256": corpus_snapshot_sha256,
            "skills_lock_sha256": skills_lock["skills_lock_sha256"],
            "acceptance": accepted,
            "correction_loops": correction_loops,
            "verification": verification,
            "review": review,
            "held_out": held_out,
            "model_calls": {
                key: value for key, value in model_calls.items() if key != "calls"
            },
            "worktree": str(worktree) if worktree is not None else None,
            "corpus_preflight": corpus_preflight,
            "held_out_validation": held_out_validation,
            "model_server_status_before": model_status,
        }
        event_log.append(
            attempt=0,
            event_type="execution_terminal",
            from_state="native_execution",
            to_state="complete" if status == "COMPLETE" else "failed",
            source_state_fingerprint=(
                str(verification.get("source_state_fingerprint"))
                if isinstance(verification.get("source_state_fingerprint"), str)
                else None
            ),
            payload={
                "status": status,
                "result_sha256": canonical_sha256(result),
                "acceptance": accepted,
            },
        )
        return result
