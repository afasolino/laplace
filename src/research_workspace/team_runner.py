"""Bounded local-model implementation runner for the five Laplace roles."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import shutil

# Git invocation is restricted to fixed worktree/apply operations.
import subprocess  # nosec B404
import sys
import time
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .engineering import (
    AgentTask,
    AgentTaskStore,
    EngineeringError,
    JsonObject,
    LocalToolRunner,
    ReferenceEvidenceError,
    TaskState,
    _inside,
    _safe_relative,
    _write_json_atomic,
    collect_cuda_evidence,
    resolve_shared_reference_root,
    retrieve_engineering_evidence,
)
from .inference import ServingCandidate
from .llm import ModelRequired
from .model_routing import (
    AuditedModelCaller,
    DualModelConfiguration,
    ModelRole,
    RoleRouter,
    RoutedCall,
    RoutingTaskMetadata,
    assess_rtl_worker_eligibility,
)
from .repair_protocol import (
    StructuredOutputError,
    build_local_patch,
    file_sha256,
    parse_replacement_plan,
    parse_reviewer_verdict,
    replacement_plan_json_schema,
    source_state,
)
from .rtl_contract import (
    RtlWorkerContract,
    parse_rtl_worker_contract,
    rtl_contract_prompt,
    rtl_worker_prompt,
)


class PatchValidationError(EngineeringError):
    """A model patch did not meet the narrow worktree policy."""


@dataclass(frozen=True)
class Worktree:
    root: Path
    base_commit: str
    task_id: str


@dataclass(frozen=True)
class TeamWorkflowOptions:
    """Explicit, auditable controls for a local five-role implementation run.

    The default is deliberately the stricter workflow.  Ablation callers can
    disable one contribution at a time, but cannot silently enable a CPU or
    an unbounded repair path.
    """

    base_commit: str = "HEAD"
    retrieval_mode: Literal["full", "project_local", "curated_only", "none"] = "full"
    role_mode: Literal["five_role", "direct"] = "five_role"
    adversarial_verification: bool = True
    reviewer_invariants: bool = True
    required_tools: tuple[str, ...] = ()


def _run_git(
    repository_root: Path, command: list[str], *, timeout_seconds: int = 60
) -> tuple[int, str, str]:
    started = time.monotonic()
    git = shutil.which("git")
    if git is None:
        raise EngineeringError("Git executable is unavailable")
    try:
        # Command verbs and arguments are fixed by WorktreeManager and apply_validated_patch.
        process = subprocess.Popen(  # nosec B603
            [git, *command],
            cwd=repository_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise EngineeringError(f"Cannot start git: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        raise EngineeringError(
            f"Git command timed out after {time.monotonic() - started:.1f}s: {stderr}"
        )
    if process.returncode != 0:
        raise EngineeringError(f"Git command failed: {stderr[-2000:]}")
    return process.returncode, stdout, stderr


class WorktreeManager:
    """Create non-overlapping, detached task worktrees without merging."""

    def __init__(self, repository_root: Path, project_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.project_root = project_root.resolve()
        self.root = self.project_root / "Data" / "AgentTeam" / "worktrees"

    def create(self, task_id: str, base_commit: str = "HEAD") -> Worktree:
        target = self.root / f"{task_id}-{uuid.uuid4().hex[:12]}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise EngineeringError("Refusing to reuse an existing task worktree")
        _, resolved, _ = _run_git(self.repository_root, ["rev-parse", base_commit])
        commit = resolved.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise EngineeringError("Git did not return an exact base commit")
        _run_git(
            self.repository_root,
            ["worktree", "add", "--detach", str(target), commit],
            timeout_seconds=180,
        )
        return Worktree(target.resolve(), commit, task_id)


def _allowed_paths(task: AgentTask) -> list[str]:
    key = "allowed_paths" if task.domain in {"python", "c"} else "files_allowed_to_change"
    raw = task.specification.get(key)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise EngineeringError(f"Task has no valid {key}")
    if not raw:
        raise EngineeringError(f"Task {key} cannot be empty")
    return list(raw)


def _diff_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if not path.startswith("b/") or path == "/dev/null":
                raise PatchValidationError(
                    "Locally generated patch must use git-style non-deleting +++ paths"
                )
            paths.append(path[2:])
        if line.startswith("--- ") and line[4:].strip() == "/dev/null":
            raise PatchValidationError("Patch deletion is not permitted")
    if not paths:
        raise PatchValidationError("Locally generated patch contains no unified-diff file headers")
    return paths


def _is_allowed(path: str, allowed_paths: list[str]) -> bool:
    relative = _safe_relative(path, label="patch path").as_posix()
    for allowed in allowed_paths:
        permitted = _safe_relative(allowed, label="allowed path").as_posix().rstrip("/")
        if relative == permitted or relative.startswith(permitted + "/"):
            return True
    return False


def apply_validated_patch(
    worktree: Worktree, patch: str, allowed_paths: list[str], log_root: Path
) -> JsonObject:
    """Apply one validated unified diff using Git, never a model shell command."""
    paths = _diff_paths(patch)
    forbidden = [path for path in paths if not _is_allowed(path, allowed_paths)]
    if forbidden:
        raise PatchValidationError(f"Patch changes paths outside task scope: {forbidden}")
    for path in paths:
        _inside(worktree.root, worktree.root / _safe_relative(path, label="patch path"))
    patch_file = log_root / f"{worktree.task_id}_{uuid.uuid4().hex}.patch"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(patch, encoding="utf-8")
    patch_file.chmod(0o444)
    # Local models occasionally emit a correct diff body with stale hunk
    # counts.  ``--recount`` recalculates only those counts; Git still checks
    # every context line and rejects a patch that does not apply.
    _, check_stdout, check_stderr = _run_git(
        worktree.root, ["apply", "--check", "--recount", str(patch_file)]
    )
    before_hashes = {
        path: file_sha256(_inside(worktree.root, worktree.root / Path(path))) for path in paths
    }
    _, apply_stdout, apply_stderr = _run_git(
        worktree.root, ["apply", "--whitespace=error", "--recount", str(patch_file)]
    )
    after_hashes = {
        path: file_sha256(_inside(worktree.root, worktree.root / Path(path))) for path in paths
    }
    report: JsonObject = {
        "status": "APPLIED",
        "worktree": str(worktree.root),
        "base_commit": worktree.base_commit,
        "changed_paths": paths,
        "patch_path": str(patch_file),
        "patch_origin": "locally_generated_from_hash_bound_replacements",
        "before_sha256": before_hashes,
        "after_sha256": after_hashes,
        "git_apply_check_stdout": check_stdout[-4000:],
        "git_apply_check_stderr": check_stderr[-4000:],
        "git_apply_stdout": apply_stdout[-4000:],
        "git_apply_stderr": apply_stderr[-4000:],
    }
    report_path = log_root / f"{worktree.task_id}_{uuid.uuid4().hex}_patch_report.json"
    _write_json_atomic(report_path, report, readonly=True)
    report["report_path"] = str(report_path)
    return report


class LocalTeamRunner:
    """Execute the persisted five-agent graph with at most two repair cycles."""

    def __init__(
        self,
        repository_root: Path,
        project_root: Path,
        candidate: ServingCandidate,
        *,
        rtl_worker_candidate: ServingCandidate | None = None,
        dual_model_configuration: DualModelConfiguration | None = None,
        routing_metadata: RoutingTaskMetadata | None = None,
        experiment_arm: str = "single_model",
        options: TeamWorkflowOptions | None = None,
        shared_reference_root: Path | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.project_root = project_root.resolve()
        self.store = AgentTaskStore(self.project_root)
        self.candidate = candidate
        if dual_model_configuration is not None and dual_model_configuration.main != candidate:
            raise EngineeringError("Dual-model main profile must match the runner candidate")
        if (
            dual_model_configuration is not None
            and rtl_worker_candidate is not None
            and dual_model_configuration.rtl_worker != rtl_worker_candidate
        ):
            raise EngineeringError("Dual-model worker profiles disagree")
        self.model_configuration = dual_model_configuration or DualModelConfiguration(
            main=candidate, rtl_worker=rtl_worker_candidate
        )
        self.rtl_worker_candidate = self.model_configuration.rtl_worker
        self.routing_metadata = routing_metadata
        self.experiment_arm = experiment_arm
        self.options = options or TeamWorkflowOptions()
        self.log_root = self.project_root / "Outputs" / "AgentTeam" / "team_logs"
        self.shared_reference_root = resolve_shared_reference_root(shared_reference_root)

    def _transition(self, task: AgentTask, target: TaskState, note: str) -> AgentTask:
        return self.store.transition(task.task_id, target, role="supervisor", note=note)

    def _role_generation(
        self,
        caller: AuditedModelCaller,
        task_metadata: RoutingTaskMetadata,
        role: ModelRole,
        prompt: str,
        validator: Callable[[str], object] | None = None,
        compact_prompt: Callable[[], str] | None = None,
        retry_index: int = 0,
    ) -> JsonObject:
        """Record a real local-model role contribution without granting it tool authority."""
        call = caller.generate(
            prompt,
            role=role,
            metadata=task_metadata,
            validator=validator,
            compact_prompt=compact_prompt,
            retry_index=retry_index,
        )
        result = call.result
        return {
            "model": result.model,
            "status": result.status,
            "text": result.text,
            "finish_reason": result.finish_reason,
            "reasoning_present": bool(result.reasoning_text),
            "reasoning_characters": len(result.reasoning_text),
            "reasoning_tokens": result.reasoning_tokens,
            "ttft_seconds": result.ttft_seconds,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "generation_seconds": call.generation_seconds,
            "response_valid": call.response_valid,
            "validation_error": call.validation_error,
            "routing": call.decision.to_json(),
            "audit_path": call.audit_path,
            "generation_budget": call.budget,
        }

    @staticmethod
    def _acceptance_matrix(task: AgentTask) -> list[JsonObject]:
        """Make every approval criterion explicit and machine-checkable."""
        requirements = task.specification.get("functional_requirements", [])
        requirement_text = (
            [item for item in requirements if isinstance(item, str)]
            if isinstance(requirements, list)
            else []
        )
        if task.domain == "python":
            checks = [
                "required_public_fixture_test",
                "adversarial_negative_path_test",
                "ruff_format",
                "ruff",
                "strict_mypy",
                "task_public_pytest",
                "task_public_coverage_pytest",
                "bandit",
            ]
        elif task.domain == "c":
            checks = [
                "self_checking_public_unit_tests",
                "gcc_or_clang_warnings",
                "cmake_build",
                "ctest",
                "address_sanitizer",
                "undefined_behavior_sanitizer",
            ]
        else:
            checks = [
                "public_self_checking_simulation",
                "adversarial_protocol_simulation",
                "verilator_lint",
                "iverilog_compile",
                "vvp_simulation",
                "yosys_synthesis",
            ]
        return [{"requirement": item, "required_evidence": checks} for item in requirement_text]

    @staticmethod
    def _current_source_context(
        worktree: Worktree,
        allowed_paths: list[str],
        domain: Literal["python", "c", "verilog", "systemverilog"],
    ) -> JsonObject:
        sources = source_state(worktree.root, allowed_paths, domain)
        fingerprint = hashlib.sha256(
            json.dumps(sources, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return {
            "current_worktree_sources": sources,
            "source_state_fingerprint": fingerprint,
        }

    def _append_artifact_history(
        self,
        task_id: str,
        *,
        role: Literal["supervisor", "researcher", "implementer", "verifier", "reviewer"],
        name: str,
        entry: JsonObject,
    ) -> None:
        task = self.store.load(task_id)
        history: list[JsonObject] = []
        prior_path = task.artifacts.get(name)
        if isinstance(prior_path, str) and Path(prior_path).is_file():
            try:
                prior_raw: object = json.loads(Path(prior_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior_raw = {}
            if isinstance(prior_raw, dict):
                prior_history = prior_raw.get("history")
                if isinstance(prior_history, list):
                    history = [item for item in prior_history if isinstance(item, dict)]
                elif prior_raw:
                    history = [dict(prior_raw)]
        history.append(dict(entry))
        payload = dict(entry)
        payload["history"] = history
        self.store.write_artifact(task_id, role=role, name=name, payload=payload)

    @staticmethod
    def _filter_evidence(evidence: JsonObject, mode: str) -> JsonObject:
        """Support retrieval ablations without changing source or provenance data."""
        filtered = dict(evidence)
        if mode == "none":
            filtered["target_project"] = []
            filtered["project_knowledge_cards"] = []
            filtered["governed_references"] = []
        elif mode == "project_local":
            filtered["governed_references"] = []
        elif mode == "curated_only":
            filtered["target_project"] = []
            filtered["project_knowledge_cards"] = []
        elif mode != "full":
            raise EngineeringError(f"Unknown retrieval mode: {mode}")
        filtered["retrieval_mode"] = mode
        target = filtered.get("target_project")
        knowledge = filtered.get("project_knowledge_cards")
        governed = filtered.get("governed_references")
        target_available = isinstance(target, list) and bool(target)
        knowledge_available = isinstance(knowledge, list) and bool(knowledge)
        governed_available = isinstance(governed, list) and bool(governed)
        filtered["retrieval_availability"] = {
            "target_project": target_available,
            "project_knowledge_cards": knowledge_available,
            "governed_references": governed_available,
        }
        if mode == "curated_only" and not governed_available:
            raise ReferenceEvidenceError(
                "BLOCKED_REFERENCE_EMPTY: curated-only execution requires at least one "
                "verified governed-reference content chunk"
            )
        return filtered

    @staticmethod
    def _public_python_tests(task: AgentTask, worktree: Worktree) -> list[str]:
        raw = task.specification.get("resolved_public_tests")
        if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
            raise EngineeringError("Python task has no explicit resolved public-test paths")
        tests: list[str] = []
        for item in raw:
            relative = _safe_relative(item, label="public Python test").as_posix()
            absolute = _inside(worktree.root, worktree.root / relative)
            if not absolute.is_file() or absolute.suffix != ".py":
                raise EngineeringError(f"Declared public Python test is missing: {relative}")
            tests.append(relative)
        return sorted(set(tests))

    @staticmethod
    def _compact_evidence_for_prompt(evidence: JsonObject) -> JsonObject:
        """Remove verbose retrieved bodies while retaining rank, hash and provenance."""
        compacted: JsonObject = {
            "compaction_notice": (
                "Retrieved bodies were omitted for context capacity. Selection order, scores, "
                "immutable identifiers, hashes and provenance remain authoritative."
            )
        }
        for key, value in evidence.items():
            if key == "researcher_model_contribution" and isinstance(value, dict):
                compacted[key] = {
                    field: value.get(field)
                    for field in ("status", "model", "audit_path", "response_valid")
                }
                continue
            if isinstance(value, list):
                records: list[object] = []
                for raw in value:
                    if not isinstance(raw, dict):
                        records.append(raw)
                        continue
                    record = {
                        field: raw.get(field)
                        for field in (
                            "source",
                            "path",
                            "file",
                            "page",
                            "section",
                            "chunk_id",
                            "reference_id",
                            "rank",
                            "score",
                            "sha256",
                            "revision",
                            "licence",
                            "provenance",
                        )
                        if field in raw
                    }
                    body = raw.get("text", raw.get("content"))
                    if isinstance(body, str):
                        record["omitted_body_sha256"] = hashlib.sha256(
                            body.encode("utf-8")
                        ).hexdigest()
                        record["omitted_body_characters"] = len(body)
                    records.append(
                        record
                        or {
                            "record_sha256": hashlib.sha256(
                                json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
                            ).hexdigest()
                        }
                    )
                compacted[key] = records
            else:
                compacted[key] = value
        return compacted

    @staticmethod
    def _compact_verification_for_review(verification: JsonObject) -> JsonObject:
        """Keep every outcome and all failing diagnostics; omit verbose passing output."""

        def compact(value: object) -> object:
            if isinstance(value, list):
                return [compact(item) for item in value]
            if not isinstance(value, dict):
                return value
            failed = value.get("status") not in {None, "PASS"} or value.get("passed") is False
            kept: JsonObject = {}
            for key, item in value.items():
                if key in {"stdout", "stderr"} and not failed:
                    if isinstance(item, str) and item:
                        kept[f"omitted_passing_{key}_sha256"] = hashlib.sha256(
                            item.encode("utf-8")
                        ).hexdigest()
                        kept[f"omitted_passing_{key}_characters"] = len(item)
                    continue
                if key == "history":
                    continue
                kept[key] = compact(item)
            return kept

        result = compact(verification)
        if not isinstance(result, dict):
            raise EngineeringError("Verifier evidence is not an object")
        result["compaction_notice"] = (
            "All commands, statuses, return codes, log paths and failing diagnostics are present. "
            "Only verbose stdout/stderr from passing commands was replaced by hashes and lengths."
        )
        return result

    def _run_python_adversarial_checks(
        self, runner: LocalToolRunner, task: AgentTask, worktree: Worktree
    ) -> JsonObject:
        """Run public-spec-derived negative tests without materializing hidden tests.

        These checks are independently written from the stated contract.  They
        live in the verifier process, not the implementation worktree, so they
        cannot change the submitted patch or expose evaluator-held tests.
        """
        source_root = str(
            _inside(worktree.root, worktree.root / Path(_allowed_paths(task)[0]).parent)
        )
        task_id = task.task_id
        scripts: dict[str, str] = {
            "py_safe_async_job": """import asyncio, sys
sys.path.insert(0, {root!r})
from job_runner import JobTimeout, run_job
cancelled = asyncio.Event()
async def work():
    try:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
        cancelled.set()
        raise
try:
    asyncio.run(run_job(work, 0.001))
except JobTimeout:
    pass
else:
    raise AssertionError('timeout must raise JobTimeout')
assert cancelled.is_set(), 'cancelled coroutine must finish cleanup'
""",
            "py_fastapi_strict_endpoint": """import sys
sys.path.insert(0, {root!r})
from endpoint import SquareRequest
from pydantic import ValidationError
for value in ('3', 3.0):
    try:
        SquareRequest(value=value)
    except ValidationError:
        pass
    else:
        raise AssertionError('coercion was accepted')
try:
    SquareRequest(value=3, extra_field=True)
except ValidationError:
    pass
else:
    raise AssertionError('extra field was accepted')
""",
            "py_sqlite_transaction": """import sqlite3, sys
sys.path.insert(0, {root!r})
from store import record_transition
with sqlite3.connect(':memory:') as connection:
    assert record_transition(connection, 'key', 'first') is True
    assert record_transition(connection, 'key', 'first') is False
    try:
        record_transition(connection, 'key', 'other')
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting provenance was accepted')
    assert connection.execute('select value from transitions where key = ?', ('key',)).fetchone() == ('first',)
""",
            "py_safe_path_cli": """import sys, tempfile
from pathlib import Path
sys.path.insert(0, {root!r})
from path_cli import write_json
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    for name in ('../outside.json', str((base / 'absolute.json').resolve())):
        try:
            write_json(base, name, {{'ok': True}})
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe output name was accepted')
""",
            "py_unseen_pydantic_policy": """import sys
sys.path.insert(0, {root!r})
from pydantic import ValidationError
from request import PolicyRequest
for candidate in ({{'retries': '2'}}, {{'retries': 2, 'unknown': True}}):
    try:
        PolicyRequest(**candidate)
    except ValidationError:
        pass
    else:
        raise AssertionError('strict validation was weakened')
""",
            "py_unseen_atomic_writer": """import sys, tempfile
from pathlib import Path
sys.path.insert(0, {root!r})
from writer import write_json_output
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for name in ('../escape.json', str((root / 'absolute.json').resolve())):
        try:
            write_json_output(root, name, {{'ok': True}})
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe output path was accepted')
""",
            "py_unseen_async_deadline": """import asyncio, sys
sys.path.insert(0, {root!r})
from deadline import DeadlineExceeded, run_with_deadline
cancelled = asyncio.Event()
async def work():
    try:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
        cancelled.set()
        raise
try:
    asyncio.run(run_with_deadline(work, 0.001))
except DeadlineExceeded:
    pass
else:
    raise AssertionError('deadline must raise DeadlineExceeded')
assert cancelled.is_set(), 'deadline cancellation did not complete'
""",
            "py_unseen_sqlite_state": """import sqlite3, sys
sys.path.insert(0, {root!r})
from state import record_state
with sqlite3.connect(':memory:') as connection:
    assert record_state(connection, 'job', 'created') is True
    assert record_state(connection, 'job', 'created') is False
    try:
        record_state(connection, 'job', 'done')
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting state was accepted')
    assert connection.execute('select value from state_entries where name = ?', ('job',)).fetchone() == ('created',)
""",
        }
        script = scripts.get(task_id)
        if script is None:
            return {
                "tool": "adversarial_python",
                "status": "FAILED",
                "reason": "No contract-derived adversarial check is registered for this task",
            }
        result = runner.run(
            "pytest",
            [sys.executable, "-c", script.format(root=source_root)],
            timeout_seconds=60,
        ).to_json()
        result["tool"] = "adversarial_python"
        result["invariants"] = self._acceptance_matrix(task)
        return result

    def _run_systemverilog_adversarial_checks(
        self, runner: LocalToolRunner, task: AgentTask, worktree: Worktree
    ) -> JsonObject:
        """Compile a verifier-owned protocol testbench from public invariants."""
        source = _inside(
            worktree.root,
            worktree.root / _safe_relative(_allowed_paths(task)[0], label="RTL source"),
        )
        suffix = ".v" if task.domain == "verilog" else ".sv"
        generic = source.parent / f"tb_adversarial{suffix}"
        if generic.is_file():
            report = runner.run_eda_flow(
                [str(source.relative_to(worktree.root))],
                top_module="tb_adversarial",
                testbench=str(generic.relative_to(worktree.root)),
                language="verilog" if task.domain == "verilog" else "systemverilog",
                require_verilator_simulation=task.domain == "systemverilog",
                required_tools=self.options.required_tools,
                timeout_seconds=60,
            )
            return {
                "tool": f"adversarial_{task.domain}",
                "status": "PASS" if report.get("passed") is True else "FAILED",
                "flow": report,
                "invariants": self._acceptance_matrix(task),
            }
        testbench_root = self.log_root / "adversarial"
        testbench_root.mkdir(parents=True, exist_ok=True)
        testbench = testbench_root / f"{task.task_id}_{uuid.uuid4().hex}.sv"
        if task.task_id == "sv_ready_valid_buffer":
            body = """module tb_adversarial;
logic clk = 0, rst_n = 0, in_valid, in_ready, out_valid, out_ready;
logic [7:0] in_data, out_data;
rv_buffer #(.WIDTH(8)) dut (.*);
always #5 clk = ~clk;
initial begin
  in_valid=0; in_data=0; out_ready=0; repeat (2) @(posedge clk); rst_n=1;
  @(negedge clk); in_valid=1; in_data=8'hA1;
  @(negedge clk); if (!out_valid || out_data !== 8'hA1) $fatal(1, "stored payload missing");
  in_data=8'hB2; out_ready=1;
  @(negedge clk); if (!out_valid || out_data !== 8'hB2) $fatal(1, "simultaneous dequeue/enqueue lost payload");
  in_valid=0; @(negedge clk); if (out_valid) $fatal(1, "buffer did not drain");
  $display("PASS adversarial ready/valid"); $finish;
end
endmodule
"""
        elif task.task_id == "sv_axi_lite_irq_regs":
            body = """module tb_adversarial;
logic clk=0, rst_n=0; logic [3:0] s_awaddr,s_araddr; logic s_awvalid,s_awready,s_wvalid,s_wready,s_bvalid,s_bready;
logic [31:0] s_wdata,s_rdata; logic [3:0] s_wstrb; logic [1:0] s_bresp,s_rresp; logic s_arvalid,s_arready,s_rvalid,s_rready,irq_input,irq;
axi_lite_irq_regs dut (.*); always #5 clk=~clk;
task automatic write_split(input logic [3:0] addr,input logic [31:0] data,input logic [3:0] strb);
begin
 @(negedge clk); s_awaddr=addr;s_awvalid=1; while(!s_awready) @(negedge clk); s_awvalid=0;
 s_wdata=data;s_wstrb=strb;s_wvalid=1; while(!s_wready) @(negedge clk); @(negedge clk);s_wvalid=0;
 while(!s_bvalid) @(negedge clk); if(s_bresp!==2'b00)$fatal(1,"write error");
end endtask
initial begin
 s_awaddr=0;s_awvalid=0;s_wdata=0;s_wstrb=0;s_wvalid=0;s_bready=1;s_araddr=0;s_arvalid=0;s_rready=1;irq_input=0;
 repeat(2) @(posedge clk); rst_n=1;
 write_split(0,32'h00000100,4'b0010); @(negedge clk);irq_input=1;@(negedge clk);irq_input=0; if(irq)$fatal(1,"WSTRB ignored");
 write_split(0,32'h1,4'b0001); @(negedge clk);irq_input=1;@(negedge clk);irq_input=0;if(!irq)$fatal(1,"IRQ enable/status broken");
 write_split(4,32'h1,4'b0001); @(negedge clk);if(irq)$fatal(1,"W1C did not clear");
 $display("PASS adversarial AXI");$finish;
end endmodule
"""
        elif task.task_id == "sv_unseen_rv_slot":
            body = """module tb_adversarial;
logic clk=0,rst_n=0,in_valid,in_ready,out_valid,out_ready; logic [7:0] in_data,out_data;
rv_slot #(.WIDTH(8)) dut (.*); always #5 clk=~clk;
initial begin
 in_valid=0;in_data=0;out_ready=0;repeat(2)@(posedge clk);rst_n=1;
 @(negedge clk);in_valid=1;in_data=8'hA1;
 @(negedge clk);in_data=8'hB2;out_ready=1;
 if(!in_ready||!out_valid||out_data!==8'hA1)$fatal(1,"replace failed");
 @(negedge clk);in_valid=0;if(!out_valid||out_data!==8'hB2)$fatal(1,"new payload lost");
 @(negedge clk);if(out_valid)$fatal(1,"drain failed");$finish;
end endmodule
"""
        elif task.task_id == "sv_unseen_w1c_event":
            body = """module tb_adversarial;
logic clk=0,rst_n=0,event_i,write_i;logic[31:0]write_data_i;logic[3:0]write_strb_i;logic pending_o,irq_o;
w1c_event dut (.*);always #5 clk=~clk;
initial begin
 event_i=0;write_i=0;write_data_i=0;write_strb_i=0;repeat(2)@(posedge clk);rst_n=1;
 @(negedge clk);event_i=1;@(negedge clk);event_i=0;if(!pending_o||irq_o)$fatal(1,"disabled event IRQ");
 @(negedge clk);write_i=1;write_data_i=32'h1;write_strb_i=4'b0010;@(negedge clk);write_i=0;if(irq_o)$fatal(1,"WSTRB broken");
 @(negedge clk);write_i=1;write_data_i=32'h1;write_strb_i=4'b0001;@(negedge clk);write_i=0;if(!irq_o)$fatal(1,"enable broken");
 @(negedge clk);write_i=1;write_data_i=32'h2;write_strb_i=4'b0001;@(negedge clk);write_i=0;if(pending_o||irq_o)$fatal(1,"W1C broken");$finish;
end endmodule
"""
        else:
            return {
                "tool": "adversarial_systemverilog",
                "status": "FAILED",
                "reason": "No contract-derived adversarial check is registered for this task",
            }
        testbench.write_text(body, encoding="utf-8")
        testbench.chmod(0o444)
        output = self.log_root / f"adversarial_{task.task_id}_{uuid.uuid4().hex}.vvp"
        compile_result = runner.run(
            "iverilog",
            [
                "iverilog",
                "-g2012",
                "-s",
                "tb_adversarial",
                "-o",
                str(output),
                str(source),
                str(testbench),
            ],
            timeout_seconds=60,
        ).to_json()
        run_result: JsonObject = {"tool": "vvp", "status": "NOT_RUN"}
        if compile_result["status"] == "PASS":
            run_result = runner.run("vvp", ["vvp", str(output)], timeout_seconds=60).to_json()
        return {
            "tool": "adversarial_systemverilog",
            "status": "PASS"
            if compile_result["status"] == "PASS" and run_result["status"] == "PASS"
            else "FAILED",
            "compile": compile_result,
            "simulation": run_result,
            "invariants": self._acceptance_matrix(task),
        }

    def _prepare(
        self,
        task: AgentTask,
        query: str,
        caller: AuditedModelCaller,
        task_metadata: RoutingTaskMetadata,
    ) -> AgentTask:
        if task.state == "request":
            task = self._transition(task, "requirements", "Task schema accepted")
        if task.state == "requirements":
            supervisor_plan: JsonObject = {
                "status": "SKIPPED_FOR_DIRECT_ABLATION",
                "reason": "One-agent ablation omits supervisor generation.",
            }
            if self.options.role_mode == "five_role":
                supervisor_plan = self._role_generation(
                    caller,
                    task_metadata,
                    "planning_supervision",
                    "You are the Laplace supervisor. Produce an interface-first implementation plan, "
                    "risk list, explicit acceptance matrix, and negative-path test strategy. Cover "
                    "invariants, errors, async lifecycle/transactions for Python, or microarchitecture, "
                    "reset and protocol stability for SystemVerilog. Do not propose shell commands or edits.\n"
                    f"Task: {task.specification}",
                )
            self.store.write_artifact(
                task.task_id,
                role="supervisor",
                name="requirements",
                payload={
                    "task_id": task.task_id,
                    "specification": task.specification,
                    "acceptance_matrix": self._acceptance_matrix(task),
                    "supervisor_model_contribution": supervisor_plan,
                },
            )
            task = self._transition(task, "plan", "Narrow task plan persisted")
        if task.state == "plan":
            self.store.write_artifact(
                task.task_id,
                role="supervisor",
                name="plan",
                payload={
                    "task_id": task.task_id,
                    "allowed_paths": _allowed_paths(task),
                    "correction_budget": 2,
                    "test_before_production_change": True,
                    "acceptance_matrix": self._acceptance_matrix(task),
                },
            )
            if self.options.role_mode == "five_role":
                test_strategy = self._role_generation(
                    caller,
                    task_metadata,
                    "planning_supervision",
                    "You are the Laplace implementer before editing production code. Produce an "
                    "executable-test strategy covering every acceptance criterion, negative path and "
                    "boundary. For SystemVerilog include reset, stalls/backpressure, simultaneous events "
                    "and assertions; for Python include interfaces, types, lifecycle and transactions. "
                    "Do not edit code or reveal/seek held-out tests.\n"
                    f"Task: {task.specification}",
                )
                self.store.write_artifact(
                    task.task_id,
                    role="implementer",
                    name="test_strategy",
                    payload={
                        "status": "GENERATED_BEFORE_PRODUCTION_PATCH",
                        "acceptance_matrix": self._acceptance_matrix(task),
                        "implementer_model_contribution": test_strategy,
                    },
                )
            task = self._transition(task, "retrieval", "Researcher may retrieve read-only evidence")
        if task.state == "retrieval":
            evidence = retrieve_engineering_evidence(
                self.repository_root,
                self.project_root,
                task,
                query=query,
                shared_reference_root=self.shared_reference_root,
            )
            evidence = self._filter_evidence(evidence, self.options.retrieval_mode)
            researcher_summary: JsonObject = {
                "status": "SKIPPED_FOR_DIRECT_ABLATION",
                "reason": "One-agent ablation omits researcher generation.",
            }
            if self.options.role_mode == "five_role":
                compact_evidence = self._compact_evidence_for_prompt(evidence)
                researcher_summary = self._role_generation(
                    caller,
                    task_metadata,
                    "retrieval_interpretation",
                    "You are the Laplace researcher. Summarize the following precedence-ordered "
                    "evidence. Project-local conventions outrank references. Identify only information "
                    "relevant to interface invariants, error cases, lifecycle/transactions, or RTL "
                    "microarchitecture and protocol behavior. Do not edit code.\n"
                    f"Evidence: {evidence}",
                    compact_prompt=lambda: (
                        "You are the Laplace researcher. Summarize the following precedence-ordered "
                        "evidence index. Project-local conventions outrank references. Retrieved bodies "
                        "were omitted only for context capacity; use the preserved ranks and provenance. "
                        "Do not edit code.\n"
                        f"Compacted evidence: {compact_evidence}"
                    ),
                )
            evidence["researcher_model_contribution"] = researcher_summary
            self.store.write_artifact(
                task.task_id, role="researcher", name="evidence_packet", payload=evidence
            )
            task = self._transition(task, "implementation", "Evidence packet persisted")
        return task

    def _prompt(
        self,
        task: AgentTask,
        evidence: JsonObject,
        *,
        defect_report: JsonObject | None,
        current_sources: JsonObject,
    ) -> str:
        if task.domain == "python":
            domain_requirements = (
                "For Python, preserve public interfaces, use current Pydantic v2 APIs, reject forbidden "
                "coercion and extra fields, preserve specified domain exceptions, and maintain strict types."
            )
        elif task.domain == "c":
            domain_requirements = (
                "For C11, preserve public headers and ownership contracts, validate sizes before arithmetic, "
                "avoid undefined behavior, release every acquired resource on every path, and remain warning- "
                "and sanitizer-clean with the committed CMake tests."
            )
        else:
            domain_requirements = (
                f"For {task.domain}, preserve ports, implement the explicit microarchitecture and cycle "
                "contract, support simultaneous handshakes and stable stalled payloads, and remain portable "
                "across lint, executable simulation, and synthesis."
            )
        repair = (
            "No prior defect report exists."
            if defect_report is None
            else "Address only this structured defect report while retaining all passing behavior: "
            + json.dumps(defect_report, sort_keys=True)
        )
        schema = {
            "schema_version": 1,
            "replacements": [
                {
                    "path": "exact allowed path",
                    "language": "python, c, verilog, or systemverilog",
                    "kind": "source or testbench",
                    "expected_sha256": "64 lowercase hex from current source state",
                    "content": "complete replacement file text",
                }
            ],
        }
        return (
            "You are the Laplace implementer. Return exactly one JSON object and no Markdown or prose. "
            "Raw unified diffs are forbidden. The required JSON shape is "
            + json.dumps(schema, separators=(",", ":"))
            + ". Every replacement must copy the exact current SHA-256 supplied below. Include only files "
            "that require a meaningful change. Do not return duplicate paths, unknown paths, shell commands, "
            "partial snippets, deletions, or unchanged content. Testbench paths require kind=testbench; all "
            "other paths require kind=source. The orchestrator generates and validates the patch locally.\n"
            f"Task specification: {json.dumps(task.specification, sort_keys=True)}\n"
            f"Allowed paths: {json.dumps(_allowed_paths(task))}\n"
            f"Acceptance matrix: {json.dumps(self._acceptance_matrix(task), sort_keys=True)}\n"
            f"Evidence in precedence order: {json.dumps(evidence, sort_keys=True)}\n"
            f"Current authoritative source state: {json.dumps(current_sources, sort_keys=True)}\n"
            f"Domain requirements: {domain_requirements}\n"
            f"Repair status: {repair}"
        )

    @staticmethod
    def _review_prompt(task: AgentTask, evidence: JsonObject, verification: JsonObject) -> str:
        schema = {
            "schema_version": 1,
            "verdict": "approve|request_changes|block",
            "reason": "specific evidence-based reason",
            "missing_evidence": ["items"],
        }
        return (
            "You are the operational Laplace reviewer. Return exactly one JSON object and no Markdown or "
            "prose. The required JSON shape is "
            + json.dumps(schema, separators=(",", ":"))
            + ". Approve only when every required deterministic verifier record and adversarial invariant "
            "passes. Request changes for a repairable implementation or evidence defect. Block only for a "
            "non-repairable scope, policy, or environment defect. Do not infer hidden tests and do not edit code.\n"
            f"Task: {json.dumps(task.specification, sort_keys=True)}\n"
            f"Acceptance matrix: {json.dumps(LocalTeamRunner._acceptance_matrix(task), sort_keys=True)}\n"
            f"Reference evidence: {json.dumps(evidence, sort_keys=True)}\n"
            f"Verifier report: {json.dumps(verification, sort_keys=True)}"
        )

    def _defect_report(
        self,
        task: AgentTask,
        verification: JsonObject | None,
        error: str,
        worktree: Worktree,
    ) -> JsonObject:
        """Turn raw tool output into a bounded, reproducible repair request."""
        failing: list[JsonObject] = []
        if isinstance(verification, dict):
            results = verification.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and item.get("status") != "PASS":
                        failing.append(
                            {
                                "tool": item.get("tool"),
                                "observed_result": str(
                                    item.get("stderr") or item.get("stdout") or item
                                ),
                            }
                        )
            adversarial = verification.get("adversarial")
            if isinstance(adversarial, dict) and adversarial.get("status") != "PASS":
                failing.append(
                    {
                        "tool": adversarial.get("tool", "adversarial"),
                        "observed_result": str(
                            adversarial.get("stderr")
                            or adversarial.get("stdout")
                            or adversarial.get("reason")
                            or adversarial
                        ),
                    }
                )
        return {
            "status": "CHANGES_REQUESTED",
            "violated_requirement": task.specification.get("functional_requirements", []),
            "minimal_failing_example": failing[:4]
            or [{"tool": "patch_or_workflow", "observed_result": error}],
            "observed_result": error or "One or more required verifier commands did not pass.",
            "expected_result": "Every acceptance-matrix command and adversarial invariant passes without regression.",
            "relevant_source_files": _allowed_paths(task),
            "allowed_files_to_modify": _allowed_paths(task),
            "mandatory_regression_commands": self._acceptance_matrix(task),
            "worktree": str(worktree.root),
        }

    def _effective_routing_metadata(self, task: AgentTask) -> RoutingTaskMetadata:
        metadata = self.routing_metadata
        if metadata is None:
            return RoutingTaskMetadata.single_model(
                task_id=task.task_id,
                experiment_arm=self.experiment_arm,
                domain=task.domain,
            )
        if metadata.task_id != task.task_id or metadata.domain != task.domain:
            raise EngineeringError("Routing metadata does not match the persisted task")
        if metadata.experiment_arm != self.experiment_arm:
            raise EngineeringError("Routing metadata experiment arm does not match the run")
        return metadata

    @staticmethod
    def _replacement_validator(
        *,
        worktree: Worktree,
        allowed_paths: list[str],
        domain: Literal["python", "c", "verilog", "systemverilog"],
    ) -> Callable[[str], None]:
        def validate(model_text: str) -> None:
            parse_replacement_plan(
                model_text,
                root=worktree.root,
                allowed_paths=allowed_paths,
                domain=domain,
            )

        return validate

    def _worker_contract(
        self,
        caller: AuditedModelCaller,
        task: AgentTask,
        metadata: RoutingTaskMetadata,
        worktree: Worktree,
        current_sources: JsonObject,
        latest_defect: JsonObject | None,
    ) -> tuple[RtlWorkerContract | None, str | None]:
        editable_path = metadata.editable_sources[0]
        states = current_sources.get("current_worktree_sources")
        source_record = (
            next(
                (
                    item
                    for item in states
                    if isinstance(item, dict) and item.get("path") == editable_path
                ),
                None,
            )
            if isinstance(states, list)
            else None
        )
        if not isinstance(source_record, dict):
            return None, "worker editable source is absent from the current source state"

        parsed: RtlWorkerContract | None = None

        def validate_contract(model_text: str) -> None:
            nonlocal parsed
            parsed = parse_rtl_worker_contract(
                model_text,
                root=worktree.root,
                task_id=task.task_id,
                language=task.domain,
                editable_path=editable_path,
                require_diagnostics=latest_defect is not None,
            )

        failures: list[str] = []
        for retry in range(caller.router.configuration.worker_contract_retries + 1):
            call = caller.generate(
                rtl_contract_prompt(
                    task_specification=task.specification,
                    current_source=dict(source_record),
                    editable_path=editable_path,
                    language="verilog" if task.domain == "verilog" else "systemverilog",
                    defect_report=latest_defect,
                ),
                role="rtl_contract_generation",
                metadata=metadata,
                retry_index=retry,
                validator=validate_contract,
            )
            if call.response_valid and parsed is not None:
                self._append_artifact_history(
                    task.task_id,
                    role="implementer",
                    name="implementation_report",
                    entry={
                        "status": "RTL_WORKER_CONTRACT_ACCEPTED",
                        "contract": parsed.to_json(),
                        "routing": call.decision.to_json(),
                        "audit_path": call.audit_path,
                        "retry_index": retry,
                    },
                )
                return parsed, None
            failures.append(call.validation_error or "invalid RTL contract")
        reason = "; ".join(failures)
        self._append_artifact_history(
            task.task_id,
            role="implementer",
            name="implementation_report",
            entry={
                "status": "RTL_WORKER_CONTRACT_REJECTED",
                "error": reason,
                "fallback": "main_model",
            },
        )
        return None, reason

    def _generate_implementation_call(
        self,
        caller: AuditedModelCaller,
        task: AgentTask,
        metadata: RoutingTaskMetadata,
        worktree: Worktree,
        evidence: JsonObject,
        current_sources: JsonObject,
        latest_defect: JsonObject | None,
        response_retry: int,
    ) -> RoutedCall:
        allowed = _allowed_paths(task)
        qwen_reasoning_repair = response_retry >= 2 and "qwen3.6" in (
            self.model_configuration.main.model.lower()
        )
        response_schema = (
            replacement_plan_json_schema(allowed_paths=allowed, domain=task.domain)
            if qwen_reasoning_repair
            else None
        )
        eligibility = assess_rtl_worker_eligibility(metadata)
        can_use_worker = eligibility.eligible and self.rtl_worker_candidate is not None
        if not can_use_worker:
            return caller.generate(
                self._prompt(
                    task,
                    evidence,
                    defect_report=latest_defect,
                    current_sources=current_sources,
                ),
                role="general_implementation",
                metadata=metadata,
                retry_index=response_retry,
                validator=self._replacement_validator(
                    worktree=worktree, allowed_paths=allowed, domain=task.domain
                ),
                compact_prompt=lambda: self._prompt(
                    task,
                    self._compact_evidence_for_prompt(evidence),
                    defect_report=latest_defect,
                    current_sources=current_sources,
                ),
                response_schema=response_schema,
                schema_name="laplace_replacement_plan" if response_schema else None,
                enable_thinking=False if qwen_reasoning_repair else None,
            )

        contract, contract_error = self._worker_contract(
            caller,
            task,
            metadata,
            worktree,
            current_sources,
            latest_defect,
        )
        worker_error = contract_error
        if contract is not None:
            role: ModelRole = (
                "bounded_rtl_repair" if latest_defect is not None else "bounded_rtl_implementation"
            )
            for retry in range(caller.router.configuration.worker_response_retries + 1):
                try:
                    worker_call = caller.generate(
                        rtl_worker_prompt(contract),
                        role=role,
                        metadata=metadata,
                        retry_index=retry,
                        validator=self._replacement_validator(
                            worktree=worktree,
                            allowed_paths=[contract.editable_path],
                            domain=task.domain,
                        ),
                    )
                except ModelRequired as exc:
                    worker_error = f"worker endpoint failed during generation: {exc}"
                    break
                if worker_call.response_valid:
                    return worker_call
                worker_error = worker_call.validation_error or "invalid worker replacement"
        return caller.generate(
            self._prompt(
                task,
                evidence,
                defect_report=latest_defect,
                current_sources=current_sources,
            ),
            role="bounded_rtl_repair"
            if latest_defect is not None
            else "bounded_rtl_implementation",
            metadata=metadata,
            retry_index=response_retry,
            fallback_reason=worker_error or "worker contract unavailable",
            validator=self._replacement_validator(
                worktree=worktree, allowed_paths=allowed, domain=task.domain
            ),
            compact_prompt=lambda: self._prompt(
                task,
                self._compact_evidence_for_prompt(evidence),
                defect_report=latest_defect,
                current_sources=current_sources,
            ),
            response_schema=response_schema,
            schema_name="laplace_replacement_plan" if response_schema else None,
            enable_thinking=False if qwen_reasoning_repair else None,
        )

    @staticmethod
    def _public_rtl_testbench(
        worktree: Worktree, task: AgentTask, allowed_paths: list[str]
    ) -> str | None:
        verification = task.specification.get("verification")
        if isinstance(verification, dict):
            raw_tests = verification.get("tests")
            if isinstance(raw_tests, list):
                for raw in raw_tests:
                    if not isinstance(raw, str) or Path(raw).suffix.lower() not in {".v", ".sv"}:
                        continue
                    candidate = _inside(
                        worktree.root,
                        worktree.root / _safe_relative(raw, label="public RTL testbench"),
                    )
                    if candidate.is_file():
                        return raw
        allowed_testbench = next(
            (
                path
                for path in allowed_paths
                if Path(path).name.startswith("tb_") and "adversarial" not in Path(path).stem
            ),
            None,
        )
        if allowed_testbench is not None:
            return allowed_testbench
        source = _inside(
            worktree.root,
            worktree.root / _safe_relative(allowed_paths[0], label="RTL source"),
        )
        candidates = sorted(source.parent.glob("tb_public.*"))
        if not candidates:
            candidates = sorted(
                path
                for path in source.parent.glob("tb_*.*")
                if "adversarial" not in path.stem and "held" not in path.stem
            )
        return str(candidates[0].relative_to(worktree.root)) if candidates else None

    def run(self, task_id: str, *, query: str) -> JsonObject:
        task = self.store.load(task_id)
        cuda = collect_cuda_evidence(LocalToolRunner(self.repository_root, self.log_root))
        if cuda["status"] != "CUDA_A6000_VERIFIED":
            task = self._transition(
                task, "blocked", "BLOCKED_GPU: local A6000 CUDA inference is unavailable"
            )
            return {"status": "BLOCKED_GPU", "task": task.to_json(), "cuda_evidence": cuda}
        task_metadata = self._effective_routing_metadata(task)
        configuration = self.model_configuration
        caller = AuditedModelCaller(RoleRouter(configuration), self.log_root / "model_calls")
        worker_needed = (
            assess_rtl_worker_eligibility(task_metadata).eligible
            and self.rtl_worker_candidate is not None
        )
        health = caller.health(include_worker=worker_needed)
        main_health = health.get("main")
        worker_health = health.get("rtl_worker")
        healthy = isinstance(main_health, dict) and main_health.get("status") == "AVAILABLE"
        if worker_needed:
            healthy = (
                healthy
                and isinstance(worker_health, dict)
                and worker_health.get("status") == "AVAILABLE"
            )
        if not healthy:
            task = self._transition(
                task, "blocked", "MODEL_REQUIRED: local serving endpoint is unavailable"
            )
            return {
                "status": "MODEL_REQUIRED",
                "task": task.to_json(),
                "health": health,
                "cuda_evidence": cuda,
            }
        try:
            task = self._prepare(task, query, caller, task_metadata)
        except ReferenceEvidenceError as exc:
            task = self.store.load(task.task_id)
            task = self._transition(task, "blocked", str(exc))
            return {
                "status": "BLOCKED_REFERENCE_EMPTY",
                "task": task.to_json(),
                "cuda_evidence": cuda,
                "error": str(exc),
            }
        except ModelRequired as exc:
            task = self._transition(
                task, "blocked", f"MODEL_REQUIRED: role generation failed: {exc}"
            )
            return {
                "status": "MODEL_REQUIRED",
                "task": task.to_json(),
                "cuda_evidence": cuda,
                "error": str(exc),
            }
        task = self.store.load(task.task_id)
        evidence_path = task.artifacts.get("evidence_packet")
        if not evidence_path:
            raise EngineeringError("Task has no evidence packet")
        evidence_file = Path(evidence_path)
        evidence_raw: object = json.loads(evidence_file.read_text(encoding="utf-8"))
        if not isinstance(evidence_raw, dict):
            raise EngineeringError("Task evidence packet is malformed")
        worktree = WorktreeManager(self.repository_root, self.project_root).create(
            task.task_id, self.options.base_commit
        )
        runner = LocalToolRunner(worktree.root, self.log_root)
        allowed = _allowed_paths(task)
        last_error = ""
        latest_verification: JsonObject | None = None
        latest_defect: JsonObject | None = None
        invalid_response_limit = 3
        for meaningful_attempt in range(3):
            current_sources = self._current_source_context(worktree, allowed, task.domain)
            patch_applied = False
            for response_retry in range(invalid_response_limit):
                generated_call = self._generate_implementation_call(
                    caller,
                    task,
                    task_metadata,
                    worktree,
                    evidence_raw,
                    current_sources,
                    latest_defect,
                    response_retry,
                )
                generated = generated_call.result
                response_entry: JsonObject = {
                    "status": "MODEL_OUTPUT_RECEIVED",
                    "meaningful_attempt": meaningful_attempt,
                    "response_retry": response_retry,
                    "model": generated.model,
                    "ttft_seconds": generated.ttft_seconds,
                    "prompt_tokens": generated.prompt_tokens,
                    "completion_tokens": generated.completion_tokens,
                    "reasoning_tokens": generated.reasoning_tokens,
                    "reasoning_present": bool(generated.reasoning_text),
                    "reasoning_characters": len(generated.reasoning_text),
                    "finish_reason": generated.finish_reason,
                    "generation_seconds": generated_call.generation_seconds,
                    "response_valid": generated_call.response_valid,
                    "validation_error": generated_call.validation_error,
                    "routing": generated_call.decision.to_json(),
                    "model_call_audit": generated_call.audit_path,
                    "source_state_fingerprint": current_sources.get("source_state_fingerprint"),
                    "model_output": generated.text,
                }
                try:
                    plan = parse_replacement_plan(
                        generated.text,
                        root=worktree.root,
                        allowed_paths=allowed,
                        domain=task.domain,
                    )
                    patch = build_local_patch(plan, root=worktree.root)
                    patch_report = apply_validated_patch(worktree, patch, allowed, self.log_root)
                except (StructuredOutputError, EngineeringError) as exc:
                    last_error = str(exc)
                    response_entry["status"] = "MODEL_OUTPUT_REJECTED"
                    response_entry["error"] = last_error
                    self._append_artifact_history(
                        task.task_id,
                        role="implementer",
                        name="implementation_report",
                        entry=response_entry,
                    )
                    latest_defect = self._defect_report(
                        task, latest_verification, last_error, worktree
                    )
                    latest_defect["rejected_response_retry"] = response_retry
                    latest_defect["source_state"] = current_sources
                    continue
                response_entry["status"] = "PATCH_APPLIED"
                response_entry["worktree"] = str(worktree.root)
                response_entry["replacement_paths"] = [item.path for item in plan.replacements]
                self._append_artifact_history(
                    task.task_id,
                    role="implementer",
                    name="implementation_report",
                    entry=response_entry,
                )
                patch_report["meaningful_attempt"] = meaningful_attempt
                patch_report["response_retry"] = response_retry
                patch_report["source_state_fingerprint"] = current_sources.get(
                    "source_state_fingerprint"
                )
                self._append_artifact_history(
                    task.task_id,
                    role="implementer",
                    name="patch_manifest",
                    entry=patch_report,
                )
                patch_applied = True
                break
            if not patch_applied:
                last_error = (
                    f"Structured model-output retry budget exhausted after {invalid_response_limit} "
                    f"rejections: {last_error}"
                )
                break

            task = self._transition(
                task, "verification", "Locally generated hash-bound patch applied"
            )
            if task.domain == "python":
                formatter = runner.run(
                    "ruff_format",
                    [sys.executable, "-m", "ruff", "format", *allowed],
                    timeout_seconds=120,
                ).to_json()
                required_tests = self._public_python_tests(task, worktree)
                verification = runner.run_python_quality_gates(
                    allowed, required_test_paths=required_tests
                )
                verification["formatter_preparation"] = formatter
                verification["passed"] = bool(verification.get("passed")) and (
                    formatter["status"] == "PASS"
                )
                adversarial = (
                    self._run_python_adversarial_checks(runner, task, worktree)
                    if self.options.adversarial_verification
                    else {"tool": "adversarial_python", "status": "SKIPPED_BY_ABLATION"}
                )
            elif task.domain == "c":
                fixture = str(Path(allowed[0]).parent)
                verification = runner.run_c_quality_gates(
                    fixture, required_tools=self.options.required_tools
                )
                adversarial = {
                    "tool": "c_negative_paths_and_sanitizers",
                    "status": "PASS" if verification.get("passed") is True else "FAILED",
                    "evidence": verification.get("report_path"),
                }
            else:
                testbench = self._public_rtl_testbench(worktree, task, allowed)
                source_files = [path for path in allowed if path != testbench]
                verification = runner.run_eda_flow(
                    source_files,
                    top_module=Path(source_files[0]).stem,
                    testbench=testbench,
                    language="verilog" if task.domain == "verilog" else "systemverilog",
                    require_verilator_simulation=task.domain == "systemverilog",
                    required_tools=self.options.required_tools,
                )
                adversarial = (
                    self._run_systemverilog_adversarial_checks(runner, task, worktree)
                    if self.options.adversarial_verification
                    else {
                        "tool": "adversarial_systemverilog",
                        "status": "SKIPPED_BY_ABLATION",
                    }
                )
            verification["adversarial"] = adversarial
            verification["acceptance_matrix"] = self._acceptance_matrix(task)
            verification["passed"] = bool(verification.get("passed")) and (
                adversarial.get("status") == "PASS"
                or adversarial.get("status") == "SKIPPED_BY_ABLATION"
            )
            verification["meaningful_attempt"] = meaningful_attempt
            latest_verification = verification
            self._append_artifact_history(
                task.task_id,
                role="verifier",
                name="verification_report",
                entry=verification,
            )
            task = self._transition(task, "review", "Verifier emitted immutable command evidence")

            reviewer_contribution: JsonObject = {
                "status": "SKIPPED_FOR_DIRECT_ABLATION",
                "reason": "One-agent direct mode omits reviewer generation.",
            }
            reviewer_verdict: JsonObject = {
                "schema_version": 1,
                "verdict": "approve",
                "reason": "Deterministic verifier controls direct-mode approval.",
                "missing_evidence": [],
            }
            reviewer_required = (
                self.options.role_mode == "five_role" and self.options.reviewer_invariants
            )
            if self.options.role_mode == "five_role":
                reviewer_errors: list[str] = []
                for reviewer_retry in range(2):
                    compact_evidence = self._compact_evidence_for_prompt(evidence_raw)
                    compact_verification = self._compact_verification_for_review(verification)
                    reviewer_contribution = self._role_generation(
                        caller,
                        task_metadata,
                        "review",
                        self._review_prompt(task, evidence_raw, verification),
                        validator=lambda text: parse_reviewer_verdict(text),
                        compact_prompt=lambda: self._review_prompt(
                            task, compact_evidence, compact_verification
                        ),
                        retry_index=reviewer_retry,
                    )
                    try:
                        parsed_verdict = parse_reviewer_verdict(
                            str(reviewer_contribution.get("text", ""))
                        )
                    except StructuredOutputError as exc:
                        reviewer_errors.append(str(exc))
                        continue
                    reviewer_verdict = parsed_verdict.to_json()
                    reviewer_verdict["response_retry"] = reviewer_retry
                    break
                else:
                    reviewer_verdict = {
                        "schema_version": 1,
                        "verdict": "request_changes",
                        "reason": "Reviewer output remained invalid after bounded retries.",
                        "missing_evidence": reviewer_errors,
                    }

            task = self.store.load(task.task_id)
            evidence_complete = verification.get("passed") is True
            reviewer_accepts = not reviewer_required or reviewer_verdict.get("verdict") == "approve"
            approved = evidence_complete and reviewer_accepts
            review: JsonObject = {
                "status": "APPROVED" if approved else "CHANGES_REQUESTED",
                "task_id": task.task_id,
                "verification_report": task.artifacts.get("verification_report"),
                "repair_cycles_used": task.correction_loops,
                "reviewer_required_for_approval": reviewer_required,
                "reviewer_approved": reviewer_verdict.get("verdict") == "approve",
                "reviewer_can_merge": False,
                "reference_library": evidence_raw.get("governed_reference_library", {}),
                "reviewer_verdict": reviewer_verdict,
                "reviewer_model_contribution": reviewer_contribution,
                "meaningful_attempt": meaningful_attempt,
            }
            self._append_artifact_history(
                task.task_id, role="reviewer", name="review_report", entry=review
            )
            if reviewer_required and reviewer_verdict.get("verdict") == "block":
                reason = str(reviewer_verdict.get("reason", "Operational reviewer blocked task"))
                task = self._transition(task, "blocked", reason)
                return {
                    "status": "BLOCKED_BY_REVIEWER",
                    "task": task.to_json(),
                    "worktree": str(worktree.root),
                    "error": reason,
                }
            if approved:
                task = self._transition(
                    task, "final_report", "Operational review accepted verifier evidence"
                )
                self.store.write_artifact(
                    task.task_id,
                    role="supervisor",
                    name="final_report",
                    payload={
                        "status": "COMPLETE",
                        "task_id": task.task_id,
                        "worktree": str(worktree.root),
                        "base_commit": worktree.base_commit,
                        "references": evidence_raw,
                        "verification": verification,
                        "review": review,
                        "residual_risks": [
                            "Patch remains isolated and is not merged automatically."
                        ],
                    },
                )
                return {
                    "status": "COMPLETE",
                    "task": self.store.load(task.task_id).to_json(),
                    "worktree": str(worktree.root),
                }

            if not evidence_complete:
                last_error = "Verifier reported failed quality gates"
            else:
                last_error = str(
                    reviewer_verdict.get("reason", "Operational reviewer requested changes")
                )
            latest_defect = self._defect_report(task, latest_verification, last_error, worktree)
            latest_defect["reviewer_verdict"] = reviewer_verdict
            latest_defect["source_state"] = self._current_source_context(
                worktree, allowed, task.domain
            )
            self._append_artifact_history(
                task.task_id,
                role="verifier",
                name="defect_report",
                entry=latest_defect,
            )
            if meaningful_attempt == 2:
                break
            task = self._transition(
                task,
                "bounded_correction",
                "Meaningful applied patch failed verification or operational review",
            )
            task = self._transition(
                task,
                "implementation",
                "Implementer receives structured defects and current source hashes",
            )
        task = self._transition(
            task, "blocked", f"Verification failed after two correction loops: {last_error}"
        )
        return {
            "status": "FAILED_AFTER_REPAIRS",
            "task": task.to_json(),
            "worktree": str(worktree.root),
            "error": last_error,
        }
