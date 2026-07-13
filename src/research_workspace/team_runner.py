"""Bounded local-model implementation runner for the five Laplace roles."""

from __future__ import annotations

import os
import re
import signal
import shutil
import difflib

# Git invocation is restricted to fixed worktree/apply operations.
import subprocess  # nosec B404
import time
import uuid
import json
from dataclasses import dataclass
from pathlib import Path

from .engineering import (
    AgentTask,
    AgentTaskStore,
    EngineeringError,
    JsonObject,
    LocalToolRunner,
    TaskState,
    _inside,
    _safe_relative,
    _write_json_atomic,
    collect_cuda_evidence,
    retrieve_engineering_evidence,
)
from .inference import ServingCandidate, backend_for
from .llm import LocalGenerationBackend, ModelRequired


class PatchValidationError(EngineeringError):
    """A model patch did not meet the narrow worktree policy."""


@dataclass(frozen=True)
class Worktree:
    root: Path
    base_commit: str
    task_id: str


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
    key = "allowed_paths" if task.domain == "python" else "files_allowed_to_change"
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
                raise PatchValidationError("Patch must use git-style non-deleting +++ paths")
            paths.append(path[2:])
        if line.startswith("--- ") and line[4:].strip() == "/dev/null":
            raise PatchValidationError("Patch deletion is not permitted")
    if not paths:
        raise PatchValidationError("Model output contains no unified-diff file headers")
    return paths


def _is_allowed(path: str, allowed_paths: list[str]) -> bool:
    relative = _safe_relative(path, label="patch path").as_posix()
    for allowed in allowed_paths:
        permitted = _safe_relative(allowed, label="allowed path").as_posix().rstrip("/")
        if relative == permitted or relative.startswith(permitted + "/"):
            return True
    return False


def _extract_diff(model_text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", model_text, re.DOTALL | re.IGNORECASE)
    patch = fenced.group(1) if fenced else model_text
    if len(patch.encode("utf-8")) > 1_000_000:
        raise PatchValidationError("Patch exceeds the one MiB task safety limit")
    return patch.strip() + "\n"


def _extract_model_patch(worktree: Worktree, model_text: str, allowed_paths: list[str]) -> str:
    """Accept a model diff or one fenced replacement for one allowed source file.

    The replacement fallback does not synthesize code: it only wraps complete
    code emitted by the local model in a standard diff, then the regular Git
    path/scope/context checks still decide whether it may apply.
    """
    patch = _extract_diff(model_text)
    if "+++ b/" in patch and "--- a/" in patch:
        return patch
    source_paths = [path for path in allowed_paths if not Path(path).name.startswith("tb_")]
    if len(source_paths) != 1:
        raise PatchValidationError(
            "Model output is not a diff and task has no unambiguous source file"
        )
    blocks = re.findall(
        r"```(?:python|py|systemverilog|verilog|sv)?\s*\n(.*?)```",
        model_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(blocks) != 1:
        raise PatchValidationError("Model output contains no unambiguous fenced source replacement")
    relative = _safe_relative(source_paths[0], label="allowed source path")
    source = _inside(worktree.root, worktree.root / relative)
    if not source.is_file():
        raise PatchValidationError("Allowed source file is missing from isolated worktree")
    replacement = blocks[0].strip() + "\n"
    if len(replacement.encode("utf-8")) > 1_000_000:
        raise PatchValidationError("Model replacement exceeds the one MiB task safety limit")
    diff = "".join(
        difflib.unified_diff(
            source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True),
            replacement.splitlines(keepends=True),
            fromfile=f"a/{relative.as_posix()}",
            tofile=f"b/{relative.as_posix()}",
        )
    )
    if not diff:
        raise PatchValidationError("Model replacement makes no source change")
    return f"diff --git a/{relative.as_posix()} b/{relative.as_posix()}\n{diff}"


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
    _, apply_stdout, apply_stderr = _run_git(
        worktree.root, ["apply", "--whitespace=error", "--recount", str(patch_file)]
    )
    report: JsonObject = {
        "status": "APPLIED",
        "worktree": str(worktree.root),
        "base_commit": worktree.base_commit,
        "changed_paths": paths,
        "patch_path": str(patch_file),
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
        self, repository_root: Path, project_root: Path, candidate: ServingCandidate
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.project_root = project_root.resolve()
        self.store = AgentTaskStore(self.project_root)
        self.candidate = candidate
        self.log_root = self.project_root / "Outputs" / "AgentTeam" / "team_logs"

    def _transition(self, task: AgentTask, target: TaskState, note: str) -> AgentTask:
        return self.store.transition(task.task_id, target, role="supervisor", note=note)

    def _role_generation(self, backend: LocalGenerationBackend, prompt: str) -> JsonObject:
        """Record a real local-model role contribution without granting it tool authority."""
        result = backend.generate(prompt, context_tokens=8192)
        return {
            "model": result.model,
            "status": result.status,
            "text": result.text,
            "ttft_seconds": result.ttft_seconds,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

    def _prepare(self, task: AgentTask, query: str, backend: LocalGenerationBackend) -> AgentTask:
        if task.state == "request":
            task = self._transition(task, "requirements", "Task schema accepted")
        if task.state == "requirements":
            supervisor_plan = self._role_generation(
                backend,
                "You are the Laplace supervisor. Produce a narrow implementation plan, risk list, "
                "and acceptance checklist. Do not propose shell commands or edits.\n"
                f"Task: {task.specification}",
            )
            self.store.write_artifact(
                task.task_id,
                role="supervisor",
                name="requirements",
                payload={
                    "task_id": task.task_id,
                    "specification": task.specification,
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
                },
            )
            task = self._transition(task, "retrieval", "Researcher may retrieve read-only evidence")
        if task.state == "retrieval":
            evidence = retrieve_engineering_evidence(
                self.repository_root, self.project_root, task, query=query
            )
            researcher_summary = self._role_generation(
                backend,
                "You are the Laplace researcher. Summarize the following precedence-ordered "
                "evidence and identify project conventions relevant to the task. Do not edit code.\n"
                f"Evidence: {evidence}",
            )
            evidence["researcher_model_contribution"] = researcher_summary
            self.store.write_artifact(
                task.task_id, role="researcher", name="evidence_packet", payload=evidence
            )
            task = self._transition(task, "implementation", "Evidence packet persisted")
        return task

    def _prompt(self, task: AgentTask, evidence: JsonObject) -> str:
        return (
            "You are the Laplace implementer. Return ONLY one unified git diff. "
            "Do not include shell commands, prose, Markdown fences, binary files, deletions, or paths outside the task scope. "
            f"Task specification: {task.specification}\n"
            f"Allowed paths: {_allowed_paths(task)}\n"
            f"Evidence in precedence order: {evidence}\n"
            "Implement the smallest complete change with tests in the allowed paths. If your decoder cannot emit a unified diff, emit exactly one fenced full replacement for the single non-test source file and nothing else."
        )

    def run(self, task_id: str, *, query: str) -> JsonObject:
        task = self.store.load(task_id)
        cuda = collect_cuda_evidence(LocalToolRunner(self.repository_root, self.log_root))
        if cuda["status"] != "CUDA_A6000_VERIFIED":
            task = self._transition(
                task, "blocked", "BLOCKED_GPU: local A6000 CUDA inference is unavailable"
            )
            return {"status": "BLOCKED_GPU", "task": task.to_json(), "cuda_evidence": cuda}
        backend = backend_for(self.candidate)
        health = backend.health()
        if health.get("status") != "AVAILABLE":
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
            task = self._prepare(task, query, backend)
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
        worktree = WorktreeManager(self.repository_root, self.project_root).create(task.task_id)
        runner = LocalToolRunner(worktree.root, self.log_root)
        allowed = _allowed_paths(task)
        last_error = ""
        for attempt in range(0, 3):
            try:
                generated = backend.generate(self._prompt(task, evidence_raw), context_tokens=8192)
                patch = _extract_model_patch(worktree, generated.text, allowed)
                patch_report = apply_validated_patch(worktree, patch, allowed, self.log_root)
                self.store.write_artifact(
                    task.task_id, role="implementer", name="patch_manifest", payload=patch_report
                )
                self.store.write_artifact(
                    task.task_id,
                    role="implementer",
                    name="implementation_report",
                    payload={
                        "status": "PATCH_APPLIED",
                        "attempt": attempt,
                        "model": generated.model,
                        "worktree": str(worktree.root),
                    },
                )
                task = self._transition(
                    task, "verification", "Validated patch applied in isolated worktree"
                )
                if task.domain == "python":
                    verification = runner.run_python_quality_gates(allowed)
                else:
                    testbench = next(
                        (path for path in allowed if Path(path).stem.startswith("tb_")), None
                    )
                    source_files = [path for path in allowed if path != testbench]
                    verification = runner.run_eda_flow(
                        source_files,
                        top_module=Path(source_files[0]).stem,
                        testbench=testbench,
                    )
                self.store.write_artifact(
                    task.task_id, role="verifier", name="verification_report", payload=verification
                )
                task = self._transition(
                    task, "review", "Verifier emitted immutable command evidence"
                )
                reviewer_contribution = self._role_generation(
                    backend,
                    "You are the Laplace reviewer. Review the task requirements and the verifier "
                    "report below. State whether the deterministic verifier evidence is sufficient; "
                    "do not edit or merge code.\n"
                    f"Task: {task.specification}\nVerifier report: {verification}",
                )
                task = self.store.load(task.task_id)
                review: JsonObject = {
                    "status": "APPROVED"
                    if verification.get("passed") is True
                    else "CHANGES_REQUESTED",
                    "task_id": task.task_id,
                    "verification_report": task.artifacts.get("verification_report"),
                    "repair_cycles_used": task.correction_loops,
                    "reviewer_can_merge": False,
                    "reviewer_model_contribution": reviewer_contribution,
                }
                self.store.write_artifact(
                    task.task_id, role="reviewer", name="review_report", payload=review
                )
                if review["status"] == "APPROVED":
                    task = self._transition(
                        task, "final_report", "Independent review accepted verifier evidence"
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
                last_error = "Verifier reported failed quality gates"
            except (EngineeringError, ModelRequired) as exc:
                last_error = str(exc)
            if attempt == 2:
                break
            task = self._transition(
                task, "bounded_correction", f"Focused repair requested: {last_error}"
            )
            task = self._transition(
                task,
                "implementation",
                "Implementer receives only the focused verifier/reviewer failure",
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
