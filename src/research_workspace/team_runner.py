"""Bounded local-model implementation runner for the five Laplace roles."""

from __future__ import annotations

import os
import re
import signal
import shutil
import difflib

# Git invocation is restricted to fixed worktree/apply operations.
import subprocess  # nosec B404
import sys
import time
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


def _normalize_added_line_whitespace(patch: str) -> str:
    """Remove only trailing spaces from model-added lines before Git validation."""
    normalized: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("+") and not line.startswith("+++ "):
            ending = "\n" if line.endswith("\n") else ""
            normalized.append("+" + line[1:].rstrip(" \t\r\n") + ending)
        else:
            normalized.append(line)
    return "".join(normalized)


def _extract_model_patch(worktree: Worktree, model_text: str, allowed_paths: list[str]) -> str:
    """Accept a model diff or one fenced replacement for one allowed source file.

    The replacement fallback does not synthesize code: it only wraps complete
    code emitted by the local model in a standard diff, then the regular Git
    path/scope/context checks still decide whether it may apply.
    """
    patch = _extract_diff(model_text)
    if "+++ b/" in patch and "--- a/" in patch:
        return _normalize_added_line_whitespace(patch)
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
    return _normalize_added_line_whitespace(
        f"diff --git a/{relative.as_posix()} b/{relative.as_posix()}\n{diff}"
    )


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
        self,
        repository_root: Path,
        project_root: Path,
        candidate: ServingCandidate,
        *,
        options: TeamWorkflowOptions | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.project_root = project_root.resolve()
        self.store = AgentTaskStore(self.project_root)
        self.candidate = candidate
        self.options = options or TeamWorkflowOptions()
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
                "project_pytest",
                "coverage_pytest",
                "bandit",
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
    def _current_source_context(worktree: Worktree, allowed_paths: list[str]) -> JsonObject:
        sources: list[JsonObject] = []
        for relative in allowed_paths:
            path = _inside(worktree.root, worktree.root / _safe_relative(relative, label="source"))
            if path.is_file():
                sources.append(
                    {
                        "path": relative,
                        "content": path.read_text(encoding="utf-8", errors="replace")[:16000],
                    }
                )
        return {"current_worktree_sources": sources}

    @staticmethod
    def _filter_evidence(evidence: JsonObject, mode: str) -> JsonObject:
        """Support retrieval ablations without changing source or provenance data."""
        filtered = dict(evidence)
        if mode == "none":
            filtered["target_project"] = []
            filtered["governed_references"] = []
        elif mode == "project_local":
            filtered["governed_references"] = []
        elif mode == "curated_only":
            filtered["target_project"] = []
        elif mode != "full":
            raise EngineeringError(f"Unknown retrieval mode: {mode}")
        filtered["retrieval_mode"] = mode
        return filtered

    @staticmethod
    def _public_python_tests(worktree: Worktree, allowed_paths: list[str]) -> list[str]:
        tests: list[str] = []
        for path in allowed_paths:
            candidate = Path(path).parent / "test_public.py"
            absolute = _inside(worktree.root, worktree.root / candidate)
            if absolute.is_file():
                tests.append(candidate.as_posix())
        return sorted(set(tests))

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

    def _prepare(self, task: AgentTask, query: str, backend: LocalGenerationBackend) -> AgentTask:
        if task.state == "request":
            task = self._transition(task, "requirements", "Task schema accepted")
        if task.state == "requirements":
            supervisor_plan: JsonObject = {
                "status": "SKIPPED_FOR_DIRECT_ABLATION",
                "reason": "One-agent ablation omits supervisor generation.",
            }
            if self.options.role_mode == "five_role":
                supervisor_plan = self._role_generation(
                    backend,
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
                    backend,
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
                self.repository_root, self.project_root, task, query=query
            )
            evidence = self._filter_evidence(evidence, self.options.retrieval_mode)
            researcher_summary: JsonObject = {
                "status": "SKIPPED_FOR_DIRECT_ABLATION",
                "reason": "One-agent ablation omits researcher generation.",
            }
            if self.options.role_mode == "five_role":
                researcher_summary = self._role_generation(
                    backend,
                    "You are the Laplace researcher. Summarize the following precedence-ordered "
                    "evidence. Project-local conventions outrank references. Identify only information "
                    "relevant to interface invariants, error cases, lifecycle/transactions, or RTL "
                    "microarchitecture and protocol behavior. Do not edit code.\n"
                    f"Evidence: {evidence}",
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
        defect_report: JsonObject | None = None,
        current_sources: JsonObject | None = None,
    ) -> str:
        domain_requirements = (
            "For Python: preserve public interfaces, validate errors explicitly, account for async "
            "lifecycle and transactions, and maintain strict types."
            if task.domain == "python"
            else "For SystemVerilog: state the intended microarchitecture in code structure, keep "
            "ready/valid or AXI payloads stable under stall, define reset behavior, and keep RTL synthesizable."
        )
        repair = ""
        if defect_report is not None:
            repair = (
                "\nThis is a bounded repair. Address only the structured defect report below; retain "
                "all already-passing behavior. The current worktree source is authoritative, not a stale "
                "earlier excerpt.\n"
                f"Defect report: {defect_report}\nCurrent sources: {current_sources}\n"
            )
        return (
            "You are the Laplace implementer. Return ONLY one unified git diff. "
            "Do not include shell commands, prose, Markdown fences, binary files, deletions, or paths outside the task scope. "
            f"Task specification: {task.specification}\n"
            f"Allowed paths: {_allowed_paths(task)}\n"
            f"Acceptance matrix: {self._acceptance_matrix(task)}\n"
            f"Evidence in precedence order: {evidence}\n"
            f"{domain_requirements}\n"
            "The test strategy is already persisted before this production change. Make the smallest complete "
            "change that will satisfy every listed invariant and public test. If the task allows a testbench, "
            "update it into a self-checking regression before or with the RTL. If your decoder cannot emit a "
            "unified diff, emit exactly one fenced full replacement for the single non-test source file and nothing else."
            f"{repair}"
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
        worktree = WorktreeManager(self.repository_root, self.project_root).create(
            task.task_id, self.options.base_commit
        )
        runner = LocalToolRunner(worktree.root, self.log_root)
        allowed = _allowed_paths(task)
        last_error = ""
        latest_verification: JsonObject | None = None
        latest_defect: JsonObject | None = None
        for attempt in range(0, 3):
            try:
                current_sources = (
                    self._current_source_context(worktree, allowed)
                    if latest_defect is not None
                    else None
                )
                generated = backend.generate(
                    self._prompt(
                        task,
                        evidence_raw,
                        defect_report=latest_defect,
                        current_sources=current_sources,
                    ),
                    context_tokens=8192,
                )
                self.store.write_artifact(
                    task.task_id,
                    role="implementer",
                    name="implementation_report",
                    payload={
                        "status": "MODEL_OUTPUT_RECEIVED",
                        "attempt": attempt,
                        "model": generated.model,
                        "ttft_seconds": generated.ttft_seconds,
                        "prompt_tokens": generated.prompt_tokens,
                        "completion_tokens": generated.completion_tokens,
                        "model_output": generated.text,
                    },
                )
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
                    formatter = runner.run(
                        "ruff_format",
                        [sys.executable, "-m", "ruff", "format", *allowed],
                        timeout_seconds=120,
                    ).to_json()
                    required_tests = self._public_python_tests(worktree, allowed)
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
                latest_verification = verification
                self.store.write_artifact(
                    task.task_id, role="verifier", name="verification_report", payload=verification
                )
                task = self._transition(
                    task, "review", "Verifier emitted immutable command evidence"
                )
                reviewer_contribution: JsonObject = {
                    "status": "SKIPPED_FOR_DIRECT_ABLATION",
                    "reason": "One-agent ablation omits reviewer generation.",
                }
                if self.options.role_mode == "five_role":
                    reviewer_contribution = self._role_generation(
                        backend,
                        "You are the Laplace reviewer. Do not approve merely because code compiles. "
                        "For each acceptance criterion, require an explicit passing verifier record. "
                        "The invariant matrix below is held-out-style guidance, not evaluator tests; do not "
                        "look for hidden tests. State missing evidence as changes requested. Do not edit or merge.\n"
                        f"Task: {task.specification}\n"
                        f"Acceptance matrix: {self._acceptance_matrix(task)}\n"
                        f"Verifier report: {verification}",
                    )
                task = self.store.load(task.task_id)
                evidence_complete = verification.get("passed") is True
                if self.options.reviewer_invariants:
                    evidence_complete = evidence_complete and isinstance(
                        verification.get("acceptance_matrix"), list
                    )
                review: JsonObject = {
                    "status": "APPROVED" if evidence_complete else "CHANGES_REQUESTED",
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
            latest_defect = self._defect_report(task, latest_verification, last_error, worktree)
            self.store.write_artifact(
                task.task_id,
                role="verifier",
                name="defect_report",
                payload=latest_defect,
            )
            if attempt == 2:
                break
            task = self._transition(
                task,
                "bounded_correction",
                "Structured repair requested; see immutable verifier defect report",
            )
            task = self._transition(
                task,
                "implementation",
                "Implementer receives structured defect evidence and current worktree context",
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
