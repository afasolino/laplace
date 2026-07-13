"""Reproducible, concurrent Codex-versus-Laplace benchmark orchestration.

The runner refuses to score a comparison when the required local CUDA lane is
unavailable.  It still writes invalid-run evidence so a missing GPU cannot be
mistaken for a zero score or a CPU substitute.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess  # nosec B404 - fixed benchmark executables only.
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml

from .engineering import (
    AgentTaskStore,
    JsonObject,
    LocalToolRunner,
    _write_json_atomic,
    collect_cuda_evidence,
    normalize_task_spec,
)
from .inference import ServingCandidate
from .team_runner import LocalTeamRunner


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _catalog_tasks(repository_root: Path) -> list[JsonObject]:
    catalog_path = repository_root / "codex_a6000" / "benchmarks" / "paired_task_catalog.yaml"
    try:
        raw: object = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot read paired task catalog: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("Paired task catalog must be an object")
    families = raw.get("task_families")
    if not isinstance(families, dict):
        raise RuntimeError("Paired task catalog has no task_families")
    selected: list[JsonObject] = []
    for domain, expected_count in (("python", 4), ("systemverilog", 2)):
        entries = families.get(domain)
        if not isinstance(entries, list) or len(entries) < expected_count:
            raise RuntimeError(
                f"Paired task catalog does not contain {expected_count} {domain} tasks"
            )
        for item in entries[:expected_count]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise RuntimeError("Paired task catalog contains a malformed task")
            selected.append(
                {
                    "task_id": item["id"],
                    "domain": domain,
                    "description": str(item.get("description", "")),
                    "risks": item.get("risks", []),
                    "timeout_s": 900,
                }
            )
    return selected


def _blocked_rows(tasks: list[JsonObject], reason: str) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for task in tasks:
        rows.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "valid_run": False,
                "invalid_run_reason": reason,
                "execution_overlap_proven": False,
                "held_out_exposed_to_lanes": False,
                "resource_symmetry": "not_started",
                "lanes": {
                    "codex_direct": {"status": "NOT_STARTED", "score": None},
                    "laplace_team": {"status": "NOT_STARTED", "score": None},
                },
            }
        )
    return rows


def _write_reports(output_root: Path, payload: JsonObject) -> JsonObject:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "codex_vs_laplace_results.json"
    csv_path = output_root / "codex_vs_laplace_results.csv"
    markdown_path = output_root / "codex_vs_laplace_summary.md"
    _write_json_atomic(json_path, payload)
    tasks = payload.get("tasks")
    rows = tasks if isinstance(tasks, list) else []
    csv_rows: list[dict[str, object]] = []
    for task in rows:
        if not isinstance(task, dict):
            continue
        lanes = task.get("lanes")
        if not isinstance(lanes, dict):
            csv_rows.append({"task_id": task.get("task_id"), "domain": task.get("domain")})
            continue
        for lane_name, lane in lanes.items():
            if not isinstance(lane_name, str) or not isinstance(lane, dict):
                continue
            score = lane.get("score")
            score_data = score if isinstance(score, dict) else {}
            csv_rows.append(
                {
                    "task_id": task.get("task_id"),
                    "domain": task.get("domain"),
                    "lane": lane_name,
                    "valid_run": task.get("valid_run"),
                    "execution_overlap_proven": task.get("execution_overlap_proven"),
                    "held_out_exposed_to_lanes": task.get("held_out_exposed_to_lanes"),
                    "lane_status": lane.get("status"),
                    "wall_time_s": lane.get("wall_time_s"),
                    "repair_cycles": lane.get("repair_cycles"),
                    "human_intervention": lane.get("human_intervention"),
                    "objective_score_0_100": score_data.get("objective_score_0_100"),
                    "functional_correctness": score_data.get("functional_correctness"),
                    "held_out_correctness": score_data.get("held_out_correctness"),
                    "typing_and_lint_quality": score_data.get("typing_and_lint_quality"),
                    "rtl_protocol_correctness": score_data.get("rtl_protocol_correctness"),
                    "robustness": score_data.get("robustness"),
                }
            )
    fieldnames = [
        "task_id",
        "domain",
        "lane",
        "valid_run",
        "execution_overlap_proven",
        "held_out_exposed_to_lanes",
        "lane_status",
        "wall_time_s",
        "repair_cycles",
        "human_intervention",
        "objective_score_0_100",
        "functional_correctness",
        "held_out_correctness",
        "typing_and_lint_quality",
        "rtl_protocol_correctness",
        "robustness",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    status = str(payload.get("status", "UNKNOWN"))
    reason = str(payload.get("reason", "No conclusion recorded."))
    report_lines = [
        "# Codex-direct versus Laplace-team comparison",
        "",
        f"Status: `{status}`.",
        "",
        reason,
        "",
        "| Task | Domain | Lane | Functional | Held-out | Quality / protocol | Robustness | Score | Time (s) | Repairs | Intervention |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in csv_rows:
        quality = item.get("typing_and_lint_quality", item.get("rtl_protocol_correctness", ""))
        report_lines.append(
            "| {task_id} | {domain} | {lane} | {functional} | {heldout} | {quality} | {robustness} | {score} | {time} | {repairs} | {intervention} |".format(
                task_id=item.get("task_id", ""),
                domain=item.get("domain", ""),
                lane=item.get("lane", ""),
                functional=item.get("functional_correctness", ""),
                heldout=item.get("held_out_correctness", ""),
                quality=quality,
                robustness=item.get("robustness", ""),
                score=item.get("objective_score_0_100", ""),
                time=item.get("wall_time_s", ""),
                repairs=item.get("repair_cycles", ""),
                intervention=item.get("human_intervention", ""),
            )
        )
    report_lines.extend(
        [
            "",
            "Codex-direct and Laplace-team scores are reported separately for Python and SystemVerilog in the JSON aggregate. Missing or unmeasured fields remain explicit rather than being treated as zero-quality evidence.",
            "",
            "No aggregate winner is reported unless all six tasks have symmetric start commits, positive overlap, hidden held-out evaluation after both lanes stop, and no lane contamination.",
        ]
    )
    markdown_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    domain: Literal["python", "systemverilog"]
    fixture: str
    sources: tuple[str, ...]
    public_tests: tuple[str, ...]
    objective: str
    requirements: tuple[str, ...]


_BENCHMARK_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        "py_safe_async_job",
        "python",
        "benchmarks/a6000_agent_team/paired_public/py_safe_async_job",
        ("job_runner.py",),
        ("test_public.py",),
        "Repair the asynchronous job runner so it enforces a positive timeout and reliably cancels a timed-out coroutine.",
        (
            "Raise JobTimeout on timeout.",
            "Await cancellation cleanup before returning control.",
            "Preserve the successful JobResult contract.",
        ),
    ),
    BenchmarkTask(
        "py_fastapi_strict_endpoint",
        "python",
        "benchmarks/a6000_agent_team/paired_public/py_fastapi_strict_endpoint",
        ("endpoint.py",),
        ("test_public.py",),
        "Make the existing FastAPI request model strict without changing valid square responses.",
        (
            "Reject coercion of strings to integer values.",
            "Reject undeclared request fields.",
            "Preserve POST /square and its successful response contract.",
        ),
    ),
    BenchmarkTask(
        "py_sqlite_transaction",
        "python",
        "benchmarks/a6000_agent_team/paired_public/py_sqlite_transaction",
        ("store.py",),
        ("test_public.py",),
        "Make the SQLite transition operation idempotent and transaction-safe.",
        (
            "Return False for an already-recorded identical transition.",
            "Reject a conflicting value for an existing key without changing stored provenance.",
            "Use an explicit transaction with rollback on failure.",
        ),
    ),
    BenchmarkTask(
        "py_safe_path_cli",
        "python",
        "benchmarks/a6000_agent_team/paired_public/py_safe_path_cli",
        ("path_cli.py",),
        ("test_public.py",),
        "Make the JSON output helper reject hostile paths and write atomically beneath its root.",
        (
            "Reject absolute paths and traversal.",
            "Resolve and validate the output under root.",
            "Use a same-directory atomic replacement.",
        ),
    ),
    BenchmarkTask(
        "sv_ready_valid_buffer",
        "systemverilog",
        "benchmarks/a6000_agent_team/paired_public/sv_ready_valid_buffer",
        ("rv_buffer.sv", "tb_rv_buffer.sv"),
        ("tb_rv_buffer.sv",),
        "Repair the parameterized ready/valid buffer for simultaneous dequeue/enqueue and backpressure stability.",
        (
            "Retain accepted input until an output handshake.",
            "Allow simultaneous dequeue and enqueue without loss or duplication.",
            "Keep output data stable while stalled and clear valid on active-low reset.",
        ),
    ),
    BenchmarkTask(
        "sv_axi_lite_irq_regs",
        "systemverilog",
        "benchmarks/a6000_agent_team/paired_public/sv_axi_lite_irq_regs",
        ("axi_lite_irq_regs.sv", "tb_axi_lite_irq_regs.sv"),
        ("tb_axi_lite_irq_regs.sv",),
        "Repair the AXI4-Lite register block for independent write channels, WSTRB, W1C status and IRQ deassertion.",
        (
            "Accept AW and W handshakes independently before issuing one response.",
            "Apply WSTRB to control writes.",
            "Implement W1C status and deassert IRQ when no asserted enabled status remains.",
            "Return SLVERR for unmapped addresses.",
        ),
    ),
)


def _task_spec(task: BenchmarkTask) -> JsonObject:
    allowed = [f"{task.fixture}/{path}" for path in task.sources]
    if task.domain == "python":
        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "repository_root": ".",
            "allowed_paths": allowed,
            "public_interfaces": [
                {
                    "name": task.sources[0],
                    "contract": "Preserve the public fixture contract while fixing the specified defect.",
                    "compatibility": "Public tests must continue to pass.",
                }
            ],
            "functional_requirements": list(task.requirements),
            "input_validation": ["Reject invalid input with an explicit exception."],
            "error_behavior": ["Do not silently weaken error or safety behavior."],
            "concurrency_and_lifecycle": ["Do not leak asynchronous work or SQLite transactions."],
            "security_and_paths": ["Do not access files outside the fixture scope."],
            "quality_requirements": {
                "python": ">=3.11",
                "typing": "strict mypy",
                "formatting": "ruff format --check",
                "lint": "ruff check",
                "tests": "public and held-out pytest",
            },
            "references": [
                {
                    "path_or_id": task.fixture,
                    "purpose": "Target-project public fixture conventions.",
                }
            ],
            "verification_commands": [
                f"PYTHONPATH={task.fixture} python -m pytest {task.fixture}/test_public.py",
                f"python -m ruff check {allowed[0]}",
                f"python -m mypy {allowed[0]}",
            ],
            "deliverables": ["narrow source patch", "verification evidence"],
            "out_of_scope": ["network", "changes outside allowed paths"],
            "assumptions": ["Held-out tests are unavailable during implementation."],
        }
    return {
        "task_id": task.task_id,
        "objective": task.objective,
        "target": {
            "class": "portable_rtl",
            "language": "SystemVerilog-2017 subset",
            "toolchain": ["verilator", "iverilog", "yosys"],
            "technology_or_device": None,
            "frequency_mhz": None,
        },
        "parameters": [],
        "interfaces": [
            {
                "name": "fixture_interface",
                "protocol": "ready_valid"
                if task.task_id == "sv_ready_valid_buffer"
                else "axi4_lite",
                "direction": "bidirectional",
                "signals": ["See the public RTL module port list."],
                "ordering": "in-order",
                "backpressure": "Protocol handshakes must remain stable until accepted.",
            }
        ],
        "clock_reset": {
            "clock_domains": ["clk"],
            "reset_semantics": "active-low asynchronous reset",
            "cdc_rdc_assumptions": "single-clock fixture; no CDC crossing",
        },
        "functional_requirements": list(task.requirements),
        "error_and_corner_behavior": ["No data loss, protocol violation or unknown response."],
        "coding_constraints": ["Synthesizable SystemVerilog only."],
        "files_allowed_to_change": allowed,
        "references": [
            {"kind": "project", "identifier": task.fixture, "purpose": "Public fixture."}
        ],
        "verification": {
            "self_checking": True,
            "tests": ["public self-checking simulation", "held-out self-checking simulation"],
            "assertions": ["protocol stability"],
            "commands": ["verilator", "iverilog", "vvp", "yosys"],
            "acceptance_criteria": ["All available checks return zero."],
        },
        "deliverables": ["RTL patch", "verification evidence"],
        "out_of_scope": ["vendor-specific IP", "network"],
        "assumptions": ["Held-out tests are unavailable during implementation."],
        "blocking_questions": [],
    }


_PYTHON_HELD_OUT: dict[str, str] = {
    "py_safe_async_job": """from __future__ import annotations

import asyncio

import pytest

from job_runner import JobTimeout, run_job


def test_timeout_cancels_the_underlying_coroutine() -> None:
    cancelled = asyncio.Event()

    async def forever() -> int:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 0

    with pytest.raises(JobTimeout):
        asyncio.run(run_job(forever, 0.01))
    assert cancelled.is_set()
""",
    "py_fastapi_strict_endpoint": """from __future__ import annotations

import pytest
from pydantic import ValidationError

from endpoint import SquareRequest


def test_request_model_rejects_coercion_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SquareRequest(value="9")
    with pytest.raises(ValidationError):
        SquareRequest(value=9, unexpected=True)
""",
    "py_sqlite_transaction": """from __future__ import annotations

import sqlite3

import pytest

from store import record_transition


def test_duplicate_is_idempotent_and_conflict_preserves_original() -> None:
    with sqlite3.connect(":memory:") as connection:
        assert record_transition(connection, "run-1", "created") is True
        assert record_transition(connection, "run-1", "created") is False
        with pytest.raises(ValueError):
            record_transition(connection, "run-1", "different")
        assert connection.execute("SELECT value FROM transitions WHERE key='run-1'").fetchone() == (
            "created",
        )
""",
    "py_safe_path_cli": """from __future__ import annotations

from pathlib import Path

import pytest

from path_cli import write_json


def test_traversal_and_absolute_outputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json(tmp_path, "../escape.json", {"x": 1})
    with pytest.raises(ValueError):
        write_json(tmp_path, str((tmp_path / "absolute.json").resolve()), {"x": 1})
    assert not (tmp_path.parent / "escape.json").exists()
""",
}

_SV_HELD_OUT: dict[str, str] = {
    "sv_ready_valid_buffer": """module tb_heldout;
    logic clk = 0;
    logic rst_n = 0;
    logic in_valid, in_ready, out_valid, out_ready;
    logic [7:0] in_data, out_data;
    rv_buffer #(.WIDTH(8)) dut (.*);
    always #5 clk = ~clk;
    initial begin
        in_valid = 0; in_data = 0; out_ready = 0;
        repeat (2) @(posedge clk); rst_n = 1;
        @(negedge clk); in_valid = 1; in_data = 8'h11;
        @(negedge clk); in_data = 8'h22; out_ready = 1;
        if (!in_ready || !out_valid || out_data !== 8'h11) $fatal(1, "simultaneous transfer broken");
        @(negedge clk); in_valid = 0;
        if (!out_valid || out_data !== 8'h22) $fatal(1, "second item lost");
        @(negedge clk); if (out_valid) $fatal(1, "second item not drained");
        $display("PASS: held-out ready-valid"); $finish;
    end
endmodule
""",
    "sv_axi_lite_irq_regs": """module tb_heldout;
    logic clk = 0, rst_n = 0;
    logic [3:0] s_awaddr, s_araddr;
    logic s_awvalid, s_awready, s_wvalid, s_wready, s_bvalid, s_bready;
    logic [31:0] s_wdata, s_rdata;
    logic [3:0] s_wstrb;
    logic [1:0] s_bresp, s_rresp;
    logic s_arvalid, s_arready, s_rvalid, s_rready, irq_input, irq;
    axi_lite_irq_regs dut (.*);
    always #5 clk = ~clk;
    task automatic write_split(input logic [3:0] addr, input logic [31:0] data, input logic [3:0] strb);
        begin
            @(negedge clk); s_awaddr = addr; s_awvalid = 1;
            while (!s_awready) @(negedge clk);
            @(negedge clk); s_awvalid = 0; s_wdata = data; s_wstrb = strb; s_wvalid = 1;
            while (!s_wready) @(negedge clk);
            @(negedge clk); s_wvalid = 0;
            while (!s_bvalid) @(negedge clk);
            if (s_bresp !== 2'b00) $fatal(1, "write response");
        end
    endtask
    initial begin
        s_awaddr=0; s_awvalid=0; s_wdata=0; s_wstrb=0; s_wvalid=0; s_bready=1;
        s_araddr=0; s_arvalid=0; s_rready=1; irq_input=0;
        repeat (2) @(posedge clk); rst_n=1;
        write_split(0, 32'h00000100, 4'b0010);
        @(negedge clk); irq_input=1; @(negedge clk); irq_input=0;
        if (irq) $fatal(1, "WSTRB incorrectly enabled IRQ");
        write_split(0, 32'h00000001, 4'b0001);
        @(negedge clk); irq_input=1; @(negedge clk); irq_input=0;
        if (!irq) $fatal(1, "enabled status did not assert IRQ");
        write_split(4, 32'h00000001, 4'b0001);
        @(negedge clk); if (irq) $fatal(1, "W1C did not deassert IRQ");
        $display("PASS: held-out AXI-Lite"); $finish;
    end
endmodule
""",
}


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    environment: dict[str, str] | None = None,
) -> JsonObject:
    """Run a fixed evaluator command and retain bounded output as typed evidence."""
    started = time.monotonic()
    env = os.environ.copy()
    if environment:
        env.update(environment)
    try:
        completed = subprocess.run(  # nosec B603 - callers construct fixed evaluator commands.
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
        status = "PASS" if returncode == 0 else "FAILED"
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + "\nTimed out."
        status = "TIMEOUT"
    return {
        "command": command,
        "returncode": returncode,
        "status": status,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": stdout[-20_000:],
        "stderr": stderr[-20_000:],
    }


def _git(repository_root: Path, arguments: list[str], *, timeout_seconds: int = 120) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is unavailable")
    completed = subprocess.run(  # nosec B603 - benchmark worktree arguments are generated locally.
        [git, *arguments],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments[:2])} failed: {completed.stderr[-1000:]}")
    return completed.stdout


def _create_worktree(repository_root: Path, target: Path, base_commit: str) -> None:
    if target.exists():
        raise RuntimeError(f"Refusing to reuse benchmark worktree {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _git(
        repository_root,
        ["worktree", "add", "--detach", str(target), base_commit],
        timeout_seconds=180,
    )


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def _await_lane(process: subprocess.Popen[str], *, timeout_seconds: int) -> JsonObject:
    started = time.monotonic()
    try:
        returncode = process.wait(timeout=timeout_seconds)
        status = "COMPLETE" if returncode == 0 else "FAILED"
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        returncode = 124
        status = "TIMEOUT"
    return {
        "returncode": returncode,
        "status": status,
        "ended_at": _now(),
        "ended_monotonic": time.monotonic(),
        "elapsed_seconds": time.monotonic() - started,
    }


def _task_environment(task: BenchmarkTask, fixture_root: Path) -> dict[str, str]:
    if task.domain == "python":
        return {"PYTHONPATH": str(fixture_root)}
    return {}


def _lane_prompt(task: BenchmarkTask, specification: JsonObject) -> str:
    return (
        "Implement the following normalized benchmark task in the current worktree. "
        "You may edit only the declared allowed paths. Run the listed public tests and applicable "
        "static checks. The correction budget is one initial implementation plus at most two repair "
        "cycles. Do not look for held-out tests, do not access another worktree, do not use network "
        "access, and finish with a concise report.\n\n"
        f"Normalized task specification:\n{json.dumps(specification, indent=2, sort_keys=True)}\n\n"
        f"Public fixture: {task.fixture}\n"
        "The evaluator will restore public tests before scoring, so do not modify tests."
    )


def _write_lane_request(
    run_root: Path,
    task: BenchmarkTask,
    specification: JsonObject,
    candidate: ServingCandidate,
    lane_root: Path,
    project_root: Path,
    result_path: Path,
) -> Path:
    request = {
        "repository_root": str(lane_root),
        "project_root": str(project_root),
        "specification": specification,
        "domain": task.domain,
        "query": task.objective,
        "candidate": candidate.to_json(),
        "result_path": str(result_path),
    }
    target = run_root / "lane_requests" / f"{task.task_id}.json"
    _write_json_atomic(target, request)
    return target


def run_laplace_lane(request_path: Path) -> JsonObject:
    """Entry point for a real local five-role lane; it never generates task code itself."""
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Laplace lane request is malformed")
    repository_root = Path(str(raw["repository_root"])).resolve()
    project_root = Path(str(raw["project_root"])).resolve()
    specification = raw.get("specification")
    domain = raw.get("domain")
    query = raw.get("query")
    candidate_raw = raw.get("candidate")
    result_path = Path(str(raw["result_path"])).resolve()
    if (
        not isinstance(specification, dict)
        or domain not in {"python", "systemverilog"}
        or not isinstance(query, str)
        or not isinstance(candidate_raw, dict)
    ):
        raise RuntimeError("Laplace lane request has invalid types")
    engine = candidate_raw.get("engine")
    text_values = {
        key: candidate_raw.get(key)
        for key in (
            "endpoint",
            "model",
            "revision",
            "quantization",
            "kernel",
            "cuda_graph_mode",
            "scheduler",
        )
    }
    prefix_caching = candidate_raw.get("prefix_caching")
    chunked_prefill = candidate_raw.get("chunked_prefill")
    if (
        engine not in {"vllm", "sglang"}
        or not all(isinstance(value, str) and value for value in text_values.values())
        or not isinstance(prefix_caching, bool)
        or not isinstance(chunked_prefill, bool)
    ):
        raise RuntimeError("Laplace lane candidate is malformed")
    candidate = ServingCandidate(
        engine=engine,
        endpoint=str(text_values["endpoint"]),
        model=str(text_values["model"]),
        revision=str(text_values["revision"]),
        quantization=str(text_values["quantization"]),
        kernel=str(text_values["kernel"]),
        prefix_caching=prefix_caching,
        chunked_prefill=chunked_prefill,
        cuda_graph_mode=str(text_values["cuda_graph_mode"]),
        scheduler=str(text_values["scheduler"]),
    )
    normalized = normalize_task_spec(repository_root, domain, specification)
    project_root.mkdir(parents=True, exist_ok=True)
    task = AgentTaskStore(project_root).create(domain, normalized)
    result = LocalTeamRunner(repository_root, project_root, candidate).run(
        task.task_id, query=query
    )
    _write_json_atomic(result_path, result)
    return result


def _patch_from_lane(lane_root: Path, base_commit: str, patch_path: Path) -> tuple[str, list[str]]:
    patch = _git(lane_root, ["diff", "--binary", base_commit], timeout_seconds=120)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch, encoding="utf-8")
    changed = [
        line[4:]
        for line in patch.splitlines()
        if line.startswith("+++ b/") and line[4:] != "/dev/null"
    ]
    return patch, changed


def _restore_public_tests(evaluation_root: Path, task: BenchmarkTask) -> None:
    """Prevent a lane from receiving credit for modifying committed public tests."""
    for name in task.public_tests:
        target = evaluation_root / task.fixture / name
        original = target.read_bytes()
        target.write_bytes(original)


def _evaluate_python(
    evaluation_root: Path,
    task: BenchmarkTask,
    control_python: str,
    timeout_seconds: int,
) -> JsonObject:
    fixture_root = evaluation_root / task.fixture
    hidden = fixture_root / "test_heldout.py"
    hidden.write_text(_PYTHON_HELD_OUT[task.task_id], encoding="utf-8")
    source = str(Path(task.fixture) / task.sources[0])
    environment = _task_environment(task, fixture_root)
    return {
        "public_tests": _run_command(
            [control_python, "-m", "pytest", "-q", str(Path(task.fixture) / task.public_tests[0])],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
            environment=environment,
        ),
        "held_out_tests": _run_command(
            [control_python, "-m", "pytest", "-q", str(hidden.relative_to(evaluation_root))],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
            environment=environment,
        ),
        "ruff_format": _run_command(
            [control_python, "-m", "ruff", "format", "--check", source],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        ),
        "ruff": _run_command(
            [control_python, "-m", "ruff", "check", source],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        ),
        "mypy": _run_command(
            [control_python, "-m", "mypy", source],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        ),
        "bandit": _run_command(
            [control_python, "-m", "bandit", "-q", "-r", source],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        ),
    }


def _evaluate_systemverilog(
    evaluation_root: Path, task: BenchmarkTask, timeout_seconds: int
) -> JsonObject:
    fixture_root = evaluation_root / task.fixture
    source = str(Path(task.fixture) / task.sources[0])
    public_tb = str(Path(task.fixture) / task.public_tests[0])
    hidden = fixture_root / "tb_heldout.sv"
    hidden.write_text(_SV_HELD_OUT[task.task_id], encoding="utf-8")
    hidden_path = str(hidden.relative_to(evaluation_root))
    compiled_public = str(Path(".paired_eval") / f"{task.task_id}_public.vvp")
    compiled_hidden = str(Path(".paired_eval") / f"{task.task_id}_hidden.vvp")
    (evaluation_root / ".paired_eval").mkdir(exist_ok=True)
    public_compile = _run_command(
        ["iverilog", "-g2012", "-o", compiled_public, source, public_tb],
        cwd=evaluation_root,
        timeout_seconds=timeout_seconds,
    )
    public_run = (
        _run_command(["vvp", compiled_public], cwd=evaluation_root, timeout_seconds=timeout_seconds)
        if public_compile["status"] == "PASS"
        else {"status": "NOT_RUN"}
    )
    held_compile = _run_command(
        ["iverilog", "-g2012", "-s", "tb_heldout", "-o", compiled_hidden, source, hidden_path],
        cwd=evaluation_root,
        timeout_seconds=timeout_seconds,
    )
    held_run = (
        _run_command(["vvp", compiled_hidden], cwd=evaluation_root, timeout_seconds=timeout_seconds)
        if held_compile["status"] == "PASS"
        else {"status": "NOT_RUN"}
    )
    return {
        "public_compile": public_compile,
        "public_tests": public_run,
        "held_out_compile": held_compile,
        "held_out_tests": held_run,
        "verilator": _run_command(
            ["verilator", "--lint-only", "--Wall", "--sv", source],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        ),
        "yosys": _run_command(
            [
                "yosys",
                "-p",
                f"read_verilog -sv {source}; hierarchy -top {Path(task.sources[0]).stem}; synth -top {Path(task.sources[0]).stem}; stat",
            ],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        ),
    }


def _passed(result: object) -> bool:
    return isinstance(result, dict) and result.get("status") == "PASS"


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _score(task: BenchmarkTask, evaluation: JsonObject, changed_paths: list[str]) -> JsonObject:
    allowed = {f"{task.fixture}/{path}" for path in task.sources}
    scope_ok = bool(changed_paths) and all(path in allowed for path in changed_paths)
    if task.domain == "python":
        held_out = _passed(evaluation.get("held_out_tests"))
        public = _passed(evaluation.get("public_tests"))
        typing = _passed(evaluation.get("mypy"))
        ruff = _passed(evaluation.get("ruff")) and _passed(evaluation.get("ruff_format"))
        security = _passed(evaluation.get("bandit"))
        score = (
            (35 if held_out else 0)
            + (15 if public else 0)
            + (10 if typing else 0)
            + (10 if ruff else 0)
            + (10 if security else 0)
            + (10 if scope_ok and public else 0)
            + (5 if scope_ok else 0)
            + (5 if held_out and security else 0)
        )
        return {
            "objective_score_0_100": score,
            "functional_correctness": public,
            "held_out_correctness": held_out,
            "typing_and_lint_quality": typing and ruff,
            "security": security,
            "maintainability_scope": scope_ok,
            "robustness": held_out and security,
        }
    held_out = _passed(evaluation.get("held_out_tests"))
    public = _passed(evaluation.get("public_tests"))
    lint = _passed(evaluation.get("verilator")) and _passed(evaluation.get("held_out_compile"))
    synthesis = _passed(evaluation.get("yosys"))
    score = (
        (35 if held_out else 0)
        + (20 if held_out else 0)
        + (10 if lint else 0)
        + (10 if synthesis else 0)
        + (10 if held_out and lint else 0)
        + (10 if public else 0)
        + (5 if scope_ok else 0)
    )
    return {
        "objective_score_0_100": score,
        "functional_correctness": public,
        "held_out_correctness": held_out,
        "rtl_protocol_correctness": held_out,
        "rtl_lint": lint,
        "synthesis": synthesis,
        "maintainability_scope": scope_ok,
        "robustness": held_out and lint,
    }


def _evaluate_lane(
    repository_root: Path,
    run_root: Path,
    task: BenchmarkTask,
    lane: str,
    implementation_root: Path,
    base_commit: str,
    control_python: str,
    timeout_seconds: int,
) -> JsonObject:
    patch_path = run_root / "patches" / task.task_id / f"{lane}.patch"
    patch, changed_paths = _patch_from_lane(implementation_root, base_commit, patch_path)
    evaluation_root = run_root / "evaluation_worktrees" / task.task_id / lane
    _create_worktree(repository_root, evaluation_root, base_commit)
    public_contents = {
        name: (evaluation_root / task.fixture / name).read_bytes() for name in task.public_tests
    }
    if patch:
        applied = _run_command(
            ["git", "apply", "--whitespace=error", str(patch_path)],
            cwd=evaluation_root,
            timeout_seconds=timeout_seconds,
        )
    else:
        applied = {"status": "PASS", "returncode": 0, "command": ["git", "apply", "<empty>"]}
    for name, content in public_contents.items():
        (evaluation_root / task.fixture / name).write_bytes(content)
    if applied.get("status") != "PASS":
        evaluation: JsonObject = {"status": "PATCH_APPLY_FAILED", "patch_apply": applied}
    elif task.domain == "python":
        evaluation = _evaluate_python(evaluation_root, task, control_python, timeout_seconds)
    else:
        evaluation = _evaluate_systemverilog(evaluation_root, task, timeout_seconds)
    score = _score(task, evaluation, changed_paths)
    return {
        "implementation_root": str(implementation_root),
        "evaluation_worktree": str(evaluation_root),
        "patch": str(patch_path),
        "patch_bytes": len(patch.encode("utf-8")),
        "diff_lines": len(patch.splitlines()),
        "changed_paths": changed_paths,
        "evaluation": evaluation,
        "score": score,
        "public_tests_modified_by_lane": any(
            f"{task.fixture}/{name}" in changed_paths for name in task.public_tests
        ),
    }


def _lane_record(
    lane: str,
    started_at: str,
    started_monotonic: float,
    log_path: Path,
    result: JsonObject,
) -> JsonObject:
    ended_monotonic = result.get("ended_monotonic")
    ended = (
        float(ended_monotonic) if isinstance(ended_monotonic, (int, float)) else started_monotonic
    )
    return {
        "lane": lane,
        "started_at": started_at,
        "started_monotonic": started_monotonic,
        "ended_at": result.get("ended_at"),
        "ended_monotonic": ended,
        "wall_time_s": max(0.0, ended - started_monotonic),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "command_log": str(log_path),
        "human_intervention": 0,
        "measurement_limits": {
            "repair_cycles": "Codex exec does not expose a reliable repair-cycle counter; recorded as unmeasured.",
            "model_generation_usage": "Codex exec does not expose generation accounting to this runner; recorded as unmeasured.",
        },
    }


def _aggregate_scores(task_rows: list[JsonObject]) -> JsonObject:
    result: JsonObject = {}
    for domain in ("python", "systemverilog"):
        domain_rows = [row for row in task_rows if row.get("domain") == domain]
        lane_summary: JsonObject = {}
        for lane in ("codex_direct", "laplace_team"):
            scores: list[float] = []
            times: list[float] = []
            held_out_passes = 0
            functional_passes = 0
            for row in domain_rows:
                lanes = row.get("lanes")
                lane_data = lanes.get(lane) if isinstance(lanes, dict) else None
                if not isinstance(lane_data, dict):
                    continue
                score = lane_data.get("score")
                score_data = score if isinstance(score, dict) else {}
                score_value = score_data.get("objective_score_0_100")
                if isinstance(score_value, (int, float)):
                    scores.append(float(score_value))
                times.append(_number(lane_data.get("wall_time_s")))
                held_out_passes += int(score_data.get("held_out_correctness") is True)
                functional_passes += int(score_data.get("functional_correctness") is True)
            lane_summary[lane] = {
                "tasks": len(domain_rows),
                "mean_score": statistics.mean(scores) if scores else None,
                "median_score": statistics.median(scores) if scores else None,
                "functional_pass_rate": functional_passes / len(domain_rows)
                if domain_rows
                else None,
                "held_out_pass_rate": held_out_passes / len(domain_rows) if domain_rows else None,
                "median_wall_time_s": statistics.median(times) if times else None,
                "score_per_minute": (
                    statistics.mean(scores) / (statistics.mean(times) / 60)
                    if scores and times and statistics.mean(times) > 0
                    else None
                ),
            }
        result[domain] = lane_summary
    return result


def run_valid_paired_benchmark(
    repository_root: Path,
    candidate: ServingCandidate,
    *,
    base_commit: str,
    timeout_seconds: int = 900,
    control_python: str | None = None,
    codex_command: str | None = None,
) -> JsonObject:
    """Run six real, overlapping Codex-direct and local-Laplace implementation pairs."""
    root = repository_root.resolve()
    if not _COMMIT_RE.fullmatch(base_commit):
        raise RuntimeError("Paired benchmark requires an exact checkpoint commit")
    cuda = collect_cuda_evidence(LocalToolRunner(root))
    if cuda["status"] != "CUDA_A6000_VERIFIED":
        raise RuntimeError("Paired benchmark refuses to run without verified A6000 CUDA")
    python_executable = control_python or sys.executable
    codex = codex_command or shutil.which("codex")
    if not codex:
        raise RuntimeError("codex executable is unavailable for the Codex-direct lane")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_root = root / "outputs" / "a6000_agent_team" / "comparison" / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    task_rows: list[JsonObject] = []
    for task in _BENCHMARK_TASKS:
        spec = _task_spec(task)
        validate_root = root
        normalized = normalize_task_spec(validate_root, task.domain, spec)
        _write_json_atomic(run_root / "task_specs" / f"{task.task_id}.json", normalized)
        lane_root = run_root / "lane_worktrees" / task.task_id
        codex_root = lane_root / "codex_direct"
        laplace_outer_root = lane_root / "laplace_outer"
        _create_worktree(root, codex_root, base_commit)
        _create_worktree(root, laplace_outer_root, base_commit)
        logs_root = run_root / "command_logs" / task.task_id
        logs_root.mkdir(parents=True, exist_ok=True)
        codex_log = logs_root / "codex_direct.log"
        laplace_log = logs_root / "laplace_team.log"
        laplace_project = run_root / "laplace_projects" / task.task_id
        laplace_result = run_root / "laplace_results" / f"{task.task_id}.json"
        request = _write_lane_request(
            run_root,
            task,
            spec,
            candidate,
            laplace_outer_root,
            laplace_project,
            laplace_result,
        )
        codex_started_at, codex_started = _now(), time.monotonic()
        with codex_log.open("w", encoding="utf-8") as codex_stream:
            codex_process = subprocess.Popen(  # nosec B603 - executable and arguments are fixed.
                [
                    codex,
                    "exec",
                    "--sandbox",
                    "workspace-write",
                    "--cd",
                    str(codex_root),
                    "--output-last-message",
                    str(logs_root / "codex_final_message.txt"),
                    _lane_prompt(task, normalized),
                ],
                cwd=codex_root,
                stdout=codex_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            laplace_started_at, laplace_started = _now(), time.monotonic()
            with laplace_log.open("w", encoding="utf-8") as laplace_stream:
                laplace_process = subprocess.Popen(  # nosec B603 - fixed module and request path.
                    [
                        python_executable,
                        "-m",
                        "research_workspace.paired_benchmark",
                        "--laplace-lane",
                        str(request),
                    ],
                    cwd=laplace_outer_root,
                    stdout=laplace_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    codex_result, laplace_result_status = tuple(
                        future.result()
                        for future in (
                            executor.submit(
                                _await_lane, codex_process, timeout_seconds=timeout_seconds
                            ),
                            executor.submit(
                                _await_lane, laplace_process, timeout_seconds=timeout_seconds
                            ),
                        )
                    )
        codex_record = _lane_record(
            "codex_direct", codex_started_at, codex_started, codex_log, codex_result
        )
        laplace_record = _lane_record(
            "laplace_team", laplace_started_at, laplace_started, laplace_log, laplace_result_status
        )
        laplace_implementation = laplace_outer_root
        if laplace_result.is_file():
            raw_result = json.loads(laplace_result.read_text(encoding="utf-8"))
            if isinstance(raw_result, dict) and isinstance(raw_result.get("worktree"), str):
                laplace_implementation = Path(raw_result["worktree"]).resolve()
            laplace_record["typed_result"] = raw_result
        overlap_seconds = max(
            0.0,
            min(
                _number(codex_record.get("ended_monotonic")),
                _number(laplace_record.get("ended_monotonic")),
            )
            - max(codex_started, laplace_started),
        )
        codex_evaluation = _evaluate_lane(
            root,
            run_root,
            task,
            "codex_direct",
            codex_root,
            base_commit,
            python_executable,
            300,
        )
        laplace_evaluation = _evaluate_lane(
            root,
            run_root,
            task,
            "laplace_team",
            laplace_implementation,
            base_commit,
            python_executable,
            300,
        )
        valid = (
            overlap_seconds > 0
            and codex_record["status"] != "TIMEOUT"
            and laplace_record["status"] != "TIMEOUT"
            and not laplace_evaluation["public_tests_modified_by_lane"]
            and not codex_evaluation["public_tests_modified_by_lane"]
        )
        typed_result = laplace_record.get("typed_result")
        typed_task = typed_result.get("task") if isinstance(typed_result, dict) else None
        repair_cycles = typed_task.get("correction_loops") if isinstance(typed_task, dict) else None
        task_rows.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "base_commit": base_commit,
                "valid_run": valid,
                "invalid_run_reason": None
                if valid
                else "timeout, no temporal overlap, or public-test modification",
                "execution_overlap_proven": overlap_seconds > 0,
                "overlap_seconds": overlap_seconds,
                "held_out_exposed_to_lanes": False,
                "resource_symmetry": {
                    "same_base_commit": True,
                    "same_normalized_spec": True,
                    "same_public_tests": True,
                    "same_timeout_seconds": timeout_seconds,
                    "same_correction_budget": 2,
                    "same_tool_access": "repository-local approved tools only",
                },
                "lanes": {
                    "codex_direct": {
                        **codex_record,
                        **codex_evaluation,
                        "repair_cycles": None,
                        "model_output_tokens": None,
                    },
                    "laplace_team": {
                        **laplace_record,
                        **laplace_evaluation,
                        "repair_cycles": repair_cycles,
                        "model_output_tokens": None,
                    },
                },
            }
        )
    valid_tasks = sum(1 for row in task_rows if row["valid_run"] is True)
    payload: JsonObject = {
        "schema_version": 2,
        "benchmark_id": "codex_vs_laplace_a6000",
        "status": "MEASURED" if valid_tasks == len(_BENCHMARK_TASKS) else "INVALID",
        "created_at": _now(),
        "run_root": str(run_root),
        "base_commit": base_commit,
        "cuda_evidence": cuda,
        "model_candidate": candidate.to_json(),
        "codex_cli": {"path": codex, "command": "codex exec", "version": None},
        "tasks": task_rows,
        "aggregate": {
            "valid_tasks": valid_tasks,
            "required_tasks": len(_BENCHMARK_TASKS),
            "winner": None,
            "by_domain": _aggregate_scores(task_rows),
            "conclusion": "Objective scores are recorded per lane. No aggregate winner is forced; invalid or missing measurements remain explicit.",
        },
    }
    paths = _write_reports(root / "outputs" / "a6000_agent_team" / "comparison", payload)
    payload["reports"] = paths
    return payload


def run_paired_quality_benchmark(repository_root: Path) -> JsonObject:
    """Preflight and report the final comparison without fabricating results."""
    root = repository_root.resolve()
    tasks = _catalog_tasks(root)
    cuda = collect_cuda_evidence(LocalToolRunner(root))
    output_root = root / "outputs" / "a6000_agent_team" / "comparison"
    if cuda["status"] != "CUDA_A6000_VERIFIED":
        reason = "Final paired run is invalid because the required A6000 CUDA lane could not start. Codex-direct was not started, preserving equal resources."
        payload: JsonObject = {
            "schema_version": 1,
            "benchmark_id": "codex_vs_laplace_a6000",
            "status": "INVALID_BLOCKED_GPU",
            "created_at": _now(),
            "reason": reason,
            "cuda_evidence": cuda,
            "tasks": _blocked_rows(tasks, reason),
            "aggregate": {
                "valid_tasks": 0,
                "required_tasks": 6,
                "winner": None,
                "conclusion": "No comparative conclusion is valid.",
            },
        }
        paths = _write_reports(output_root, payload)
        payload["reports"] = paths
        return payload
    payload = {
        "schema_version": 1,
        "benchmark_id": "codex_vs_laplace_a6000",
        "status": "PRECONDITIONS_MET_RUNNER_REQUIRED",
        "created_at": _now(),
        "cuda_evidence": cuda,
        "tasks": tasks,
        "reason": "CUDA preflight passed. Launch the dedicated asynchronous benchmark runner with clean worktrees and held-out tests outside both lanes.",
        "aggregate": {"valid_tasks": 0, "required_tasks": 6, "winner": None},
    }
    paths = _write_reports(output_root, payload)
    payload["reports"] = paths
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-paired-benchmark")
    parser.add_argument("--laplace-lane", type=Path)
    args = parser.parse_args(argv)
    if args.laplace_lane is None:
        parser.error("--laplace-lane is required for the process entry point")
    try:
        result = run_laplace_lane(args.laplace_lane)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
