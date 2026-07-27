#!/usr/bin/env python3
"""Execute and preserve the orchestration-upgrade CPU certification smokes."""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.engineering import LocalToolRunner, resolve_eda_executable
from research_workspace.multilanguage_ablation import _activate_isolated_tools
from research_workspace.verification_gates import VerificationGateRegistry


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run(
    root: Path,
    output: Path,
    name: str,
    command: Sequence[str],
) -> dict[str, object]:
    started = datetime.now(UTC).isoformat()
    completed = subprocess.run(  # nosec B603
        list(command),
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    log_path = output / f"{name}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "command": list(command),
        "started_at": started,
        "ended_at": datetime.now(UTC).isoformat(),
        "returncode": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAILED",
        "log_path": str(log_path),
    }


def _fixture_source() -> str:
    return """module smoke_dut(
  input logic clk,
  input logic rst_n,
  input logic in_bit,
  output logic out_bit
);
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) out_bit <= 1'b0;
  else out_bit <= in_bit;
end
endmodule
"""


def _passing_testbench(module: str, marker: str = "PASS") -> str:
    return f"""module {module};
logic clk=0,rst_n=0,in_bit,out_bit;
smoke_dut dut(.*);
always #1 clk=~clk;
initial begin
  in_bit=0; repeat(2) @(posedge clk); rst_n=1;
  @(negedge clk); in_bit=1; @(negedge clk);
  if(out_bit!==1'b1) $fatal(1,"registered value missing");
  $display("{marker} deterministic RTL fixture"); $finish;
end
endmodule
"""


def _verification_fixture(root: Path, output: Path) -> dict[str, object]:
    fixture = output / "verification_fixture"
    logs = output / "verification_fixture_logs"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "smoke_dut.sv").write_text(_fixture_source(), encoding="utf-8")
    (fixture / "tb_public.sv").write_text(
        _passing_testbench("tb_public"), encoding="utf-8"
    )
    (fixture / "tb_adversarial.sv").write_text(
        _passing_testbench("tb_adversarial"), encoding="utf-8"
    )
    runner = LocalToolRunner(fixture, logs)
    public = runner.run_eda_flow(
        ["smoke_dut.sv"],
        top_module="smoke_dut",
        testbench="tb_public.sv",
        language="systemverilog",
        require_verilator_simulation=True,
        required_tools=(
            "iverilog",
            "vvp",
            "verilator",
            "verilator_simulation",
            "yosys",
        ),
        run_id="cpu-smoke",
        task_id="verification-fixture",
    )
    adversarial = runner.run_eda_flow(
        ["smoke_dut.sv"],
        top_module="smoke_dut",
        testbench="tb_adversarial.sv",
        language="systemverilog",
        require_verilator_simulation=True,
        required_tools=(
            "iverilog",
            "vvp",
            "verilator",
            "verilator_simulation",
            "yosys",
        ),
        run_id="cpu-smoke",
        task_id="verification-adversarial",
    )
    gates_raw = public.get("gate_results")
    gates = (
        {str(key): dict(value) for key, value in gates_raw.items() if isinstance(value, dict)}
        if isinstance(gates_raw, dict)
        else {}
    )
    gates["adversarial_protocol_simulation"] = {
        "tool": "vvp",
        "gate": "adversarial_protocol_simulation",
        "status": "PASS" if adversarial.get("passed") is True else "FAILED",
        "returncode": 0 if adversarial.get("passed") is True else 1,
        "executed": True,
        "log_path": next(
            (
                item.get("log_path")
                for item in adversarial.get("results", [])
                if isinstance(item, dict) and item.get("gate") == "vvp_simulation"
            ),
            None,
        ),
    }
    final = VerificationGateRegistry.evaluate(
        "systemverilog",
        gates,
        available_tools={
            "iverilog": resolve_eda_executable("iverilog") is not None,
            "vvp": resolve_eda_executable("vvp") is not None,
            "verilator": resolve_eda_executable("verilator") is not None,
            "yosys": resolve_eda_executable("yosys") is not None,
        },
    ).to_json()

    (fixture / "tb_nonzero.sv").write_text(
        """module tb_nonzero; logic clk=0,rst_n=1,in_bit=0,out_bit;
smoke_dut dut(.*); initial begin #1; $fatal(1,"intentional"); end endmodule
""",
        encoding="utf-8",
    )
    nonzero = runner.run_eda_flow(
        ["smoke_dut.sv"],
        top_module="smoke_dut",
        testbench="tb_nonzero.sv",
        language="systemverilog",
        require_verilator_simulation=True,
        run_id="cpu-smoke",
        task_id="nonzero-binary",
    )
    (fixture / "tb_missing_marker.sv").write_text(
        _passing_testbench("tb_missing_marker", marker="DONE"),
        encoding="utf-8",
    )
    missing_marker = runner.run_eda_flow(
        ["smoke_dut.sv"],
        top_module="smoke_dut",
        testbench="tb_missing_marker.sv",
        language="systemverilog",
        require_verilator_simulation=True,
        run_id="cpu-smoke",
        task_id="missing-marker",
    )
    (fixture / "broken_synthesis.sv").write_text(
        "module broken_synthesis(input logic a, output logic y); assign y = ; endmodule\n",
        encoding="utf-8",
    )
    yosys = resolve_eda_executable("yosys")
    broken_synthesis = (
        runner.run(
            "yosys",
            [
                str(yosys),
                "-p",
                "read_verilog -sv broken_synthesis.sv; synth -top broken_synthesis",
            ],
        ).to_json()
        if yosys is not None
        else {"status": "MISSING"}
    )
    skipped_gates = dict(gates)
    skipped_gates.pop("verilator_simulation", None)
    skipped = VerificationGateRegistry.evaluate(
        "systemverilog",
        skipped_gates,
        available_tools={
            "iverilog": True,
            "vvp": True,
            "verilator": True,
            "yosys": True,
        },
    ).to_json()
    nonzero_gate = (
        nonzero.get("gate_results", {}).get("verilator_simulation", {})
        if isinstance(nonzero.get("gate_results"), dict)
        else {}
    )
    marker_gate = (
        missing_marker.get("gate_results", {}).get("verilator_simulation", {})
        if isinstance(missing_marker.get("gate_results"), dict)
        else {}
    )
    acceptance = {
        "require_verilator_simulation": public.get("require_verilator_simulation") is True,
        "verilator_simulation_executed": (
            public.get("verilator_simulation_executed") is True
        ),
        "missing_tools_empty": final.get("missing_tools") == [],
        "missing_required_results_empty": final.get("missing_required_results") == [],
        "verification_passed": final.get("passed") is True,
        "skipped_verilator_rejected": (
            skipped.get("passed") is False
            and "verilator_simulation" in skipped.get("missing_required_results", [])
        ),
        "nonzero_binary_rejected": (
            isinstance(nonzero_gate, dict)
            and nonzero_gate.get("executed") is True
            and nonzero_gate.get("status") == "FAILED"
            and nonzero_gate.get("returncode") not in {None, 0}
        ),
        "missing_marker_rejected": (
            isinstance(marker_gate, dict)
            and marker_gate.get("executed") is True
            and marker_gate.get("status") == "FAILED"
            and marker_gate.get("success_marker_present") is False
        ),
        "broken_synthesis_rejected": broken_synthesis.get("status") == "FAILED",
    }
    return {
        "status": "PASS" if all(acceptance.values()) else "FAILED",
        "acceptance": acceptance,
        "verification": final,
        "public_flow": public,
        "adversarial_flow": adversarial,
        "failure_injections": {
            "skipped_verilator": skipped,
            "nonzero_binary": nonzero,
            "missing_marker": missing_marker,
            "broken_synthesis": broken_synthesis,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _activate_isolated_tools(root)
    python = Path(sys.executable)
    checks = [
        ("compile", [str(python), "-m", "compileall", "-q", "src", "tests"]),
        (
            "targeted_pytest",
            [
                str(python),
                "-m",
                "pytest",
                "tests/test_reproducibility.py",
                "tests/test_execution_records.py",
                "tests/test_verification_gate_registry.py",
                "tests/test_rtl_worker_correction_evidence.py",
                "tests/test_research_plane.py",
                "tests/test_operator_service.py",
                "tests/test_operator_api.py",
                "tests/test_model_servers.py",
                "tests/test_notifications.py",
                "tests/test_certification_bundle.py",
                "tests/test_orchestration_certification.py",
                "-q",
            ],
        ),
        (
            "full_pytest_cpu",
            [
                str(python),
                "-m",
                "pytest",
                "--ignore=tests/test_operator_gui_e2e.py",
                "-q",
            ],
        ),
        ("ruff", [str(python), "-m", "ruff", "check", "src", "tests"]),
        ("mypy", [str(python), "-m", "mypy", "src/research_workspace"]),
        ("bandit_high", [str(python), "-m", "bandit", "-q", "-lll", "-r", "src/research_workspace"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    results = [_run(root, output, name, command) for name, command in checks]
    fixture = _verification_fixture(root, output)
    _write_json(output / "smoke_4_verification.json", fixture)
    passed = all(item["status"] == "PASS" for item in results) and fixture["status"] == "PASS"
    summary = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAILED",
        "created_at": datetime.now(UTC).isoformat(),
        "checks": results,
        "smoke_2_skills_context": "covered_by_targeted_pytest",
        "smoke_3_event_resume": "covered_by_targeted_pytest_interruptions",
        "smoke_4_verification": str(output / "smoke_4_verification.json"),
        "smoke_5_reviewer": "covered_by_targeted_pytest",
        "smoke_6_no_effect": "covered_by_targeted_pytest",
        "smoke_10_fixture_research": "covered_by_targeted_pytest",
    }
    _write_json(output / "cpu_smoke_summary.json", summary)
    with (output / "test_results.txt").open("w", encoding="utf-8") as combined:
        for result in results:
            combined.write(
                f"{result['name']}: {result['status']} returncode={result['returncode']}\n"
            )
            combined.write(Path(str(result["log_path"])).read_text(encoding="utf-8"))
            combined.write("\n")
        combined.write(
            f"real_verification_fixture: {fixture['status']}\n"
            f"acceptance={json.dumps(fixture['acceptance'], sort_keys=True)}\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
