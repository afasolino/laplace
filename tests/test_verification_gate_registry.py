from __future__ import annotations

from research_workspace.verification_gates import (
    VerificationGateRegistry,
    classify_no_effect_correction,
)


EXPECTED_SYSTEMVERILOG_GATES = (
    "self_checking_public_simulation",
    "adversarial_protocol_simulation",
    "verilator_lint",
    "verilator_simulation",
    "iverilog_compile",
    "vvp_simulation",
    "yosys_synthesis",
)


def test_systemverilog_registry_is_authoritative_and_ordered() -> None:
    assert VerificationGateRegistry.required_gates("systemverilog") == (
        EXPECTED_SYSTEMVERILOG_GATES
    )
    assert VerificationGateRegistry.required_tools("systemverilog") == (
        "vvp",
        "verilator",
        "iverilog",
        "yosys",
    )


def test_registry_reports_missing_and_failed_gates() -> None:
    gate_results = {
        gate: {"tool": "fixture", "status": "PASS", "returncode": 0}
        for gate in EXPECTED_SYSTEMVERILOG_GATES
    }
    gate_results["yosys_synthesis"]["status"] = "FAILED"
    del gate_results["verilator_simulation"]

    result = VerificationGateRegistry.evaluate(
        "systemverilog",
        gate_results,
        available_tools={
            "vvp": True,
            "verilator": True,
            "iverilog": True,
            "yosys": True,
        },
    )

    assert result.passed is False
    assert result.missing_required_results == ("verilator_simulation",)
    assert "verilator_simulation" not in result.executed_gates


def test_registry_accepts_only_complete_passing_matrix() -> None:
    tool_for_gate = {
        item.gate_id: item.tool
        for item in VerificationGateRegistry.definitions("systemverilog")
    }
    results = {
        gate: {"tool": tool_for_gate[gate], "status": "PASS", "returncode": 0}
        for gate in EXPECTED_SYSTEMVERILOG_GATES
    }

    summary = VerificationGateRegistry.evaluate(
        "systemverilog",
        results,
        available_tools={tool: True for tool in set(tool_for_gate.values())},
    )

    assert summary.passed is True
    assert summary.missing_tools == ()
    assert summary.missing_required_results == ()


def test_no_effect_converges_only_with_passing_gates_and_approval() -> None:
    converged = classify_no_effect_correction(
        verification_passed=True, reviewer_verdict="approve"
    )
    failed = classify_no_effect_correction(
        verification_passed=False, reviewer_verdict="approve"
    )
    conflict = classify_no_effect_correction(
        verification_passed=True,
        reviewer_verdict="request_changes",
        reviewer_change_already_present=True,
    )

    assert converged.classification == "converged_no_change"
    assert converged.complete is True
    assert failed.classification == "no_effect_correction"
    assert failed.complete is False
    assert conflict.classification == "reviewer_state_conflict"
    assert conflict.reconcile_review is True
