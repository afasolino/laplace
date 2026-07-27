"""Authoritative verification-gate registry and report projection.

The registry is deliberately independent from the runner.  Tool execution
produces named gate records; this module alone decides which records are
required and whether a verification report is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

Domain = Literal["python", "c", "verilog", "systemverilog"]
JsonObject = dict[str, object]

GateScope = Literal["public", "adversarial", "final"]


@dataclass(frozen=True)
class GateDefinition:
    gate_id: str
    tool: str
    scope: Literal["public", "adversarial"]
    functional: bool

    def to_json(self) -> JsonObject:
        return {
            "gate_id": self.gate_id,
            "tool": self.tool,
            "scope": self.scope,
            "functional": self.functional,
        }


@dataclass(frozen=True)
class VerificationSummary:
    domain: Domain
    scope: GateScope
    required_gates: tuple[str, ...]
    required_tools: tuple[str, ...]
    executed_gates: tuple[str, ...]
    executed_tools: tuple[str, ...]
    missing_tools: tuple[str, ...]
    missing_required_results: tuple[str, ...]
    passed: bool
    gate_matrix: tuple[JsonObject, ...]

    def to_json(self) -> JsonObject:
        return {
            "domain": self.domain,
            "scope": self.scope,
            "required_gates": list(self.required_gates),
            "required_tools": list(self.required_tools),
            "executed_gates": list(self.executed_gates),
            "executed_tools": list(self.executed_tools),
            "missing_tools": list(self.missing_tools),
            "missing_required_results": list(self.missing_required_results),
            "passed": self.passed,
            "verification_status": "PASSED" if self.passed else "FAILED",
            "gate_matrix": list(self.gate_matrix),
        }


@dataclass(frozen=True)
class NoEffectDisposition:
    classification: Literal[
        "converged_no_change",
        "reviewer_state_conflict",
        "no_effect_correction",
    ]
    complete: bool
    reconcile_review: bool


def classify_no_effect_correction(
    *,
    verification_passed: bool,
    reviewer_verdict: str,
    reviewer_change_already_present: bool = False,
) -> NoEffectDisposition:
    """Classify a byte-identical correction without hiding failed evidence."""
    if verification_passed and reviewer_verdict == "approve":
        return NoEffectDisposition("converged_no_change", True, False)
    if verification_passed and reviewer_change_already_present:
        return NoEffectDisposition("reviewer_state_conflict", False, True)
    return NoEffectDisposition("no_effect_correction", False, False)


class VerificationGateRegistry:
    """Single source of truth for deterministic gate and tool requirements."""

    _SYSTEMVERILOG = (
        GateDefinition("self_checking_public_simulation", "vvp", "public", True),
        GateDefinition("adversarial_protocol_simulation", "vvp", "adversarial", True),
        GateDefinition("verilator_lint", "verilator", "public", False),
        GateDefinition("verilator_simulation", "verilator", "public", True),
        GateDefinition("iverilog_compile", "iverilog", "public", False),
        GateDefinition("vvp_simulation", "vvp", "public", True),
        GateDefinition("yosys_synthesis", "yosys", "public", False),
    )
    _VERILOG = tuple(
        item for item in _SYSTEMVERILOG if item.gate_id != "verilator_simulation"
    )
    _PYTHON = (
        GateDefinition("explicit_public_fixture_test", "pytest", "public", True),
        GateDefinition("adversarial_negative_path_test", "pytest", "adversarial", True),
        GateDefinition("ruff_format_check", "ruff", "public", False),
        GateDefinition("ruff_lint", "ruff", "public", False),
        GateDefinition("strict_mypy", "mypy", "public", False),
        GateDefinition("pytest", "pytest", "public", True),
        GateDefinition("coverage_pytest", "coverage", "public", True),
        GateDefinition("bandit", "bandit", "public", False),
    )
    _C = (
        GateDefinition("self_checking_public_unit_tests", "ctest", "public", True),
        GateDefinition("gcc_or_clang_warnings", "gcc", "public", False),
        GateDefinition("cmake_build", "cmake", "public", False),
        GateDefinition("ctest", "ctest", "public", True),
        GateDefinition("address_sanitizer", "asan", "adversarial", True),
        GateDefinition("undefined_behavior_sanitizer", "ubsan", "adversarial", True),
        GateDefinition("static_analysis_when_available", "clang_tidy", "public", False),
    )

    @classmethod
    def definitions(cls, domain: Domain, *, scope: GateScope = "final") -> tuple[GateDefinition, ...]:
        definitions = {
            "python": cls._PYTHON,
            "c": cls._C,
            "verilog": cls._VERILOG,
            "systemverilog": cls._SYSTEMVERILOG,
        }[domain]
        if scope == "final":
            return definitions
        return tuple(item for item in definitions if item.scope == scope)

    @classmethod
    def required_gates(cls, domain: Domain, *, scope: GateScope = "final") -> tuple[str, ...]:
        return tuple(item.gate_id for item in cls.definitions(domain, scope=scope))

    @classmethod
    def required_tools(cls, domain: Domain, *, scope: GateScope = "final") -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.tool for item in cls.definitions(domain, scope=scope)))

    @classmethod
    def acceptance_matrix(
        cls, domain: Domain, requirements: Sequence[str]
    ) -> list[JsonObject]:
        gates = list(cls.required_gates(domain))
        return [
            {"requirement": requirement, "required_evidence": gates}
            for requirement in requirements
        ]

    @classmethod
    def evaluate(
        cls,
        domain: Domain,
        gate_results: Mapping[str, JsonObject],
        *,
        scope: GateScope = "final",
        available_tools: Mapping[str, bool] | None = None,
    ) -> VerificationSummary:
        definitions = cls.definitions(domain, scope=scope)
        required_gates = tuple(item.gate_id for item in definitions)
        required_tools = tuple(dict.fromkeys(item.tool for item in definitions))
        executed_gates = tuple(
            gate
            for gate in required_gates
            if gate in gate_results and gate_results[gate].get("executed", True) is True
        )
        tool_by_gate = {item.gate_id: item.tool for item in definitions}
        executed_tools = tuple(
            dict.fromkeys(
                [
                    *(tool_by_gate[gate] for gate in executed_gates),
                    *(
                        str(gate_results[gate]["tool"])
                        for gate in executed_gates
                        if isinstance(gate_results[gate].get("tool"), str)
                    ),
                ]
            )
        )
        missing_tools = tuple(
            tool
            for tool in required_tools
            if available_tools is not None and available_tools.get(tool) is not True
        )
        missing_results = tuple(
            gate for gate in required_gates if gate not in executed_gates
        )
        matrix: list[JsonObject] = []
        for definition in definitions:
            raw = gate_results.get(definition.gate_id)
            status = raw.get("status") if isinstance(raw, dict) else "MISSING"
            matrix.append(
                {
                    **definition.to_json(),
                    "status": status,
                    "executed": raw is not None,
                    "returncode": raw.get("returncode") if isinstance(raw, dict) else None,
                    "log_path": raw.get("log_path") if isinstance(raw, dict) else None,
                }
            )
        passed = (
            not missing_tools
            and not missing_results
            and all(gate_results[item].get("status") == "PASS" for item in required_gates)
        )
        return VerificationSummary(
            domain=domain,
            scope=scope,
            required_gates=required_gates,
            required_tools=required_tools,
            executed_gates=executed_gates,
            executed_tools=executed_tools,
            missing_tools=missing_tools,
            missing_required_results=missing_results,
            passed=passed,
            gate_matrix=tuple(matrix),
        )
