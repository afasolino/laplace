"""Strict handoff contract from the main engineering model to the RTL worker."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .engineering import Domain, JsonObject, _inside, _safe_relative
from .repair_protocol import (
    StructuredOutputError,
    _require_exact_keys,
    _single_json_object,
    file_sha256,
)


RtlLanguage = Literal["verilog", "systemverilog"]


@dataclass(frozen=True)
class RtlWorkerContract:
    value: JsonObject

    @property
    def editable_path(self) -> str:
        return str(self.value["editable_path"])

    @property
    def language(self) -> RtlLanguage:
        value = self.value["language"]
        if value == "verilog":
            return "verilog"
        return "systemverilog"

    def to_json(self) -> JsonObject:
        return dict(self.value)


def _non_empty_strings(value: object, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise StructuredOutputError(f"RTL contract {label} must be a non-empty string list")
    return list(value)


def _validate_parameters(value: object) -> None:
    if not isinstance(value, list):
        raise StructuredOutputError("RTL contract parameters must be a list")
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise StructuredOutputError(f"RTL contract parameter {index} must be an object")
        item = dict(raw)
        _require_exact_keys(
            item,
            {"name", "type", "default", "constraints"},
            label=f"RTL contract parameter {index}",
        )
        if not all(
            isinstance(item.get(key), str) and str(item[key]).strip()
            for key in ("name", "type", "constraints")
        ):
            raise StructuredOutputError(f"RTL contract parameter {index} is incomplete")
        name = str(item["name"])
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name) is None:
            raise StructuredOutputError(f"RTL contract parameter {index} has an invalid name")
        if name in seen:
            raise StructuredOutputError(f"RTL contract parameter is duplicated: {name}")
        seen.add(name)


def _validate_ports(value: object) -> set[str]:
    if not isinstance(value, list) or not value:
        raise StructuredOutputError("RTL contract ports must be a non-empty list")
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise StructuredOutputError(f"RTL contract port {index} must be an object")
        item = dict(raw)
        _require_exact_keys(
            item,
            {"name", "direction", "width", "signed", "description"},
            label=f"RTL contract port {index}",
        )
        name = item.get("name")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name) is None:
            raise StructuredOutputError(f"RTL contract port {index} has an invalid name")
        if name in seen:
            raise StructuredOutputError(f"RTL contract port is duplicated: {name}")
        seen.add(name)
        if item.get("direction") not in {"input", "output", "inout"}:
            raise StructuredOutputError(f"RTL contract port {name} has an invalid direction")
        if not isinstance(item.get("width"), str) or not str(item["width"]).strip():
            raise StructuredOutputError(f"RTL contract port {name} has no explicit width")
        if not isinstance(item.get("signed"), bool):
            raise StructuredOutputError(f"RTL contract port {name} signed must be boolean")
        if not isinstance(item.get("description"), str) or not str(item["description"]).strip():
            raise StructuredOutputError(f"RTL contract port {name} has no description")
    return seen


def _validate_clock_reset(value: object, *, port_names: set[str]) -> None:
    if not isinstance(value, dict):
        raise StructuredOutputError("RTL contract clock_reset must be an object")
    item = dict(value)
    _require_exact_keys(item, {"clock", "reset"}, label="RTL contract clock_reset")
    clock = item.get("clock")
    reset = item.get("reset")
    if not isinstance(clock, dict) or not isinstance(reset, dict):
        raise StructuredOutputError("RTL contract clock and reset must be objects")
    clock_item = dict(clock)
    reset_item = dict(reset)
    _require_exact_keys(clock_item, {"name", "edge"}, label="RTL contract clock")
    _require_exact_keys(
        reset_item,
        {"name", "active_level", "synchronous", "reset_values"},
        label="RTL contract reset",
    )
    if clock_item.get("edge") not in {"posedge", "negedge", "none"}:
        raise StructuredOutputError("RTL contract clock edge is invalid")
    if clock_item.get("name") is not None and not isinstance(clock_item.get("name"), str):
        raise StructuredOutputError("RTL contract clock name must be text or null")
    clock_name = clock_item.get("name")
    if (clock_item.get("edge") == "none") != (clock_name is None):
        raise StructuredOutputError("RTL contract clock name and edge none must agree")
    if isinstance(clock_name, str) and (not clock_name or clock_name not in port_names):
        raise StructuredOutputError("RTL contract clock name must identify a declared port")
    if reset_item.get("active_level") not in {"high", "low", "none"}:
        raise StructuredOutputError("RTL contract reset active_level is invalid")
    if reset_item.get("name") is not None and not isinstance(reset_item.get("name"), str):
        raise StructuredOutputError("RTL contract reset name must be text or null")
    reset_name = reset_item.get("name")
    if (reset_item.get("active_level") == "none") != (reset_name is None):
        raise StructuredOutputError("RTL contract reset name and active_level none must agree")
    if isinstance(reset_name, str) and (not reset_name or reset_name not in port_names):
        raise StructuredOutputError("RTL contract reset name must identify a declared port")
    if not isinstance(reset_item.get("synchronous"), bool):
        raise StructuredOutputError("RTL contract reset synchronous must be boolean")
    if not isinstance(reset_item.get("reset_values"), dict):
        raise StructuredOutputError("RTL contract reset_values must be an object")


def _validate_diagnostics(value: object) -> None:
    if not isinstance(value, list):
        raise StructuredOutputError("RTL contract diagnostics must be a list")
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise StructuredOutputError(f"RTL contract diagnostic {index} must be an object")
        item = dict(raw)
        _require_exact_keys(
            item,
            {"attempt", "tool", "observed", "expected", "source_locations"},
            label=f"RTL contract diagnostic {index}",
        )
        if not isinstance(item.get("attempt"), int) or int(item["attempt"]) < 1:
            raise StructuredOutputError(f"RTL contract diagnostic {index} attempt is invalid")
        for key in ("tool", "observed", "expected"):
            if not isinstance(item.get(key), str) or not str(item[key]).strip():
                raise StructuredOutputError(
                    f"RTL contract diagnostic {index} {key} must be non-empty text"
                )
        if not isinstance(item.get("source_locations"), list) or not all(
            isinstance(location, str) for location in item["source_locations"]
        ):
            raise StructuredOutputError(
                f"RTL contract diagnostic {index} source_locations must be strings"
            )


def parse_rtl_worker_contract(
    model_text: str,
    *,
    root: Path,
    task_id: str,
    language: Domain,
    editable_path: str,
    require_diagnostics: bool,
) -> RtlWorkerContract:
    """Reject incomplete, stale, ambiguous, or out-of-scope main-model handoffs."""
    if language not in {"verilog", "systemverilog"}:
        raise StructuredOutputError("RTL worker contracts require Verilog or SystemVerilog")
    value = _single_json_object(model_text, label="RTL contract")
    expected = {
        "schema_version",
        "task_id",
        "module_name",
        "language",
        "editable_path",
        "current_source",
        "parameters",
        "ports",
        "clock_reset",
        "functional_requirements",
        "cycle_requirements",
        "handshake_and_events",
        "corner_cases",
        "synthesis_constraints",
        "forbidden_constructs",
        "verification",
        "diagnostics",
    }
    _require_exact_keys(value, expected, label="RTL contract")
    if value.get("schema_version") != 1:
        raise StructuredOutputError("RTL contract schema_version must equal 1")
    if value.get("task_id") != task_id:
        raise StructuredOutputError("RTL contract task_id does not match the active task")
    module_name = value.get("module_name")
    if (
        not isinstance(module_name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", module_name) is None
    ):
        raise StructuredOutputError("RTL contract module_name is invalid")
    if value.get("language") != language:
        raise StructuredOutputError("RTL contract language does not match the active task")
    normalized_path = _safe_relative(editable_path, label="contract editable path").as_posix()
    if value.get("editable_path") != normalized_path:
        raise StructuredOutputError("RTL contract editable_path does not match deterministic scope")
    source = _inside(root, root / Path(normalized_path))
    if not source.is_file():
        raise StructuredOutputError("RTL contract editable source does not exist")
    source_value = value.get("current_source")
    if not isinstance(source_value, dict):
        raise StructuredOutputError("RTL contract current_source must be an object")
    source_item = dict(source_value)
    _require_exact_keys(source_item, {"sha256", "content"}, label="RTL contract current_source")
    if source_item.get("sha256") != file_sha256(source):
        raise StructuredOutputError("RTL contract current_source hash is stale")
    content = source.read_text(encoding="utf-8", errors="strict")
    if source_item.get("content") != content:
        raise StructuredOutputError("RTL contract current_source content is stale")
    _validate_parameters(value.get("parameters"))
    port_names = _validate_ports(value.get("ports"))
    _validate_clock_reset(value.get("clock_reset"), port_names=port_names)
    for key in (
        "functional_requirements",
        "cycle_requirements",
        "handshake_and_events",
        "corner_cases",
        "synthesis_constraints",
        "forbidden_constructs",
    ):
        _non_empty_strings(value.get(key), label=key)
    verification = value.get("verification")
    if not isinstance(verification, dict):
        raise StructuredOutputError("RTL contract verification must be an object")
    verification_item = dict(verification)
    _require_exact_keys(
        verification_item,
        {"commands", "acceptance_criteria"},
        label="RTL contract verification",
    )
    _non_empty_strings(verification_item.get("commands"), label="verification commands")
    commands = " ".join(str(item).lower() for item in verification_item["commands"])
    required_tools = {"iverilog", "vvp", "yosys"}
    if language == "systemverilog":
        required_tools.add("verilator")
    missing_tools = sorted(tool for tool in required_tools if tool not in commands)
    if missing_tools:
        raise StructuredOutputError(
            "RTL contract verification commands omit deterministic tools: "
            + ", ".join(missing_tools)
        )
    _non_empty_strings(
        verification_item.get("acceptance_criteria"), label="verification acceptance criteria"
    )
    _validate_diagnostics(value.get("diagnostics"))
    if require_diagnostics and not value.get("diagnostics"):
        raise StructuredOutputError("RTL repair contract must contain diagnostics")
    return RtlWorkerContract(dict(value))


def rtl_contract_prompt(
    *,
    task_specification: JsonObject,
    current_source: JsonObject,
    editable_path: str,
    language: RtlLanguage,
    defect_report: JsonObject | None,
) -> str:
    """Build the main-model prompt for a worker-ready, tool-bounded handoff."""
    schema_path = "codex_a6000/templates/rtl_worker_contract.schema.json"
    return (
        "You are the main Laplace RTL architect. Return exactly one JSON object and no prose. "
        f"It must satisfy {schema_path}. Resolve every module-local implementation decision; if the "
        "task still needs repository exploration, multi-file architecture, CDC design, UVM, software/RTL "
        "co-design, AXI subsystem integration, or testbench design, return an invalid empty object so the "
        "orchestrator safely falls back to the main implementation model. Verification commands are evidence "
        "for the worker but the worker must never execute tools. Preserve the exact current source content and "
        "SHA-256. Diagnostics must be empty for the first implementation and populated for a repair.\n"
        f"Language: {language}\nEditable path: {editable_path}\n"
        f"Task specification: {json.dumps(task_specification, sort_keys=True)}\n"
        f"Current source: {json.dumps(current_source, sort_keys=True)}\n"
        f"Structured defect report: {json.dumps(defect_report or {}, sort_keys=True)}"
    )


def rtl_worker_prompt(contract: RtlWorkerContract) -> str:
    """Give the specialist only the complete contract and replacement protocol."""
    replacement_shape = {
        "schema_version": 1,
        "replacements": [
            {
                "path": contract.editable_path,
                "language": contract.language,
                "kind": "source",
                "expected_sha256": "copy exact current_source.sha256",
                "content": "complete replacement source file",
            }
        ],
    }
    return (
        "You are a bounded RTL implementation worker. Implement or repair exactly the one module in the "
        "contract. Do not explore a repository, design a testbench, execute tools, change architecture, add "
        "files, or review the result. Return exactly one JSON object and no Markdown or prose using this "
        f"shape: {json.dumps(replacement_shape, separators=(',', ':'))}. The replacement must be synthesizable "
        "and must copy the supplied current source SHA-256 exactly.\n"
        f"RTL contract: {json.dumps(contract.to_json(), sort_keys=True)}"
    )
