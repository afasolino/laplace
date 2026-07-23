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

    @property
    def module_name(self) -> str:
        return str(self.value["module_name"])

    @property
    def current_source_sha256(self) -> str:
        current = self.value["current_source"]
        if not isinstance(current, dict):
            raise StructuredOutputError("RTL worker contract has no current source hash")
        value = current.get("sha256")
        if not isinstance(value, str):
            raise StructuredOutputError("RTL worker contract has no current source hash")
        return value

    @property
    def current_source_content(self) -> str:
        current = self.value["current_source"]
        if not isinstance(current, dict):
            raise StructuredOutputError("RTL worker contract has no current source content")
        value = current.get("content")
        if not isinstance(value, str):
            raise StructuredOutputError("RTL worker contract has no current source content")
        return value

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


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_$]*"


def _strip_hdl_comments(source: str) -> str:
    """Remove comments while preserving string literals and source spacing."""
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            output.extend("  ")
            index += 2
            while index < len(source) and source[index] != "\n":
                output.append(" ")
                index += 1
            continue
        if char == "/" and next_char == "*":
            output.extend("  ")
            index += 2
            while index < len(source):
                if source[index] == "*" and index + 1 < len(source) and source[index + 1] == "/":
                    output.extend("  ")
                    index += 2
                    break
                output.append("\n" if source[index] == "\n" else " ")
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _balanced_text(source: str, start: int, *, opening: str = "(", closing: str = ")") -> tuple[str, int]:
    if start >= len(source) or source[start] != opening:
        raise StructuredOutputError(f"Expected {opening!r} in RTL module declaration")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return source[start + 1 : index], index + 1
    raise StructuredOutputError(f"Unterminated {opening}{closing} region in RTL declaration")


def _split_top_level(value: str, delimiter: str = ",") -> list[str]:
    items: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    in_string = False
    escaped = False
    for index, char in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in depths:
            depths[char] += 1
        elif char in closing:
            depths[closing[char]] -= 1
        elif char == delimiter and all(depth == 0 for depth in depths.values()):
            item = value[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    item = value[start:].strip()
    if item:
        items.append(item)
    return items


def _split_assignment(value: str) -> tuple[str, str]:
    parts = _split_top_level(value, "=")
    if len(parts) == 1:
        return parts[0].strip(), "unspecified"
    if len(parts) != 2:
        raise StructuredOutputError("Ambiguous parameter assignment in RTL declaration")
    return parts[0].strip(), parts[1].strip()


def _module_declaration(source: str) -> tuple[str, str, str]:
    clean = _strip_hdl_comments(source)
    match = re.search(rf"\bmodule\s+({_IDENTIFIER})\b", clean)
    if match is None:
        raise StructuredOutputError("Cannot find one ANSI RTL module declaration")
    module_name = match.group(1)
    cursor = match.end()
    while cursor < len(clean) and clean[cursor].isspace():
        cursor += 1
    parameters = ""
    if cursor < len(clean) and clean[cursor] == "#":
        cursor += 1
        while cursor < len(clean) and clean[cursor].isspace():
            cursor += 1
        parameters, cursor = _balanced_text(clean, cursor)
        while cursor < len(clean) and clean[cursor].isspace():
            cursor += 1
    ports, _ = _balanced_text(clean, cursor)
    if re.search(rf"\bmodule\s+{_IDENTIFIER}\b", clean[_:]) is not None:
        raise StructuredOutputError("RTL worker source must contain exactly one module")
    return module_name, parameters, ports


def _declared_parameters(parameter_text: str) -> list[JsonObject]:
    if not parameter_text.strip():
        return []
    result: list[JsonObject] = []
    inherited_type = "implicit"
    for raw in _split_top_level(parameter_text):
        value = re.sub(r"^\s*(?:parameter|localparam)\b", "", raw, count=1).strip()
        left, default = _split_assignment(value)
        match = re.search(rf"({_IDENTIFIER})\s*$", left)
        if match is None:
            raise StructuredOutputError(f"Cannot parse RTL parameter declaration: {raw}")
        name = match.group(1)
        declared_type = left[: match.start()].strip()
        if declared_type:
            inherited_type = declared_type
        result.append(
            {
                "name": name,
                "type": inherited_type,
                "default": default,
                "constraints": "Preserve the public parameter name, type, and default semantics.",
            }
        )
    return result


def _declared_ports(port_text: str) -> list[JsonObject]:
    result: list[JsonObject] = []
    inherited_direction: str | None = None
    inherited_prefix = ""
    for raw in _split_top_level(port_text):
        value = raw.strip().rstrip(";")
        direction_match = re.match(r"^(input|output|inout)\b", value)
        if direction_match is not None:
            inherited_direction = direction_match.group(1)
            value = value[direction_match.end() :].strip()
        elif inherited_direction is None:
            raise StructuredOutputError("Non-ANSI or directionless first RTL port is unsupported")
        name_match = re.search(rf"({_IDENTIFIER})\s*(?:\[[^\]]+\]\s*)*$", value)
        if name_match is None:
            raise StructuredOutputError(f"Cannot parse RTL port declaration: {raw}")
        name = name_match.group(1)
        prefix = value[: name_match.start()].strip()
        if prefix:
            inherited_prefix = prefix
        else:
            prefix = inherited_prefix
        widths = re.findall(r"\[[^\]]+\]", prefix)
        result.append(
            {
                "name": name,
                "direction": inherited_direction,
                "width": " ".join(widths) if widths else "1",
                "signed": re.search(r"\bsigned\b", prefix) is not None,
                "description": (
                    f"Public {inherited_direction} port {name}; preserve the declared interface semantics."
                ),
            }
        )
    if not result:
        raise StructuredOutputError("RTL module declaration has no ports")
    return result


def _strings(value: object, *, fallback: str) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
        if items:
            return items
    return [fallback]


def _interface_requirements(task_specification: JsonObject) -> tuple[list[str], list[str]]:
    cycle: list[str] = []
    handshake: list[str] = []
    interfaces = task_specification.get("interfaces")
    if isinstance(interfaces, list):
        for raw in interfaces:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "interface")
            protocol = str(raw.get("protocol") or "unspecified")
            ordering = str(raw.get("ordering") or "").strip()
            backpressure = str(raw.get("backpressure") or "").strip()
            if ordering:
                cycle.append(f"{name}: {ordering}")
            if backpressure:
                cycle.append(f"{name}: {backpressure}")
            handshake.append(f"{name} uses the {protocol} protocol.")
            signals = raw.get("signals")
            if isinstance(signals, list):
                signal_text = [str(item).strip() for item in signals if isinstance(item, str) and item.strip()]
                if signal_text:
                    handshake.append(f"{name} signals: {'; '.join(signal_text)}")
    if not cycle:
        cycle.append("Preserve the cycle ordering encoded by the public module interface and tests.")
    if not handshake:
        handshake.append("Preserve all public event and handshake semantics under stalls.")
    return cycle, handshake


def _clock_reset_contract(
    *, source: str, task_specification: JsonObject, ports: list[JsonObject]
) -> JsonObject:
    names = [str(port["name"]) for port in ports]
    clock_domains: list[str] = []
    raw_clock_reset = task_specification.get("clock_reset")
    if isinstance(raw_clock_reset, dict):
        raw_domains = raw_clock_reset.get("clock_domains")
        if isinstance(raw_domains, list):
            clock_domains = [str(item) for item in raw_domains if isinstance(item, str)]
    clock_name = next((name for name in clock_domains if name in names), None)
    if clock_name is None:
        clock_name = next((name for name in names if re.search(r"(?:^|_)(?:clk|clock)(?:$|_)", name, re.I)), None)
    reset_name = next(
        (
            name
            for name in names
            if re.search(r"(?:^|_)(?:rst|reset)(?:_n|n)?(?:$|_)", name, re.I)
        ),
        None,
    )
    reset_low = bool(reset_name and re.search(r"(?:_n|n)$", reset_name, re.I))
    asynchronous = False
    if clock_name and reset_name:
        event_controls = re.findall(r"@\s*\(([^)]*)\)", _strip_hdl_comments(source))
        asynchronous = any(clock_name in event and reset_name in event for event in event_controls)
    return {
        "clock": {
            "name": clock_name,
            "edge": "posedge" if clock_name is not None else "none",
        },
        "reset": {
            "name": reset_name,
            "active_level": "low" if reset_low else "high" if reset_name else "none",
            "synchronous": not asynchronous if reset_name is not None else True,
            "reset_values": {},
        },
    }


def _diagnostics(defect_report: JsonObject | None) -> list[JsonObject]:
    if defect_report is None:
        return []
    attempt = defect_report.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        attempt = 1
    tool = defect_report.get("tool")
    tool_text = str(tool).strip() if isinstance(tool, str) else "deterministic verification"
    observed = json.dumps(defect_report, sort_keys=True, separators=(",", ":"))
    if len(observed) > 4000:
        observed = observed[:3997] + "..."
    locations: list[str] = []
    for key in ("source_locations", "paths", "files"):
        raw = defect_report.get(key)
        if isinstance(raw, list):
            locations.extend(str(item) for item in raw if isinstance(item, str))
    return [
        {
            "attempt": attempt,
            "tool": tool_text or "deterministic verification",
            "observed": observed or "Previous deterministic verification failed.",
            "expected": "All deterministic verification commands and acceptance criteria pass.",
            "source_locations": locations,
        }
    ]


def build_rtl_worker_contract(
    *,
    root: Path,
    task_id: str,
    task_specification: JsonObject,
    current_source: JsonObject,
    editable_path: str,
    language: RtlLanguage,
    defect_report: JsonObject | None,
) -> RtlWorkerContract:
    """Construct the specialist handoff from trusted task metadata and source state."""
    normalized_path = _safe_relative(editable_path, label="contract editable path").as_posix()
    source_path = _inside(root, root / Path(normalized_path))
    if not source_path.is_file():
        raise StructuredOutputError("RTL worker editable source does not exist")
    content = source_path.read_text(encoding="utf-8", errors="strict")
    expected_hash = file_sha256(source_path)
    if current_source.get("path") != normalized_path:
        raise StructuredOutputError("Current source record path does not match worker scope")
    if current_source.get("sha256") != expected_hash or current_source.get("content") != content:
        raise StructuredOutputError("Current source record is stale")
    module_name, parameter_text, port_text = _module_declaration(content)
    ports = _declared_ports(port_text)
    cycle_requirements, handshake_requirements = _interface_requirements(task_specification)
    verification = task_specification.get("verification")
    verification_item = dict(verification) if isinstance(verification, dict) else {}
    contract: JsonObject = {
        "schema_version": 1,
        "task_id": task_id,
        "module_name": module_name,
        "language": language,
        "editable_path": normalized_path,
        "current_source": {"sha256": expected_hash, "content": content},
        "parameters": _declared_parameters(parameter_text),
        "ports": ports,
        "clock_reset": _clock_reset_contract(
            source=content, task_specification=task_specification, ports=ports
        ),
        "functional_requirements": _strings(
            task_specification.get("functional_requirements"),
            fallback="Implement the exact public task specification.",
        ),
        "cycle_requirements": cycle_requirements,
        "handshake_and_events": handshake_requirements,
        "corner_cases": _strings(
            task_specification.get("error_and_corner_behavior"),
            fallback="Handle reset, boundary, simultaneous-event, and stall cases deterministically.",
        ),
        "synthesis_constraints": _strings(
            task_specification.get("coding_constraints"),
            fallback="Use a portable synthesizable RTL subset.",
        ),
        "forbidden_constructs": [
            "Do not add files, modules, testbenches, delays, force/release, DPI, UVM, or tool commands.",
            "Do not change the public module name, parameters, ports, clocking, or reset contract.",
        ],
        "verification": {
            "commands": _strings(
                verification_item.get("commands"),
                fallback="iverilog compile; vvp simulation; verilator lint; yosys synthesis",
            ),
            "acceptance_criteria": _strings(
                verification_item.get("acceptance_criteria"),
                fallback="All deterministic simulation, lint, and synthesis checks pass.",
            ),
        },
        "diagnostics": _diagnostics(defect_report),
    }
    return parse_rtl_worker_contract(
        json.dumps(contract, sort_keys=True),
        root=root,
        task_id=task_id,
        language=language,
        editable_path=normalized_path,
        require_diagnostics=defect_report is not None,
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


def rtl_worker_prompt(
    contract: RtlWorkerContract,
    *,
    retry_index: int = 0,
    prior_error: str | None = None,
) -> str:
    """Use CodeV's native reasoning/final-answer protocol for one bounded RTL file."""
    retry_instruction = ""
    if retry_index > 0:
        retry_instruction = (
            "This is a deterministic correction attempt. The previous answer was rejected for: "
            f"{prior_error or 'invalid output'}. Correct that issue, keep the source concise, and do not "
            "repeat declarations or source blocks.\n"
        )
    return (
        "You are a bounded RTL implementation worker. Implement exactly the one module specified by the "
        "contract. Think through the implementation inside exactly one <think>...</think> block. Then emit "
        "exactly one <answer>...</answer> block containing exactly one "
        f"fenced {contract.language} code block with the complete replacement source for module "
        f"{contract.module_name}. Do not emit JSON, a diff, a path, shell commands, testbench code, or prose "
        "outside those tags. Preserve the public module name, parameters, ports, clock/reset behavior, and "
        "all contract constraints. Do not explore a repository, execute tools, add modules, add files, or "
        "change architecture. The orchestrator will validate and wrap the source deterministically.\n"
        f"{retry_instruction}"
        f"RTL contract: {json.dumps(contract.to_json(), sort_keys=True)}"
    )


def parse_codev_rtl_answer(model_text: str, *, contract: RtlWorkerContract) -> str:
    """Extract one complete RTL module from CodeV's native tagged response."""
    if not isinstance(model_text, str) or not model_text.strip():
        raise StructuredOutputError("RTL worker response is empty")
    text = model_text.strip()
    answer_match = re.fullmatch(
        r"(?:<think>.*?</think>\s*)?<answer>(?P<answer>.*?)</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    payload = answer_match.group("answer").strip() if answer_match else text
    if "<think>" in payload.lower() or "<answer>" in payload.lower():
        raise StructuredOutputError("RTL worker response has malformed reasoning/answer tags")

    fence_matches = list(
        re.finditer(
            r"```(?P<label>systemverilog|verilog|sv)?[ \t]*\n(?P<source>.*?)```",
            payload,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if fence_matches:
        if len(fence_matches) != 1:
            raise StructuredOutputError("RTL worker response must contain one code block")
        outside = (
            payload[: fence_matches[0].start()] + payload[fence_matches[0].end() :]
        ).strip()
        if outside:
            raise StructuredOutputError("RTL worker answer contains prose outside the code block")
        source = fence_matches[0].group("source").strip()
    else:
        source = payload.strip()

    if not source:
        raise StructuredOutputError("RTL worker answer contains no source")
    if "```" in source or "\x00" in source:
        raise StructuredOutputError("RTL worker source contains invalid framing")

    current_length = len(contract.current_source_content)
    maximum_length = min(200_000, max(16_384, current_length * 4, current_length + 8_192))
    if len(source) > maximum_length:
        raise StructuredOutputError(
            f"RTL worker source exceeds the deterministic {maximum_length}-character limit"
        )

    stripped = _strip_hdl_comments(source)
    module_names = re.findall(rf"\bmodule\s+({_IDENTIFIER})\b", stripped)
    if module_names != [contract.module_name]:
        raise StructuredOutputError(
            "RTL worker answer must define exactly the contracted module and no other module"
        )
    if len(re.findall(r"\bendmodule\b", stripped)) != 1:
        raise StructuredOutputError("RTL worker answer must contain exactly one endmodule")
    return source.rstrip() + "\n"


def codev_replacement_plan(model_text: str, *, contract: RtlWorkerContract) -> str:
    """Wrap validated native CodeV source in the repository replacement protocol."""
    source = parse_codev_rtl_answer(model_text, contract=contract)
    return json.dumps(
        {
            "schema_version": 1,
            "replacements": [
                {
                    "path": contract.editable_path,
                    "language": contract.language,
                    "kind": "source",
                    "expected_sha256": contract.current_source_sha256,
                    "content": source,
                }
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
