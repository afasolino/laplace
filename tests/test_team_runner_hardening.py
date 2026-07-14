from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_workspace.engineering import (
    AgentTaskStore,
    ReferenceLibrary,
    expand_engineering_query,
    normalize_task_spec,
    retrieve_engineering_evidence,
)
from research_workspace.repair_protocol import (
    StructuredOutputError,
    build_local_patch,
    parse_replacement_plan,
    parse_reviewer_verdict,
    source_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _plan(path: Path, *, content: str, **overrides: object) -> str:
    replacement: dict[str, object] = {
        "path": path.name,
        "language": "python",
        "kind": "source",
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content": content,
    }
    replacement.update(overrides)
    return json.dumps({"schema_version": 1, "replacements": [replacement]})


def test_structured_replacement_records_current_hash_and_builds_local_diff(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    state = source_state(tmp_path, ["module.py"], "python")
    assert state[0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    plan = parse_replacement_plan(
        _plan(source, content="value = 2\n"),
        root=tmp_path,
        allowed_paths=["module.py"],
        domain="python",
    )
    patch = build_local_patch(plan, root=tmp_path)
    assert "diff --git a/module.py b/module.py" in patch
    assert "+value = 2" in patch


@pytest.mark.parametrize(
    ("model_text", "message"),
    [
        (
            "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n",
            "Raw model-generated patches",
        ),
        ("```json\n{}\n```\nextra", "mixes JSON with prose"),
        ("```json\n{}\n```\n```json\n{}\n```", "multiple fenced blocks"),
    ],
)
def test_structured_replacement_rejects_ambiguous_or_raw_patch_output(
    tmp_path: Path, model_text: str, message: str
) -> None:
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    with pytest.raises(StructuredOutputError, match=message):
        parse_replacement_plan(
            model_text, root=tmp_path, allowed_paths=["module.py"], domain="python"
        )


def test_structured_replacement_rejects_stale_duplicate_unknown_kind_and_noop(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    current_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cases = [
        (
            {
                "schema_version": 1,
                "replacements": [
                    {
                        "path": "module.py",
                        "language": "python",
                        "kind": "source",
                        "expected_sha256": "0" * 64,
                        "content": "value = 2\n",
                    }
                ],
            },
            "stale",
        ),
        (
            {
                "schema_version": 1,
                "replacements": [
                    {
                        "path": "module.py",
                        "language": "python",
                        "kind": "source",
                        "expected_sha256": current_hash,
                        "content": "value = 2\n",
                    },
                    {
                        "path": "module.py",
                        "language": "python",
                        "kind": "source",
                        "expected_sha256": current_hash,
                        "content": "value = 3\n",
                    },
                ],
            },
            "duplicated",
        ),
        (
            {
                "schema_version": 1,
                "replacements": [
                    {
                        "path": "other.py",
                        "language": "python",
                        "kind": "source",
                        "expected_sha256": current_hash,
                        "content": "value = 2\n",
                    }
                ],
            },
            "outside task scope",
        ),
        (
            {
                "schema_version": 1,
                "replacements": [
                    {
                        "path": "module.py",
                        "language": "python",
                        "kind": "testbench",
                        "expected_sha256": current_hash,
                        "content": "value = 2\n",
                    }
                ],
            },
            "kind must be",
        ),
        (
            {
                "schema_version": 1,
                "replacements": [
                    {
                        "path": "module.py",
                        "language": "python",
                        "kind": "source",
                        "expected_sha256": current_hash,
                        "content": "value = 1\n",
                    }
                ],
            },
            "no source change",
        ),
    ]
    for payload, message in cases:
        with pytest.raises(StructuredOutputError, match=message):
            parse_replacement_plan(
                json.dumps(payload),
                root=tmp_path,
                allowed_paths=["module.py"],
                domain="python",
            )


def test_reviewer_verdict_is_machine_readable_and_invalid_output_never_approves() -> None:
    verdict = parse_reviewer_verdict(
        json.dumps(
            {
                "schema_version": 1,
                "verdict": "approve",
                "reason": "All deterministic evidence passed.",
                "missing_evidence": [],
            }
        )
    )
    assert verdict.verdict == "approve"
    with pytest.raises(StructuredOutputError):
        parse_reviewer_verdict("The code looks fine, approve it.")
    with pytest.raises(StructuredOutputError):
        parse_reviewer_verdict(
            json.dumps(
                {
                    "schema_version": 1,
                    "verdict": "approve",
                    "reason": "Fine",
                    "missing_evidence": [],
                    "unexpected": True,
                }
            )
        )


def _systemverilog_task(task_id: str = "rv-query") -> dict[str, object]:
    return {
        "task_id": task_id,
        "objective": "Repair a ready/valid slot for simultaneous dequeue and enqueue.",
        "target": {
            "class": "portable_rtl",
            "language": "SystemVerilog",
            "toolchain": ["verilator", "iverilog", "yosys"],
        },
        "files_allowed_to_change": [
            "benchmarks/a6000_agent_team/paired_public/sv_ready_valid_buffer/rv_buffer.sv"
        ],
        "interfaces": [
            {
                "name": "stream",
                "protocol": "ready_valid",
                "direction": "bidirectional",
                "signals": ["valid", "ready", "data"],
                "ordering": "in order",
                "backpressure": "supported",
            }
        ],
        "clock_reset": {
            "clock_domains": ["clk"],
            "reset_semantics": "active-low synchronous reset",
            "cdc_rdc_assumptions": "single clock domain",
        },
        "functional_requirements": [
            "Accept replacement on simultaneous dequeue and enqueue.",
            "Keep payload stable while stalled.",
        ],
        "error_and_corner_behavior": ["Reset empty."],
        "verification": {
            "self_checking": True,
            "tests": ["public ready/valid simulation"],
            "assertions": ["stable payload while stalled"],
            "commands": ["iverilog", "vvp", "verilator", "yosys"],
            "acceptance_criteria": ["replace while full", "stall stability", "reset empty"],
        },
        "deliverables": ["RTL", "testbench"],
        "out_of_scope": ["vendor IP"],
        "blocking_questions": [],
    }


def test_ready_valid_query_expansion_and_project_knowledge_card(tmp_path: Path) -> None:
    task = AgentTaskStore(tmp_path).create(
        "systemverilog",
        normalize_task_spec(REPOSITORY_ROOT, "systemverilog", _systemverilog_task()),
    )
    expanded, terms = expand_engineering_query(task, task.specification["objective"])
    assert "simultaneous enqueue dequeue" in expanded
    assert any("consume and replace" in item for item in terms)
    evidence = retrieve_engineering_evidence(
        REPOSITORY_ROOT,
        tmp_path,
        task,
        query=str(task.specification["objective"]),
    )
    cards = evidence["project_knowledge_cards"]
    assert isinstance(cards, list) and cards
    assert cards[0]["path"].endswith("systemverilog_handshake_axi_w1c.md")


def test_governed_search_enforces_path_diversity(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    licence = source_root / "LICENSE"
    licence.write_text("Fixture licence\n", encoding="utf-8")
    files = []
    for index, phrase in enumerate(
        (
            "ready valid simultaneous enqueue dequeue skid buffer",
            "AXI4-Lite independent AW W WSTRB write response",
            "write one to clear pending enable interrupt IRQ",
        )
    ):
        path = source_root / f"guide_{index}.md"
        path.write_text((phrase + "\n") * 30, encoding="utf-8")
        files.append((path, f"10_rtl_patterns/diverse_{index}", tuple(phrase.split())))
    library = ReferenceLibrary(tmp_path / "Library", "systemverilog", shared=True)
    library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / "systemverilog_sources.yaml"
    )
    library.register_local(
        reference_id="diverse_fixture",
        repository="https://example.invalid/diverse.git",
        commit="1" * 40,
        licence_identifier="MIT",
        licence_path=licence,
        selected_files=files,
        permitted_use="reference_only_no_copy",
        attribution="Fixture",
    )
    chunks = library.search_chunks(
        "ready valid AXI WSTRB W1C pending interrupt",
        limit=3,
        token_budget=2000,
        max_chunks_per_path=1,
    )
    paths = [str(item["path"]) for item in chunks]
    assert len(chunks) == 3
    assert len(set(paths)) == 3
    assert all(item["matched_query_terms"] for item in chunks)
