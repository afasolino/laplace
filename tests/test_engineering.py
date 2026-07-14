from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from research_workspace.agent_mcp import McpService, run_stdio
from research_workspace.engineering import (
    AgentTaskStore,
    EngineeringError,
    LocalToolRunner,
    ReferenceEvidenceError,
    ReferenceLibrary,
    SchemaValidationError,
    normalize_task_spec,
    retrieve_engineering_evidence,
)
from research_workspace.inference import ServingCandidate
from research_workspace.team_runner import (
    LocalTeamRunner,
    PatchValidationError,
    Worktree,
    _extract_model_patch,
    apply_validated_patch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _python_task(task_id: str = "typed-task") -> dict[str, object]:
    return {
        "task_id": task_id,
        "objective": "Add a narrow, typed local operation.",
        "repository_root": ".",
        "allowed_paths": ["src/research_workspace/engineering.py"],
        "public_interfaces": [{"name": "operation", "contract": "returns a typed report"}],
        "functional_requirements": ["Validate every input before it reaches a tool."],
        "error_behavior": ["Raise an explicit safe error for invalid input."],
        "quality_requirements": {
            "python": ">=3.11",
            "typing": "strict mypy",
            "formatting": "ruff format",
            "lint": "ruff check",
            "tests": "pytest",
        },
        "verification_commands": ["python -m pytest"],
        "deliverables": ["implementation", "tests"],
        "out_of_scope": ["network access"],
    }


def test_task_normalization_and_bounded_persisted_state_machine(tmp_path: Path) -> None:
    normalized = normalize_task_spec(REPOSITORY_ROOT, "python", _python_task())
    store = AgentTaskStore(tmp_path)
    task = store.create("python", normalized)
    assert (
        store.transition(task.task_id, "requirements", role="supervisor", note="valid").state
        == "requirements"
    )
    with pytest.raises(EngineeringError, match="Only the supervisor"):
        store.transition(task.task_id, "plan", role="researcher", note="invalid")
    store.transition(task.task_id, "plan", role="supervisor", note="narrow plan")
    store.transition(task.task_id, "retrieval", role="supervisor", note="retrieve")
    store.transition(task.task_id, "implementation", role="supervisor", note="worktree")
    store.transition(task.task_id, "verification", role="supervisor", note="verify")
    store.transition(task.task_id, "review", role="supervisor", note="review")
    store.transition(task.task_id, "bounded_correction", role="supervisor", note="first repair")
    assert store.load(task.task_id).correction_loops == 1


def test_patch_rejection_can_use_the_bounded_repair_budget(tmp_path: Path) -> None:
    normalized = normalize_task_spec(REPOSITORY_ROOT, "python", _python_task())
    store = AgentTaskStore(tmp_path)
    task = store.create("python", normalized)
    for target in ("requirements", "plan", "retrieval", "implementation"):
        task = store.transition(task.task_id, target, role="supervisor", note=target)
    repaired = store.transition(
        task.task_id, "bounded_correction", role="supervisor", note="patch validation rejected"
    )
    assert repaired.correction_loops == 1
    assert repaired.state == "bounded_correction"


def test_task_schema_rejects_missing_required_contract() -> None:
    invalid = _python_task()
    invalid.pop("verification_commands")
    with pytest.raises(SchemaValidationError, match="missing required property"):
        normalize_task_spec(REPOSITORY_ROOT, "python", invalid)


def test_governed_reference_fixture_is_read_only_hash_verified_and_indexed(tmp_path: Path) -> None:
    source_root = tmp_path / "fixture"
    source_root.mkdir()
    licence = source_root / "LICENSE"
    licence.write_text("Fixture licence text\n", encoding="utf-8")
    guide = source_root / "guide.md"
    guide.write_text("Use explicit transactions and provenance hashes.\n", encoding="utf-8")
    library = ReferenceLibrary(tmp_path, "python")
    initialized = library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / "python_sources.yaml"
    )
    assert initialized["initialized"] is True
    verified = library.register_local(
        reference_id="fixture_python",
        repository="https://example.invalid/fixture.git",
        commit="a" * 40,
        licence_identifier="MIT",
        licence_path=licence,
        selected_files=[(guide, "50_typing_validation/fixture", ("typing", "transactions"))],
        permitted_use="reference_only_no_copy",
        attribution="Fixture author",
    )
    assert verified["status"] == "VERIFIED"
    selected = library.select(["transactions"])
    assert selected["references"]
    ingested = library.ingest(tmp_path / "Data" / "Metadata" / "workspace.db")
    assert ingested["counts"] == {"indexed": 1, "unchanged": 0, "skipped": 0}
    local_reference = (
        tmp_path
        / "Data"
        / "References"
        / "Python"
        / "50_typing_validation"
        / "fixture"
        / "guide.md"
    )
    local_reference.chmod(0o644)
    local_reference.write_text("tampered\n", encoding="utf-8")
    assert library.verify("fixture_python")["status"] == "FAILED"


def test_research_evidence_prefers_target_project_before_governed_references(
    tmp_path: Path,
) -> None:
    store = AgentTaskStore(tmp_path)
    task = store.create(
        "python", normalize_task_spec(REPOSITORY_ROOT, "python", _python_task("precedence"))
    )
    evidence = retrieve_engineering_evidence(
        REPOSITORY_ROOT, tmp_path, task, query="typed explicit operation"
    )
    assert evidence["precedence"][0] == "target_project"
    assert evidence["target_project"]


def test_shared_governed_reference_is_retrieved_from_fresh_task_project(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "approved-source"
    source_root.mkdir()
    licence = source_root / "LICENSE"
    licence.write_text("Fixture licence text\n", encoding="utf-8")
    guide = source_root / "axi_guide.sv"
    guide.write_text(
        "AXI4-Lite writes must apply WSTRB byte enables and write-one-to-clear semantics.\n",
        encoding="utf-8",
    )
    shared_root = tmp_path / "FormalScience" / "Library"
    library = ReferenceLibrary(shared_root, "systemverilog", shared=True)
    library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / "systemverilog_sources.yaml"
    )
    library.register_local(
        reference_id="fixture_axi",
        repository="https://example.invalid/axi.git",
        commit="b" * 40,
        licence_identifier="MIT",
        licence_path=licence,
        selected_files=[(guide, "20_interfaces/axi", ("axi4_lite", "wstrb", "w1c"))],
        permitted_use="reference_only_no_copy",
        attribution="Fixture author",
    )
    fresh_project = tmp_path / "fresh-task-project"
    task = AgentTaskStore(fresh_project).create(
        "systemverilog",
        normalize_task_spec(
            REPOSITORY_ROOT,
            "systemverilog",
            {
                "task_id": "shared-reference-task",
                "objective": "Implement WSTRB and W1C behavior.",
                "target": {
                    "class": "portable_rtl",
                    "language": "SystemVerilog",
                    "toolchain": ["verilator", "iverilog", "yosys"],
                },
                "files_allowed_to_change": [
                    "benchmarks/a6000_agent_team/paired_public/sv_axi_lite_irq_regs/axi_lite_irq_regs.sv"
                ],
                "interfaces": [
                    {
                        "name": "s_axi",
                        "protocol": "AXI4-Lite",
                        "direction": "slave",
                        "signals": ["AW", "W", "B", "AR", "R"],
                        "ordering": "independent address and data channels",
                        "backpressure": "supported",
                    }
                ],
                "clock_reset": {
                    "clock_domains": ["aclk"],
                    "reset_semantics": "active-low synchronous reset",
                    "cdc_rdc_assumptions": "single clock domain",
                },
                "functional_requirements": ["Respect WSTRB and W1C semantics."],
                "error_and_corner_behavior": ["No protocol deadlock."],
                "verification": {
                    "self_checking": True,
                    "tests": ["byte strobes and W1C clear"],
                    "assertions": ["responses remain stable while stalled"],
                    "commands": ["verilator --lint-only"],
                    "acceptance_criteria": ["public simulation passes"],
                },
                "deliverables": ["rtl"],
                "out_of_scope": ["AXI4 full"],
                "blocking_questions": [],
            },
        ),
    )
    evidence = retrieve_engineering_evidence(
        REPOSITORY_ROOT,
        fresh_project,
        task,
        query="AXI4-Lite WSTRB W1C",
        shared_reference_root=shared_root,
    )
    governed = evidence["governed_references"]
    assert isinstance(governed, list) and governed
    first = governed[0]
    assert isinstance(first, dict)
    assert first["reference_id"] == "fixture_axi"
    assert first["chunk_id"].startswith("reference:fixture_axi:")
    assert "write-one-to-clear" in str(first["content"])
    assert evidence["governed_reference_library"]["snapshot_hash"] == library.snapshot_hash()


def test_curated_only_fails_closed_when_governed_content_is_empty() -> None:
    evidence: dict[str, object] = {
        "target_project": [{"path": "local.py"}],
        "governed_references": [],
    }
    with pytest.raises(ReferenceEvidenceError, match="BLOCKED_REFERENCE_EMPTY"):
        LocalTeamRunner._filter_evidence(evidence, "curated_only")


def test_governed_reference_search_excludes_irrelevant_chunks(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    licence = source_root / "LICENSE"
    licence.write_text("Fixture licence\n", encoding="utf-8")
    async_guide = source_root / "async.md"
    async_guide.write_text(
        "Await cancellation cleanup before raising a timeout from an asynchronous job.\n",
        encoding="utf-8",
    )
    unrelated = source_root / "optics.md"
    unrelated.write_text("Thin lenses form images using focal distance.\n", encoding="utf-8")
    shared_root = tmp_path / "Library"
    library = ReferenceLibrary(shared_root, "python", shared=True)
    library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / "python_sources.yaml"
    )
    library.register_local(
        reference_id="fixture_python_topics",
        repository="https://example.invalid/python.git",
        commit="c" * 40,
        licence_identifier="MIT",
        licence_path=licence,
        selected_files=[
            (async_guide, "20_architecture_patterns/async", ("async", "cancellation", "timeout")),
            (unrelated, "20_architecture_patterns/optics", ("optics", "lens")),
        ],
        permitted_use="reference_only_no_copy",
        attribution="Fixture author",
    )
    chunks = library.search_chunks("async cancellation timeout", limit=1)
    assert len(chunks) == 1
    assert chunks[0]["path"] == "20_architecture_patterns/async/async.md"


def test_shared_reference_registration_is_idempotent_and_snapshot_is_stable(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    licence = source_root / "LICENSE"
    licence.write_text("Fixture licence\n", encoding="utf-8")
    guide = source_root / "typing.md"
    guide.write_text("Use explicit return types and reject invalid inputs.\n", encoding="utf-8")
    shared_root = tmp_path / "Library"
    library = ReferenceLibrary(shared_root, "python", shared=True)
    library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / "python_sources.yaml"
    )
    arguments = {
        "reference_id": "fixture_idempotent",
        "repository": "https://example.invalid/idempotent.git",
        "commit": "d" * 40,
        "licence_identifier": "MIT",
        "licence_path": licence,
        "selected_files": [(guide, "50_typing_validation/typing", ("typing", "validation"))],
        "permitted_use": "reference_only_no_copy",
        "attribution": "Fixture author",
    }
    first = library.register_local(**arguments)
    first_hash = library.snapshot_hash()
    second = library.register_local(**arguments)
    assert first["status"] == "VERIFIED"
    assert second["status"] == "VERIFIED"
    assert first_hash == library.snapshot_hash()


def test_mcp_tools_list_is_a_harmless_stdio_discovery_call(tmp_path: Path) -> None:
    input_stream = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n')
    output_stream = io.StringIO()
    assert run_stdio(McpService(REPOSITORY_ROOT, tmp_path), input_stream, output_stream) == 0
    response = json.loads(output_stream.getvalue())
    assert response["result"]["tools"][0]["name"] == "normalize_software_task"


def test_mcp_research_uses_configured_shared_reference_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    licence = source_root / "LICENSE"
    licence.write_text("Fixture licence\n", encoding="utf-8")
    guide = source_root / "validation.md"
    guide.write_text(
        "A typed explicit operation validates every input before invoking a tool.\n",
        encoding="utf-8",
    )
    shared_root = tmp_path / "FormalScience" / "Library"
    library = ReferenceLibrary(shared_root, "python", shared=True)
    library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / "python_sources.yaml"
    )
    library.register_local(
        reference_id="fixture_mcp_shared",
        repository="https://example.invalid/mcp.git",
        commit="f" * 40,
        licence_identifier="MIT",
        licence_path=licence,
        selected_files=[(guide, "50_typing_validation/input", ("typing", "validation", "tools"))],
        permitted_use="reference_only_no_copy",
        attribution="Fixture author",
    )
    project_root = tmp_path / "project"
    task = AgentTaskStore(project_root).create(
        "python", normalize_task_spec(REPOSITORY_ROOT, "python", _python_task("mcp-shared"))
    )
    monkeypatch.setenv("LAPLACE_SHARED_REFERENCE_ROOT", str(shared_root))
    result = McpService(REPOSITORY_ROOT, project_root).call(
        "research_task",
        {"task_id": task.task_id, "query": "typed explicit operation validation"},
    )
    governed = result["governed_references"]
    assert isinstance(governed, list) and governed
    library_record = result["governed_reference_library"]
    assert isinstance(library_record, dict)
    assert library_record["shared"] is True
    assert library_record["snapshot_hash"] == library.snapshot_hash()


def test_eda_flow_runs_lint_self_checking_simulation_and_synthesis() -> None:
    result = LocalToolRunner(REPOSITORY_ROOT).run_eda_flow(
        ["benchmarks/a6000_agent_team/rtl/rv_skid_buffer.sv"],
        top_module="rv_skid_buffer",
        testbench="benchmarks/a6000_agent_team/rtl/tb_rv_skid_buffer.sv",
    )
    assert result["passed"] is True
    assert {entry["tool"] for entry in result["results"]} >= {
        "verilator",
        "iverilog",
        "vvp",
        "yosys",
    }


def test_validated_patch_cannot_escape_allowed_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "allowed.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "allowed.py"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    worktree = Worktree(tmp_path, commit, "patch-fixture")
    valid = """diff --git a/allowed.py b/allowed.py
index 1e8b314..5b40bd0 100644
--- a/allowed.py
+++ b/allowed.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
    report = apply_validated_patch(worktree, valid, ["allowed.py"], tmp_path / "logs")
    assert report["status"] == "APPLIED"
    assert (tmp_path / "allowed.py").read_text(encoding="utf-8") == "value = 2\n"
    escaped = valid.replace("allowed.py", "outside.py")
    with pytest.raises(PatchValidationError, match="outside task scope"):
        apply_validated_patch(worktree, escaped, ["allowed.py"], tmp_path / "logs")


def test_validated_patch_recounts_only_a_stale_hunk_length(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "allowed.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "allowed.py"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    stale_count_patch = """diff --git a/allowed.py b/allowed.py
index 1e8b314..5b40bd0 100644
--- a/allowed.py
+++ b/allowed.py
@@ -1,9 +1,11 @@
-value = 1
+value = 3
"""
    report = apply_validated_patch(
        Worktree(tmp_path, commit, "stale-hunk"),
        stale_count_patch,
        ["allowed.py"],
        tmp_path / "logs",
    )
    assert report["status"] == "APPLIED"
    assert (tmp_path / "allowed.py").read_text(encoding="utf-8") == "value = 3\n"


def test_fenced_model_replacement_is_wrapped_then_git_validated(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "allowed.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "allowed.py"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    worktree = Worktree(tmp_path, commit, "replacement")
    patch = _extract_model_patch(worktree, "```python\nvalue = 4  \n```", ["allowed.py"])
    assert "+value = 4  \n" not in patch
    apply_validated_patch(worktree, patch, ["allowed.py"], tmp_path / "logs")
    assert (tmp_path / "allowed.py").read_text(encoding="utf-8") == "value = 4\n"


def test_local_team_records_gpu_block_without_cpu_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = AgentTaskStore(tmp_path).create(
        "python", normalize_task_spec(REPOSITORY_ROOT, "python", _python_task("gpu-block"))
    )
    candidate = ServingCandidate(
        engine="vllm",
        endpoint="http://127.0.0.1:8001",
        model="fixture",
        revision="fixture",
        quantization="awq",
        kernel="flashinfer",
        prefix_caching=True,
        chunked_prefill=True,
        cuda_graph_mode="full",
        scheduler="continuous_batching",
    )
    monkeypatch.setattr(
        "research_workspace.team_runner.collect_cuda_evidence",
        lambda runner: {"status": "BLOCKED_GPU", "reason": "unit-test fixture"},
    )
    result = LocalTeamRunner(REPOSITORY_ROOT, tmp_path, candidate).run(
        task.task_id, query="typed explicit operation"
    )
    assert result["status"] == "BLOCKED_GPU"
    assert result["task"]["state"] == "blocked"
