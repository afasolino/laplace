from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from research_workspace.engineering import (
    AgentTaskStore,
    LocalToolRunner,
    ReferenceLibrary,
    normalize_task_spec,
    retrieve_engineering_evidence,
)
from research_workspace.governed_corpus import (
    load_installed_external_manifest,
    prepare_corpus_overlay,
    validate_corpus_retrieval,
)
from research_workspace.inference import ServingCandidate
from research_workspace.llm import GenerationResult, ModelInvocationError
from research_workspace.llm import VllmProvider
from research_workspace.model_routing import (
    AuditedModelCaller,
    ContextBudgetError,
    DualModelConfiguration,
    RoleRouter,
    RoutingTaskMetadata,
    assess_rtl_worker_eligibility,
    serving_candidate_from_json,
)
from research_workspace.model_artifacts import (
    validate_local_artifacts,
    validate_profile_alignment,
    validate_quantization_lock,
    validate_serving_environments,
)
from research_workspace.multilanguage_ablation import (
    _assert_phase_manifest_compatible,
    _configuration_fingerprint,
    _execute_task_lane,
    _result_set_root,
    _run_lane_subprocess,
    _task_spec,
    build_plan,
    load_benchmark_manifest,
    load_experiment_configuration,
    merge_phase_results,
    package_results,
    phase_status,
    preflight_report,
    runtime_prerequisites,
    selective_retry_plan,
    validate_phase_setup,
    validate_runtime,
    validate_held_out_pack,
)
from research_workspace.repair_protocol import StructuredOutputError
from research_workspace.rtl_contract import parse_rtl_worker_contract, rtl_worker_prompt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "codex_a6000"
    / "experiments"
    / "multilanguage_dual_model_ablation_v1"
    / "experiment.json"
)


def _candidate(port: int, model: str) -> ServingCandidate:
    return ServingCandidate(
        engine="vllm",
        endpoint=f"http://127.0.0.1:{port}",
        model=model,
        revision="fixture-revision",
        quantization="INT4",
        kernel="fixture",
        prefix_caching=True,
        chunked_prefill=True,
        cuda_graph_mode="fixture",
        scheduler="fixture",
        context_tokens=4096,
        max_output_tokens=512,
    )


def _eligible(arm: str = "C") -> RoutingTaskMetadata:
    return RoutingTaskMetadata(
        task_id="rtl-one",
        experiment_arm=arm,
        domain="verilog",
        task_kind="implementation",
        rtl_scope="bounded_module",
        worker_eligible=True,
        editable_sources=("rtl_one.v",),
        module_count=1,
        synthesizable=True,
        explicit_ports=True,
        cycle_behavior_specified=True,
        deterministic_verification=True,
    )


def test_serving_profiles_are_loopback_only_and_decoding_is_configurable() -> None:
    candidate = _candidate(8102, "main")
    assert candidate.context_tokens == 4096
    assert candidate.to_json()["max_output_tokens"] == 512
    with pytest.raises(ValueError, match="loopback"):
        ServingCandidate(
            engine="vllm",
            endpoint="http://192.0.2.1:8000",
            model="forbidden",
            revision="fixture",
            quantization="INT4",
            kernel="fixture",
            prefix_caching=True,
            chunked_prefill=True,
            cuda_graph_mode="fixture",
            scheduler="fixture",
        )
    invalid_profile = candidate.to_json()
    invalid_profile["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected fields"):
        serving_candidate_from_json(invalid_profile)


def test_openai_compatible_health_requires_exact_served_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, model: str) -> None:
            self.model = model

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": [{"id": self.model}]}).encode()

    served = "exact-model"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response(served))
    provider = VllmProvider("http://127.0.0.1:8102", "exact-model")
    assert provider.health()["status"] == "AVAILABLE"
    served = "wrong-model"
    assert provider.health()["status"] == "MODEL_MISMATCH"


def test_role_router_uses_worker_only_for_eligible_bounded_rtl_roles() -> None:
    main = _candidate(8102, "main")
    worker = _candidate(8103, "worker")
    router = RoleRouter(DualModelConfiguration(main=main, rtl_worker=worker))
    metadata = _eligible()
    assert assess_rtl_worker_eligibility(metadata).eligible is True
    assert router.select("bounded_rtl_implementation", metadata).selected == "rtl_worker"
    assert router.select("bounded_rtl_repair", metadata).selected == "rtl_worker"
    for role in (
        "planning_supervision",
        "retrieval_interpretation",
        "general_implementation",
        "rtl_contract_generation",
        "review",
    ):
        assert router.select(role, metadata).selected == "main"
    integration = RoutingTaskMetadata(
        **{
            **metadata.__dict__,
            "rtl_scope": "protocol_integration",
            "worker_eligible": False,
        }
    )
    decision = router.select("bounded_rtl_implementation", integration)
    assert decision.selected == "main"
    assert "protocol_integration" in decision.reason


class _FixtureBackend:
    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        assert prompt and context_tokens == 4096
        assert max_tokens is not None and max_tokens <= 512
        return GenerationResult("valid", "worker", None, None, "mocked", 11, 3)

    def token_count(self, prompt: str) -> int:
        assert prompt
        return 11

    def health(self) -> dict[str, str]:
        return {"status": "AVAILABLE", "backend": "fixture"}

    def model_identity(self) -> dict[str, str]:
        return {"backend": "fixture", "model": "worker"}


def test_every_routed_call_writes_complete_audit_record(tmp_path: Path) -> None:
    main = _candidate(8102, "main")
    worker = _candidate(8103, "worker")
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=main, rtl_worker=worker)),
        tmp_path,
        backend_factory=lambda candidate: _FixtureBackend(),
    )
    call = caller.generate(
        "bounded prompt",
        role="bounded_rtl_implementation",
        metadata=_eligible(),
        validator=lambda text: None if text == "valid" else (_ for _ in ()).throw(ValueError()),
    )
    assert call.response_valid is True
    audit = json.loads(Path(call.audit_path).read_text(encoding="utf-8"))
    assert audit["task_id"] == "rtl-one"
    assert audit["experiment_arm"] == "C"
    assert audit["routing"]["selected"] == "rtl_worker"
    assert audit["prompt_tokens"] == 11
    assert audit["completion_tokens"] == 3
    assert audit["generation_seconds"] >= 0
    assert audit["response_valid"] is True
    assert audit["retry_index"] == 0
    assert audit["fallback_used"] is False
    assert audit["requested_completion_tokens"] == 512
    assert audit["effective_completion_token_cap"] == 512
    assert audit["context_limit"] == 4096


class _BudgetBackend:
    def __init__(self, token_counts: dict[str, int], error: Exception | None = None) -> None:
        self.token_counts = token_counts
        self.error = error
        self.max_tokens: list[int | None] = []

    def token_count(self, prompt: str) -> int:
        return self.token_counts[prompt]

    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        self.max_tokens.append(max_tokens)
        if self.error is not None:
            raise self.error
        return GenerationResult("valid", "main", None, None, "mocked", None, 3)

    def health(self) -> dict[str, str]:
        return {"status": "AVAILABLE", "backend": "budget-fixture"}

    def model_identity(self) -> dict[str, str]:
        return {"backend": "budget-fixture", "model": "main"}


def test_reviewer_completion_is_capped_to_remaining_context(tmp_path: Path) -> None:
    backend = _BudgetBackend({"review": 3300})
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate(8102, "main"))),
        tmp_path,
        backend_factory=lambda candidate: backend,
    )
    call = caller.generate("review", role="review", metadata=_eligible("A"))
    assert backend.max_tokens == [284]
    assert call.budget["requested_completion_tokens"] == 512
    assert call.budget["effective_completion_token_cap"] == 284
    assert call.budget["context_limit"] == 4096
    assert call.budget["safety_margin_tokens"] == 512


def test_context_compaction_is_invoked_before_generation(tmp_path: Path) -> None:
    backend = _BudgetBackend({"large": 3900, "compact": 100})
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate(8102, "main"))),
        tmp_path,
        backend_factory=lambda candidate: backend,
    )
    call = caller.generate(
        "large",
        role="review",
        metadata=_eligible("A"),
        compact_prompt=lambda: "compact",
    )
    assert backend.max_tokens == [512]
    assert call.budget["compaction_occurred"] is True
    assert call.budget["prompt_tokens"] == 100


def test_irreducible_context_overflow_is_typed_and_audited(tmp_path: Path) -> None:
    backend = _BudgetBackend({"large": 3900, "still-large": 3800})
    caller = AuditedModelCaller(
        RoleRouter(DualModelConfiguration(main=_candidate(8102, "main"))),
        tmp_path,
        backend_factory=lambda candidate: backend,
    )
    with pytest.raises(ContextBudgetError) as caught:
        caller.generate(
            "large",
            role="review",
            metadata=_eligible("A"),
            compact_prompt=lambda: "still-large",
        )
    assert caught.value.category == "context_overflow"
    assert caught.value.budget["context_rejection_reason"]
    audit = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert audit["status"] == "CONTEXT_REJECTED"
    assert audit["response_valid"] is False


def test_http_400_model_error_becomes_a_typed_terminal_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path / "output",
        overlay_root=tmp_path / "corpus",
    )
    task = load_benchmark_manifest(configuration.manifest_path)[0]
    arm = configuration.arms[0]
    monkeypatch.setattr(
        "research_workspace.team_runner.LocalTeamRunner.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ModelInvocationError(
                "HTTP 400 maximum context length exceeded",
                category="http_error",
                http_status=400,
            )
        ),
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.gpu_memory_snapshot",
        lambda *args, **kwargs: {"status": "UNAVAILABLE"},
    )
    lane = _execute_task_lane(REPOSITORY_ROOT, configuration, task, arm, tmp_path / "lane-project")[
        "lane"
    ]
    assert isinstance(lane, dict)
    assert lane["status"] == "TERMINAL_FAILURE"
    assert lane["failure_category"] == "http_error"
    assert lane["terminal"] is True
    assert lane["resumability"]["skip_on_resume"] is False


def test_python_task_verification_is_scoped_and_sanitizes_experiment_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "candidate.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )
    (tmp_path / "test_public.py").write_text(
        "import os\nfrom candidate import add\n\n"
        "def test_public_contract() -> None:\n"
        "    assert 'LAPLACE_ABLATION_BASE_REVISION' not in os.environ\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (tmp_path / "test_unrelated.py").write_text(
        "def test_unrelated_repository_failure() -> None:\n    assert False\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LAPLACE_ABLATION_BASE_REVISION", "a" * 40)
    report = LocalToolRunner(tmp_path, tmp_path / "logs").run_python_quality_gates(
        ["candidate.py"], required_test_paths=["test_public.py"], timeout_seconds=60
    )
    assert report["passed"] is True
    assert report["repository_wide_tests_executed"] is False
    assert report["required_test_paths"] == ["test_public.py"]
    commands = [item["command"] for item in report["results"]]
    assert all("test_unrelated.py" not in command for command in commands)


def _contract(source: Path) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "task_id": "rtl-one",
            "module_name": "rtl_one",
            "language": "verilog",
            "editable_path": source.name,
            "current_source": {
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "content": source.read_text(encoding="utf-8"),
            },
            "parameters": [],
            "ports": [
                {
                    "name": "clk",
                    "direction": "input",
                    "width": "1",
                    "signed": False,
                    "description": "Rising-edge clock.",
                },
                {
                    "name": "rst_n",
                    "direction": "input",
                    "width": "1",
                    "signed": False,
                    "description": "Asynchronous active-low reset.",
                },
            ],
            "clock_reset": {
                "clock": {"name": "clk", "edge": "posedge"},
                "reset": {
                    "name": "rst_n",
                    "active_level": "low",
                    "synchronous": False,
                    "reset_values": {"state": "0"},
                },
            },
            "functional_requirements": ["Increment on an accepted event."],
            "cycle_requirements": ["Update on the rising edge only."],
            "handshake_and_events": ["event_i is sampled for one cycle."],
            "corner_cases": ["Reset and event together reset state."],
            "synthesis_constraints": ["Synthesizable Verilog-2001."],
            "forbidden_constructs": ["No delays or force/release."],
            "verification": {
                "commands": ["iverilog -g2001 and vvp", "yosys synth"],
                "acceptance_criteria": ["Public simulation and synthesis pass."],
            },
            "diagnostics": [],
        }
    )


def test_rtl_contract_is_complete_hash_bound_and_worker_prompt_has_no_tool_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rtl_one.v"
    source.write_text("module rtl_one(input wire clk); endmodule\n", encoding="utf-8")
    contract = parse_rtl_worker_contract(
        _contract(source),
        root=tmp_path,
        task_id="rtl-one",
        language="verilog",
        editable_path="rtl_one.v",
        require_diagnostics=False,
    )
    prompt = rtl_worker_prompt(contract)
    assert "exactly the one module" in prompt
    assert "Do not explore a repository" in prompt
    assert "never execute tools" not in prompt
    source.write_text("module rtl_one(input wire clk); wire changed; endmodule\n", encoding="utf-8")
    with pytest.raises(StructuredOutputError, match="stale"):
        parse_rtl_worker_contract(
            _contract(tmp_path / "snapshot.v")
            if (tmp_path / "snapshot.v").exists()
            else _contract_text_with_stale(source),
            root=tmp_path,
            task_id="rtl-one",
            language="verilog",
            editable_path="rtl_one.v",
            require_diagnostics=False,
        )


def _contract_text_with_stale(source: Path) -> str:
    value = json.loads(_contract(source))
    value["current_source"]["sha256"] = "0" * 64
    return json.dumps(value)


def test_manifest_validates_all_task_schemas_and_plan_does_not_probe_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LAPLACE_ABLATION_BASE_REVISION", raising=False)
    configuration = load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH)
    tasks = load_benchmark_manifest(configuration.manifest_path)
    assert len(tasks) == 32
    assert sum(task.routing.worker_eligible for task in tasks) == 11
    for task in tasks:
        normalize_task_spec(REPOSITORY_ROOT, task.language, _task_spec(task))
    monkeypatch.setattr(
        "research_workspace.model_routing.AuditedModelCaller.health",
        lambda self, include_worker: (_ for _ in ()).throw(AssertionError("endpoint probed")),
    )
    plan = build_plan(REPOSITORY_ROOT, configuration)
    assert plan["task_count"] == 32
    assert plan["endpoint_status"] == "NOT_PROBED_PLAN_ONLY"
    assert len(plan["tasks"]) == 32
    assert [phase.arm_ids for phase in configuration.phases] == [("A",), ("B",), ("C",)]
    assert configuration.base_revision is None
    phase_one_plan = build_plan(REPOSITORY_ROOT, configuration, phase_id="phase1")
    assert phase_one_plan["selected_phase"] == "phase1"
    assert not any(
        "phase2_main" in item or "phase2_rtl_worker" in item
        for item in phase_one_plan["missing_prerequisites"]
    )
    phase_two_plan = build_plan(REPOSITORY_ROOT, configuration, phase_id="phase2")
    assert not any("phase2_rtl_worker" in item for item in phase_two_plan["missing_prerequisites"])
    phase_three_plan = build_plan(REPOSITORY_ROOT, configuration, phase_id="phase3")
    assert phase_three_plan["selected_phase"] == "phase3"


def test_preflight_is_scoped_to_requested_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH)
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.runtime_prerequisites",
        lambda *args, **kwargs: {"passed": True, "missing": []},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation._tool_version",
        lambda name: {"available": True, "command": [name], "version": "test"},
    )
    report = preflight_report(REPOSITORY_ROOT, configuration, phase_id="phase1")
    assert report["selected_phase"] == "phase1"
    assert set(report["phases"]) == {"phase1"}
    assert report["phases"]["phase1"]["runtime"]["status"] == "NOT_PROBED"


def test_phase_setup_validation_does_not_require_other_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH)
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.runtime_prerequisites",
        lambda *args, **kwargs: {"passed": True, "missing": []},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_local_artifacts",
        lambda *args, **kwargs: {"status": "ALL_MODEL_ARTIFACTS_AVAILABLE"},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_held_out_pack",
        lambda *args, **kwargs: {"status": "VALID_ISOLATED_HELD_OUT_PACK"},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.prepare_corpus_overlay",
        lambda *args, **kwargs: {"status": "PREPARED"},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_corpus_retrieval",
        lambda *args, **kwargs: {"status": "VERIFIED_NON_EMPTY"},
    )
    monkeypatch.setenv(configuration.held_out_environment_variable, str(tmp_path))
    result = validate_phase_setup(REPOSITORY_ROOT, configuration, "phase1")
    assert result["status"] == "PHASE_CONFIGURATION_READY"
    assert result["phase_id"] == "phase1"
    assert result["runtime"]["status"] == "NOT_PROBED"


def test_phase_validation_selects_only_required_model_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH)
    selected: list[set[str]] = []

    def artifacts(_root: Path, artifact_ids: set[str]) -> dict[str, object]:
        selected.append(artifact_ids)
        return {"status": "ALL_MODEL_ARTIFACTS_AVAILABLE"}

    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_local_artifacts", artifacts
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.runtime_prerequisites",
        lambda *args, **kwargs: {"passed": True, "missing": []},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_held_out_pack",
        lambda *args, **kwargs: {"status": "VALID_ISOLATED_HELD_OUT_PACK"},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.prepare_corpus_overlay",
        lambda *args, **kwargs: {"status": "PREPARED"},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_corpus_retrieval",
        lambda *args, **kwargs: {"status": "VERIFIED_NON_EMPTY"},
    )
    monkeypatch.setenv(configuration.held_out_environment_variable, str(tmp_path))
    assert validate_phase_setup(REPOSITORY_ROOT, configuration, "phase2")["status"] == (
        "PHASE_CONFIGURATION_READY"
    )
    assert selected[-1] == {"phase2_main"}
    assert validate_phase_setup(REPOSITORY_ROOT, configuration, "phase3")["status"] == (
        "PHASE_CONFIGURATION_READY"
    )
    assert selected[-1] == {"phase2_main", "phase2_rtl_worker"}


def test_serving_environment_validation_checks_version_and_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "vllm"
    binary = environment / "bin"
    binary.mkdir(parents=True)
    (binary / "python").symlink_to(sys.executable)
    executable = binary / "vllm"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--version" ]; then echo 1.2.3; exit 0; fi\n'
        "echo --host --port --served-model-name --tensor-parallel-size "
        "--max-model-len --max-num-seqs --gpu-memory-utilization "
        "--enable-prefix-caching --enable-chunked-prefill --quantization\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        "research_workspace.model_artifacts.load_model_artifacts",
        lambda path: {
            "phase1_main": {
                "serving": {
                    "environment_path": str(environment),
                    "executable": str(executable),
                    "backend_version": "1.2.3",
                    "extra_args": ["--quantization", "awq_marlin"],
                }
            }
        },
    )
    monkeypatch.setattr(
        "research_workspace.model_artifacts.subprocess.run",
        _serving_environment_subprocess(executable),
    )
    result = validate_serving_environments(tmp_path, {"phase1_main"})
    environment_result = result["environments"][0]
    assert result["status"] == "SERVING_ENVIRONMENTS_READY"
    assert environment_result["available"] is True
    assert environment_result["arguments_supported"] is True
    assert environment_result["missing_arguments"] == []


def _serving_environment_subprocess(
    executable: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == str(executable) and command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="1.2.3\n", stderr="")
        if command[0] == str(executable) and command[1:] == ["serve", "--help"]:
            arguments = (
                "--host --port --served-model-name --tensor-parallel-size --max-model-len "
                "--max-num-seqs --gpu-memory-utilization --enable-prefix-caching "
                "--enable-chunked-prefill --quantization"
            )
            return subprocess.CompletedProcess(command, 0, stdout=arguments, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="1.2.3\n", stderr="")

    return run


def test_task_lane_timeout_is_killable_and_retained_as_typed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path / "experiment-output",
        default_timeout_seconds=17,
    )
    task = load_benchmark_manifest(configuration.manifest_path)[0]
    arm = configuration.arms[0]
    process = object()
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "research_workspace.paired_benchmark._await_lane",
        lambda received, *, timeout_seconds: (
            {
                "status": "TIMEOUT",
                "returncode": 124,
                "elapsed_seconds": float(timeout_seconds),
            }
            if received is process
            else (_ for _ in ()).throw(AssertionError("wrong process"))
        ),
    )
    result = _run_lane_subprocess(REPOSITORY_ROOT, configuration, task, arm)
    assert result["status"] == "TIMEOUT"
    assert result["returncode"] == 124
    assert result["timeout_seconds"] == 17
    bundle = result["bundle"]
    assert isinstance(bundle, dict)
    assert bundle["status"] == "TERMINAL_FAILURE"
    assert bundle["lane"]["failure_category"] == "timeout"
    assert bundle["lane"]["terminal"] is True
    assert Path(str(result["command_log"])).is_file()


def _base_reference_fixture(tmp_path: Path, domain: str, content: str) -> None:
    source = tmp_path / f"{domain}-source"
    source.mkdir()
    licence = source / "LICENSE"
    licence.write_text("Fixture licence\n", encoding="utf-8")
    guide = source / ("guide.py" if domain == "python" else "guide.sv")
    guide.write_text(content, encoding="utf-8")
    library = ReferenceLibrary(tmp_path / "base", domain, shared=True)  # type: ignore[arg-type]
    library.initialize(
        REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / f"{domain}_sources.yaml"
    )
    topic = "50_typing_validation/fixture" if domain == "python" else "20_interfaces/fixture"
    library.register_local(
        reference_id=f"fixture_{domain}",
        repository="https://example.invalid/fixture.git",
        commit=("a" if domain == "python" else "b") * 40,
        licence_identifier="MIT",
        licence_path=licence,
        selected_files=[(guide, topic, tuple(content.lower().split()))],
        permitted_use="reference_only_no_copy",
        attribution="Fixture",
    )


def test_fresh_projects_retrieve_non_empty_language_separated_corpus(tmp_path: Path) -> None:
    _base_reference_fixture(
        tmp_path,
        "python",
        "strict Pydantic validation asyncio cancellation SQLite transaction",
    )
    _base_reference_fixture(
        tmp_path,
        "systemverilog",
        "SystemVerilog AXI4-Lite WSTRB W1C ready valid assertions",
    )
    overlay = tmp_path / "overlay"
    prepare_corpus_overlay(REPOSITORY_ROOT, tmp_path / "base", overlay, require_external=False)
    validation = validate_corpus_retrieval(overlay, require_external=False)
    assert validation["status"] == "VERIFIED_NON_EMPTY"
    records = validation["domains"]
    assert isinstance(records, list)
    assert {record["domain"] for record in records} == {
        "python",
        "c",
        "verilog",
        "systemverilog",
    }
    assert all(record["retrieved_chunk_count"] > 0 for record in records)
    expected_reference_fragment = {
        "python": "python",
        "c": "_c_",
        "verilog": "verilog",
        "systemverilog": "systemverilog",
    }
    for record in records:
        assert all(
            expected_reference_fragment[record["domain"]]
            in str(item.get("reference_id", "")).lower()
            for item in record["retrieved"]
        )
        assert all(
            item.get("reference_id")
            and item.get("chunk_id")
            and item.get("rank")
            and item.get("score") is not None
            and item.get("revision")
            and item.get("licence_identifier")
            and item.get("sha256")
            for item in record["retrieved"]
        )


def test_external_c_and_verilog_corpus_is_installed_and_hash_verified() -> None:
    manifest = load_installed_external_manifest(REPOSITORY_ROOT)
    sources = manifest["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 6
    assert all(
        isinstance(source, dict)
        and len(str(source.get("resolved_commit", ""))) == 40
        and source.get("files")
        for source in sources
    )


def test_pinned_model_profiles_align_and_unavailable_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    experiment = CONFIG_PATH.parent
    alignment = validate_profile_alignment(experiment)
    assert alignment["status"] == "VALID_MODEL_PROFILES"
    local = validate_local_artifacts(experiment, verify_hashes=False)
    by_id = {
        str(record["artifact_id"]): record
        for record in local["artifacts"]
        if isinstance(record, dict)
    }
    assert by_id["phase1_main"]["available"] is True
    assert all(isinstance(by_id[artifact_id]["available"], bool) for artifact_id in by_id)

    missing_experiment = tmp_path / "experiment"
    missing_experiment.mkdir()
    metadata = json.loads((experiment / "model_artifacts.json").read_text(encoding="utf-8"))
    phase2_main = next(
        item for item in metadata["artifacts"] if item["artifact_id"] == "phase2_main"
    )
    phase2_main["output_path"] = str(tmp_path / "missing-model")
    phase2_main["verification"]["artifact_manifest"] = str(
        tmp_path / "missing-model" / "artifact_manifest.json"
    )
    (missing_experiment / "model_artifacts.json").write_text(json.dumps(metadata), encoding="utf-8")
    unavailable = validate_local_artifacts(missing_experiment, {"phase2_main"})
    assert unavailable["status"] == "MODEL_ARTIFACTS_INCOMPLETE"
    assert unavailable["artifacts"][0]["available"] is False


def test_c_quality_gate_uses_gcc_without_requiring_optional_cmake(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "value.c").write_text("int value(void) { return 7; }\n", encoding="utf-8")
    (fixture / "test_public.c").write_text(
        "int value(void); int main(void) { return value() == 7 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    result = LocalToolRunner(tmp_path, tmp_path / "logs").run_c_quality_gates(
        "fixture", required_tools=("gcc",), sanitizers=False
    )
    assert result["passed"] is True
    assert result["missing_tools"] == []


def test_fresh_c_task_retrieval_does_not_return_verilog_chunks(tmp_path: Path) -> None:
    _base_reference_fixture(tmp_path, "python", "strict validation transaction")
    _base_reference_fixture(tmp_path, "systemverilog", "AXI WSTRB W1C assertions")
    overlay = tmp_path / "overlay"
    prepare_corpus_overlay(REPOSITORY_ROOT, tmp_path / "base", overlay, require_external=False)
    task = next(
        item
        for item in load_benchmark_manifest(
            load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH).manifest_path
        )
        if item.task_id == "c_bounded_copy"
    )
    project = tmp_path / "fresh-project"
    normalized = normalize_task_spec(REPOSITORY_ROOT, "c", _task_spec(task))
    persisted = AgentTaskStore(project).create("c", normalized)
    evidence = retrieve_engineering_evidence(
        REPOSITORY_ROOT,
        project,
        persisted,
        query=task.objective,
        shared_reference_root=overlay,
    )
    governed = evidence["governed_references"]
    assert isinstance(governed, list) and governed
    assert all("/Verilog/" not in str(item.get("path")) for item in governed)


def test_empty_result_packaging_is_explicit_and_does_not_invent_measurements(
    tmp_path: Path,
) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH), output_root=tmp_path
    )
    reports = package_results(configuration)
    payload = json.loads(Path(str(reports["json"])).read_text(encoding="utf-8"))
    assert payload["task_arm_results"] == []
    assert payload["statistical_generality_claim"] is False
    assert payload["comparison_context"]["arm_phase_mapping"] == {
        "A": "phase1",
        "B": "phase2",
        "C": "phase3",
    }
    assert "across_serialized_phases" in payload["comparison_context"]["C_minus_B"]
    assert Path(str(reports["csv"])).is_file()
    assert Path(str(reports["markdown"])).is_file()


def test_infrastructure_failures_are_excluded_from_model_quality_metrics(tmp_path: Path) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path,
    )
    pair_root = _result_set_root(configuration) / "pairs" / "A"
    pair_root.mkdir(parents=True)
    common = {
        "schema_version": 2,
        "terminal": True,
        "arm_id": "A",
        "language": "python",
        "category": "implementation",
        "execution_phase": "phase1",
        "total_task_seconds": 1.0,
        "model_calls": {},
    }
    (pair_root / "candidate.json").write_text(
        json.dumps(
            {
                **common,
                "task_id": "candidate",
                "status": "COMPLETE_EVALUATED",
                "outcome_kind": "candidate_result",
                "held_out": {"status": "PASS", "score": 100.0},
            }
        ),
        encoding="utf-8",
    )
    (pair_root / "infra.json").write_text(
        json.dumps(
            {
                **common,
                "task_id": "infra",
                "status": "TERMINAL_FAILURE",
                "outcome_kind": "infrastructure_failure",
                "held_out": {
                    "status": "NOT_RUN_INFRASTRUCTURE_FAILURE",
                    "score": None,
                    "included_in_model_quality_metrics": False,
                },
            }
        ),
        encoding="utf-8",
    )
    reports = package_results(configuration)
    payload = json.loads(Path(str(reports["json"])).read_text(encoding="utf-8"))
    assert len(payload["infrastructure_failures"]) == 1
    assert payload["model_quality_task_arm_results"] == 1
    aggregate = next(
        item
        for item in payload["aggregates"]
        if item.get("arm_id") == "A" and item.get("language") == "python"
    )
    assert aggregate["tasks"] == 1
    assert aggregate["mean_held_out_score"] == 100.0


def test_offline_quantization_lock_satisfies_backend_metadata(tmp_path: Path) -> None:
    lock = tmp_path / "quantization.lock"
    lock.write_text(
        """compressed-tensors==0.17.1 \\
    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
datasets==5.0.0 \\
    --hash=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
huggingface-hub==1.23.0 \\
    --hash=sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
llmcompressor==0.12.0 \\
    --hash=sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
torch==2.12.0 \\
    --hash=sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
transformers==5.10.1 \\
    --hash=sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
""",
        encoding="utf-8",
    )
    experiment = CONFIG_PATH.parent
    result = validate_quantization_lock(experiment, lock)
    assert result["status"] == "QUANTIZATION_LOCK_COMPATIBLE"
    assert result["resolved_versions"]["transformers"] == "5.10.1"
    statuses = {str(item["status"]) for item in result["source_architecture_validation"]}
    assert statuses <= {"SUPPORTED", "DEFERRED_SOURCE_ARTIFACT_MISSING"}
    assert result["source_artifacts_ready"] is (statuses == {"SUPPORTED"})


@pytest.mark.parametrize(
    ("phase_id", "arm_id"),
    (("phase1", "A"), ("phase2", "B"), ("phase3", "C")),
)
def test_phase_resume_status_counts_only_fingerprint_compatible_results(
    tmp_path: Path, phase_id: str, arm_id: str
) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path,
    )
    tasks = load_benchmark_manifest(configuration.manifest_path)
    fingerprint = _configuration_fingerprint(configuration, {}, "f" * 64)
    pairs = _result_set_root(configuration) / "pairs" / arm_id
    pairs.mkdir(parents=True)
    for task in tasks:
        (pairs / f"{task.task_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "COMPLETE_EVALUATED",
                    "terminal": True,
                    "outcome_kind": "candidate_result",
                    "experiment_id": "multilanguage_dual_model_ablation_v1",
                    "task_id": task.task_id,
                    "language": task.language,
                    "category": task.category,
                    "arm_id": arm_id,
                    "base_revision": "a" * 40,
                    "execution_phase": phase_id,
                    "configuration_fingerprint": fingerprint,
                    "held_out_manifest_sha256": "f" * 64,
                    "started_at": "2026-07-16T00:00:00+00:00",
                    "ended_at": "2026-07-16T00:00:01+00:00",
                    "total_task_seconds": 1.0,
                    "model_calls": {},
                    "held_out": {"status": "PASS", "score": 100.0},
                    "resumability": {"skip_on_resume": True},
                    "lane_result": {},
                }
            ),
            encoding="utf-8",
        )
    manifest = _result_set_root(configuration) / "phases" / phase_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    assert phase_status(configuration, phase_id)["status"] == "COMPLETE"
    first = pairs / f"{tasks[0].task_id}.json"
    value = json.loads(first.read_text(encoding="utf-8"))
    value["base_revision"] = "b" * 40
    first.write_text(json.dumps(value), encoding="utf-8")
    status = phase_status(configuration, phase_id)
    assert status["status"] == "INCOMPLETE"
    assert status["remaining_pairs"] == 1
    value["base_revision"] = "a" * 40
    value["configuration_fingerprint"] = {**fingerprint, "experiment_sha256": "0" * 64}
    first.write_text(json.dumps(value), encoding="utf-8")
    status = phase_status(configuration, phase_id)
    assert status["status"] == "INCOMPLETE"
    assert status["remaining_pairs"] == 1


def test_legacy_result_schema_and_old_output_root_are_not_resumable(tmp_path: Path) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path,
    )
    task = load_benchmark_manifest(configuration.manifest_path)[0]
    fingerprint = _configuration_fingerprint(configuration, {}, "f" * 64)
    legacy = tmp_path / "pairs" / "A" / f"{task.task_id}.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE_EVALUATED",
                "task_id": task.task_id,
                "arm_id": "A",
                "base_revision": "a" * 40,
                "execution_phase": "phase1",
                "configuration_fingerprint": fingerprint,
                "held_out_manifest_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert phase_status(configuration, "phase1")["completed_pairs"] == 0
    assert "failure-accounting-v2" in str(_result_set_root(configuration))


def test_thirty_one_of_thirty_two_pairs_cannot_complete_a_phase(tmp_path: Path) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path,
    )
    tasks = load_benchmark_manifest(configuration.manifest_path)
    fingerprint = _configuration_fingerprint(configuration, {}, "f" * 64)
    pair_root = _result_set_root(configuration) / "pairs" / "A"
    pair_root.mkdir(parents=True)
    for task in tasks[:-1]:
        (pair_root / f"{task.task_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "COMPLETE_EVALUATED",
                    "terminal": True,
                    "outcome_kind": "candidate_result",
                    "experiment_id": "multilanguage_dual_model_ablation_v1",
                    "task_id": task.task_id,
                    "language": task.language,
                    "category": task.category,
                    "arm_id": "A",
                    "base_revision": "a" * 40,
                    "execution_phase": "phase1",
                    "configuration_fingerprint": fingerprint,
                    "held_out_manifest_sha256": "f" * 64,
                    "started_at": "2026-07-16T00:00:00+00:00",
                    "ended_at": "2026-07-16T00:00:01+00:00",
                    "total_task_seconds": 1.0,
                    "model_calls": {},
                    "held_out": {"status": "PASS", "score": 100.0},
                    "resumability": {"skip_on_resume": True},
                    "lane_result": {},
                }
            ),
            encoding="utf-8",
        )
    (pair_root / f"{tasks[-1].task_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "TERMINAL_FAILURE",
                "terminal": True,
                "outcome_kind": "infrastructure_failure",
                "task_id": tasks[-1].task_id,
                "arm_id": "A",
                "base_revision": "a" * 40,
                "execution_phase": "phase1",
                "configuration_fingerprint": fingerprint,
                "held_out_manifest_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest = _result_set_root(configuration) / "phases" / "phase1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"status": "COMPLETE", "fingerprint": fingerprint}), encoding="utf-8"
    )
    status = phase_status(configuration, "phase1")
    assert status["status"] == "INCOMPLETE"
    assert status["completed_pairs"] == 31
    assert status["remaining_pairs"] == 1
    assert status["attempted_terminal_pairs"] == 32
    assert status["terminal_failure_pairs"] == 1


def test_selective_retry_plan_retains_complete_and_retries_only_typed_failures(
    tmp_path: Path,
) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path,
    )
    tasks = load_benchmark_manifest(configuration.manifest_path)
    fingerprint = _configuration_fingerprint(configuration, {}, "f" * 64)
    pair_root = _result_set_root(configuration) / "pairs" / "B"
    pair_root.mkdir(parents=True)
    for index, task in enumerate(tasks):
        complete = index < 8
        row: dict[str, object] = {
            "schema_version": 2,
            "status": "COMPLETE_EVALUATED" if complete else "TERMINAL_FAILURE",
            "terminal": True,
            "outcome_kind": "candidate_result" if complete else "infrastructure_failure",
            "experiment_id": "multilanguage_dual_model_ablation_v1",
            "task_id": task.task_id,
            "arm_id": "B",
            "base_revision": "a" * 40,
            "execution_phase": "phase2",
            "configuration_fingerprint": fingerprint,
            "held_out_manifest_sha256": "f" * 64,
            "started_at": "2026-07-16T00:00:00+00:00",
            "ended_at": "2026-07-16T00:00:01+00:00",
            "total_task_seconds": 1.0,
            "model_calls": {},
            "held_out": {"status": "PASS" if complete else "NOT_RUN"},
            "resumability": {
                "classification": (
                    "COMPATIBLE_COMPLETE_RESULT" if complete else "RETRYABLE_TERMINAL_FAILURE"
                ),
                "skip_on_resume": complete,
            },
            "lane_result": {},
        }
        if not complete:
            row.update(
                {
                    "route": {},
                    "stage": "implementation",
                    "failure_category": "truncated_response",
                    "error": "retry budget exhausted",
                    "retry_counts": {},
                    "deterministic_verification": {"status": "NOT_RUN"},
                    "reviewer_status": {"status": "NOT_RUN"},
                }
            )
        (pair_root / f"{task.task_id}.json").write_text(json.dumps(row), encoding="utf-8")
    manifest = _result_set_root(configuration) / "phases" / "phase2" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"status": "INCOMPLETE", "fingerprint": fingerprint}),
        encoding="utf-8",
    )
    plan = selective_retry_plan(configuration, "phase2")
    assert plan["mutations_performed"] is False
    assert plan["counts"] == {
        "retained_completed": 8,
        "reset_for_retry": 24,
        "unattempted": 0,
        "untouched": 0,
    }
    retained = {item["task_id"] for item in plan["retained_completed_pairs"]}
    retried = {item["task_id"] for item in plan["reset_for_retry_pairs"]}
    assert retained.isdisjoint(retried)

    non_retryable = pair_root / f"{tasks[-1].task_id}.json"
    row = json.loads(non_retryable.read_text(encoding="utf-8"))
    row["resumability"] = {
        "classification": "NON_RETRYABLE_TERMINAL_FAILURE",
        "skip_on_resume": True,
    }
    non_retryable.write_text(json.dumps(row), encoding="utf-8")
    plan = selective_retry_plan(configuration, "phase2")
    assert plan["counts"]["reset_for_retry"] == 23
    assert plan["counts"]["untouched"] == 1
    assert plan["untouched_pairs"][0]["reason"] == "non_retryable_existing_result"


def test_phase_fingerprint_rejects_changed_benchmark_or_corpus() -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40
    )
    expected = _configuration_fingerprint(configuration, {"c": "one"}, "f" * 64)
    assert set(expected["model_configuration_hashes"]) == {"A", "B", "C"}
    assert len(str(expected["model_artifacts_sha256"])) == 64
    _assert_phase_manifest_compatible(
        configuration, {"fingerprint": expected}, {"c": "one"}, "f" * 64
    )
    with pytest.raises(Exception, match="incompatible"):
        _assert_phase_manifest_compatible(
            configuration, {"fingerprint": expected}, {"c": "changed"}, "f" * 64
        )


def test_three_phase_dependencies_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40
    )
    states = {"phase1": "COMPLETE", "phase2": "INCOMPLETE"}
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.build_plan",
        lambda *args, **kwargs: {"missing_prerequisites": []},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation._git_worktree_clean", lambda *args: True
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation._git_object_exists", lambda *args: True
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.phase_status",
        lambda config, phase: {
            "status": states[phase],
            "manifest": {"status": "COMPLETE", "fingerprint": {}},
        },
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation._assert_phase_manifest_compatible",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation._corpus_snapshot_hashes", lambda *args: {}
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation._sha256_file", lambda *args: "f" * 64
    )
    monkeypatch.setenv(configuration.held_out_environment_variable, "/tmp/heldout/manifest")
    phase_two = runtime_prerequisites(REPOSITORY_ROOT, configuration, phase_id="phase2")
    assert phase_two["passed"] is True
    phase_three = runtime_prerequisites(REPOSITORY_ROOT, configuration, phase_id="phase3")
    assert phase_three["passed"] is False
    assert "phase2_not_complete_or_incompatible" in phase_three["missing"]
    states["phase2"] = "COMPLETE"
    phase_three = runtime_prerequisites(REPOSITORY_ROOT, configuration, phase_id="phase3")
    assert phase_three["passed"] is True


def test_runtime_endpoint_probe_is_scoped_to_the_selected_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.runtime_prerequisites",
        lambda *args, **kwargs: {"passed": True, "missing": []},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.validate_held_out_pack",
        lambda *args, **kwargs: {"status": "VALID_ISOLATED_HELD_OUT_PACK"},
    )
    monkeypatch.setattr(
        "research_workspace.multilanguage_ablation.collect_cuda_evidence",
        lambda *args, **kwargs: {"status": "CUDA_A6000_VERIFIED"},
    )
    monkeypatch.setenv(configuration.held_out_environment_variable, "/tmp/evaluator-pack")
    worker_flags: list[bool] = []

    def health(self: object, *, include_worker: bool) -> dict[str, object]:
        worker_flags.append(include_worker)
        result: dict[str, object] = {"main": {"status": "AVAILABLE"}}
        if include_worker:
            result["rtl_worker"] = {"status": "AVAILABLE"}
        return result

    monkeypatch.setattr("research_workspace.model_routing.AuditedModelCaller.health", health)
    assert validate_runtime(REPOSITORY_ROOT, configuration, "phase1")["status"] == "RUNTIME_READY"
    assert worker_flags == [False]
    worker_flags.clear()
    assert validate_runtime(REPOSITORY_ROOT, configuration, "phase2")["status"] == "RUNTIME_READY"
    assert worker_flags == [False]
    worker_flags.clear()
    assert validate_runtime(REPOSITORY_ROOT, configuration, "phase3")["status"] == "RUNTIME_READY"
    assert worker_flags == [True]


def test_merge_requires_and_packages_all_serialized_task_arm_pairs(tmp_path: Path) -> None:
    configuration = replace(
        load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH, base_revision="a" * 40),
        output_root=tmp_path / "results",
        overlay_root=tmp_path / "corpus",
    )
    for domain in ("python", "c", "verilog", "systemverilog"):
        ReferenceLibrary(configuration.overlay_root, domain, shared=True).initialize(  # type: ignore[arg-type]
            REPOSITORY_ROOT / "codex_a6000" / "reference_sources" / f"{domain}_sources.yaml"
        )
    corpus_hashes = {
        domain: ReferenceLibrary(
            configuration.overlay_root,
            domain,
            shared=True,  # type: ignore[arg-type]
        ).snapshot_hash()
        or "UNINITIALIZED"
        for domain in ("python", "c", "verilog", "systemverilog")
    }
    fingerprint = _configuration_fingerprint(configuration, corpus_hashes, "f" * 64)
    tasks = load_benchmark_manifest(configuration.manifest_path)
    for arm in ("A", "B", "C"):
        pair_root = _result_set_root(configuration) / "pairs" / arm
        pair_root.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            (pair_root / f"{task.task_id}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "COMPLETE_EVALUATED",
                        "terminal": True,
                        "outcome_kind": "candidate_result",
                        "experiment_id": "multilanguage_dual_model_ablation_v1",
                        "task_id": task.task_id,
                        "arm_id": arm,
                        "base_revision": configuration.base_revision,
                        "execution_phase": {"A": "phase1", "B": "phase2", "C": "phase3"}[arm],
                        "configuration_fingerprint": fingerprint,
                        "held_out_manifest_sha256": "f" * 64,
                        "language": task.language,
                        "category": task.category,
                        "held_out": {"score": 100},
                        "started_at": "2026-07-16T00:00:00+00:00",
                        "ended_at": "2026-07-16T00:00:01+00:00",
                        "total_task_seconds": 1.0,
                        "model_calls": {},
                        "resumability": {"skip_on_resume": True},
                        "lane_result": {},
                    }
                ),
                encoding="utf-8",
            )
    for phase_id in ("phase1", "phase2", "phase3"):
        path = _result_set_root(configuration) / "phases" / phase_id / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "phase_id": phase_id,
                    "expected_pairs": 32,
                    "fingerprint": fingerprint,
                }
            ),
            encoding="utf-8",
        )
    merged = merge_phase_results(configuration)
    assert merged["status"] == "MERGED_COMPLETE"
    assert merged["task_arm_pairs"] == 96
    removed_path = _result_set_root(configuration) / "pairs" / "C" / f"{tasks[0].task_id}.json"
    removed = removed_path.read_text(encoding="utf-8")
    removed_path.unlink()
    with pytest.raises(Exception, match="Cannot merge"):
        merge_phase_results(configuration)
    removed_path.write_text(removed, encoding="utf-8")
    unexpected = _result_set_root(configuration) / "pairs" / "C" / "stale-task.json"
    unexpected.write_text(removed, encoding="utf-8")
    with pytest.raises(Exception, match="exactly 96"):
        merge_phase_results(configuration)


def _fake_three_phase_controls(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    control_log = tmp_path / "control.log"
    server_log = tmp_path / "server.log"
    python = tmp_path / "fake-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${FAKE_CONTROL_LOG}"\n'
        "for phase in phase1 phase2 phase3; do\n"
        '  if [[ " $* " == *" phase-status "* && " $* " == *" --phase ${phase} "* '
        '&& " $* " == *" --require-complete "* ]]; then\n'
        '      grep -qx "${phase}" "${FAKE_PHASE_STATE}" 2>/dev/null\n'
        "      exit $?\n"
        "  fi\n"
        "done\n"
        "for phase in phase1 phase2 phase3; do\n"
        '  if [[ " $* " == *" run-${phase} "* ]]; then\n'
        '      if [ "${FAKE_FAIL_PHASE:-}" = "${phase}" ]; then exit 2; fi\n'
        '      printf \'%s\\n\' "${phase}" >>"${FAKE_PHASE_STATE}"\n'
        "      exit 0\n"
        "  fi\n"
        "done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    manager = tmp_path / "fake-server-manager"
    manager.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s %s\\n\' "${LAPLACE_SERVER_OWNER_TOKEN:-external}" "$1" '
        '>>"${FAKE_SERVER_LOG}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    manager.chmod(0o755)
    return python, manager, control_log, server_log


def _run_fake_managed_all(tmp_path: Path, *, fail_phase: str = "") -> tuple[list[str], list[str]]:
    python, manager, control_log, server_log = _fake_three_phase_controls(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "LAPLACE_PYTHON": str(python),
            "LAPLACE_SERVER_MANAGER": str(manager),
            "LAPLACE_ABLATION_BASE_REVISION": "a" * 40,
            "LAPLACE_ABLATION_HELD_OUT_ROOT": str(tmp_path / "heldout"),
            "LAPLACE_ABLATION_OUTPUT_ROOT": str(tmp_path / "output"),
            "FAKE_CONTROL_LOG": str(control_log),
            "FAKE_SERVER_LOG": str(server_log),
            "FAKE_PHASE_STATE": str(tmp_path / "completed_phases"),
            "FAKE_FAIL_PHASE": fail_phase,
        }
    )
    completed = subprocess.run(
        [str(REPOSITORY_ROOT / "scripts/run_multilanguage_dual_model_ablation.sh"), "all"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == (2 if fail_phase else 0)
    controls = control_log.read_text(encoding="utf-8").splitlines()
    servers = server_log.read_text(encoding="utf-8").splitlines()
    return controls, servers


def test_managed_all_orders_phases_and_reuses_qwen_main(tmp_path: Path) -> None:
    controls, servers = _run_fake_managed_all(tmp_path)
    positions = {
        marker: next(index for index, line in enumerate(controls) if marker in line)
        for marker in (
            "validate-phase1",
            "run-phase1",
            "validate-phase2",
            "run-phase2",
            "validate-phase3",
            "run-phase3",
            "merge-report",
        )
    }
    assert list(positions.values()) == sorted(positions.values())
    actions = [line.split(maxsplit=1)[1] for line in servers]
    assert actions == [
        "start-phase1",
        "check-phase1",
        "stop-phase1",
        "start-phase2",
        "check-phase2",
        "start-phase3",
        "check-phase3",
        "stop-phase3",
    ]
    tokens = {line.split(maxsplit=1)[0] for line in servers}
    assert len(tokens) == 1
    assert "external" not in tokens


def test_managed_all_failure_prevents_next_phase(tmp_path: Path) -> None:
    controls, servers = _run_fake_managed_all(tmp_path, fail_phase="phase2")
    assert any("run-phase2" in line for line in controls)
    assert not any("validate-phase3" in line or "run-phase3" in line for line in controls)
    assert not any("merge-report" in line for line in controls)
    assert all("start-phase3" not in line for line in servers)
    assert any("stop-phase2" in line for line in servers)


def test_phase2_phase3_serial_launcher_verifies_orders_stops_and_merges(tmp_path: Path) -> None:
    python, manager, control_log, server_log = _fake_three_phase_controls(tmp_path)
    phase_state = tmp_path / "completed_phases"
    phase_state.write_text("phase1\n", encoding="utf-8")
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / "manifest.json").write_text("{}\n", encoding="utf-8")
    ffmpeg = tmp_path / "ffmpeg-lib"
    ffmpeg.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "LAPLACE_PYTHON": str(python),
            "LAPLACE_SERVER_MANAGER": str(manager),
            "LAPLACE_ABLATION_BASE_REVISION": "a" * 40,
            "LAPLACE_ABLATION_HELD_OUT_ROOT": str(heldout),
            "LAPLACE_ABLATION_OUTPUT_ROOT": str(tmp_path / "output"),
            "LAPLACE_VLLM_EXECUTABLE": "/bin/true",
            "LAPLACE_FFMPEG_LIBRARY_PATH": str(ffmpeg),
            "FAKE_CONTROL_LOG": str(control_log),
            "FAKE_SERVER_LOG": str(server_log),
            "FAKE_PHASE_STATE": str(phase_state),
        }
    )
    completed = subprocess.run(
        [str(REPOSITORY_ROOT / "scripts/run_phase2_phase3_serial.sh")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    controls = control_log.read_text(encoding="utf-8").splitlines()
    positions = {
        marker: next(index for index, line in enumerate(controls) if marker in line)
        for marker in ("run-phase2", "run-phase3", "merge-report")
    }
    assert list(positions.values()) == sorted(positions.values())
    actions = [line.split(maxsplit=1)[1] for line in server_log.read_text().splitlines()]
    assert actions.index("stop-phase2") < actions.index("start-phase3")
    assert actions[-1] == "stop-phase3"
    assert phase_state.read_text(encoding="utf-8").splitlines() == [
        "phase1",
        "phase2",
        "phase3",
    ]


def test_server_manager_accepts_absolute_vllm_override() -> None:
    environment = os.environ.copy()
    environment["LAPLACE_VLLM_EXECUTABLE"] = "/opt/laplace/vllm-cu129/bin/vllm"
    completed = subprocess.run(
        [
            str(REPOSITORY_ROOT / "scripts/manage_multilanguage_model_servers.sh"),
            "command-phase2-main",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "executable=/opt/laplace/vllm-cu129/bin/vllm" in completed.stdout
    assert "command=/opt/laplace/vllm-cu129/bin/vllm serve" in completed.stdout


def test_serving_preflight_validates_phase2_vllm_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "vllm-cu129"
    executable = environment / "bin" / "vllm"
    python = environment / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    python.write_text("#!/usr/bin/env bash\necho 0.25.0+cu129\n", encoding="utf-8")
    executable.chmod(0o755)
    python.chmod(0o755)
    monkeypatch.setenv("LAPLACE_VLLM_EXECUTABLE", str(executable))
    result = validate_serving_environments(CONFIG_PATH.parent, {"phase2_main"}, probe_cli=False)
    record = result["environments"][0]
    assert record["available"] is True
    assert record["environment_path"] == str(environment)
    assert record["executable"] == str(executable)


def test_server_shutdown_requires_pid_command_and_orchestration_ownership() -> None:
    script = (REPOSITORY_ROOT / "scripts/manage_multilanguage_model_servers.sh").read_text(
        encoding="utf-8"
    )
    signal_index = script.index('kill -TERM "${PID}"')
    assert script.index('RECORDED_TOKEN="$(sed -n') < signal_index
    assert script.index('*"${MODEL_PATH}"*"--host 127.0.0.1"*"--port ${PORT}"*') < signal_index


def test_managed_launcher_stops_polling_when_server_process_dies() -> None:
    script = (REPOSITORY_ROOT / "scripts/run_multilanguage_dual_model_ablation.sh").read_text(
        encoding="utf-8"
    )
    process_check = '"${SERVER_MANAGER}" "check-${PHASE}"'
    endpoint_check = "validate-runtime --phase"
    assert process_check in script
    assert script.index(process_check) < script.index(endpoint_check)
    assert "server stopped before its endpoint became ready" in script


def test_held_out_pack_is_hash_checked_and_must_be_external(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(REPOSITORY_ROOT, CONFIG_PATH)
    tasks = load_benchmark_manifest(configuration.manifest_path)
    pack = tmp_path / "evaluator-pack"
    manifest_tasks: dict[str, object] = {}
    for task in tasks:
        directory = pack / task.task_id
        directory.mkdir(parents=True)
        if task.language == "python":
            names = ["test_heldout.py"]
        elif task.language == "c":
            names = ["CMakeLists.txt", "test_heldout.c"]
        elif task.language == "verilog":
            names = ["tb_heldout.v"]
        else:
            names = ["tb_heldout.sv"]
        hashes: dict[str, str] = {}
        for name in names:
            content = f"// evaluator fixture for {task.task_id}\n"
            (directory / name).write_text(content, encoding="utf-8")
            hashes[name] = hashlib.sha256(content.encode()).hexdigest()
        manifest_tasks[task.task_id] = {
            "directory": task.task_id,
            "files": hashes,
        }
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "multilanguage_dual_model_ablation_v1",
                "tasks": manifest_tasks,
            }
        ),
        encoding="utf-8",
    )
    result = validate_held_out_pack(REPOSITORY_ROOT, configuration, pack)
    assert result["status"] == "VALID_ISOLATED_HELD_OUT_PACK"
    with pytest.raises(Exception, match="outside the repository"):
        validate_held_out_pack(tmp_path, configuration, pack)
