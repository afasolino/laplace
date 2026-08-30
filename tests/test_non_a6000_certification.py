from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_non_a6000_certification.py"
    spec = importlib.util.spec_from_file_location("run_non_a6000_certification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_has_explicit_categories_and_deferred_inventory() -> None:
    module = _module()
    manifest = module.classification_manifest("python")
    assert set(manifest) == {
        "cross_platform_deterministic",
        "linux_posix_required",
        "interactive_e2e",
        "optional_dependency",
        "gpu_smoke",
        "a6000_required",
        "external_live",
        "windows_privilege_required",
    }
    assert any(item["check_id"] == "qwen38_vllm_production_and_mtp" for item in manifest["a6000_required"])
    assert all(item["execute"] is False for item in manifest["a6000_required"] + manifest["external_live"])
    cross_platform = next(
        item
        for item in manifest["cross_platform_deterministic"]
        if item["check_id"] == "cross_platform_pytest"
    )
    marker_index = cross_platform["command"].index("-m")
    marker_index = cross_platform["command"].index("-m", marker_index + 1)
    assert cross_platform["command"][marker_index + 1] == "cross_platform_deterministic"
    deferred = module.deferred_test_inventory()
    assert deferred
    assert all(item["status"] == "DEFERRED" for item in deferred)
    assert all(item["category"] != "cross_platform_deterministic" for item in deferred)


def test_runner_rejects_non_campaign_output(tmp_path: Path) -> None:
    module = _module()
    try:
        module.run_certification(tmp_path / "outside")
    except ValueError as exc:
        assert str(exc) == "output_must_be_repository_local_campaign_artifact"
    else:
        raise AssertionError("runner accepted an output path outside .runtime/v3-non-a6000")


def test_unknown_test_node_is_rejected_instead_of_defaulting_to_cross_platform() -> None:
    from research_workspace.certification_taxonomy import category_for_nodeid

    try:
        category_for_nodeid("tests/test_future_unclassified.py::test_new")
    except ValueError as exc:
        assert str(exc) == "unknown_test_classification:tests/test_future_unclassified.py::test_new"
    else:
        raise AssertionError("unknown test node was classified")


def test_runner_uses_branch_independent_provenance() -> None:
    module = _module()
    assert "WINDOWS_CROSS_PLATFORM_CHECKS" not in module.__dict__
    assert "origin/feature/laplace-v3" not in module.__dict__.get("_provenance").__code__.co_consts


def test_routine_ci_uses_taxonomy_and_measured_coverage_threshold() -> None:
    workflow = (ROOT / ".github/workflows/unit-and-integration-tests.yml").read_text(
        encoding="utf-8"
    )

    assert "-m cross_platform_deterministic" in workflow
    assert "--cov-fail-under=63.6" in workflow
