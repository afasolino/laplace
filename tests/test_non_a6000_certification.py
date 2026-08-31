from __future__ import annotations

import importlib.util
import subprocess
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
    integration = (ROOT / ".github/workflows/lint-and-types.yml").read_text(
        encoding="utf-8"
    )
    release = (ROOT / ".github/workflows/release-candidate.yml").read_text(
        encoding="utf-8"
    )

    assert "-m cross_platform_deterministic" in workflow
    assert "-m linux_posix_required" in workflow
    assert "--cov-fail-under=63.6" in workflow
    assert "-m optional_dependency" in integration
    assert "include-hidden-files: true" in release


def test_non_a6000_report_cannot_be_misread_as_final_v3_certification() -> None:
    script = (ROOT / "scripts/run_non_a6000_certification.py").read_text(encoding="utf-8")
    assert '"certification_scope": "non_a6000_phase_only"' in script
    assert '"final_v3_certification": False' in script


def test_dirty_provenance_binds_exact_content_not_only_status(tmp_path: Path) -> None:
    module = _module()
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.test")
    tracked = repository / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-qm", "base")

    assert module._provenance(repository)["diff_sha256"] is None

    tracked.write_text("first\n", encoding="utf-8")
    first = module._provenance(repository)
    tracked.write_text("second\n", encoding="utf-8")
    second = module._provenance(repository)
    assert first["status_sha256"] == second["status_sha256"]
    assert first["diff_sha256"] != second["diff_sha256"]

    tracked.write_text("base\n", encoding="utf-8")
    untracked = repository / "untracked.txt"
    untracked.write_text("alpha\n", encoding="utf-8")
    third = module._provenance(repository)
    untracked.write_text("beta\n", encoding="utf-8")
    fourth = module._provenance(repository)
    assert third["status_sha256"] == fourth["status_sha256"]
    assert third["diff_sha256"] != fourth["diff_sha256"]
