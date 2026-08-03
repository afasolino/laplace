from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_live_production_gpu_certification as live_gpu  # noqa: E402
from run_live_production_gpu_certification import (  # noqa: E402
    ADMIN_CAPABILITIES,
    STABLE,
    _changed_worktree,
    _prepare_output,
    _record_unexpected_failure,
    _validate_static_preflight,
    _verify_python_worktree,
    _verify_systemverilog_worktree,
)
from run_release_candidate_v8_certification import (  # noqa: E402
    _copy_verified_live_screenshots,
    _validated_live_result,
)


@pytest.mark.parametrize(
    ("overrides", "category"),
    (
        ({"stable_clean": False}, "stable_checkout_not_clean"),
        (
            {"artifacts_available": False},
            "local_model_artifact_verification_failed",
        ),
        (
            {"occupied_endpoints": ("http://127.0.0.1:8201",)},
            "unrelated_target_endpoint_active",
        ),
        (
            {"runtime_paths_available": False},
            "required_local_runtime_path_missing",
        ),
    ),
)
def test_static_preflight_refuses_unsafe_inputs(
    overrides: dict[str, object],
    category: str,
) -> None:
    values: dict[str, object] = {
        "stable_clean": True,
        "artifacts_available": True,
        "occupied_endpoints": (),
        "runtime_paths_available": True,
    }
    values.update(overrides)
    with pytest.raises(RuntimeError, match=category):
        _validate_static_preflight(**values)  # type: ignore[arg-type]


def test_output_preflight_rejects_stable_and_existing_without_resume(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="must_not_use_stable"):
        _prepare_output(STABLE / "outputs/v8-forbidden", resume=False)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="exists_without_safe_resume"):
        _prepare_output(existing, resume=False)


def test_output_resume_requires_safe_terminal_record(tmp_path: Path) -> None:
    output = tmp_path / "resume"
    output.mkdir()
    (output / "live_production_gpu_results.json").write_text(
        json.dumps({"status": "BLOCKED_BY_SPECDEC_ACTIVE"}) + "\n",
        encoding="utf-8",
    )
    _prepare_output(output, resume=True)
    (output / "owned_profile_process.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ownership_record_present"):
        _prepare_output(output, resume=True)


def test_live_admin_has_every_independent_capability() -> None:
    assert set(ADMIN_CAPABILITIES) == {
        "chat",
        "agent",
        "research",
        "operator",
        "admin",
        "personal_corpus",
        "shared_corpus_ingest",
        "repository_admin",
        "model_admin",
    }


def test_unexpected_live_failure_writes_terminal_shutdown_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-live"
    output.mkdir()
    monkeypatch.setattr(live_gpu, "_endpoint_down", lambda url: True)

    def unavailable() -> object:
        raise RuntimeError("fixture GPU observer unavailable")

    monkeypatch.setattr(live_gpu, "observe_gpu", unavailable)
    result = _record_unexpected_failure(output, RuntimeError("fixture failure"))
    assert result["status"] == "FAIL"
    assert result["failure_category"] == "RuntimeError"
    assert result["safe_shutdown"] == {
        "status": "PASS",
        "quality_endpoint_down": True,
        "codev_endpoint_down": True,
        "final_gpu": {"status": "UNAVAILABLE", "error_type": "RuntimeError"},
        "final_coordination": {
            "status": "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
            "reason": "post_failure_gpu_observation_unavailable",
        },
    }
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAIL"
    assert any(
        item["path"] == "live_production_gpu_results.json"
        for item in manifest["files"]
    )


def test_invalid_live_result_is_bounded_and_fails_the_validity_gate() -> None:
    bounded, valid = _validated_live_result(
        {"schema_version": 1, "status": "FAIL", "detail": "fixture failure"}
    )
    assert valid is False
    assert bounded == {
        "schema_version": 1,
        "status": "NOT_RUN_DUE_TO_EARLIER_P0_DEFECT",
        "reason": "supplied live result had an invalid status",
        "supplied_result_valid": False,
    }


def test_live_screenshots_are_hash_bound_and_copied(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    screenshot = live_root / "screenshots/live_quality.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"synthetic live screenshot")
    digest = hashlib.sha256(screenshot.read_bytes()).hexdigest()
    result_path = live_root / "live_production_gpu_results.json"
    result_path.write_text("{}\n", encoding="utf-8")
    (live_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "screenshots/live_quality.png",
                        "sha256": digest,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "final"
    output.mkdir()
    evidence, valid = _copy_verified_live_screenshots(
        {"screenshots": ["screenshots/live_quality.png"]},
        result_path,
        output,
    )
    assert valid is True
    assert evidence == ["live_screenshots/live_quality.png"]
    assert (output / evidence[0]).read_bytes() == screenshot.read_bytes()
    screenshot.write_bytes(b"tampered")
    tampered_output = tmp_path / "tampered-final"
    tampered_output.mkdir()
    _, tampered_valid = _copy_verified_live_screenshots(
        {"screenshots": ["screenshots/live_quality.png"]},
        result_path,
        tampered_output,
    )
    assert tampered_valid is False


@pytest.mark.parametrize(
    "screenshots",
    (
        ["screenshots/live_quality.png", "screenshots/live_quality.png"],
        ["outside.png"],
    ),
)
def test_live_screenshots_reject_duplicates_and_path_escape(
    tmp_path: Path,
    screenshots: list[str],
) -> None:
    live_root = tmp_path / "live"
    (live_root / "screenshots").mkdir(parents=True)
    (live_root / "screenshots/live_quality.png").write_bytes(b"fixture")
    (live_root / "outside.png").write_bytes(b"fixture")
    result_path = live_root / "live_production_gpu_results.json"
    result_path.write_text("{}\n", encoding="utf-8")
    (live_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": item,
                        "sha256": hashlib.sha256(b"fixture").hexdigest(),
                    }
                    for item in screenshots
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "final"
    output.mkdir()
    _, valid = _copy_verified_live_screenshots(
        {"screenshots": screenshots},
        result_path,
        output,
    )
    assert valid is False


def test_python_worktree_verification_uses_only_isolated_changed_tree(
    tmp_path: Path,
) -> None:
    state = tmp_path / "external-state"
    worktree = state / "tiered_serving/worktrees/session-python"
    (worktree / "python").mkdir(parents=True)
    (worktree / "python/value.py").write_text(
        "def value() -> int:\n    return 2\n",
        encoding="utf-8",
    )
    (worktree / "python/test_value.py").write_text(
        "from python.value import value\n\n"
        "def test_value() -> None:\n    assert value() == 2\n",
        encoding="utf-8",
    )
    assert _changed_worktree(state, "python/value.py", "return 2") == worktree
    verification = _verify_python_worktree(state)
    assert verification["status"] == "PASS", verification


def test_systemverilog_verification_requires_every_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "external-state"
    worktree = state / "tiered_serving/worktrees/session-sv"
    (worktree / "rtl").mkdir(parents=True)
    (worktree / "rtl/example.sv").write_text(
        "module example(input logic a, output logic y);\n"
        "assign y = ~a;\nendmodule\n",
        encoding="utf-8",
    )
    (worktree / "rtl/tb_example.sv").write_text(
        "module tb_example; endmodule\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(live_gpu.shutil, "which", lambda name: f"/fixture/{name}")

    def verifier(
        command: list[str] | tuple[str, ...],
        *,
        worktree: Path,
        timeout: int = 120,
    ) -> dict[str, object]:
        del worktree, timeout
        return {
            "status": "PASS",
            "returncode": 0,
            "output_tail": (
                "SYSTEMVERILOG_VERIFY_PASS"
                if command[0].endswith("/vvp")
                else "fixture pass"
            ),
        }

    monkeypatch.setattr(live_gpu, "_run_verifier", verifier)
    assert _verify_systemverilog_worktree(state)["status"] == "PASS"
