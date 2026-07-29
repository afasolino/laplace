from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_production_gpu_certification import (  # noqa: E402
    STABLE,
    _prepare_output,
    _validate_static_preflight,
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
