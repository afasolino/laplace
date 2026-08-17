from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from research_workspace.production_model import select


def test_qwen38_promotion_fails_closed_and_qwen36_rollback_works(tmp_path: Path) -> None:
    root = Path.cwd()
    with pytest.raises(RuntimeError, match="qwen38_promotion_not_certified"):
        select("qwen38", root, tmp_path / "active.json")

    output = select("qwen36", root, tmp_path / "active.json")
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["routes"]["economy"]["model_id"].startswith("laplace-codev")
    assert selected["default_profile_id"] == "P1_fp8_kv"


def test_qwen38_selects_mtp_only_after_its_independent_gate(tmp_path: Path) -> None:
    source_root = Path.cwd()
    root = tmp_path / "repo"
    config = root / "configs"
    (config / "model_manifests").mkdir(parents=True)
    for name in (
        "selected_serving_profiles.qwen38.json",
        "selected_serving_profiles.qwen38-mtp.json",
    ):
        shutil.copyfile(source_root / "configs" / name, config / name)
    manifest = {
        "promotion_allowed": True,
        "certification_status": "PASSED",
        "artifact_sha256": "a" * 64,
        "base_revision": "base-revision",
        "artifact_revision": "artifact-revision",
        "tokenizer_revision": "tokenizer-revision",
        "mtp": {"status": "NOT_CERTIFIED"},
    }
    manifest_path = config / "model_manifests/qwen38_27b_a6000.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    non_mtp = select("qwen38", root, tmp_path / "non-mtp.json")
    assert json.loads(non_mtp.read_text(encoding="utf-8"))["default_profile_id"] == (
        "P6_qwen38_w4a16"
    )

    manifest["mtp"] = {"status": "PASSED"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mtp = select("qwen38", root, tmp_path / "mtp.json")
    assert json.loads(mtp.read_text(encoding="utf-8"))["default_profile_id"] == (
        "P7_qwen38_w4a16_mtp"
    )
