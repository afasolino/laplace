from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from research_workspace.production_model import select, verify_qwen38_artifact


def _write_promotable_manifest(root: Path, *, mtp_status: str) -> Path:
    config = root / "configs"
    artifact = root / ".models/qwen38"
    artifact.mkdir(parents=True)
    model = artifact / "model.safetensors"
    model.write_bytes(b"packed-qwen38")
    file_record = {
        "path": model.name,
        "size": model.stat().st_size,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    artifact_sha = hashlib.sha256(
        json.dumps([file_record], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    recipe = config / "quantization/recipe.json"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("{}\n", encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe.read_bytes()).hexdigest()
    artifact_manifest = {
        "schema_version": 1,
        "artifact_sha256": artifact_sha,
        "base_revision": "base-revision",
        "recipe_sha256": recipe_sha,
        "files": [file_record],
    }
    (artifact / "laplace_artifact_manifest.json").write_text(
        json.dumps(artifact_manifest), encoding="utf-8"
    )
    manifest = {
        "schema_version": 3,
        "promotion_allowed": True,
        "certification_status": "PASSED",
        "certification_evidence": {},
        "artifact_path": str(artifact),
        "artifact_manifest": ".models/qwen38/laplace_artifact_manifest.json",
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": model.stat().st_size,
        "base_revision": "base-revision",
        "artifact_revision": f"local-awq:{artifact_sha}",
        "tokenizer_revision": "tokenizer-revision",
        "source_provenance": {
            "recipe_path": "configs/quantization/recipe.json",
            "recipe_sha256": recipe_sha,
        },
        "quantization": {"status": "ARTIFACT_VERIFIED"},
        "mtp": {"status": mtp_status},
    }
    manifest_path = config / "model_manifests/qwen38_27b_a6000.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model


def test_qwen38_promotion_fails_closed_and_qwen36_rollback_works(tmp_path: Path) -> None:
    source_root = Path.cwd()
    root = tmp_path / "repo"
    config = root / "configs"
    (config / "model_manifests").mkdir(parents=True)
    for name in (
        "selected_serving_profiles.qwen38.json",
        "selected_serving_profiles.qwen38-mtp.json",
        "selected_serving_profiles.qwen36.rollback.json",
    ):
        shutil.copyfile(source_root / "configs" / name, config / name)
    (config / "model_manifests/qwen38_27b_a6000.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "promotion_allowed": False,
                "certification_status": "NOT_RUN",
            }
        ),
        encoding="utf-8",
    )
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
    _write_promotable_manifest(root, mtp_status="NOT_CERTIFIED")

    non_mtp = select("qwen38", root, tmp_path / "non-mtp.json")
    assert json.loads(non_mtp.read_text(encoding="utf-8"))["default_profile_id"] == (
        "P6_qwen38_w4a16"
    )

    manifest_path = config / "model_manifests/qwen38_27b_a6000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mtp"] = {"status": "PASSED"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mtp = select("qwen38", root, tmp_path / "mtp.json")
    assert json.loads(mtp.read_text(encoding="utf-8"))["default_profile_id"] == (
        "P7_qwen38_w4a16_mtp"
    )


def test_qwen38_selection_rejects_artifact_corruption(tmp_path: Path) -> None:
    source_root = Path.cwd()
    root = tmp_path / "repo"
    config = root / "configs"
    (config / "model_manifests").mkdir(parents=True)
    for name in (
        "selected_serving_profiles.qwen38.json",
        "selected_serving_profiles.qwen38-mtp.json",
    ):
        shutil.copyfile(source_root / "configs" / name, config / name)
    model = _write_promotable_manifest(root, mtp_status="NOT_CERTIFIED")
    model.write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="qwen38_artifact_file_hash_mismatch"):
        select("qwen38", root, tmp_path / "active.json")


def _move_artifact_outside_repository(root: Path, external_parent: Path) -> Path:
    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source = root / ".models/qwen38"
    external_parent.mkdir(parents=True, exist_ok=True)
    target = external_parent / "qwen38"
    shutil.move(str(source), str(target))

    manifest["artifact_path"] = str(target)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_qwen38_repository_local_artifact_remains_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "configs/model_manifests").mkdir(parents=True)
    _write_promotable_manifest(root, mtp_status="PASSED")

    monkeypatch.delenv("LAPLACE_APPROVED_MODEL_ROOTS", raising=False)
    result = verify_qwen38_artifact(root)

    assert Path(str(result["artifact_path"])).resolve() == (root / ".models/qwen38").resolve()


def test_qwen38_explicit_external_approved_root_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "configs/model_manifests").mkdir(parents=True)
    _write_promotable_manifest(root, mtp_status="PASSED")
    approved = tmp_path / "approved-models"
    artifact = _move_artifact_outside_repository(root, approved)

    monkeypatch.setenv("LAPLACE_APPROVED_MODEL_ROOTS", str(approved))
    result = verify_qwen38_artifact(root)

    assert Path(str(result["artifact_path"])).resolve() == artifact.resolve()


def test_qwen38_external_artifact_without_approval_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "configs/model_manifests").mkdir(parents=True)
    _write_promotable_manifest(root, mtp_status="PASSED")
    _move_artifact_outside_repository(root, tmp_path / "external-models")

    monkeypatch.delenv("LAPLACE_APPROVED_MODEL_ROOTS", raising=False)
    with pytest.raises(RuntimeError, match="qwen38_artifact_path_outside_repository"):
        verify_qwen38_artifact(root)


def test_qwen38_approved_root_does_not_allow_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "configs/model_manifests").mkdir(parents=True)
    _write_promotable_manifest(root, mtp_status="PASSED")

    outside = tmp_path / "outside"
    artifact = _move_artifact_outside_repository(root, outside)

    approved = tmp_path / "approved"
    approved.mkdir()
    escaped = approved / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)

    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_path"] = str(escaped / artifact.name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setenv("LAPLACE_APPROVED_MODEL_ROOTS", str(approved))
    with pytest.raises(RuntimeError, match="qwen38_artifact_path_outside_repository"):
        verify_qwen38_artifact(root)
