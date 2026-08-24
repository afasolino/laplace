#!/usr/bin/env python3
"""Validate Qwen3.8 evidence and atomically promote P6 or certified P7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess  # nosec B404 - fixed local Git command
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from research_workspace.production_model import select, verify_qwen38_artifact


ROOT = Path(__file__).resolve().parents[1]
P6 = "P6_qwen38_w4a16"
P7 = "P7_qwen38_w4a16_mtp"
MANDATORY_PROFILE_GATES = (
    "model_identity",
    "normal_inference",
    "streaming",
    "reasoning",
    "tool_calling",
    "multi_turn",
    "cancellation",
    "context_window",
    "runtime_stability",
    "quantized_kernel",
    "gpu_headroom",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p6-certification", type=Path, required=True)
    parser.add_argument("--p6-production-gate", type=Path, required=True)
    parser.add_argument("--p7-certification", type=Path)
    parser.add_argument("--p7-production-gate", type=Path)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="write the certified manifest and active selector; otherwise validate only",
    )
    return parser


def _load(path: Path, error: str) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(error)
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()


def _profile_sha256(root: Path, profile_id: str) -> str:
    return _sha256(root / "configs/serving_profile_candidates" / f"{profile_id}.json")


def _profile_evidence_state(
    path: Path,
    *,
    profile_id: str,
    profile_sha256: str,
    artifact_sha256: str,
    repository_revision: str,
) -> tuple[bool, dict[str, object]]:
    evidence = _load(path, "profile_certification_malformed")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("profile_id") != profile_id
        or evidence.get("profile_sha256") != profile_sha256
        or evidence.get("artifact_sha256") != artifact_sha256
        or evidence.get("repository_revision") != repository_revision
        or evidence.get("unrelated_processes_signalled") is not False
    ):
        raise RuntimeError(f"profile_certification_identity_mismatch:{profile_id}")
    if evidence.get("status") != "PASSED":
        return False, evidence
    gates = evidence.get("gates")
    required = [*MANDATORY_PROFILE_GATES]
    if profile_id == P7:
        required.append("mtp")
    release = evidence.get("release")
    passed = (
        isinstance(gates, dict)
        and all(
            isinstance(gates.get(name), dict) and gates[name].get("status") == "PASS"
            for name in required
        )
        and isinstance(release, dict)
        and release.get("status") == "RELEASED_OWNED_PROFILE"
        and evidence.get("endpoint_down_after_release") is True
    )
    if not passed:
        raise RuntimeError(f"profile_certification_incomplete:{profile_id}")
    return True, evidence


def _production_evidence_state(
    path: Path,
    *,
    profile_id: str,
    profile_sha256: str,
    artifact_sha256: str,
    repository_revision: str,
    profile_certification_sha256: str,
) -> tuple[bool, dict[str, object]]:
    evidence = _load(path, "production_gate_malformed")
    gpu = evidence.get("gpu")
    release = evidence.get("release")
    quality_release = release.get("quality") if isinstance(release, dict) else None
    codev_release = release.get("codev") if isinstance(release, dict) else None
    if (
        evidence.get("schema_version") != 1
        or evidence.get("selected_profile_id") != profile_id
        or evidence.get("profile_sha256") != profile_sha256
        or evidence.get("artifact_sha256") != artifact_sha256
        or evidence.get("repository_revision") != repository_revision
        or evidence.get("profile_certification_sha256") != profile_certification_sha256
        or evidence.get("unrelated_processes_signalled") is not False
    ):
        raise RuntimeError(f"production_gate_identity_mismatch:{profile_id}")
    if evidence.get("status") != "PASSED":
        return False, evidence
    passed = (
        isinstance(gpu, dict)
        and isinstance(gpu.get("minimum_free_headroom_mib"), int)
        and gpu["minimum_free_headroom_mib"] >= 2_048
        and isinstance(quality_release, dict)
        and quality_release.get("status") == "RELEASED_OWNED_PROFILE"
        and isinstance(codev_release, dict)
        and codev_release.get("status") == "STOPPED_OWNED_CODEV"
        and evidence.get("target_endpoints_down_after_release") is True
    )
    if not passed:
        raise RuntimeError(f"production_gate_incomplete:{profile_id}")
    return True, evidence


def _evidence_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}


def _already_promoted_p6_revision(
    root: Path,
    artifact: Mapping[str, object],
    *,
    p6_certification: Path,
    p6_production_gate: Path,
    p7_certification: Path,
    p7_production_gate: Path,
    repository_revision: str,
) -> str | None:
    """Return the prior P6 revision only when persisted state proves promotion."""

    selected = _load(
        root / "configs/selected_serving_profiles.json",
        "selected_serving_profiles_malformed",
    )
    selected_profile = selected.get("default_profile_id")
    if (
        selected_profile not in {P6, P7}
        or selected.get("high_context_profile_id") != selected_profile
        or artifact.get("certification_status") != "PASSED"
        or artifact.get("promotion_allowed") is not True
    ):
        return None
    p6_cert_raw = _load(p6_certification, "profile_certification_malformed")
    p6_prod_raw = _load(p6_production_gate, "production_gate_malformed")
    p6_revision = p6_cert_raw.get("repository_revision")
    if (
        not isinstance(p6_revision, str)
        or not re.fullmatch(r"[a-f0-9]{40}", p6_revision)
        or p6_prod_raw.get("repository_revision") != p6_revision
    ):
        return None
    recorded = artifact.get("certification_evidence")
    p6_recorded = recorded.get("P6") if isinstance(recorded, Mapping) else None
    if not isinstance(p6_recorded, Mapping):
        return None
    expected = {
        "profile_certification": _evidence_record(p6_certification),
        "production_gate": _evidence_record(p6_production_gate),
    }
    if p6_recorded != expected:
        return None

    certified_revision = artifact.get("certified_repository_revision")
    if certified_revision not in {p6_revision, repository_revision}:
        return None
    p6_selection = (
        f"{expected['production_gate']['path']}#sha256={expected['production_gate']['sha256']}"
    )
    if certified_revision == p6_revision:
        return p6_revision if selected.get("selection_evidence") == p6_selection else None

    p7_recorded = recorded.get("P7") if isinstance(recorded, Mapping) else None
    p7_expected = {
        "profile_certification": _evidence_record(p7_certification),
        "production_gate": _evidence_record(p7_production_gate),
    }
    if p7_recorded != p7_expected:
        return None
    if selected_profile == P7:
        p7_selection = (
            f"{p7_expected['production_gate']['path']}"
            f"#sha256={p7_expected['production_gate']['sha256']}"
        )
        if selected.get("selection_evidence") != p7_selection:
            return None
    elif selected.get("selection_evidence") != p6_selection:
        return None
    return p6_revision


def evaluate_evidence(
    root: Path,
    *,
    p6_certification: Path,
    p6_production_gate: Path,
    p7_certification: Path | None,
    p7_production_gate: Path | None,
    artifact: dict[str, object],
    repository_revision: str,
) -> dict[str, object]:
    """Return a promotion decision; P7 failure never blocks a valid P6."""

    artifact_sha256 = str(artifact["artifact_sha256"])
    p6_profile_sha = _profile_sha256(root, P6)
    p6_cert_path = p6_certification.resolve(strict=True)
    p6_prod_path = p6_production_gate.resolve(strict=True)
    p6_cert_raw = _load(p6_cert_path, "profile_certification_malformed")
    p6_revision = repository_revision
    if p6_cert_raw.get("repository_revision") != repository_revision:
        if p7_certification is None or p7_production_gate is None:
            raise RuntimeError("prior_p6_evidence_requires_current_p7_upgrade")
        promoted_revision = _already_promoted_p6_revision(
            root,
            artifact,
            p6_certification=p6_cert_path,
            p6_production_gate=p6_prod_path,
            p7_certification=p7_certification.resolve(strict=True),
            p7_production_gate=p7_production_gate.resolve(strict=True),
            repository_revision=repository_revision,
        )
        if promoted_revision != p6_cert_raw.get("repository_revision"):
            raise RuntimeError("prior_p6_promotion_identity_mismatch")
        p6_revision = promoted_revision
    p6_cert_passed, _ = _profile_evidence_state(
        p6_cert_path,
        profile_id=P6,
        profile_sha256=p6_profile_sha,
        artifact_sha256=artifact_sha256,
        repository_revision=p6_revision,
    )
    if not p6_cert_passed:
        raise RuntimeError("mandatory_p6_profile_certification_failed")
    p6_prod_passed, _ = _production_evidence_state(
        p6_prod_path,
        profile_id=P6,
        profile_sha256=p6_profile_sha,
        artifact_sha256=artifact_sha256,
        repository_revision=p6_revision,
        profile_certification_sha256=_sha256(p6_cert_path),
    )
    if not p6_prod_passed:
        raise RuntimeError("mandatory_p6_production_gate_failed")

    mtp_status = "NOT_RUN"
    p7_cert_passed = False
    p7_prod_passed = False
    p7_records: dict[str, object] = {}
    if p7_production_gate is not None and p7_certification is None:
        raise RuntimeError("p7_production_gate_requires_p7_certification")
    if p7_certification is not None:
        p7_cert_path = p7_certification.resolve(strict=True)
        p7_records["profile_certification"] = _evidence_record(p7_cert_path)
        p7_cert_passed, _ = _profile_evidence_state(
            p7_cert_path,
            profile_id=P7,
            profile_sha256=_profile_sha256(root, P7),
            artifact_sha256=artifact_sha256,
            repository_revision=repository_revision,
        )
        mtp_status = "PROFILE_PASSED_PRODUCTION_NOT_RUN" if p7_cert_passed else "FAILED"
        if p7_production_gate is not None:
            p7_prod_path = p7_production_gate.resolve(strict=True)
            p7_records["production_gate"] = _evidence_record(p7_prod_path)
            p7_prod_passed, _ = _production_evidence_state(
                p7_prod_path,
                profile_id=P7,
                profile_sha256=_profile_sha256(root, P7),
                artifact_sha256=artifact_sha256,
                repository_revision=repository_revision,
                profile_certification_sha256=_sha256(p7_cert_path),
            )
            mtp_status = "PASSED" if p7_cert_passed and p7_prod_passed else "FAILED"

    mtp_enabled = p7_cert_passed and p7_prod_passed
    return {
        "selected_profile_id": P7 if mtp_enabled else P6,
        "mtp_enabled": mtp_enabled,
        "mtp_runtime_status": mtp_status,
        "p6_evidence_repository_revision": p6_revision,
        "evidence": {
            "P6": {
                "profile_certification": _evidence_record(p6_cert_path),
                "production_gate": _evidence_record(p6_prod_path),
            },
            "P7": p7_records,
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8"),
    )


def _atomic_bytes(path: Path, value: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _promoted_manifest(
    manifest: dict[str, object], decision: dict[str, object], repository_revision: str
) -> dict[str, object]:
    updated = json.loads(json.dumps(manifest))
    updated["certification_status"] = "PASSED"
    updated["promotion_allowed"] = True
    updated["certified_at_utc"] = datetime.now(UTC).isoformat()
    updated["certified_repository_revision"] = repository_revision
    updated["certification_evidence"] = decision["evidence"]
    updated.pop("external_blocker", None)
    candidates = updated.get("serving_candidates")
    if not isinstance(candidates, dict):
        raise RuntimeError("qwen38_serving_candidates_invalid")
    p6 = candidates.get(P6)
    p7 = candidates.get(P7)
    if not isinstance(p6, dict) or not isinstance(p7, dict):
        raise RuntimeError("qwen38_serving_candidates_invalid")
    p6["live_status"] = "PASSED"
    p6["evidence"] = decision["evidence"]["P6"]
    mtp_enabled = decision["mtp_enabled"] is True
    p7["live_status"] = "PASSED" if mtp_enabled else decision["mtp_runtime_status"]
    p7["evidence"] = decision["evidence"]["P7"]
    mtp = updated.get("mtp")
    if not isinstance(mtp, dict):
        raise RuntimeError("qwen38_mtp_manifest_invalid")
    mtp["runtime_status"] = decision["mtp_runtime_status"]
    mtp["status"] = "PASSED" if mtp_enabled else "NOT_CERTIFIED"
    if mtp_enabled:
        mtp["asset_status"] = "PRESENT_CERTIFIED"
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    artifact = verify_qwen38_artifact(root)
    repository_revision = _git_revision(root)
    decision = evaluate_evidence(
        root,
        p6_certification=arguments.p6_certification,
        p6_production_gate=arguments.p6_production_gate,
        p7_certification=arguments.p7_certification,
        p7_production_gate=arguments.p7_production_gate,
        artifact=artifact,
        repository_revision=repository_revision,
    )
    if not arguments.promote:
        print(json.dumps({"status": "VALIDATED_NOT_PROMOTED", **decision}, sort_keys=True))
        return 0

    manifest_path = root / "configs/model_manifests/qwen38_27b_a6000.json"
    selected_path = root / "configs/selected_serving_profiles.json"
    original_manifest = manifest_path.read_bytes()
    original_selected = selected_path.read_bytes()
    promoted = _promoted_manifest(artifact, decision, repository_revision)
    selected_gate = decision["evidence"]["P7" if decision["mtp_enabled"] else "P6"][
        "production_gate"
    ]
    selection_evidence = f"{selected_gate['path']}#sha256={selected_gate['sha256']}"
    try:
        _atomic_json(manifest_path, promoted)
        select("qwen38", root, selection_evidence=selection_evidence)
    except Exception:
        _atomic_bytes(manifest_path, original_manifest)
        _atomic_bytes(selected_path, original_selected)
        raise
    print(
        json.dumps(
            {
                "status": "PROMOTED",
                "selected_profile_id": decision["selected_profile_id"],
                "mtp_enabled": decision["mtp_enabled"],
                "manifest": str(manifest_path),
                "selector": str(selected_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
