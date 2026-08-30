#!/usr/bin/env python3
"""Validate least-privilege, pinned, non-GPU workflow contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "browser-fixture-tests.yml": "browser-fixture-tests",
    "documentation.yml": "documentation",
    "lint-and-types.yml": "lint-and-types",
    "migration-tests.yml": "migration-tests",
    "package-build.yml": "package-build",
    "release-candidate.yml": "release-candidate",
    "security.yml": "security",
    "unit-and-integration-tests.yml": "unit-and-integration-tests",
}
ACTION_PIN = re.compile(r"uses:\s*[^@\s]+@[a-f0-9]{40}(?:\s*#.*)?$")
SCRIPT_COMMAND = re.compile(r"python(?:\s+-m)?\s+(scripts/[A-Za-z0-9_.-]+\.py)")
FORBIDDEN_RUN = re.compile(
    r"\b(?:nvidia-smi|vllm|ollama|cuda|rocm|tensorrt|model\s+download)\b",
    re.IGNORECASE,
)


def validate() -> dict[str, object]:
    root = ROOT / ".github/workflows"
    findings: list[dict[str, str]] = []
    files = {path.name: path for path in root.glob("*.yml")}
    if set(files) != set(EXPECTED):
        findings.append({"file": ".github/workflows", "category": "workflow_set_mismatch"})
    for name, expected_workflow_name in sorted(EXPECTED.items()):
        path = files.get(name)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            raw: object = yaml.safe_load(text)
        except yaml.YAMLError:
            raw = None
        if not isinstance(raw, dict) or raw.get("name") != expected_workflow_name:
            findings.append({"file": name, "category": "workflow_yaml_or_name_invalid"})
        if "pull_request_target:" in text:
            findings.append({"file": name, "category": "pull_request_target_forbidden"})
        if "${{ secrets." in text:
            findings.append({"file": name, "category": "workflow_secret_reference_forbidden"})
        if not re.search(r"(?m)^permissions:\n  contents: read$", text):
            findings.append({"file": name, "category": "least_privilege_permissions_missing"})
        if len(re.findall(r"(?m)^\s+runs-on:", text)) != len(
            re.findall(r"(?m)^\s+timeout-minutes:", text)
        ):
            findings.append({"file": name, "category": "job_timeout_missing"})
        for line in text.splitlines():
            if "uses:" in line and ACTION_PIN.search(line.strip()) is None:
                findings.append({"file": name, "category": "action_not_commit_pinned"})
            if line.lstrip().startswith("- run:") and FORBIDDEN_RUN.search(line):
                findings.append({"file": name, "category": "gpu_or_model_command_forbidden"})
        for script in SCRIPT_COMMAND.findall(text):
            if not (ROOT / script).is_file():
                findings.append(
                    {"file": name, "category": f"workflow_script_missing:{script}"}
                )
        if "pytest" in text and "LAPLACE_FIXTURE_ONLY" not in text:
            findings.append({"file": name, "category": "fixture_mode_not_explicit"})
        if "upload-artifact" in text and not (
            "path: outputs/ci/" in text or "path: .runtime/v3-non-a6000/" in text
        ):
            findings.append({"file": name, "category": "artifact_path_not_sanitized"})
        if name != "release-candidate.yml" and not all(
            pattern in text for pattern in ('"feature/**"', '"review/**"')
        ):
            findings.append({"file": name, "category": "active_branch_trigger_missing"})
    matrix = files.get("unit-and-integration-tests.yml")
    if matrix is not None:
        text = matrix.read_text(encoding="utf-8")
        for required in ("ubuntu-24.04", "windows-2025", '"3.11"', '"3.12"'):
            if required not in text:
                findings.append({"file": matrix.name, "category": "matrix_incomplete"})
        if "-m cross_platform_deterministic" not in text:
            findings.append({"file": matrix.name, "category": "taxonomy_selection_missing"})
        if "--cov-fail-under=63.6" not in text:
            findings.append({"file": matrix.name, "category": "coverage_regression_gate_missing"})
    release = files.get("release-candidate.yml")
    if release is not None:
        release_text = release.read_text(encoding="utf-8")
        if "run_non_a6000_certification.py" not in release_text:
            findings.append(
                {"file": release.name, "category": "non_a6000_certification_command_missing"}
            )
        if "run_release_candidate_v8_certification.py" in release_text:
            findings.append(
                {"file": release.name, "category": "stale_release_candidate_command"}
            )
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "workflow_count": len(files),
        "expected_workflow_count": len(EXPECTED),
        "findings": findings,
        "gpu_commands_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = validate()
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
