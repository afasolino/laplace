#!/usr/bin/env python3
"""Run the deterministic non-A6000 Laplace certification boundary.

This runner executes only hardware-independent checks.  GPU/model-server and
live-service checks are represented as explicit deferred inventory entries and
are never silently converted into passing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess  # nosec B404 - fixed argv, shell=False
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.certification_taxonomy import (
    A6000_REQUIRED,
    CATEGORIES,
    CROSS_PLATFORM_DETERMINISTIC,
    DEFERRED_CATEGORIES,
    EXTERNAL_LIVE,
    GPU_SMOKE,
    NODEID_CATEGORIES,
    OPTIONAL_DEPENDENCY,
)


_TIMEOUT_SECONDS = 7_200


@dataclass(frozen=True, slots=True)
class CheckSpec:
    check_id: str
    category: str
    command: tuple[str, ...]
    reason: str
    execute: bool = False

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError("unknown_check_category")
        if not self.check_id or not self.command or not self.reason:
            raise ValueError("invalid_check_spec")


DETERMINISTIC_CHECKS = (
    CheckSpec(
        "pytest_collection",
        "cross_platform_deterministic",
        (
            "{python}",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--strict-markers",
            "-p",
            "no:cacheprovider",
            "-p",
            "anyio",
        ),
        "Complete test collection without model or GPU execution.",
        True,
    ),
    CheckSpec(
        "compileall",
        "cross_platform_deterministic",
        ("{python}", "-m", "compileall", "-q", "src", "scripts", "tests"),
        "Python syntax and bytecode compilation.",
        True,
    ),
    CheckSpec(
        "cross_platform_pytest",
        "cross_platform_deterministic",
        (
            "{python}",
            "-m",
            "pytest",
            "-q",
            "--strict-markers",
            "-m",
            CROSS_PLATFORM_DETERMINISTIC,
            "-p",
            "no:cacheprovider",
            "-p",
            "anyio",
        ),
        "Only tests marked cross_platform_deterministic; deferred categories are deselected.",
        True,
    ),
    CheckSpec(
        "ruff",
        "cross_platform_deterministic",
        ("{python}", "-m", "ruff", "check", "src", "tests", "scripts"),
        "Static lint gate.",
        True,
    ),
    CheckSpec(
        "mypy",
        "cross_platform_deterministic",
        ("{python}", "-m", "mypy", "--strict", "--platform", "linux", "src/research_workspace"),
        "Strict type gate using the authoritative Linux target platform.",
        True,
    ),
    CheckSpec(
        "git_diff_check",
        "cross_platform_deterministic",
        ("git", "diff", "--check"),
        "Whitespace and patch-integrity gate.",
        True,
    ),
)

SAFE_OPTIONAL_CHECKS = (
    CheckSpec(
        "generic_nvidia_resource_detection",
        GPU_SMOKE,
        ("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"),
        "Optional generic NVIDIA detection only; not used as A6000 evidence.",
    ),
    CheckSpec(
        "lightweight_cuda_smoke",
        GPU_SMOKE,
        ("{python}", "-c", "import torch; print(torch.cuda.is_available())"),
        "Optional lightweight CUDA smoke only; no model serving or performance claim.",
    ),
)

OPTIONAL_DEPENDENCY_CHECKS = (
    CheckSpec(
        "v3_optional_dependencies",
        OPTIONAL_DEPENDENCY,
        ("{python}", "-m", "pip", "check"),
        "The v3 extra is declared in pyproject.toml; unavailable optional packages are reported, not installed implicitly by this gate.",
    ),
)

DEFERRED_CHECKS = (
    CheckSpec(
        "qwen38_vllm_production_and_mtp",
        A6000_REQUIRED,
        ("scripts/certify_qwen38_production.py", "scripts/benchmark_qwen38_mtp_sweep.py"),
        "Deferred: Qwen3.8-27B production vLLM serving and MTP 3->4->6->8 optimization require the RTX A6000 topology.",
    ),
    CheckSpec(
        "a6000_vram_and_throughput_optimization",
        A6000_REQUIRED,
        ("scripts/certify_qwen38_profile.py",),
        "Deferred: no A6000 VRAM, throughput, or production memory conclusion may be inferred from the RTX 5060.",
    ),
    CheckSpec(
        "siliconmind_v1_live_certification",
        A6000_REQUIRED,
        ("future SiliconMind-V1 Qwen3 4B live certification",),
        "Deferred: configuration candidate only; live RTL-specialist promotion remains an A6000-phase task.",
    ),
    CheckSpec(
        "flash_next_runtime_and_coexistence",
        A6000_REQUIRED,
        ("future Flash-Next runtime/coexistence certification",),
        "Deferred: Flash-Next coexistence and runtime certification require the target production host.",
    ),
    CheckSpec(
        "manager_ability_ab_experiment",
        EXTERNAL_LIVE,
        ("future manager ability A/B experiment",),
        "Deferred: manager capability comparison requires the live production model/runtime.",
    ),
    CheckSpec(
        "final_whole_stack_performance",
        A6000_REQUIRED,
        ("future whole-stack performance certification",),
        "Deferred: final whole-stack performance certification is outside the non-A6000 phase.",
    ),
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _command(spec: CheckSpec, python: str) -> list[str]:
    return [python if item == "{python}" else item for item in spec.command]


def _git_value(root: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", *argv],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git_{argv[0]}_failed")
    return completed.stdout.strip()


def _git_optional(root: Path, *argv: str) -> str | None:
    completed = subprocess.run(
        ["git", *argv], cwd=root, capture_output=True, text=True, check=False, timeout=60
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _provenance(root: Path) -> dict[str, object]:
    branch = _git_optional(root, "branch", "--show-current")
    upstream = _git_optional(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    status = _git_value(root, "status", "--porcelain=v1")
    return {
        "head_sha": _git_value(root, "rev-parse", "HEAD"),
        "branch": branch or None,
        "detached": branch is None,
        "upstream": upstream,
        "upstream_sha": _git_optional(root, "rev-parse", upstream) if upstream else None,
        "clean": not bool(status),
        "diff_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None,
    }


def classification_manifest(python: str) -> dict[str, list[dict[str, object]]]:
    """Return the immutable check-category inventory used by the runner."""

    checks = (
        *DETERMINISTIC_CHECKS,
        *SAFE_OPTIONAL_CHECKS,
        *OPTIONAL_DEPENDENCY_CHECKS,
        *DEFERRED_CHECKS,
    )
    result = {category: [] for category in CATEGORIES}
    for spec in checks:
        item = asdict(spec)
        item["command"] = _command(spec, python)
        result[spec.category].append(item)
    return result


def deferred_test_inventory() -> list[dict[str, str]]:
    """Report explicit non-cross-platform tests as deferred, never passed."""

    return [
        {"nodeid": nodeid, "category": category, "status": "DEFERRED"}
        for nodeid, category in sorted(NODEID_CATEGORIES.items())
        if category in DEFERRED_CATEGORIES
    ]


def _run_check(root: Path, output: Path, spec: CheckSpec, python: str) -> dict[str, object]:
    command = _command(spec, python)
    started = datetime.now(UTC).isoformat()
    log_path = output / f"{spec.check_id}.log"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if spec.check_id == "pytest_collection" or spec.check_id.endswith("_pytest"):
        environment["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_SECONDS,
        )
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        return {
            "check_id": spec.check_id,
            "category": spec.category,
            "command": command,
            "reason": spec.reason,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "log": str(log_path),
            "started_at_utc": started,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return {
            "check_id": spec.check_id,
            "category": spec.category,
            "command": command,
            "reason": spec.reason,
            "status": "FAIL",
            "returncode": 2,
            "log": str(log_path),
            "started_at_utc": started,
            "error": type(exc).__name__,
        }


def run_certification(output_root: Path, *, root: Path | None = None) -> dict[str, object]:
    """Run non-A6000 checks and persist a bounded certification record."""

    repository = (root or _repository_root()).resolve(strict=True)
    output = output_root.resolve()
    campaign_root = repository / ".runtime" / "v3-non-a6000"
    try:
        output.relative_to(campaign_root)
    except ValueError as exc:
        raise ValueError("output_must_be_repository_local_campaign_artifact") from exc
    output.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    checks = [
        _run_check(repository, output, spec, python)
        for spec in DETERMINISTIC_CHECKS
    ]
    deferred = classification_manifest(python)
    environment = {
        "python_executable": python,
        "python_version": sys.version,
        "platform": platform.platform(),
        "provenance": _provenance(repository),
        "gpu_or_model_server_started": False,
        "live_model_inference_run": False,
    }
    result = {
        "schema_version": 2,
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "checks": checks,
        "categories": deferred,
        "deferred_tests": deferred_test_inventory(),
        "deferred_a6000_or_live": [asdict(spec) for spec in DEFERRED_CHECKS],
    }
    (output / "classification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_certification(arguments.output_root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"output_root": str(arguments.output_root.resolve()), "status": result["status"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
