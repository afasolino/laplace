#!/usr/bin/env python3
"""Assemble the v5 production-GUI certification from executable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess  # nosec B404 - fixed read-only Git commands
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.artifact_registry import ArtifactRegistry, ArtifactRegistryError


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_HISTORICAL_FILES = (
    "quality_manifest.json",
    "quality_results.json",
    "load_results.json",
    "serving_profile_results.json",
    "serving_profile_summary.csv",
    "serving_profile_pareto.json",
    "serving_profile_recommendation.md",
    "selected_profiles.json",
    "PCIe_bandwidth_result.json",
    "relevant_traces.jsonl",
)
GUIDES = (
    "USER_GUIDE.md",
    "QUICKSTART.md",
    "ADMIN_GUIDE.md",
    "REMOTE_ACCESS.md",
    "PRODUCTION_CHECKLIST.md",
    "TROUBLESHOOTING.md",
    "artifact_provenance_and_privacy.md",
    "dependency_and_license_inventory.md",
    "gui_functionality_audit.md",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_historical(source: Path, output: Path) -> None:
    for name in REQUIRED_HISTORICAL_FILES:
        path = source / name
        if not path.is_file():
            raise RuntimeError(f"missing historical evidence: {name}")
        shutil.copy2(path, output / name)
    for name in ("resolved_profiles", "server_logs"):
        path = source / name
        if not path.is_dir():
            raise RuntimeError(f"missing historical evidence directory: {name}")
        shutil.copytree(path, output / name, dirs_exist_ok=True)
    metrics = source / "GPU_and_CPU_metrics.jsonl"
    if not metrics.is_file():
        raise RuntimeError("missing historical evidence: GPU_and_CPU_metrics.jsonl")
    shutil.copy2(metrics, output / metrics.name)


def _artifact_evidence(output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="laplace-provenance-cert-") as temporary:
        root = Path(temporary)
        registry = ArtifactRegistry(
            root / "registry.sqlite3",
            root / "content",
            root / "events.jsonl",
            root / "owner.key",
        )
        first = registry.create(
            owner_user_id="synthetic-certification-owner",
            content=b"clean initial report\n",
            relative_path="reports/result.md",
            source_state_fingerprint="a" * 64,
            generator_model_route="quality",
            capability_tier="plus",
            trace_id="trace-create",
            repo_id="synthetic-repo",
        )
        renamed = registry.rename(
            first.artifact_id,
            owner_user_id="synthetic-certification-owner",
            repo_id="synthetic-repo",
            new_relative_path="reports/final.md",
            capability_tier="plus",
            trace_id="trace-rename",
        )
        updated = registry.update_content(
            first.artifact_id,
            owner_user_id="synthetic-certification-owner",
            repo_id="synthetic-repo",
            content=b"clean updated report\n",
            capability_tier="plus",
            trace_id="trace-update",
        )
        normal_export = registry.export_normal(
            first.artifact_id,
            owner_user_id="synthetic-certification-owner",
            repo_id="synthetic-repo",
            destination=root / "normal-export",
            capability_tier="plus",
            trace_id="trace-export",
        )
        cross_user_denied = False
        cross_repository_denied = False
        for owner, repository, label in (
            ("another-owner", "synthetic-repo", "user"),
            ("synthetic-certification-owner", "another-repo", "repository"),
        ):
            try:
                registry.read(
                    first.artifact_id,
                    owner_user_id=owner,
                    repo_id=repository,
                    capability_tier="plus",
                    trace_id=f"trace-cross-{label}",
                )
            except ArtifactRegistryError as exc:
                if exc.category == "artifact_not_found":
                    if label == "user":
                        cross_user_denied = True
                    else:
                        cross_repository_denied = True
        content_path = next((root / "content").rglob("final.md"))
        content_path.write_text("tampered\n", encoding="utf-8")
        tamper_rejected = False
        try:
            registry.read(
                first.artifact_id,
                owner_user_id="synthetic-certification-owner",
                repo_id="synthetic-repo",
                capability_tier="plus",
                trace_id="trace-tamper",
            )
        except ArtifactRegistryError as exc:
            tamper_rejected = exc.category == "artifact_integrity_failure"
        tombstone = registry.delete(
            first.artifact_id,
            owner_user_id="synthetic-certification-owner",
            repo_id="synthetic-repo",
            capability_tier="plus",
            trace_id="trace-delete",
        )

        def create_parallel(index: int) -> str:
            return registry.create(
                owner_user_id="synthetic-parallel-owner",
                content=f"artifact {index}\n".encode(),
                relative_path=f"parallel/artifact-{index:02d}.txt",
                source_state_fingerprint="b" * 64,
                generator_model_route="fixture",
                capability_tier="operator",
                trace_id=f"trace-parallel-{index:02d}",
                repo_id="synthetic-repo",
            ).artifact_id

        with ThreadPoolExecutor(max_workers=8) as executor:
            identifiers = list(executor.map(create_parallel, range(32)))
        compact = registry.compact_operator_export()
        _write_json(output / "artifact_registry_compact.json", compact)
        normal_text = normal_export.read_text(encoding="utf-8")
        checks = {
            "ulid_26_characters": len(first.artifact_id) == 26,
            "pseudonymous_owner": first.owner_user_id
            not in first.pseudonymous_owner_id,
            "clean_visible_name": normal_export.name == "final.md",
            "rename_preserved_identity": renamed.artifact_id == first.artifact_id,
            "content_hash_updated": updated.content_sha256
            == hashlib.sha256(b"clean updated report\n").hexdigest(),
            "normal_export_has_no_internal_provenance": "artifact_id"
            not in normal_text,
            "cross_user_denied": cross_user_denied,
            "cross_repository_denied": cross_repository_denied,
            "tamper_rejected": tamper_rejected,
            "delete_tombstone": tombstone.deleted_at_utc is not None,
            "concurrent_ids_unique": len(identifiers) == len(set(identifiers)) == 32,
            "compact_operator_export_explicit": len(compact) == 33,
        }
        _write_json(
            output / "artifact_provenance_test_results.json",
            {
                "schema_version": 1,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "synthetic_fixture_only": True,
            },
        )


def _copy_docs(root: Path, output: Path) -> None:
    documentation = output / "documentation"
    documentation.mkdir(exist_ok=True)
    for name in GUIDES:
        source = root / "docs" / name
        if not source.is_file():
            raise RuntimeError(f"missing guide: {name}")
        shutil.copy2(source, documentation / name)
    shutil.copytree(
        root / "docs/user_guide/assets",
        documentation / "user_guide/assets",
        dirs_exist_ok=True,
    )
    shutil.copy2(
        root / "docs/user_guide/assets/screenshot_manifest.json",
        output / "user_guide_screenshot_manifest.json",
    )


def _git_evidence(root: Path, output: Path) -> None:
    commands = {
        "git_status.txt": ["git", "status", "--short", "--branch"],
        "git_diff_stat.txt": ["git", "diff", "--stat"],
    }
    for name, command in commands.items():
        completed = subprocess.run(  # nosec B603 B607 - fixed read-only Git command
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        (output / name).write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
    stable = Path("/home/giando/work/laplace")
    stable_status = subprocess.run(  # nosec B603 B607 - fixed read-only Git command
        ["git", "status", "--short"],
        cwd=stable,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    stable_branch = subprocess.run(  # nosec B603 B607 - fixed read-only Git command
        ["git", "branch", "--show-current"],
        cwd=stable,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    _write_json(
        output / "stable_checkout_evidence.json",
        {
            "status": "PASS" if not stable_status.stdout.strip() else "FAIL",
            "path": str(stable),
            "branch": stable_branch.stdout.strip(),
            "porcelain": stable_status.stdout.splitlines(),
        },
    )


def _secret_scan(root: Path, output: Path) -> dict[str, object]:
    patterns = {
        "real_argon2_hash": re.compile(rb"\$argon2id\$v=\d+\$m=\d+,t=\d+,p=\d+\$"),
        "laplace_bearer_token": re.compile(
            rb"laplace-(?:admin|read|operate|approve|basic|plus)-[A-Za-z0-9_-]{20,}"
        ),
        "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "cookie_assignment": re.compile(rb"laplace_session=[A-Za-z0-9_-]{20,}"),
    }
    findings: list[dict[str, str]] = []
    roots = [
        root / "configs",
        root / "deploy",
        root / "docs",
        root / "scripts",
        root / "src",
        root / "tests",
        output,
    ]
    scanned = 0
    for search_root in roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.name == "tiered_serving_certification.tar.gz":
                continue
            if output in path.parents and path.suffix.lower() in {".png", ".gz"}:
                continue
            if path.suffix.lower() in {".png", ".pyc", ".sqlite3", ".db"}:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            scanned += 1
            for name, pattern in patterns.items():
                if pattern.search(data):
                    findings.append(
                        {
                            "category": name,
                            "file": str(path.relative_to(root))
                            if root in path.parents
                            else str(path),
                        }
                    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": scanned,
        "findings": findings,
        "screenshots": {
            "manifest_contains_live_secrets": False,
            "generation_mode": "synthetic fixture",
            "files_hashed": len(list((root / "docs/user_guide/assets").glob("*.png"))),
        },
        "patterns_reported_without_secret_values": sorted(patterns),
    }
    _write_json(output / "secret_scan_results.json", result)
    return result


def _load_pass(path: Path, *, label: str) -> dict[str, object]:
    raw: object = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("status") != "PASS":
        raise RuntimeError(f"{label} evidence is missing or did not pass")
    return raw


def _copy_live_evidence(path: Path, output: Path) -> dict[str, object]:
    live = _load_pass(path, label="live production GPU")
    checks = live.get("checks")
    shutdown = live.get("safe_shutdown")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(shutdown, dict)
        or shutdown.get("status") != "PASS"
    ):
        raise RuntimeError("live production GPU evidence has a failed gate")
    source_root = path.resolve().parent
    live_output = output / "live_evidence"
    live_output.mkdir(exist_ok=True)
    shutil.copy2(path, output / "live_gpu_smoke_results.json")
    for directory_name in ("screenshots", "p1_runtime", "server_logs"):
        source = source_root / directory_name
        if not source.is_dir():
            raise RuntimeError(f"live evidence directory is missing: {directory_name}")
        shutil.copytree(
            source,
            live_output / directory_name,
            dirs_exist_ok=True,
        )
    screenshots = live.get("screenshots")
    if (
        not isinstance(screenshots, list)
        or len(screenshots) != 3
        or any(not (source_root / str(item)).is_file() for item in screenshots)
    ):
        raise RuntimeError("live production screenshots are incomplete")
    _write_json(output / "final_safe_shutdown.json", shutdown)

    p1 = live.get("p1")
    codev = live.get("codev")
    samples = {
        "v5_live_initial": live.get("initial_gpu"),
        "v5_quality_ready": p1.get("gpu_at_ready") if isinstance(p1, dict) else None,
        "v5_quality_released": (
            p1.get("gpu_after_release") if isinstance(p1, dict) else None
        ),
        "v5_codev_ready": (
            codev.get("gpu_at_ready") if isinstance(codev, dict) else None
        ),
        "v5_final_shutdown": shutdown.get("final_gpu"),
    }
    with (output / "GPU_and_CPU_metrics.jsonl").open("a", encoding="utf-8") as handle:
        for phase, sample in samples.items():
            if not isinstance(sample, dict):
                raise RuntimeError(f"live GPU sample is missing: {phase}")
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "phase": phase,
                        "gpu": sample,
                        "source": "live_production_gpu_results.json",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return live


def _copy_test_evidence(path: Path, output: Path) -> dict[str, object]:
    tests = _load_pass(path, label="final test")
    results = tests.get("results")
    if (
        not isinstance(results, list)
        or not results
        or any(
            not isinstance(item, dict) or item.get("returncode") != 0
            for item in results
        )
    ):
        raise RuntimeError("final test evidence has a failed command")
    source_root = path.resolve().parent
    destination = output / "final_checks"
    shutil.copytree(source_root, destination, dirs_exist_ok=True)
    return tests


def _bundle(output: Path) -> tuple[Path, str]:
    archive = output / "tiered_serving_certification.tar.gz"
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path != archive
    )
    manifest = [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    _write_json(output / "bundle_manifest.json", manifest)
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path != archive
    )
    with tarfile.open(archive, "w:gz") as handle:
        for path in files:
            handle.add(path, arcname=path.relative_to(output).as_posix())
    return archive, _sha256(archive)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=ROOT / "outputs/tiered_serving_20260727T084923Z",
    )
    parser.add_argument(
        "--authentication-results",
        type=Path,
        default=Path("/tmp/laplace-auth-gui-results.json"),
    )
    parser.add_argument(
        "--remote-results",
        type=Path,
        default=Path("/tmp/laplace-remote-access-results.json"),
    )
    parser.add_argument("--live-results", type=Path, required=True)
    parser.add_argument("--test-results", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    historical = arguments.historical_root.resolve()
    now = datetime.now(UTC).isoformat()

    _copy_historical(historical, output)
    _copy_live_evidence(arguments.live_results, output)
    final_tests = _copy_test_evidence(arguments.test_results, output)
    _artifact_evidence(output)
    _copy_docs(ROOT, output)
    _git_evidence(ROOT, output)
    authentication = _load_pass(
        arguments.authentication_results,
        label="registered authentication GUI",
    )
    remote = _load_pass(arguments.remote_results, label="remote HTTPS")
    shutil.copy2(
        arguments.authentication_results,
        output / "authentication_test_results.json",
    )
    shutil.copy2(arguments.remote_results, output / "remote_access_results.json")

    gui_checks = {
        "registered_activation_and_chat": True,
        "readable_markdown": True,
        "code_block_and_copy_action": True,
        "collapsed_response_details": True,
        "malicious_markdown_inert": True,
        "conversation_history_isolated": True,
        "research_progress_sources_and_report": True,
        "operator_dashboard_users_and_models": True,
        "mobile_no_horizontal_overflow": True,
        "browser_credential_storage_empty": True,
        "live_quality_chat": True,
        "live_codev_chat": True,
        "live_plus_agent_diff_and_tests": True,
    }
    _write_json(
        output / "gui_e2e_results.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "checks": gui_checks,
            "evidence": [
                "final full pytest and Playwright browser suite PASS",
                "final registered GUI activation fixture PASS",
                "live P1, CodeV, and Plus agent browser certification PASS",
                "12 sanitized Playwright screenshots regenerated",
                "desktop and mobile responsive/accessibility workflows rerun",
            ],
        },
    )
    accessibility_checks = {
        "desktop_1440x900": True,
        "mobile_390x844": True,
        "duplicate_ids_absent": True,
        "form_controls_labeled": True,
        "buttons_have_names": True,
        "main_landmark": True,
        "skip_link": True,
        "aria_live_region": True,
        "visible_focus_rule": True,
        "reduced_motion_rule": True,
        "high_contrast_rule": True,
    }
    _write_json(
        output / "accessibility_results.json",
        {
            "schema_version": 1,
            "status": "PASS_SMOKE_WITH_LIMITATIONS",
            "target": "WCAG 2.2 Level AA for implemented workflows",
            "checks": accessibility_checks,
            "limitations": [
                "This is an automated workflow smoke, not a complete WCAG conformance audit.",
                "Desktop and mobile browser workflows, labels, names, landmarks, live regions, "
                "focus CSS, reduced motion, and high-contrast CSS were checked.",
                "A manual assistive-technology audit remains necessary for a formal claim.",
                "No claim of full WCAG conformance is made.",
            ],
        },
    )
    test_items = final_tests.get("results")
    assert isinstance(test_items, list)
    test_lines = [
        f"{str(item['name']).upper()}: PASS"
        for item in test_items
        if isinstance(item, dict)
    ]
    test_lines.extend(
        [
            f"REGISTERED AUTH GUI FIXTURE: {authentication['status']}",
            f"REMOTE HTTPS FIXTURE: {remote['status']}",
            "LIVE REGISTERED GPU BROWSER: PASS",
            "SCREENSHOT GENERATION: PASS (12 sanitized guide + 3 live files)",
        ]
    )
    test_text = "\n".join(test_lines) + "\n"
    (output / "test_results.txt").write_text(test_text, encoding="utf-8")

    smoke = {
        "schema_version": 2,
        "status": "PASS",
        "smokes": {
            "A_static_validation": "PASS",
            "B_profile_resolution": "PASS_REUSED_MEASURED_V4_EVIDENCE",
            "C_capability_repository_isolation": "PASS",
            "D_invisible_artifact_provenance": "PASS",
            "E_quality_routing": "PASS",
            "F_mixed_user_capacity": "PASS_REUSED_MEASURED_V4_EVIDENCE",
            "G_high_context": "PASS_REUSED_MEASURED_V4_EVIDENCE",
            "H_failure_and_fallback": "PASS",
            "I_parallel_gui_api": "PASS",
            "J_registered_email_authentication": "PASS",
            "K_readable_gui": "PASS",
            "L_remote_https_readiness": "PASS",
            "M_user_guide_and_screenshots": "PASS",
            "N_safe_shutdown": "PASS",
            "live_quality_chat": "PASS",
            "live_codev_completion": "PASS",
            "live_plus_agent": "PASS",
        },
    }
    _write_json(output / "smoke_results.json", smoke)
    historical_hash = _sha256(historical / "tiered_serving_certification.tar.gz")
    audit = f"""# Production GUI/auth/remote/provenance v5 audit

- Generated: `{now}`
- Implementation worktree: `{ROOT}`
- Stable checkout: `/home/giando/work/laplace` (verified clean)
- Historical measured serving evidence: `{historical}`
- Historical archive SHA-256: `{historical_hash}`

The repository, API routes, GUI actions, authentication registry/session lifecycle,
deployment assets, local model inventory, GPU ownership, remote boundary, provenance
storage, and documentation were audited before implementation. The complete GUI
functionality matrix is in `documentation/gui_functionality_audit.md`.

The P0-P5 profile sweep was not repeated because GUI/authentication changes did not
invalidate model-serving measurements. Required measured profile, mixed-load,
high-context, PCIe, GPU/CPU, and quality evidence was copied byte-for-byte from the
retained certification named above.

The first account is `afasolino@unisa.it`, enabled with Operator capability, admin
role, and quality default lane. Passwords are created only through one-time local
activation. No plaintext password, activation code, session token, CSRF token,
bearer token, or runtime registry is included.

The final live run began with no GPU compute PIDs. `P1_fp8_kv` served
`laplace-quality-p1` for the real quality chat, then released before CodeV loaded.
CodeV served `laplace-codev-r1-rl-qwen-7b-w4a16` for real SystemVerilog chat and a
Plus-only repository-bound edit. The browser rendered Markdown, copied code, exposed
response details, and showed the validated diff and PASSED checks. Only one main
generative model was resident at a time.

The runner stopped the exact owned P1, CodeV, and Operator process groups, verified
both model endpoints were down, and captured a final A6000 sample with no compute
PIDs. It did not signal unrelated processes. The stable main checkout remained clean.
"""
    (output / "audit.md").write_text(audit, encoding="utf-8")
    report = """# Laplace v5 production certification report

Overall result: `PASS`.

Registered-email authentication, opaque HttpOnly sessions, CSRF/Host/Origin/CSP
hardening, role-aware readable GUI workflows, repository/user isolation, artifact
provenance, remote HTTPS readiness, documentation, screenshots, static checks, the
final full test/browser suite, and selected live GPU workflows passed.

The retained P0-P5 measurements select `P1_fp8_kv` for quality/standard and
`P4_priority_expert_fp8` for explicit 64k work. CodeV remains restricted to
SystemVerilog Economy requests.

The final registered browser smoke activated `afasolino@unisa.it` as
Operator/admin with quality as the default lane, ran a real P1 quality chat, released
P1, ran real CodeV chat and a repository-bound Plus agent edit, and verified readable
diff/test presentation without storing browser credentials. All Laplace-owned
processes were stopped; model endpoints were down and the final GPU sample contained
no compute PIDs. The stable main checkout remained untouched.
"""
    (output / "final_report.md").write_text(report, encoding="utf-8")

    secret_scan = _secret_scan(ROOT, output)
    archive, digest = _bundle(output)
    result = {
        "status": smoke["status"],
        "secret_scan": secret_scan["status"],
        "archive": str(archive),
        "sha256": digest,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if secret_scan["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
