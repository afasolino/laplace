#!/usr/bin/env python3
"""Validate v8 documentation links, commands, claims, and screenshot manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = (
    "README.md",
    "docs/QUICKSTART.md",
    "docs/USER_GUIDE.md",
    "docs/ADMIN_GUIDE.md",
    "docs/OPERATING_MODES.md",
    "docs/PRODUCT_ARCHITECTURE.md",
    "docs/CONFIGURATION_REFERENCE.md",
    "docs/MIGRATIONS.md",
    "docs/CI.md",
    "docs/RELEASE_POLICY.md",
    "docs/DESKTOP_REPOSITORY_SYNC.md",
    "docs/DATA_GOVERNANCE.md",
    "docs/BACKUP_AND_RECOVERY.md",
    "docs/EVALUATION_METHODOLOGY.md",
    "docs/RELIABILITY_TESTING.md",
    "docs/THREAT_MODEL_V7.md",
    "docs/SECURITY_RESIDUAL_RISKS.md",
    "docs/TROUBLESHOOTING.md",
    "docs/RELEASE_CANDIDATE_REVIEW_V8.md",
    "docs/DEFECT_REGISTER_V8.md",
    "docs/GO_NO_GO_CRITERIA_V8.md",
    "docs/MIGRATION_REHEARSAL_V8.md",
    "docs/CI_REMOTE_VALIDATION_V8.md",
    "docs/PACKAGE_AUDIT_V8.md",
    "docs/OPERATIONAL_REHEARSAL_V8.md",
    "docs/DESKTOP_SYNC_REVIEW_V8.md",
    "docs/RELEASE_CANDIDATE_RUNBOOK_V8.md",
    "docs/LIVE_GPU_CERTIFICATION_RUNBOOK_V8.md",
    "docs/SPECDEC_GPU_COORDINATION.md",
)
SCREENSHOT_MANIFESTS = (
    "docs/user_guide/assets/screenshot_manifest.json",
    "docs/user_guide/assets/agent_personal_corpus_gui_v6_screenshot_manifest.json",
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\\s+['\"][^'\"]*['\"])?\)")
SCRIPT = re.compile(r"(scripts/[A-Za-z0-9_.-]+\\.(?:py|sh|ps1))")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _link_findings(path: Path, text: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for raw in LINK.findall(text):
        target = unquote(raw.strip("<>"))
        parsed = urlsplit(target)
        if parsed.scheme in {"http", "https", "mailto"} or target.startswith("#"):
            continue
        if parsed.scheme or target.startswith(("/", "\\")):
            findings.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "category": "unsafe_or_unknown_link",
                    "target": target,
                }
            )
            continue
        relative = parsed.path.replace("\\", "/")
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.exists():
            findings.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "category": "broken_local_link",
                    "target": target,
                }
            )
    return findings


def _script_findings(path: Path, text: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for script in sorted(set(SCRIPT.findall(text))):
        if not (ROOT / script).is_file():
            findings.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "category": "stale_script_command",
                    "target": script,
                }
            )
    return findings


def _screenshot_findings() -> tuple[list[dict[str, object]], int]:
    findings: list[dict[str, object]] = []
    verified = 0
    for relative in SCREENSHOT_MANIFESTS:
        path = ROOT / relative
        if not path.is_file():
            findings.append({"path": relative, "category": "screenshot_manifest_missing"})
            continue
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = None
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            findings.append({"path": relative, "category": "screenshot_manifest_invalid"})
            continue
        if raw.get("contains_live_secrets") is not False:
            findings.append({"path": relative, "category": "screenshot_secret_flag_invalid"})
        if raw.get("contains_private_paths") is not False:
            findings.append({"path": relative, "category": "screenshot_path_flag_invalid"})
        entries = raw.get("files")
        if not isinstance(entries, list):
            findings.append({"path": relative, "category": "screenshot_entries_invalid"})
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                findings.append({"path": relative, "category": "screenshot_entry_invalid"})
                continue
            image = (path.parent / entry["file"]).resolve()
            synthetic = entry.get("synthetic_data") is True or entry.get("fixture_data_only") is True
            if (
                not image.is_relative_to(path.parent)
                or not image.is_file()
                or not synthetic
                or image.stat().st_size != entry.get("size_bytes")
                or _sha256(image) != entry.get("sha256")
            ):
                findings.append(
                    {
                        "path": relative,
                        "category": "screenshot_hash_or_fixture_invalid",
                        "target": entry["file"],
                    }
                )
            else:
                verified += 1
    return findings, verified


def check_documentation() -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for relative in REQUIRED_DOCS:
        if not (ROOT / relative).is_file():
            findings.append({"path": relative, "category": "required_document_missing"})
    markdown = _markdown_files()
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        findings.extend(_link_findings(path, text))
        findings.extend(_script_findings(path, text))
    screenshot_findings, screenshot_count = _screenshot_findings()
    findings.extend(screenshot_findings)

    required_phrases = {
        "docs/RELEASE_POLICY.md": (
            "2.0.0",
            "C8",
            "`BLOCKED`",
        ),
        "docs/MIGRATIONS.md": ("--dry-run", "--rollback-backup-id"),
        "docs/CONFIGURATION_REFERENCE.md": ("--validate-config",),
        "docs/CI.md": ("run_release_candidate_v8_certification.py",),
        "docs/OPERATING_MODES.md": ("standalone Core", "Operator/Zetsu"),
        "docs/SPECDEC_GPU_COORDINATION.md": (
            "BLOCKED_BY_SPECDEC_ACTIVE",
            "YIELDED_TO_SPECDEC",
        ),
        "docs/GO_NO_GO_CRITERIA_V8.md": (
            "GO_FOR_CONTROLLED_LIVE_GPU_CERTIFICATION",
            "GO_FOR_RELEASE_REVIEW_AFTER_LIVE_CERTIFICATION",
        ),
    }
    for relative, phrases in required_phrases.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for phrase in phrases:
            if phrase.lower() not in text.lower():
                findings.append(
                    {
                        "path": relative,
                        "category": "required_documentation_phrase_missing",
                        "target": phrase,
                    }
                )
    prohibited = re.compile(
        r"(?i)(?:v7\\s+GPU(?:/live-model)?\\s+(?:tests?|certification)\\s*:\\s*PASS|"
        r"GPU/live-model certification:\\s*PASS)"
    )
    for path in markdown:
        if prohibited.search(path.read_text(encoding="utf-8")):
            findings.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "category": "false_gpu_pass_claim",
                }
            )
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "markdown_files_checked": len(markdown),
        "screenshot_manifests_checked": len(SCREENSHOT_MANIFESTS),
        "screenshot_files_verified": screenshot_count,
        "findings": findings,
        "commands_checked_against_repository": True,
        "external_links_fetched": False,
        "gpu_claim_policy": "v8 live PASS requires the guarded conditional gate",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = check_documentation()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
