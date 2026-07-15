#!/usr/bin/env python3
"""Acquire only the pinned selected C/Verilog files and emit immutable governance hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_git(arguments: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(  # nosec B603, B607
        ["git", *arguments], cwd=cwd, check=True, timeout=600
    )


def git_output(arguments: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True, timeout=60).strip()


def snapshot_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    plan_path = (
        arguments.plan or root / "codex_a6000/governed_corpus/external_acquisition_plan.json"
    )
    plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or not isinstance(plan.get("sources"), list):
        raise SystemExit("invalid external acquisition plan")
    if set(plan) != {"schema_version", "installed_manifest", "sources"}:
        raise SystemExit("external acquisition plan has unexpected keys")
    installed_manifest = Path(str(plan["installed_manifest"]))
    if installed_manifest.is_absolute() or ".." in installed_manifest.parts:
        raise SystemExit("installed external manifest path is unsafe")
    seen: set[str] = set()
    domains: set[str] = set()
    expected_source_keys = {
        "reference_id",
        "domain",
        "repository",
        "revision_kind",
        "revision",
        "licence_identifier",
        "licence_source",
        "permitted_use",
        "attribution",
        "files",
    }
    for source in plan["sources"]:
        if not isinstance(source, dict) or set(source) != expected_source_keys:
            raise SystemExit("external acquisition source has invalid keys")
        reference_id = source.get("reference_id")
        if (
            not isinstance(reference_id, str)
            or not re.fullmatch(r"[a-z0-9_]+", reference_id)
            or reference_id in seen
        ):
            raise SystemExit("external acquisition reference id is unsafe or duplicated")
        seen.add(reference_id)
        if source.get("domain") not in {"c", "verilog"}:
            raise SystemExit("external acquisition domain must be C or Verilog")
        domains.add(source["domain"])
        if source.get("revision_kind") not in {"commit", "release"}:
            raise SystemExit("external acquisition revision kind is invalid")
        revision = source.get("revision")
        if not isinstance(revision, str) or not revision:
            raise SystemExit("external acquisition revision is missing")
        if source["revision_kind"] == "commit" and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit("external acquisition commit must be an exact hash")
        repository = source.get("repository")
        if not isinstance(repository, str) or not repository.startswith("https://"):
            raise SystemExit("external acquisition repositories must use HTTPS")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise SystemExit("external acquisition source has no selected files")
        for selected in files:
            if not isinstance(selected, dict) or set(selected) != {
                "source",
                "selected_topic",
                "topics",
            }:
                raise SystemExit("external selected-file record is invalid")
            for field in ("source", "selected_topic"):
                path = Path(str(selected[field]))
                if path.is_absolute() or ".." in path.parts:
                    raise SystemExit(f"external selected {field} path is unsafe")
            if not isinstance(selected.get("topics"), list) or not selected["topics"]:
                raise SystemExit("external selected file must have topics")
    if domains != {"c", "verilog"}:
        raise SystemExit("external acquisition plan must cover C and Verilog")
    if arguments.validate_only:
        print(json.dumps({"status": "VALID_ACQUISITION_PLAN", "sources": len(seen)}))
        return 0
    target_root = root / "codex_a6000/governed_corpus/external"
    installed: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="laplace-governed-acquisition-") as temporary:
        staging = Path(temporary)
        snapshot_staging = staging / "snapshots"
        for source in plan["sources"]:
            print(f"acquiring {source['reference_id']} at {source['revision']}", flush=True)
            source_root = staging / "repositories" / source["reference_id"]
            source_root.mkdir(parents=True)
            checked_git(["init", "--quiet"], cwd=source_root)
            checked_git(["remote", "add", "origin", source["repository"]], cwd=source_root)
            checked_git(["sparse-checkout", "init", "--no-cone"], cwd=source_root)
            selected_paths = [
                source["licence_source"],
                *(selected["source"] for selected in source["files"]),
            ]
            checked_git(["sparse-checkout", "set", "--no-cone", *selected_paths], cwd=source_root)
            checked_git(
                [
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "origin",
                    source["revision"],
                ],
                cwd=source_root,
            )
            checked_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=source_root)
            actual = git_output(["rev-parse", "HEAD"], cwd=source_root)
            if not re.fullmatch(r"[0-9a-f]{40}", actual):
                raise SystemExit(f"resolved commit is invalid for {source['reference_id']}")
            if source["revision_kind"] == "commit" and actual != source["revision"]:
                raise SystemExit(f"commit mismatch for {source['reference_id']}")
            destination = snapshot_staging / source["reference_id"] / source["revision"]
            final_destination = target_root / source["reference_id"] / source["revision"]
            licence_source = source_root / source["licence_source"]
            if not licence_source.is_file():
                raise SystemExit(f"licence file missing for {source['reference_id']}")
            licence_destination = destination / "LICENSE"
            licence_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(licence_source, licence_destination)
            licence_destination.chmod(0o444)
            files: list[dict[str, Any]] = []
            for selected in source["files"]:
                selected_source = source_root / selected["source"]
                if not selected_source.is_file():
                    raise SystemExit(
                        f"selected file missing: {source['reference_id']}/{selected['source']}"
                    )
                selected_destination = destination / "selected" / selected["source"]
                selected_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(selected_source, selected_destination)
                selected_destination.chmod(0o444)
                files.append(
                    {
                        "path": str(
                            (final_destination / "selected" / selected["source"]).relative_to(root)
                        ),
                        "selected_topic": selected["selected_topic"],
                        "topics": selected["topics"],
                        "sha256": sha256(selected_destination),
                    }
                )
            installed.append(
                {
                    "reference_id": source["reference_id"],
                    "domain": source["domain"],
                    "repository": source["repository"],
                    "revision_kind": source["revision_kind"],
                    "revision": source["revision"],
                    "resolved_commit": actual,
                    "licence_identifier": source["licence_identifier"],
                    "licence_path": str((final_destination / "LICENSE").relative_to(root)),
                    "licence_sha256": sha256(licence_destination),
                    "permitted_use": source["permitted_use"],
                    "attribution": source["attribution"],
                    "files": files,
                }
            )
            print(f"staged {source['reference_id']} at resolved commit {actual}", flush=True)
        for source in plan["sources"]:
            staged = snapshot_staging / source["reference_id"] / source["revision"]
            destination = target_root / source["reference_id"] / source["revision"]
            if destination.exists():
                existing_hashes = snapshot_hashes(destination)
                staged_hashes = snapshot_hashes(staged)
                if any(
                    staged_hashes.get(path) != digest for path, digest in existing_hashes.items()
                ):
                    raise SystemExit(
                        f"existing governed snapshot differs: {source['reference_id']}"
                    )
                for relative_path in sorted(set(staged_hashes) - set(existing_hashes)):
                    source_path = staged / relative_path
                    destination_path = destination / relative_path
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                if snapshot_hashes(destination) != staged_hashes:
                    raise SystemExit(
                        f"governed snapshot completion failed: {source['reference_id']}"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(staged, destination)
    manifest_path = root / plan["installed_manifest"]
    manifest_content = (
        json.dumps(
            {"schema_version": 1, "release": "external-multilanguage-v1", "sources": installed},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(manifest_content, encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
