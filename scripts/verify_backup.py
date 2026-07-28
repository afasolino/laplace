#!/usr/bin/env python3
"""Verify a strict backup manifest against an explicit restored fixture tree."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from research_workspace.governance import BackupManifest, GovernanceStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--restored-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    manifest = BackupManifest.model_validate_json(
        arguments.manifest.read_text(encoding="utf-8")
    )
    GovernanceStore.verify_backup_manifest(manifest, arguments.restored_root)
    print(f"verified backup {manifest.backup_id}: {len(manifest.entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

