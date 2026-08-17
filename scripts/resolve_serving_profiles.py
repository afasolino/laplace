#!/usr/bin/env python3
"""Resolve every serving profile against the exact installed local vLLM CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from research_workspace.serving_profiles import (
    InstalledServingCapabilities,
    load_profiles,
    resolve_all,
)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--profile-root",
        type=Path,
        help="Profile directory relative to the repository (defaults to certified profiles).",
    )
    parser.add_argument(
        "--vllm",
        type=Path,
        default=Path("/home/giando/work/laplace/.venv-vllm-cu129/bin/vllm"),
    )
    parser.add_argument(
        "--ffmpeg-lib",
        type=Path,
        default=Path("/home/giando/work/laplace/.runtime/ffmpeg7/lib"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-model", action="store_true")
    return parser


def capture_installed(vllm: Path, ffmpeg_lib: Path) -> InstalledServingCapabilities:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(ffmpeg_lib)
    version = subprocess.run(
        [str(vllm), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout
    help_text = subprocess.run(
        [str(vllm), "serve", "--help=all"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout
    return InstalledServingCapabilities.from_help(
        version=version,
        help_text=help_text,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    capabilities = capture_installed(arguments.vllm, arguments.ffmpeg_lib)
    repository = arguments.repository_root.resolve()
    profile_root = (
        arguments.profile_root.resolve()
        if arguments.profile_root is not None
        else repository / "configs/serving_profiles"
    )
    profiles = load_profiles(profile_root)
    resolved = resolve_all(
        profiles,
        capabilities,
        executable=arguments.vllm.resolve(),
        require_model=not arguments.allow_missing_model,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "status": "RESOLVED",
                "installed": {
                    "version": capabilities.version,
                    "help_sha256": capabilities.help_sha256,
                    "flags": sorted(capabilities.flags),
                },
                "profiles": [item.to_json() for item in resolved],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
