#!/usr/bin/env python3
"""Persistent, ownership-safe supervisor for the sole P8 quality model."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Sequence

from research_workspace.production_model import (
    PRODUCTION_PROFILE_ID,
    assert_qwen38_promotable,
)
from research_workspace.serving_profile_runtime import ServingProfileRuntime, ServingRuntimeError
from research_workspace.serving_profiles import (
    InstalledServingCapabilities,
    ServingProfile,
    load_profiles,
    resolve_profile,
)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--ffmpeg-lib", type=Path, required=True)
    return parser


def selected_profile(repository_root: Path) -> ServingProfile:
    root = repository_root.resolve()
    assert_qwen38_promotable(root)
    profiles = load_profiles(root / "configs/serving_profiles")
    if len(profiles) != 1 or profiles[0].profile_id != PRODUCTION_PROFILE_ID:
        raise RuntimeError("production_profile_set_invalid")
    return profiles[0]


def installed_capabilities(vllm: Path, ffmpeg_lib: Path) -> InstalledServingCapabilities:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(ffmpeg_lib)
    version = subprocess.run(
        [str(vllm), "--version"], capture_output=True, text=True, check=True,
        timeout=120, env=environment,
    ).stdout
    help_text = subprocess.run(
        [str(vllm), "serve", "--help=all"], capture_output=True, text=True, check=True,
        timeout=120, env=environment,
    ).stdout
    return InstalledServingCapabilities.from_help(version=version, help_text=help_text)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository_root.resolve()
    profile = selected_profile(repository)
    executable = arguments.vllm.resolve(strict=True)
    capabilities = installed_capabilities(executable, arguments.ffmpeg_lib.resolve())
    resolved = resolve_profile(
        profile, capabilities, executable=executable, require_model=True,
        repository_root=repository,
    )
    runtime = ServingProfileRuntime(
        arguments.state_root, ffmpeg_library_path=arguments.ffmpeg_lib.resolve()
    )
    if runtime.ownership_path.exists():
        runtime.release_owned()
    owned = runtime.start(resolved)
    try:
        def abort_start(_signum: int, _frame: object) -> None:
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, abort_start)
        signal.signal(signal.SIGINT, abort_start)
        readiness = runtime.wait_ready(resolved)
        print(json.dumps({
            "status": "READY",
            "profile_id": profile.profile_id,
            "pid": owned.pid,
            "resolution_sha256": resolved.resolution_sha256,
            "readiness": readiness,
        }, sort_keys=True), flush=True)
        if "NOTIFY_SOCKET" in os.environ:
            subprocess.run(
                ["/usr/bin/systemd-notify", "--ready", f"--status={profile.profile_id} ready"],
                check=True, timeout=15,
            )
        stopped = threading.Event()

        def request_stop(_signum: int, _frame: object) -> None:
            stopped.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stopped.wait(1):
            if not Path(f"/proc/{owned.pid}").exists():
                raise RuntimeError("selected_model_process_exited")
    finally:
        try:
            runtime.release_owned()
        except ServingRuntimeError as exc:
            if exc.category != "no_owned_profile":
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
