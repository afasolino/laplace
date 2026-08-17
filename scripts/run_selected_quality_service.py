#!/usr/bin/env python3
"""Persistent, ownership-safe supervisor for the selected quality model."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Sequence

from research_workspace.production_model import assert_qwen38_promotable
from research_workspace.serving_profile_runtime import (
    ServingProfileRuntime,
    ServingRuntimeError,
)
from research_workspace.serving_profiles import (
    InstalledServingCapabilities,
    ServingProfile,
    load_profiles,
    resolve_profile,
)


def _pid_alive(pid_path: Path) -> bool:
    try:
        value = pid_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return value.isdigit() and Path(f"/proc/{value}").is_dir()


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
    selected: object = json.loads(
        (root / "configs/selected_serving_profiles.json").read_text(encoding="utf-8")
    )
    if not isinstance(selected, dict) or selected.get("schema_version") != 1:
        raise RuntimeError("selected_serving_profiles_invalid")
    profile_id = selected.get("default_profile_id")
    if not isinstance(profile_id, str):
        raise RuntimeError("selected_serving_profiles_invalid")
    if profile_id.startswith(("P6_qwen38", "P7_qwen38")):
        assert_qwen38_promotable(root)
    profiles = (
        *load_profiles(root / "configs/serving_profiles"),
        *load_profiles(root / "configs/serving_profile_candidates"),
    )
    try:
        return next(item for item in profiles if item.profile_id == profile_id)
    except StopIteration as exc:
        raise RuntimeError(f"selected_profile_unknown:{profile_id}") from exc


def installed_capabilities(vllm: Path, ffmpeg_lib: Path) -> InstalledServingCapabilities:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(ffmpeg_lib)
    version = subprocess.run(  # nosec B603
        [str(vllm), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout
    help_text = subprocess.run(  # nosec B603
        [str(vllm), "serve", "--help=all"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout
    return InstalledServingCapabilities.from_help(version=version, help_text=help_text)


def _legacy_qwen36(repository: Path) -> int:
    """Supervise the already-certified Phase 3 Qwen3.6 process unchanged."""

    manager = repository / "scripts/manage_multilanguage_model_servers.sh"
    environment = dict(os.environ)
    started = subprocess.run(  # nosec B603
        [str(manager), "start-phase3-main"],
        cwd=repository,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if started.returncode != 0:
        raise RuntimeError(f"qwen36_start_failed:{started.stderr[-500:]}")
    pid_path = repository / "outputs/a6000_agent_team/model_servers/phase2_main.pid"
    try:
        def abort_start(_signum: int, _frame: object) -> None:
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, abort_start)
        signal.signal(signal.SIGINT, abort_start)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            health = subprocess.run(  # nosec B603
                [str(manager), "health-phase2"],
                cwd=repository,
                env=environment,
                check=False,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if health.returncode == 0:
                break
            if not _pid_alive(pid_path):
                raise RuntimeError("qwen36_process_exited_before_ready")
            time.sleep(2)
        else:
            raise RuntimeError("qwen36_startup_timeout")

        stopped = threading.Event()

        def request_stop(_signum: int, _frame: object) -> None:
            stopped.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if "NOTIFY_SOCKET" in environment:
            subprocess.run(  # nosec B603
                ["/usr/bin/systemd-notify", "--ready", "--status=Qwen3.6 ready"],
                check=True,
                timeout=15,
            )
        print(
            json.dumps({"status": "READY", "profile": "qwen36_phase2_main"}),
            flush=True,
        )
        while not stopped.wait(1):
            if not _pid_alive(pid_path):
                raise RuntimeError("qwen36_process_exited")
    finally:
        subprocess.run(  # nosec B603
            [str(manager), "stop-phase3-main"],
            cwd=repository,
            env=environment,
            check=False,
            timeout=120,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository_root.resolve()
    profile = selected_profile(repository)
    if "Qwen3.6" in profile.model_path:
        return _legacy_qwen36(repository)
    executable = arguments.vllm.resolve(strict=True)
    capabilities = installed_capabilities(executable, arguments.ffmpeg_lib.resolve())
    resolved = resolve_profile(
        profile,
        capabilities,
        executable=executable,
        require_model=True,
    )
    runtime = ServingProfileRuntime(
        arguments.state_root,
        ffmpeg_library_path=arguments.ffmpeg_lib.resolve(),
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
        print(
            json.dumps(
                {
                    "status": "READY",
                    "profile_id": profile.profile_id,
                    "pid": owned.pid,
                    "resolution_sha256": resolved.resolution_sha256,
                    "readiness": readiness,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if "NOTIFY_SOCKET" in os.environ:
            subprocess.run(  # nosec B603
                [
                    "/usr/bin/systemd-notify",
                    "--ready",
                    f"--status={profile.profile_id} ready",
                ],
                check=True,
                timeout=15,
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
