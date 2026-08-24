#!/usr/bin/env python3
"""Persistent supervisor for the owned loopback-only CodeV process."""

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repository = _parser().parse_args(argv).repository_root.resolve()
    manager = repository / "scripts/manage_multilanguage_model_servers.sh"
    model_manager = repository / "scripts/manage_multilanguage_models.py"
    environment = dict(os.environ)
    control_python = Path(
        environment.get("LAPLACE_CONTROL_PLANE_PYTHON", str(repository / ".venv/bin/python"))
    )
    if not control_python.is_absolute() or not control_python.is_file():
        raise RuntimeError("codev_control_plane_python_invalid")
    started = subprocess.run(  # nosec B603
        [str(manager), "start-phase3-worker"],
        cwd=repository,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if started.returncode != 0:
        raise RuntimeError(f"codev_start_failed:{started.stderr[-500:]}")
    pid_path = (
        repository
        / "outputs/a6000_agent_team/model_servers/phase2_rtl_worker.pid"
    )
    try:
        def abort_start(_signum: int, _frame: object) -> None:
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, abort_start)
        signal.signal(signal.SIGINT, abort_start)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            health = subprocess.run(  # nosec B603
                [
                    str(control_python),
                    str(model_manager),
                    "endpoint",
                    "--artifact",
                    "phase2_rtl_worker",
                ],
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
                raise RuntimeError("codev_process_exited_before_ready")
            time.sleep(2)
        else:
            raise RuntimeError("codev_startup_timeout")

        stopped = threading.Event()

        def request_stop(_signum: int, _frame: object) -> None:
            stopped.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if "NOTIFY_SOCKET" in environment:
            subprocess.run(  # nosec B603
                ["/usr/bin/systemd-notify", "--ready", "--status=CodeV ready"],
                check=True,
                timeout=15,
            )
        print(json.dumps({"status": "READY", "model": "CodeV"}), flush=True)
        while not stopped.wait(1):
            if not _pid_alive(pid_path):
                raise RuntimeError("codev_process_exited")
    finally:
        subprocess.run(  # nosec B603
            [str(manager), "stop-phase3-worker"],
            cwd=repository,
            env=environment,
            check=False,
            timeout=120,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
