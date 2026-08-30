"""Policy-free bounded termination for owned subprocess groups."""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - accepts an already-owned process only
from typing import Any


def stop_process_tree(
    process: subprocess.Popen[Any],
    *,
    graceful_timeout_seconds: float = 2.0,
    final_timeout_seconds: float = 2.0,
) -> None:
    """Stop an owned process group without signalling an unrelated process."""

    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=graceful_timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=final_timeout_seconds)
    except subprocess.TimeoutExpired:
        pass
