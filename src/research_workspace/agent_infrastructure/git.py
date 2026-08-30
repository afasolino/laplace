"""Bounded fixed-argv Git execution shared by repository-agent components."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # nosec B404 - callers supply fixed Git argument sequences
import time
from collections.abc import Callable, Sequence
from pathlib import Path

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class GitExecutionError(RuntimeError):
    """A bounded Git process could not be started, completed, or validated."""


def run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    runner: CommandRunner = subprocess.run,
    timeout_seconds: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Run Git without a shell and return its captured outcome unchanged."""

    return runner(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1.0, timeout_seconds),
    )


def run_git_bounded(
    repository_root: Path, command: Sequence[str], *, timeout_seconds: int = 60
) -> tuple[int, str, str]:
    """Run one fixed Git command with bounded process-group termination."""

    started = time.monotonic()
    git = shutil.which("git")
    if git is None:
        raise GitExecutionError("Git executable is unavailable")
    try:
        process = subprocess.Popen(  # nosec B603 - callers pass fixed Git argv
            [git, *command],
            cwd=repository_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise GitExecutionError(f"Cannot start git: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        raise GitExecutionError(
            f"Git command timed out after {time.monotonic() - started:.1f}s: {stderr}"
        )
    if process.returncode != 0:
        raise GitExecutionError(f"Git command failed: {stderr[-2000:]}")
    return process.returncode, stdout, stderr
