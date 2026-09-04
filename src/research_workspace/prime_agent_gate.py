"""Advisory Prime Agent quality gate over a caller-normalized verification plan.

This process is intentionally not authoritative. Prime Agent may use it to obtain
fast repair feedback, but Zetsu reruns the bound verifier through its existing
candidate-assurance path before a result can be promoted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404 - argv comes from a Laplace-normalized plan
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
_MAX_CAPTURE_BYTES = 32 * 1024
_MAX_PLAN_BYTES = 256 * 1024


class PrimeAgentGateError(RuntimeError):
    """The advisory gate specification is malformed or unsafe."""


def _load(path: Path) -> JsonObject:
    try:
        if path.stat().st_size > _MAX_PLAN_BYTES:
            raise PrimeAgentGateError("prime_gate_spec_too_large")
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimeAgentGateError("prime_gate_spec_invalid") from exc
    if not isinstance(raw, dict):
        raise PrimeAgentGateError("prime_gate_spec_invalid")
    return cast(JsonObject, raw)


def _inside(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise PrimeAgentGateError("prime_gate_cwd_invalid")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PrimeAgentGateError("prime_gate_cwd_escape") from exc
    if not candidate.is_dir():
        raise PrimeAgentGateError("prime_gate_cwd_invalid")
    return candidate


def _argv(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 64
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > 1_000
            or "\x00" in item
            for item in value
        )
    ):
        raise PrimeAgentGateError("prime_gate_argv_invalid")
    return tuple(cast(list[str], value))


def _environment(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PrimeAgentGateError("prime_gate_environment_invalid")
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or "\x00" in key
            or "\x00" in item
        ):
            raise PrimeAgentGateError("prime_gate_environment_invalid")
        result[key] = item
    return result


def _scratch_root(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PrimeAgentGateError("prime_gate_scratch_invalid")
    path = Path(value).resolve()
    if not path.is_dir():
        raise PrimeAgentGateError("prime_gate_scratch_invalid")
    return path




def _validate_workspace_symlinks(root: Path) -> None:
    """Reject absolute or escaping links before making the advisory verifier copy."""

    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            raw_target = Path(os.readlink(path))
        except OSError as exc:
            raise PrimeAgentGateError("prime_gate_symlink_unreadable") from exc
        if raw_target.is_absolute():
            raise PrimeAgentGateError("prime_gate_absolute_symlink_forbidden")
        resolved_target = (path.parent / raw_target).resolve(strict=False)
        try:
            resolved_target.relative_to(root)
        except ValueError as exc:
            raise PrimeAgentGateError("prime_gate_symlink_escape") from exc


def run_gate(spec_path: Path) -> int:
    spec = _load(spec_path.resolve())
    root_raw = spec.get("workspace")
    if not isinstance(root_raw, str):
        raise PrimeAgentGateError("prime_gate_workspace_invalid")
    root = Path(root_raw).resolve()
    if not root.is_dir():
        raise PrimeAgentGateError("prime_gate_workspace_invalid")
    plan = spec.get("plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 16:
        raise PrimeAgentGateError("prime_gate_plan_invalid")
    env = _environment(spec.get("environment"))
    scratch = _scratch_root(spec.get("scratch_root"))
    _validate_workspace_symlinks(root)

    # Prime's gate is advisory only. Run it against a disposable copy so a test
    # that writes caches or other files can never become part of the candidate.
    with tempfile.TemporaryDirectory(prefix="prime-gate-", dir=scratch) as temporary:
        clone = Path(temporary) / "workspace"
        shutil.copytree(
            root,
            clone,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        for index, raw_step in enumerate(plan, start=1):
            if not isinstance(raw_step, Mapping):
                raise PrimeAgentGateError("prime_gate_plan_invalid")
            cwd = _inside(clone, raw_step.get("cwd", "."))
            argv = _argv(raw_step.get("argv"))
            try:
                completed = subprocess.run(  # nosec B603
                    list(argv),
                    cwd=cwd,
                    env=env,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"prime advisory gate step {index}: launch failed: {type(exc).__name__}")
                return 2
            stdout = completed.stdout[-_MAX_CAPTURE_BYTES:].decode("utf-8", errors="replace")
            stderr = completed.stderr[-_MAX_CAPTURE_BYTES:].decode("utf-8", errors="replace")
            if stdout:
                print(stdout, end="" if stdout.endswith("\n") else "\n")
            if stderr:
                print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
            if completed.returncode != 0:
                return completed.returncode if 0 < completed.returncode < 126 else 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-prime-agent-gate")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return run_gate(args.spec)
    except PrimeAgentGateError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
