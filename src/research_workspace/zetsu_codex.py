"""Codex integration managed through Codex's own MCP CLI.

No Codex TOML is parsed or rewritten by this module.  Configuration ownership is
returned to the mature Codex CLI, while the configured stdio command points at
the official-MCP-SDK Laplace bridge.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeAlias

from .zetsu_config import DEFAULT_ENDPOINT
from .zetsu_mcp import MCP_LATEST_PROTOCOL_VERSION
from .zetsu_runtime import default_state_root

JsonObject: TypeAlias = dict[str, object]
Runner: TypeAlias = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class ZetsuCodexError(RuntimeError):
    pass


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _codex() -> str:
    executable = shutil.which("codex")
    if executable is None:
        raise ZetsuCodexError("codex_executable_missing")
    return executable


def _bridge() -> str:
    executable = shutil.which("laplace-zetsu-mcp")
    if executable is None:
        raise ZetsuCodexError("laplace_zetsu_mcp_executable_missing:install_v3_extra")
    return executable


def server_command(
    repository: Path,
    state_root: Path,
    endpoint: str,
    *,
    bridge_executable: str | None = None,
) -> list[str]:
    return [
        bridge_executable or _bridge(),
        "--repo",
        str(repository.expanduser().resolve()),
        "--state-root",
        str(state_root.expanduser().resolve()),
        "--endpoint",
        endpoint,
    ]


def _get(codex: str, runner: Runner) -> subprocess.CompletedProcess[str]:
    return runner([codex, "mcp", "get", "zetsu", "--json"])


def install(
    repository: Path,
    state_root: Path,
    endpoint: str,
    *,
    replace: bool,
    runner: Runner = _run,
    codex_executable: str | None = None,
    bridge_executable: str | None = None,
) -> JsonObject:
    codex = codex_executable or _codex()
    existing = _get(codex, runner)
    if existing.returncode == 0:
        if not replace:
            raise ZetsuCodexError("zetsu_mcp_already_configured:rerun_with_--replace")
        removed = runner([codex, "mcp", "remove", "zetsu"])
        if removed.returncode != 0:
            raise ZetsuCodexError(f"codex_mcp_remove_failed:{removed.stderr.strip()[:300]}")

    command = server_command(
        repository,
        state_root,
        endpoint,
        bridge_executable=bridge_executable,
    )
    added = runner(
        [
            codex,
            "mcp",
            "add",
            "zetsu",
            "--env",
            f"CODEX_MCP_PROTOCOL_VERSION={MCP_LATEST_PROTOCOL_VERSION}",
            "--",
            *command,
        ]
    )
    if added.returncode != 0:
        raise ZetsuCodexError(f"codex_mcp_add_failed:{added.stderr.strip()[:300]}")
    checked = _get(codex, runner)
    if checked.returncode != 0:
        raise ZetsuCodexError("codex_mcp_registration_not_readable")
    return {
        "status": "CONFIGURED",
        "server": "zetsu",
        "transport": "stdio",
        "repository": str(repository.expanduser().resolve()),
        "endpoint": endpoint,
        "credential_storage": "laplace_local_state_or_environment;not_codex_config",
        "codex": checked.stdout.strip(),
    }


def status(
    *,
    repository: Path | None = None,
    runner: Runner = _run,
    codex_executable: str | None = None,
) -> JsonObject:
    codex = codex_executable or _codex()
    result = _get(codex, runner)
    payload: JsonObject = {
        "status": "CONFIGURED" if result.returncode == 0 else "NOT_CONFIGURED",
        "codex": result.stdout.strip() if result.returncode == 0 else result.stderr.strip(),
    }
    if repository is not None:
        payload["repository"] = str(repository.expanduser().resolve())
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-zetsu-codex")
    sub = parser.add_subparsers(dest="command", required=True)
    install_parser = sub.add_parser("install")
    install_parser.add_argument("--repo", type=Path, default=Path.cwd())
    install_parser.add_argument("--state-root", type=Path, default=default_state_root())
    install_parser.add_argument("--endpoint", default=os.environ.get("LAPLACE_ZETSU_ENDPOINT", DEFAULT_ENDPOINT))
    install_parser.add_argument("--replace", action="store_true")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--repo", type=Path, default=Path.cwd())
    sub.add_parser("remove")
    launch = sub.add_parser("launch")
    launch.add_argument("codex_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "install":
            payload = install(
                args.repo,
                args.state_root,
                args.endpoint,
                replace=args.replace,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "status":
            print(json.dumps(status(repository=args.repo), indent=2, sort_keys=True))
            return 0
        codex = _codex()
        if args.command == "remove":
            result = _run([codex, "mcp", "remove", "zetsu"])
            if result.returncode != 0:
                raise ZetsuCodexError(f"codex_mcp_remove_failed:{result.stderr.strip()[:300]}")
            print("zetsu MCP removed from Codex configuration")
            return 0
        if args.command == "launch":
            state = status()
            if state["status"] != "CONFIGURED":
                raise ZetsuCodexError("zetsu_mcp_not_configured:run_install_first")
            forwarded = list(args.codex_args)
            if forwarded[:1] == ["--"]:
                forwarded.pop(0)
            os.execvp(codex, [codex, *forwarded])
            raise AssertionError("os.execvp returned unexpectedly")
        raise ZetsuCodexError("unsupported_command")
    except (ZetsuCodexError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
