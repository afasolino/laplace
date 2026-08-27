import subprocess
from pathlib import Path

import pytest

from research_workspace.zetsu_codex import ZetsuCodexError, install, server_command


def completed(argv: list[str], returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def test_server_command_contains_no_bearer_secret(tmp_path: Path) -> None:
    command = server_command(
        tmp_path / "repo",
        tmp_path / "state",
        "http://127.0.0.1:8765/mcp",
        bridge_executable="/venv/bin/laplace-zetsu-mcp",
    )
    joined = " ".join(command)
    assert "--repo" in command
    assert "--state-root" in command
    assert "Bearer" not in joined
    assert "TOKEN=" not in joined


def test_install_uses_codex_native_mcp_cli_and_replace(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv):  # type: ignore[no-untyped-def]
        value = list(argv)
        calls.append(value)
        if value[1:4] == ["mcp", "get", "zetsu"] and len(calls) == 1:
            return completed(value, stdout='{"name":"zetsu"}')
        if value[1:4] == ["mcp", "get", "zetsu"]:
            return completed(value, stdout='{"name":"zetsu","transport":"stdio"}')
        return completed(value)

    value = install(
        tmp_path / "repo",
        tmp_path / "state",
        "http://127.0.0.1:8765/mcp",
        replace=True,
        runner=runner,
        codex_executable="/usr/bin/codex",
        bridge_executable="/venv/bin/laplace-zetsu-mcp",
    )
    assert value["transport"] == "stdio"
    assert calls[1] == ["/usr/bin/codex", "mcp", "remove", "zetsu"]
    assert calls[2][:4] == ["/usr/bin/codex", "mcp", "add", "zetsu"]
    assert "--env" in calls[2]
    assert "CODEX_MCP_PROTOCOL_VERSION=2026-07-28" in calls[2]
    assert "--" in calls[2]
    assert "LAPLACE_ZETSU_TOKEN" not in " ".join(calls[2])


def test_install_refuses_overwrite_without_replace(tmp_path: Path) -> None:
    def runner(argv):  # type: ignore[no-untyped-def]
        return completed(list(argv), stdout='{"name":"zetsu"}')

    with pytest.raises(ZetsuCodexError, match="already_configured"):
        install(
            tmp_path / "repo",
            tmp_path / "state",
            "http://127.0.0.1:8765/mcp",
            replace=False,
            runner=runner,
            codex_executable="codex",
            bridge_executable="laplace-zetsu-mcp",
        )
