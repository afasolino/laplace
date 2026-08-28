from __future__ import annotations

import tomllib
from pathlib import Path

from research_workspace.zetsu_codex import server_command

ROOT = Path(__file__).resolve().parents[1]
LAPLACE_BASE = "0c88eb3b748f36701d5246ab5b13b6b384d5f972"
HERMES_REFERENCE = "5cc47c994beb243407bb4c8ba47d2ab421cda9cf"

FORBIDDEN_REPOSITORY_TOOLSETS = (
    "terminal",
    "file",
    "code_execution",
    "debugging",
    "coding",
    "hermes-cli",
    "hermes-acp",
    "hermes-api-server",
    "hermes-cron",
    "all",
    "*",
)


def test_existing_stdio_bridge_command_is_hermes_compatible_and_secret_free(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    bridge = server_command(
        repo,
        state,
        "http://127.0.0.1:8765/mcp",
        bridge_executable="laplace-zetsu-mcp",
    )
    assert bridge == [
        "laplace-zetsu-mcp",
        "--repo",
        str(repo.resolve()),
        "--state-root",
        str(state.resolve()),
        "--endpoint",
        "http://127.0.0.1:8765/mcp",
    ]

    hermes = [
        "hermes",
        "mcp",
        "add",
        "zetsu",
        "--command",
        bridge[0],
        "--args",
        *bridge[1:],
    ]
    assert hermes[6] == "--args"
    assert hermes[-1] == "http://127.0.0.1:8765/mcp"
    lowered = " ".join(hermes).casefold()
    for forbidden in ("--env", "--auth", "--token", "bearer ", "authorization:"):
        assert forbidden not in lowered


def test_laplace_already_installs_stdio_bridge_entrypoint() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]
    assert scripts["laplace-zetsu-mcp"] == "research_workspace.zetsu_sdk_stdio:main"

    bridge_source = (
        ROOT / "src/research_workspace/zetsu_sdk_stdio.py"
    ).read_text(encoding="utf-8")
    assert "load_local_plus_token" in bridge_source


def test_hermes_contract_pins_current_reviewed_revisions() -> None:
    text = (
        ROOT / "docs/integrations/HERMES_ZETSU.md"
    ).read_text(encoding="utf-8")
    assert "Step 3.5 decision: **EXTERNALIZE**." in text
    assert LAPLACE_BASE in text
    assert HERMES_REFERENCE in text
    assert "hermes mcp add zetsu" in text
    assert "`--args` is `argparse.REMAINDER`" in text
    assert "hermes skills list" in text
    assert "hermes skills list --source local" not in text
    assert "hermes chat --continue" in text
    assert "hermes chat --resume <session-id>" in text
    assert "hermes memory status" in text
    assert "hermes cron run" in text
    assert 'cronjob(action="run"' in text
    assert "`mcp-zetsu` as descriptive" in text
    assert "`mcp__zetsu__<tool>`" in text
    assert "Step 3.5 live certification result" in text
    assert "res_ce758d677813d2f0bf0c18e495785fdb" in text
    assert 'enabled_toolsets=["zetsu", "skills"]' in text


def test_governed_session_has_explicit_no_bypass_tool_boundary() -> None:
    text = (
        ROOT / "docs/integrations/HERMES_ZETSU.md"
    ).read_text(encoding="utf-8")
    assert '--toolsets "zetsu,skills"' in text
    assert (
        '--toolsets "zetsu,skills,session_search,memory,cronjob"'
        in text
    )
    assert 'enabled_toolsets=["zetsu", "skills"]' in text
    for toolset in FORBIDDEN_REPOSITORY_TOOLSETS:
        assert toolset in text
    assert (
        '--toolsets "zetsu,skills,session_search,memory,delegation,cronjob"'
        not in text
    )
    assert "Delegation is **not** included in the default governed toolset" in text
    assert "do **not** enable any of:" in text
    assert "Do not use Hermes `-w` / `--worktree`" in text
    assert '--toolsets "mcp-zetsu,skills"' not in text
    assert 'enabled_toolsets=["mcp-zetsu", "skills"]' not in text
    assert "mcp_zetsu_<tool>" not in text


def test_hermes_specific_skill_is_separate_and_fail_closed() -> None:
    hermes_skill = ROOT / "integrations/hermes/laplace-governed/SKILL.md"
    codex_skill = ROOT / ".agents/skills/zetsu/SKILL.md"
    assert hermes_skill.is_file()
    assert codex_skill.is_file()

    hermes_text = hermes_skill.read_text(encoding="utf-8")
    codex_text = codex_skill.read_text(encoding="utf-8")

    assert "name: laplace-governed" in hermes_text
    assert "zetsu,skills" in hermes_text
    assert 'enabled_toolsets=["zetsu", "skills"]' in hermes_text
    assert "Mutation authority is transient" in hermes_text
    assert "child/subagent self-report is not proof" in hermes_text
    assert "version: 0.3.0" in hermes_text
    assert "mcp__zetsu__<tool>" in hermes_text
    assert "persist" in hermes_text and "YOLO mode" in hermes_text
    assert "mcp-zetsu,skills" not in hermes_text
    for toolset in FORBIDDEN_REPOSITORY_TOOLSETS:
        assert toolset in hermes_text

    assert hermes_text != codex_text
    assert "name: laplace-governed" not in codex_text


def test_externalization_adds_no_second_hermes_runtime_or_selector() -> None:
    forbidden_paths = (
        "src/research_workspace/hermes_externalization.py",
        "src/research_workspace/hermes_runtime.py",
        "src/research_workspace/hermes_adapter.py",
        "scripts/v3_5_hermes_externalize.py",
    )
    for relative in forbidden_paths:
        assert not (ROOT / relative).exists()

    zetsu = (
        ROOT / "src/research_workspace/zetsu_agent.py"
    ).read_text(encoding="utf-8")
    assert "HERMES_EXTERNALIZATION" not in zetsu
    assert "hermes_externalization" not in zetsu


def test_contract_does_not_prescribe_credential_or_repo_skill_bypass() -> None:
    text = (
        ROOT / "docs/integrations/HERMES_ZETSU.md"
    ).read_text(encoding="utf-8")
    assert "do not pass bearer" in text.casefold()
    assert 'hermes skills trust "$PWD"' not in text
    assert "--worktree` or a repository cron `workdir`" in text
