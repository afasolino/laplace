from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from research_workspace.zetsu_config import ZetsuConfigError, configure, remove, status


def test_configure_is_idempotent_and_preserves_other_codex_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    config = tmp_path / "codex-home" / "config.toml"
    config.parent.mkdir()
    config.write_text('[mcp_servers.other]\nurl = "https://other.example/mcp"\n', encoding="utf-8")

    first = configure(repo, endpoint="https://laplace.example/mcp")
    second = configure(repo, endpoint="https://laplace.example/mcp")

    text = config.read_text(encoding="utf-8")
    assert text.count("BEGIN LAPLACE ZETSU MANAGED") == 1
    assert "[mcp_servers.other]" in text
    assert 'url = "https://other.example/mcp"' in text
    assert "[mcp_servers.zetsu]" in text
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["zetsu"]["default_tools_approval_mode"] == "writes"
    assert parsed["mcp_servers"]["zetsu"]["enabled_tools"] == [
        "search",
        "get_evidence",
        "project_context",
        "experiment_context",
        "delegate",
        "agent_task",
        "rtl_task",
        "verify",
    ]
    assert first == second
    assert first.configured
    assert first.skill_installed
    installed_skill = (repo / ".agents/skills/zetsu/SKILL.md").read_text(encoding="utf-8")
    assert "must choose `verification_argv` before delegation" in installed_skill
    assert "direct\n`pytest`, `ruff`, or `mypy` executable" in installed_skill
    assert "never wrap it with\n`python -m`" in installed_skill


def test_status_reads_values_only_from_managed_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    config = tmp_path / "codex-home" / "config.toml"
    config.parent.mkdir()
    config.write_text('[mcp_servers.other]\nurl = "https://wrong.example/mcp"\n', encoding="utf-8")
    configure(repo, endpoint="https://right.example/mcp", token_env_var="RIGHT_TOKEN")

    value = status(repo)

    assert value.endpoint == "https://right.example/mcp"
    assert value.token_env_var == "RIGHT_TOKEN"


def test_remove_preserves_other_codex_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    config = tmp_path / "codex-home" / "config.toml"
    config.parent.mkdir()
    config.write_text("[features]\nfoo = true\n", encoding="utf-8")
    configure(repo, endpoint="https://laplace.example/mcp")

    value = remove(repo)

    assert not value.configured
    assert not value.skill_installed
    assert config.read_text(encoding="utf-8") == "[features]\nfoo = true\n"


def test_configure_rejects_non_https_remote_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    with pytest.raises(ZetsuConfigError, match="zetsu_endpoint_must_use_https_or_loopback"):
        configure(repo, endpoint="http://laplace.example/mcp")


def test_configure_does_not_overwrite_user_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    skill = repo / ".agents" / "skills" / "zetsu" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("user owned\n", encoding="utf-8")
    with pytest.raises(ZetsuConfigError, match="zetsu_skill_path_owned_by_user"):
        configure(repo, endpoint="https://laplace.example/mcp")


def test_configure_refuses_preexisting_unmanaged_zetsu_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    config = tmp_path / "codex-home" / "config.toml"
    config.parent.mkdir()
    original = '[mcp_servers.zetsu]\nurl = "https://user.example/mcp"\n'
    config.write_text(original, encoding="utf-8")

    with pytest.raises(ZetsuConfigError, match="zetsu_codex_config_owned_by_user"):
        configure(repo, endpoint="https://laplace.example/mcp")

    assert config.read_text(encoding="utf-8") == original


def test_configure_failure_is_non_destructive_when_skill_is_user_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    config = tmp_path / "codex-home" / "config.toml"
    config.parent.mkdir()
    config.write_text("[features]\nfoo = true\n", encoding="utf-8")
    skill = repo / ".agents" / "skills" / "zetsu" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("user owned\n", encoding="utf-8")

    with pytest.raises(ZetsuConfigError, match="zetsu_skill_path_owned_by_user"):
        configure(repo, endpoint="https://laplace.example/mcp")

    assert config.read_text(encoding="utf-8") == "[features]\nfoo = true\n"


def test_configure_repairs_owned_v1_configuration_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    config = codex_home / "config.toml"
    config.parent.mkdir()
    config.write_text(
        "# BEGIN LAPLACE ZETSU MANAGED v1\n"
        "[mcp_servers.zetsu]\n"
        'url = "https://stale.example/mcp"\n'
        'bearer_token_env_var = "OLD_TOKEN"\n'
        "# END LAPLACE ZETSU MANAGED v1\n",
        encoding="utf-8",
    )
    skill = repo / ".agents/skills/zetsu/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("<!-- managed-by: laplace-zetsu v1 -->\nstale\n", encoding="utf-8")

    repaired = configure(
        repo,
        endpoint="https://laplace.example/mcp",
        token_env_var="LAPLACE_ZETSU_TOKEN",
    )

    assert repaired.compatible
    assert config.read_text(encoding="utf-8").count("MANAGED v4") == 2
    assert "stale.example" not in config.read_text(encoding="utf-8")
    assert "managed-by: laplace-zetsu v4" in skill.read_text(encoding="utf-8")
