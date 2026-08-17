from __future__ import annotations

import json
from pathlib import Path

from research_workspace import zetsu_cli


def test_zetsu_cli_offline_lifecycle(tmp_path: Path, capsys, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    assert (
        zetsu_cli.main(
            ["configure", "--repo", str(repo), "--endpoint", "https://laplace.example/mcp", "--json"]
        )
        == 0
    )
    configured = json.loads(capsys.readouterr().out)
    assert configured["configured"] is True

    assert zetsu_cli.main(["test", "--repo", str(repo), "--offline", "--json"]) == 0
    tested = json.loads(capsys.readouterr().out)
    assert tested["ok"] is True
    assert tested["detail"] == "offline"

    assert zetsu_cli.main(["remove", "--repo", str(repo), "--json"]) == 0
    removed = json.loads(capsys.readouterr().out)
    assert removed["configured"] is False
