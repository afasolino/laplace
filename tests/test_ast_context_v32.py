from pathlib import Path

import pytest

from research_workspace.ast_context import AstContextError, render_ast_context


def test_real_grep_ast_go_and_no_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.go").write_text(
        "package main\nfunc target() int { return 7 }\n", encoding="utf-8"
    )
    value = render_ast_context(repo, "sample.go", "target")
    assert value["provider"] == "grep-ast"
    assert value["match_count"] == 1
    no_match = render_ast_context(repo, "sample.go", "does_not_exist")
    assert no_match["match_count"] == 0


def test_real_grep_ast_unsupported_language_is_bounded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.v32unsupported").write_text("needle\n", encoding="utf-8")
    with pytest.raises(AstContextError, match=r"^ast_context_upstream_error:"):
        render_ast_context(repo, "sample.v32unsupported", "needle")
