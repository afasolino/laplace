from pathlib import Path

import pytest

from research_workspace.ast_context import AstContextError, render_ast_context, tool_definition


class FakeTreeContext:
    def __init__(self, filename: str, code: str, **_kwargs: object) -> None:
        self.code = code
        self.lines: set[int] = set()

    def grep(self, pattern: str, ignore_case: bool) -> set[int]:
        needle = pattern.casefold() if ignore_case else pattern
        return {
            index
            for index, line in enumerate(self.code.splitlines())
            if needle in (line.casefold() if ignore_case else line)
        }

    def add_lines_of_interest(self, lines: set[int]) -> None:
        self.lines = lines

    def add_context(self) -> None:
        return None

    def format(self) -> str:
        return "\n".join(f"{line + 1:3}█match" for line in sorted(self.lines))


def test_ast_context_is_bounded_and_repository_relative(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    value = render_ast_context(
        repo,
        "sample.py",
        "alpha",
        tree_context_factory=FakeTreeContext,
    )
    assert value["provider"] == "grep-ast"
    assert value["path"] == "sample.py"
    assert value["match_count"] == 1
    assert "match" in str(value["context"])


def test_ast_context_refuses_escape_and_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(AstContextError, match="path_escape"):
        render_ast_context(repo, "../outside.py", "x", tree_context_factory=FakeTreeContext)
    (repo / "link.py").symlink_to(outside)
    with pytest.raises(AstContextError, match="symlink_refused"):
        render_ast_context(repo, "link.py", "x", tree_context_factory=FakeTreeContext)


def test_ast_context_schema_is_closed_and_read_only() -> None:
    definition = tool_definition()
    schema = definition["inputSchema"]
    assert isinstance(schema, dict)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"path", "pattern"}
    annotations = definition["annotations"]
    assert isinstance(annotations, dict)
    assert annotations["readOnlyHint"] is True


def test_real_grep_ast_provider_renders_python_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text(
        "class Example:\n    def target(self):\n        return 7\n",
        encoding="utf-8",
    )
    value = render_ast_context(repo, "sample.py", "target")
    assert value["provider"] == "grep-ast"
    assert value["match_count"] == 1
    assert "target" in str(value["context"])
