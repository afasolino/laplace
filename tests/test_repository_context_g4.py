from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.laplace_core import LaplaceCore
from research_workspace.repository_context import (
    RepositoryContextService,
    RepositoryContextStaleError,
    RepositoryContextValidationError,
)


def _mixed_repository(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/base.py").write_text(
        "class Base:\n    pass\n\ndef helper(value: str) -> str:\n    return value\n",
        encoding="utf-8",
    )
    (root / "pkg/use.py").write_text(
        "from pkg.base import Base\n\ndef run(value: str) -> str:\n    item = Base()\n    return helper(value)\n",
        encoding="utf-8",
    )
    (root / "native.h").write_text("#define NATIVE_FLAG 1\nstruct Native { int value; };\n", encoding="utf-8")
    (root / "native.c").write_text(
        '#include "native.h"\nint native(void) { return NATIVE_FLAG; }\n', encoding="utf-8"
    )
    (root / "child.sv").write_text("module child(input logic clk); endmodule\n", encoding="utf-8")
    (root / "top.sv").write_text(
        "`include \"child.sv\"\nmodule top(input logic clk);\n  child u_child (clk);\nendmodule\n",
        encoding="utf-8",
    )
    (root / "math.cpp").write_text(
        "class Calculator {};\nint calculate(int value) { return value; }\n", encoding="utf-8"
    )


def test_symbols_and_edges_cover_mixed_language_repository(tmp_path: Path) -> None:
    _mixed_repository(tmp_path)
    service = RepositoryContextService(tmp_path)
    index = service.index()

    assert {symbol.name for symbol in index.symbols} >= {
        "Base",
        "helper",
        "run",
        "native",
        "Calculator",
        "calculate",
        "child",
        "top",
        "NATIVE_FLAG",
    }
    import_edges = [edge for edge in index.edges if edge.kind == "import"]
    include_edges = [edge for edge in index.edges if edge.kind == "include"]
    instantiation_edges = [edge for edge in index.edges if edge.kind == "instantiates"]
    assert any(edge.source_path == "pkg/use.py" and edge.target_path == "pkg/base.py" for edge in import_edges)
    assert any(edge.source_path == "native.c" and edge.target_path == "native.h" for edge in include_edges)
    assert any(edge.source_path == "top.sv" and edge.target_path == "child.sv" for edge in include_edges)
    assert any(edge.name == "child" and edge.target_path == "child.sv" for edge in instantiation_edges)
    assert [symbol.name for symbol in service.find_symbol("Base")] == ["Base"]
    assert any(edge.source_path == "pkg/use.py" for edge in service.find_references("Base"))


def test_repo_map_is_bounded_ranked_and_advisory(tmp_path: Path) -> None:
    _mixed_repository(tmp_path)
    service = RepositoryContextService(tmp_path)
    focused = service.build_repo_map(query="child", focus_paths=("top.sv",), token_budget=80)
    repeated = service.build_repo_map(query="child", focus_paths=("top.sv",), token_budget=80)

    assert focused == repeated
    assert focused.estimated_tokens <= 80
    assert focused.snapshot_hash == service.index().snapshot_hash
    assert "# Laplace RepoMap advisory" in focused.text
    assert "top.sv" in focused.text
    assert focused.to_json()["authority"] == "advisory"
    assert focused.to_json()["exact_file_reads_required_for_mutation_or_verification"] is True
    core = LaplaceCore(tmp_path, object(), object(), repository_context=service)  # type: ignore[arg-type]
    assert core.repo_map(query="child", token_budget=80).snapshot_hash == focused.snapshot_hash


def test_edit_rename_add_delete_invalidate_cache_and_reject_stale_context(tmp_path: Path) -> None:
    _mixed_repository(tmp_path)
    service = RepositoryContextService(tmp_path)
    original = service.build_context(query="run", focus_paths=("pkg/use.py",), token_budget=120)
    original_hash = original.snapshot_hash

    use_path = tmp_path / "pkg/use.py"
    use_path.write_text(use_path.read_text(encoding="utf-8").replace("def run", "def execute"), encoding="utf-8")
    with pytest.raises(RepositoryContextStaleError, match="repository_context_stale"):
        service.assert_fresh(original)
    changed = service.index()
    assert changed.snapshot_hash != original_hash
    assert [symbol.name for symbol in service.find_symbol("execute")] == ["execute"]
    assert service.find_symbol("run") == ()

    new_file = tmp_path / "pkg/new.py"
    new_file.write_text("def added() -> None:\n    return None\n", encoding="utf-8")
    assert service.find_symbol("added")[0].path == "pkg/new.py"
    new_file.unlink()
    assert service.find_symbol("added") == ()
    base = tmp_path / "pkg/base.py"
    base.rename(tmp_path / "pkg/renamed.py")
    assert service.find_symbol("Base")[0].path == "pkg/renamed.py"


def test_paths_and_budgets_fail_closed(tmp_path: Path) -> None:
    _mixed_repository(tmp_path)
    service = RepositoryContextService(tmp_path)
    with pytest.raises(RepositoryContextValidationError, match="path_outside_repository"):
        service.build_repo_map(focus_paths=("../outside.py",))
    with pytest.raises(RepositoryContextValidationError, match="invalid_repo_map_token_budget"):
        service.build_repo_map(token_budget=0)
    with pytest.raises(RepositoryContextValidationError, match="invalid_symbol_query"):
        service.find_symbol("not a symbol")
