from __future__ import annotations

from pathlib import Path

from research_workspace.repository_context import RepositoryContextService


def test_complex_native_and_rtl_constructs_are_explicitly_advisory(tmp_path: Path) -> None:
    (tmp_path / "native.hpp").write_text(
        "#define ENABLED 1\n"
        "namespace demo {\n"
        "template <typename T> class Box { T value; };\n"
        "T\nqualified(\n  T value\n) { return value; }\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "rtl.sv").write_text(
        "`ifdef SYNTHESIS\n"
        "package types; import other_pkg::*; endpackage\n"
        "interface bus(input logic clk); modport master(input clk); endinterface\n"
        "module #(parameter WIDTH=8) top(input logic clk);\n"
        "  child #( .WIDTH(WIDTH) ) u_child ( .clk(clk) );\n"
        "  generate if (WIDTH > 1) begin : gen_bus end endgenerate\n"
        "endmodule\n",
        encoding="utf-8",
    )
    service = RepositoryContextService(tmp_path)
    index = service.index()
    repo_map = service.build_repo_map(query="Box top", token_budget=512).to_json()

    assert {symbol.name for symbol in index.symbols} >= {"ENABLED", "demo", "types", "bus"}
    assert repo_map["authority"] == "advisory"
    assert repo_map["parser_contract"] == "lightweight_advisory_regex_v1"
    assert repo_map["complete_semantic_parsing"] is False
    assert "multiline_c_cpp_declarations" in repo_map["known_unsupported_constructs"]
    assert "rtl_generate_elaboration" in repo_map["known_unsupported_constructs"]
    assert repo_map["exact_file_reads_required_for_mutation_or_verification"] is True


def test_hash_backed_cache_reuses_unchanged_index_and_invalidates_changed_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.c"
    source.write_text("int answer(void) { return 41; }\n", encoding="utf-8")
    service = RepositoryContextService(tmp_path)
    first = service.index()
    second = service.index()
    assert first is second
    assert [symbol.name for symbol in first.symbols] == ["answer"]
    source.write_text("int answer(void) { return 42; }\n", encoding="utf-8")
    changed = service.index()
    assert changed is not first
    assert changed.snapshot_hash != first.snapshot_hash
    assert changed.files[0].source_sha256 != first.files[0].source_sha256
