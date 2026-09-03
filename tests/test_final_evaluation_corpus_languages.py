from __future__ import annotations

from research_workspace.personal_corpus import PersonalCorpusPolicy, _support_label


def test_c_and_cpp_extensions_are_accepted() -> None:
    accepted = set(PersonalCorpusPolicy().public()["accepted_extensions"])
    assert {".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ipp"} <= accepted
    assert _support_label("src/kernel.c") == "c_reference"
    assert _support_label("src/kernel.cpp") == "cpp_reference"
    assert _support_label("include/kernel.hpp") == "cpp_reference"
