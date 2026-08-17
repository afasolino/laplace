from __future__ import annotations

from dataclasses import dataclass

import pytest

from research_workspace.zetsu_context import compact_evidence, normalize_budget


@dataclass
class Evidence:
    filename: str
    page: int | None
    section: str | None
    chunk_id: str
    text: str
    score: float
    document_class: str = "document"


def test_compact_evidence_assigns_stable_ids_and_budget() -> None:
    items = [
        Evidence("one.pdf", 1, "A", "chunk-1", "alpha " * 300, 0.9),
        Evidence("two.pdf", 2, "B", "chunk-2", "beta " * 300, 0.8),
    ]

    packet = compact_evidence("query", items, max_results=2, max_chars=700)

    assert packet.grounded
    assert packet.evidence[0].evidence_id == "E1"
    assert packet.evidence[0].chunk_id == "chunk-1"
    assert packet.truncated
    assert len(packet.evidence[0].excerpt) < 700


def test_compact_evidence_honors_result_limit() -> None:
    items = [Evidence(f"{i}.txt", None, None, f"c{i}", "text", 1.0) for i in range(3)]
    packet = compact_evidence("q", items, max_results=1, max_chars=1000)
    assert len(packet.evidence) == 1
    assert packet.truncated


def test_budget_bounds() -> None:
    with pytest.raises(ValueError):
        normalize_budget(511)
    with pytest.raises(ValueError):
        normalize_budget(24001)
