"""Apply the non-A6000 certification taxonomy to collected tests."""

from __future__ import annotations

import pytest

from research_workspace.certification_taxonomy import category_for_nodeid


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply one explicit platform category to every collected test."""

    del config
    for item in items:
        category = category_for_nodeid(item.nodeid)
        item.add_marker(getattr(pytest.mark, category))
