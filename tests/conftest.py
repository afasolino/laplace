"""Apply the non-A6000 certification taxonomy to collected tests."""

from __future__ import annotations

import pytest

from research_workspace.certification_taxonomy import category_for_nodeid


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply one explicit platform category to every collected test."""

    del config
    for item in items:
        try:
            category = category_for_nodeid(item.nodeid)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        declared = {marker.name for marker in item.own_markers if marker.name in {
            "cross_platform_deterministic", "linux_posix_required", "interactive_e2e",
            "optional_dependency", "gpu_smoke", "a6000_required", "external_live",
            "windows_privilege_required",
        }}
        if declared and declared != {category}:
            raise pytest.UsageError(f"multiple_test_classifications:{item.nodeid}")
        item.add_marker(getattr(pytest.mark, category))
