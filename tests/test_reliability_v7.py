from __future__ import annotations

from research_workspace.reliability import (
    FAILURE_SCENARIOS,
    GPU_BLOCKED_STATUS,
    SOAK_SCENARIOS,
    run_cpu_soak,
    run_failure_matrix,
)


def test_bounded_cpu_soak_covers_required_scenarios_and_cleans_up() -> None:
    report = run_cpu_soak(iterations=16, max_seconds=10)
    assert report.status == "PASS"
    assert report.fixture_root_removed is True
    assert report.external_network_used is False
    assert report.listeners_opened == ()
    assert report.gpu_status == GPU_BLOCKED_STATUS
    assert {result.scenario for result in report.results} == set(SOAK_SCENARIOS)
    assert all(result.status == "PASS" for result in report.results)


def test_failure_matrix_covers_required_scenarios_and_cleans_up() -> None:
    report = run_failure_matrix(max_seconds=10)
    assert report.status == "PASS"
    assert report.fixture_root_removed is True
    assert report.production_state_touched is False
    assert report.gpu_status == GPU_BLOCKED_STATUS
    assert {result.scenario for result in report.results} == set(FAILURE_SCENARIOS)
    assert all(result.status == "PASS" for result in report.results)

