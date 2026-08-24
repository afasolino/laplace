from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.idle_consolidation import (
    ABEvidence,
    ConsolidationBudgetError,
    ConsolidationConflictError,
    ConsolidationCorruptionError,
    ConsolidationError,
    IdleConsolidator,
    MaintenanceBudget,
    ProposalKind,
)
from research_workspace.laplace_core import LaplaceCore


def _event(event_id: str, sequence: int, event_type: str, **payload: object) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_sequence": sequence,
        "event_type": event_type,
        "payload": payload,
    }


def _memory(
    memory_id: str,
    content: str,
    *,
    contradiction_key: str | None = None,
    updated_at_utc: str = "2020-01-01T00:00:00Z",
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "owner_id": "user-a",
        "project_id": "project-a",
        "kind": "episodic",
        "content": content,
        "state": "ACTIVE",
        "updated_at_utc": updated_at_utc,
        "contradiction_key": contradiction_key,
    }


def _evidence(*, security_regression: bool = False) -> ABEvidence:
    return ABEvidence(
        baseline_result_sha256="a" * 64,
        candidate_result_sha256="b" * 64,
        frozen_task_ids=("frozen-1",),
        development_task_ids=("development-1",),
        held_out_task_ids=("heldout-1",),
        baseline_correct=True,
        candidate_correct=True,
        security_regression=security_regression,
        observed_at_utc="2026-08-24T20:00:00Z",
    )


def test_controlled_cycles_propose_all_categories_and_restart_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "consolidation"
    service = IdleConsolidator(root, budget=MaintenanceBudget(min_interval_seconds=60))
    events = [
        _event("evt-1", 1, "failure", category="timeout"),
        _event("evt-2", 2, "failure", category="timeout"),
        _event("evt-3", 3, "completion", fact="the deterministic gate passed"),
    ]
    memories = [
        _memory("mem-1", "old result", contradiction_key="result"),
        _memory("mem-2", "new result", contradiction_key="result"),
        _memory("mem-3", "duplicate result"),
        _memory("mem-4", "duplicate result"),
    ]

    report = service.run_cycle(
        "cycle-1",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=events,
        memories=memories,
        window_id="window-1",
        now_utc="2026-08-24T20:00:00Z",
    )
    kinds = {proposal.kind for proposal in report.proposals}
    assert kinds == {
        ProposalKind.EPISODIC_SUMMARY,
        ProposalKind.DURABLE_FACT,
        ProposalKind.RECURRING_FAILURE,
        ProposalKind.CANDIDATE_SKILL,
        ProposalKind.PROCESS_IMPROVEMENT,
        ProposalKind.CODE_CHANGE,
        ProposalKind.CONTRADICTION,
        ProposalKind.OBSOLETE_MEMORY,
    }
    assert report.shadow_only
    assert all(proposal.active is False and proposal.requires_human_approval for proposal in report.proposals)
    assert memories[0]["state"] == "ACTIVE"
    assert service.run_cycle(
        "cycle-1",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=events,
        memories=memories,
        window_id="window-1",
        now_utc="2026-08-24T20:00:01Z",
    ).replayed

    restarted = IdleConsolidator(root, budget=MaintenanceBudget(min_interval_seconds=60))
    assert restarted.maintenance_status()["active_production_mutations"] is False
    assert len(restarted.proposals(owner_id="user-a", project_id="project-a")) == len(report.proposals)
    duplicate_report = restarted.run_cycle(
        "cycle-2",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=events,
        memories=memories,
        window_id="window-2",
        now_utc="2026-08-24T20:01:01Z",
    )
    assert duplicate_report.proposals == ()
    assert duplicate_report.cycle.duplicate_proposal_ids


def test_crash_before_commit_is_atomic_and_restartable(tmp_path: Path) -> None:
    stages: list[str] = []

    def crash(stage: str) -> None:
        stages.append(stage)
        if stage == "before_commit":
            raise RuntimeError("simulated maintenance crash")

    service = IdleConsolidator(tmp_path / "state", failure_hook=crash)
    with pytest.raises(RuntimeError, match="simulated"):
        service.run_cycle(
            "cycle-crash",
            owner_id="user-a",
            project_id="project-a",
            session_id="session-a",
            trajectory_events=[_event("evt-1", 1, "completion", fact="x")],
            window_id="window-crash",
            now_utc="2026-08-24T20:00:00Z",
        )
    assert stages == ["after_input_validation", "after_analysis", "before_commit"]
    assert not (tmp_path / "state" / "consolidation.json").exists()
    restarted = IdleConsolidator(tmp_path / "state")
    assert not restarted.maintenance_status()["cycles"]
    assert restarted.run_cycle(
        "cycle-crash",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=[_event("evt-1", 1, "completion", fact="x")],
        window_id="window-crash",
        now_utc="2026-08-24T20:00:00Z",
    ).cycle.status == "SHADOW"


def test_ab_evidence_requires_security_and_correctness_then_human_approval(tmp_path: Path) -> None:
    service = IdleConsolidator(tmp_path / "state")
    service.run_cycle(
        "cycle-ab",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=[_event("evt-1", 1, "failure", category="repeat"), _event("evt-2", 2, "failure", category="repeat")],
        window_id="window-ab",
        now_utc="2026-08-24T20:00:00Z",
    )
    candidate = service.propose_harness_improvement(
        "cycle-ab",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        description="Tighten the deterministic frozen verification gate.",
        source_event_ids=("evt-1", "evt-2"),
        now_utc="2026-08-24T20:00:01Z",
    )
    assert candidate.details["evaluation_contract"]
    with pytest.raises(ConsolidationConflictError):
        service.propose_harness_improvement(
            "cycle-ab",
            owner_id="user-a",
            project_id="project-a",
            session_id="session-a",
            description="A second candidate is not allowed.",
        )
    evidence = service.record_ab_evidence(candidate.proposal_id, _evidence())
    assert evidence["decision"] == "awaiting_human_approval"
    approved = service.approve_improvement(candidate.proposal_id, approver_id="human-reviewer")
    assert approved["decision"] == "human_approved_shadow_only"
    assert service.maintenance_status()["active_production_mutations"] is False

    rejected_service = IdleConsolidator(tmp_path / "rejected")
    rejected_report = rejected_service.run_cycle(
        "cycle-rejected",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=[_event("evt-1", 1, "failure", category="repeat"), _event("evt-2", 2, "failure", category="repeat")],
        window_id="window-rejected",
        now_utc="2026-08-24T20:00:00Z",
    )
    rejected_candidate = next(proposal for proposal in rejected_report.proposals if proposal.kind is ProposalKind.CODE_CHANGE)
    assert rejected_service.record_ab_evidence(rejected_candidate.proposal_id, _evidence(security_regression=True))["decision"] == "rejected_correctness_or_security_regression"
    with pytest.raises(ConsolidationConflictError):
        rejected_service.approve_improvement(rejected_candidate.proposal_id, approver_id="human-reviewer")


def test_bounds_scope_and_core_integration(tmp_path: Path) -> None:
    bounded = IdleConsolidator(tmp_path / "bounded", budget=MaintenanceBudget(max_trajectory_events=1))
    with pytest.raises(ConsolidationBudgetError):
        bounded.run_cycle(
            "cycle-limit",
            owner_id="user-a",
            project_id="project-a",
            session_id="session-a",
            trajectory_events=[_event("evt-1", 1, "completion"), _event("evt-2", 2, "completion")],
        )
    with pytest.raises(ConsolidationConflictError):
        IdleConsolidator(tmp_path / "scope").run_cycle(
            "cycle-scope",
            owner_id="user-a",
            project_id="project-a",
            session_id="session-a",
            memories=[{**_memory("mem-1", "private"), "owner_id": "other-user"}],
        )
    core = LaplaceCore(tmp_path / "core", object(), object())  # type: ignore[arg-type]
    report = core.run_idle_consolidation(
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        cycle_id="core-cycle",
        trajectory_events=[_event("evt-1", 1, "completion", fact="core")],
        now_utc="2026-08-24T20:00:00Z",
    )
    assert report.shadow_only
    assert core.consolidation is core.consolidation


def test_replay_conflict_and_malformed_state_fail_closed(tmp_path: Path) -> None:
    service = IdleConsolidator(tmp_path / "state")
    service.run_cycle(
        "cycle-replay",
        owner_id="user-a",
        project_id="project-a",
        session_id="session-a",
        trajectory_events=[_event("evt-1", 1, "completion")],
        now_utc="2026-08-24T20:00:00Z",
    )
    with pytest.raises(ConsolidationConflictError):
        service.run_cycle(
            "cycle-replay",
            owner_id="user-a",
            project_id="project-a",
            session_id="session-a",
            trajectory_events=[_event("evt-2", 1, "completion")],
            now_utc="2026-08-24T20:00:01Z",
        )
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "consolidation.json").write_text(
        json.dumps({"schema_version": 1, "revision": 0, "cycles": {}, "proposals": {"bad": {}}, "windows": {}, "improvements": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ConsolidationCorruptionError):
        IdleConsolidator(unsafe)
    with pytest.raises(ConsolidationError):
        IdleConsolidator(tmp_path / "sensitive").run_cycle(
            "cycle-sensitive",
            owner_id="user-a",
            project_id="project-a",
            session_id="session-a",
            trajectory_events=[_event("evt-sensitive", 1, "completion", token="must not persist")],
        )
