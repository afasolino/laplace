from __future__ import annotations

import json
from pathlib import Path

from research_workspace.maintenance_cli import main


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_shadow_maintenance_cli_requires_frozen_evidence_and_human_approval(
    tmp_path: Path, capsys
) -> None:
    state = tmp_path / "state"
    events = _write(
        tmp_path / "events.json",
        [
            {"event_id": "evt-one", "event_sequence": 1, "event_type": "failure", "payload": {"category": "repeat"}},
            {"event_id": "evt-two", "event_sequence": 2, "event_type": "failure", "payload": {"category": "repeat"}},
        ],
    )
    base = [
        "--state-root",
        str(state),
    ]
    assert main(
        [
            *base,
            "run",
            "--cycle-id",
            "cycle-one",
            "--owner-id",
            "owner",
            "--project-id",
            "project",
            "--session-id",
            "session",
            "--events-json",
            str(events),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            *base,
            "propose-harness",
            "--cycle-id",
            "cycle-one",
            "--owner-id",
            "owner",
            "--project-id",
            "project",
            "--session-id",
            "session",
            "--description",
            "Add a regression for the repeat failure.",
            "--source-event-id",
            "evt-one",
            "--source-event-id",
            "evt-two",
        ]
    ) == 0
    proposal = json.loads(capsys.readouterr().out)
    proposal_id = proposal["proposal_id"]
    evidence = _write(
        tmp_path / "evidence.json",
        {
            "baseline_result_sha256": "a" * 64,
            "candidate_result_sha256": "b" * 64,
            "frozen_task_ids": ["frozen-one"],
            "development_task_ids": ["development-one"],
            "held_out_task_ids": ["heldout-one"],
            "baseline_correct": True,
            "candidate_correct": True,
            "security_regression": False,
            "correctness_regression": False,
            "observed_at_utc": "2026-08-26T00:00:00Z",
        },
    )
    assert main([*base, "record-ab", "--proposal-id", proposal_id, "--evidence-json", str(evidence)]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "awaiting_human_approval"
    assert main([*base, "approve", "--proposal-id", proposal_id, "--approver-id", "reviewer"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "human_approved_shadow_only"
    assert main([*base, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["mode"] == "SHADOW"
    assert status["active_production_mutations"] is False
    assert status["auto_promotion"] is False


def test_maintenance_cli_rejects_malformed_evidence_without_approval(tmp_path: Path, capsys) -> None:
    evidence = _write(tmp_path / "evidence.json", {"unexpected": True})
    assert main(
        [
            "--state-root",
            str(tmp_path / "state"),
            "record-ab",
            "--proposal-id",
            "proposal-one",
            "--evidence-json",
            str(evidence),
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out)["failure_category"] == "maintenance_ab_evidence_invalid"
