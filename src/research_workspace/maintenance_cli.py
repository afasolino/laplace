"""Local, shadow-only maintenance CLI for governed Laplace improvement proposals."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from .idle_consolidation import (
    ABEvidence,
    ConsolidationError,
    IdleConsolidator,
    ProposalState,
)

JsonObject: TypeAlias = dict[str, object]


def _object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsolidationError("maintenance_json_input_invalid") from exc
    if not isinstance(value, Mapping):
        raise ConsolidationError("maintenance_json_input_invalid")
    return {str(key): item for key, item in value.items()}


def _array(path: Path) -> list[JsonObject]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsolidationError("maintenance_json_input_invalid") from exc
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ConsolidationError("maintenance_json_input_invalid")
    return [
        {str(key): item for key, item in record.items()}
        for record in value
    ]


def _evidence(path: Path) -> ABEvidence:
    value = _object(path)
    expected = {
        "baseline_result_sha256",
        "candidate_result_sha256",
        "frozen_task_ids",
        "development_task_ids",
        "held_out_task_ids",
        "baseline_correct",
        "candidate_correct",
        "security_regression",
        "correctness_regression",
        "observed_at_utc",
    }
    if set(value) != expected:
        raise ConsolidationError("maintenance_ab_evidence_invalid")
    task_ids: dict[str, tuple[str, ...]] = {}
    for name in ("frozen_task_ids", "development_task_ids", "held_out_task_ids"):
        raw = value[name]
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ConsolidationError("maintenance_ab_evidence_invalid")
        task_ids[name] = tuple(raw)
    flags: dict[str, bool] = {}
    for name in (
        "baseline_correct",
        "candidate_correct",
        "security_regression",
        "correctness_regression",
    ):
        raw = value[name]
        if not isinstance(raw, bool):
            raise ConsolidationError("maintenance_ab_evidence_invalid")
        flags[name] = raw
    return ABEvidence(
        baseline_result_sha256=str(value["baseline_result_sha256"]),
        candidate_result_sha256=str(value["candidate_result_sha256"]),
        frozen_task_ids=task_ids["frozen_task_ids"],
        development_task_ids=task_ids["development_task_ids"],
        held_out_task_ids=task_ids["held_out_task_ids"],
        baseline_correct=flags["baseline_correct"],
        candidate_correct=flags["candidate_correct"],
        security_regression=flags["security_regression"],
        correctness_regression=flags["correctness_regression"],
        observed_at_utc=str(value["observed_at_utc"]),
    )


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laplace-maintenance",
        description=(
            "Create and review shadow-only maintenance proposals. This command never "
            "edits source, executes a model, activates skills, or promotes a change."
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.cwd() / ".laplace-state" / "consolidation",
        help="Local durable maintenance state; defaults below the current repository.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")

    run = commands.add_parser("run")
    run.add_argument("--cycle-id", required=True)
    run.add_argument("--owner-id", required=True)
    run.add_argument("--project-id", required=True)
    run.add_argument("--session-id", required=True)
    run.add_argument("--events-json", type=Path, required=True)
    run.add_argument("--memories-json", type=Path)
    run.add_argument("--window-id", default="default")

    proposals = commands.add_parser("proposals")
    proposals.add_argument("--owner-id", required=True)
    proposals.add_argument("--project-id", required=True)
    proposals.add_argument("--state", choices=tuple(state.value for state in ProposalState))

    propose = commands.add_parser("propose-harness")
    propose.add_argument("--cycle-id", required=True)
    propose.add_argument("--owner-id", required=True)
    propose.add_argument("--project-id", required=True)
    propose.add_argument("--session-id", required=True)
    propose.add_argument("--description", required=True)
    propose.add_argument("--source-event-id", action="append", default=[])
    propose.add_argument("--source-memory-id", action="append", default=[])

    record = commands.add_parser("record-ab")
    record.add_argument("--proposal-id", required=True)
    record.add_argument("--evidence-json", type=Path, required=True)

    approve = commands.add_parser("approve")
    approve.add_argument("--proposal-id", required=True)
    approve.add_argument("--approver-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    service = IdleConsolidator(arguments.state_root)
    try:
        if arguments.command == "status":
            result: object = service.maintenance_status()
        elif arguments.command == "run":
            result = service.run_cycle(
                arguments.cycle_id,
                owner_id=arguments.owner_id,
                project_id=arguments.project_id,
                session_id=arguments.session_id,
                trajectory_events=_array(arguments.events_json),
                memories=(_array(arguments.memories_json) if arguments.memories_json else ()),
                window_id=arguments.window_id,
            ).to_json()
        elif arguments.command == "proposals":
            state = ProposalState(arguments.state) if arguments.state else None
            result = {
                "mode": "SHADOW",
                "proposals": [
                    proposal.to_json()
                    for proposal in service.proposals(
                        owner_id=arguments.owner_id,
                        project_id=arguments.project_id,
                        state=state,
                    )
                ],
            }
        elif arguments.command == "propose-harness":
            result = service.propose_harness_improvement(
                arguments.cycle_id,
                owner_id=arguments.owner_id,
                project_id=arguments.project_id,
                session_id=arguments.session_id,
                description=arguments.description,
                source_event_ids=arguments.source_event_id,
                source_memory_ids=arguments.source_memory_id,
            ).to_json()
        elif arguments.command == "record-ab":
            result = service.record_ab_evidence(arguments.proposal_id, _evidence(arguments.evidence_json))
        else:
            result = service.approve_improvement(
                arguments.proposal_id,
                approver_id=arguments.approver_id,
            )
    except ConsolidationError as exc:
        _emit({"status": "ERROR", "failure_category": str(exc)})
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
