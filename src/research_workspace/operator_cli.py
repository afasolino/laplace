"""Machine-readable CLI for the local Operator Plane service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .model_servers import ModelServerSafetyError
from .operator_service import OperatorService, OperatorServiceError


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _load_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorServiceError(
            "invalid_json_input", {"path": str(path), "error_type": type(exc).__name__}
        ) from exc
    if not isinstance(value, dict):
        raise OperatorServiceError(
            "invalid_json_input", {"path": str(path), "reason": "expected object"}
        )
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-operator")
    repository_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=repository_root / "outputs/operator_plane",
    )
    parser.add_argument(
        "--role",
        choices=("read", "operate", "approve", "admin"),
        default="read",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")

    runs = commands.add_parser("run")
    run_commands = runs.add_subparsers(dest="run_command", required=True)
    run_prepare = run_commands.add_parser("prepare")
    run_prepare.add_argument("--config", type=Path, required=True)
    run_prepare.add_argument("--run-id")
    run_prepare.add_argument("--json", action="store_true")
    run_status = run_commands.add_parser("status")
    run_status.add_argument("--run-id", required=True)
    run_status.add_argument("--json", action="store_true")
    run_start = run_commands.add_parser("start")
    run_start.add_argument("--run-id", required=True)
    run_start.add_argument("--approval-id")
    run_start.add_argument("--json", action="store_true")

    approvals = commands.add_parser("approval")
    approval_commands = approvals.add_subparsers(
        dest="approval_command", required=True
    )
    approval_request = approval_commands.add_parser("request")
    approval_request.add_argument("--action", required=True)
    approval_request.add_argument("--entity-id", required=True)
    approval_request.add_argument("--payload", type=Path)
    approval_request.add_argument("--json", action="store_true")
    approval_decide = approval_commands.add_parser("decide")
    approval_decide.add_argument("--approval-id", required=True)
    decision = approval_decide.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    approval_decide.add_argument("--json", action="store_true")

    servers = commands.add_parser("model-servers")
    server_commands = servers.add_subparsers(
        dest="server_command", required=True
    )
    for action in ("status", "preflight", "start", "stop"):
        command = server_commands.add_parser(action)
        command.add_argument("--approval-id")
        command.add_argument("--json", action="store_true")

    events = commands.add_parser("events")
    events.add_argument("--after-sequence", type=int, default=0)
    events.add_argument("--limit", type=int, default=200)
    events.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    service = OperatorService(arguments.repository_root, arguments.state_root)
    try:
        if arguments.command == "status":
            result: object = service.summary(actor_role=arguments.role)
        elif arguments.command == "events":
            result = {
                "status": "OK",
                "events": service.events(
                    actor_role=arguments.role,
                    after_sequence=arguments.after_sequence,
                    limit=arguments.limit,
                ),
            }
        elif arguments.command == "run":
            if arguments.run_command == "prepare":
                result = service.prepare_run(
                    _load_object(arguments.config),
                    actor_role=arguments.role,
                    run_id=arguments.run_id,
                )
            elif arguments.run_command == "status":
                result = service.get_run(
                    arguments.run_id, actor_role=arguments.role
                )
            else:
                result = service.start_run(
                    arguments.run_id,
                    approval_id=arguments.approval_id,
                    actor_role=arguments.role,
                )
        elif arguments.command == "approval":
            if arguments.approval_command == "request":
                payload = (
                    _load_object(arguments.payload)
                    if arguments.payload is not None
                    else {}
                )
                result = service.request_approval(
                    arguments.action,
                    arguments.entity_id,
                    payload,
                    actor_role=arguments.role,
                )
            else:
                result = service.decide_approval(
                    arguments.approval_id,
                    approve=bool(arguments.approve),
                    actor_role=arguments.role,
                )
        else:
            action = arguments.server_command
            if action == "preflight":
                result = service.model_servers.admission.preflight(
                    output_path=service.model_servers.preflight_path,
                    specs=service.model_servers.specs,
                )
            else:
                result = service.model_server_action(
                    action,
                    approval_id=arguments.approval_id,
                    actor_role=arguments.role,
                )
    except (OperatorServiceError, ModelServerSafetyError) as exc:
        _emit(
            {
                "status": "ERROR",
                "failure_category": exc.category,
                "evidence": exc.evidence,
            }
        )
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

