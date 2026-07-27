#!/usr/bin/env python3
"""Execute one approved live CodeV certification run through OperatorService."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_workspace.operator_service import OperatorService
from research_workspace.orchestration_certification import LiveCertificationExecutor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--held-out-root", type=Path, required=True)
    arguments = parser.parse_args()
    service = OperatorService(arguments.repository_root, arguments.state_root)
    executor = LiveCertificationExecutor(
        arguments.repository_root,
        base_revision=arguments.base_revision,
        held_out_root=arguments.held_out_root,
        model_servers=service.model_servers,
    )
    result = service.start_run(
        arguments.run_id,
        approval_id=arguments.approval_id,
        actor_role="operate",
        executor=executor,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    status = result.get("status")
    if status == "COMPLETE":
        return 0
    if status == "IDEMPOTENT_EXISTING_STATE":
        return 0 if result.get("state") == "COMPLETE" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
