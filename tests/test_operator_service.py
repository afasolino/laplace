from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_workspace.operator_service import (
    OperatorService,
    OperatorServiceError,
)


ROOT = Path(__file__).resolve().parents[1]


class _ModelServers:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def status(self) -> dict[str, object]:
        return {"status": "OBSERVED", "servers": []}

    def start(self) -> dict[str, object]:
        self.actions.append("start")
        return {"status": "STARTED_HEALTHY_SERVERS"}

    def release_owned(self) -> dict[str, object]:
        self.actions.append("stop")
        return {"status": "RELEASED_LAPLACE_OWNED_SERVERS"}


def _configuration(*, gpu: bool = True, arm: str = "arm_c") -> dict[str, object]:
    return {
        "task_id": "rtl_counter",
        "arm_id": arm,
        "model_route": "main+codev",
        "corpus_snapshot_sha256": "1" * 64,
        "skills_lock_sha256": "2" * 64,
        "smoke_profile": "codev-live",
        "request_sha256": "3" * 64,
        "gpu_required": gpu,
    }


def test_run_requires_approval_and_repeated_terminal_start_is_idempotent(
    tmp_path: Path,
) -> None:
    service = OperatorService(
        ROOT,
        tmp_path,
        model_servers=_ModelServers(),  # type: ignore[arg-type]
    )
    prepared = service.prepare_run(
        _configuration(), actor_role="operate", run_id="run-fixed"
    )
    assert prepared["status"] == "PREPARED"
    with pytest.raises(OperatorServiceError) as caught:
        service.start_run(
            "run-fixed", approval_id=None, actor_role="operate"
        )
    assert caught.value.category == "approval_required"

    approval = service.request_approval(
        "START_GPU_RUN",
        "run-fixed",
        {"configuration_sha256": prepared["configuration_sha256"]},
        actor_role="operate",
    )
    approval_id = str(approval["approval_id"])
    service.decide_approval(
        approval_id, approve=True, actor_role="approve"
    )
    calls: list[str] = []

    def execute(*args: Any) -> dict[str, object]:
        calls.append(str(args[0].run_id))
        return {"outcome": "PASS"}

    first = service.start_run(
        "run-fixed",
        approval_id=approval_id,
        actor_role="operate",
        executor=execute,
    )
    second = service.start_run(
        "run-fixed",
        approval_id=approval_id,
        actor_role="operate",
        executor=execute,
    )
    assert first["status"] == "COMPLETE"
    assert second["status"] == "IDEMPOTENT_EXISTING_STATE"
    assert calls == ["run-fixed"]
    assert (
        Path(str(prepared["project_path"])) / "result_compact.json"
    ).is_file()


def test_same_run_id_with_different_configuration_is_conflict(
    tmp_path: Path,
) -> None:
    service = OperatorService(
        ROOT,
        tmp_path,
        model_servers=_ModelServers(),  # type: ignore[arg-type]
    )
    service.prepare_run(_configuration(), actor_role="operate", run_id="run-fixed")
    with pytest.raises(OperatorServiceError) as caught:
        service.prepare_run(
            _configuration(arm="arm_b"),
            actor_role="operate",
            run_id="run-fixed",
        )
    assert caught.value.category == "run_identity_conflict"
    assert "existing_configuration_sha256" in caught.value.evidence


def test_approvals_and_actions_are_idempotent_and_evented(tmp_path: Path) -> None:
    model_servers = _ModelServers()
    service = OperatorService(
        ROOT,
        tmp_path,
        model_servers=model_servers,  # type: ignore[arg-type]
    )
    approval = service.request_approval(
        "START_MODEL_SERVERS", "phase3", {}, actor_role="operate"
    )
    duplicate = service.request_approval(
        "START_MODEL_SERVERS", "phase3", {}, actor_role="operate"
    )
    assert duplicate["status"] == "IDEMPOTENT_EXISTING_APPROVAL"
    approval_id = str(approval["approval_id"])
    service.decide_approval(approval_id, approve=True, actor_role="approve")
    service.model_server_action(
        "start", approval_id=approval_id, actor_role="operate"
    )
    assert model_servers.actions == ["start"]
    events = service.events(actor_role="read")
    assert [event["action"] for event in events] == [
        "APPROVAL_REQUESTED",
        "APPROVAL_APPROVED",
        "MODEL_SERVERS_START",
    ]


def test_read_role_cannot_mutate_and_summary_is_machine_readable(
    tmp_path: Path,
) -> None:
    service = OperatorService(
        ROOT,
        tmp_path,
        model_servers=_ModelServers(),  # type: ignore[arg-type]
    )
    with pytest.raises(OperatorServiceError) as caught:
        service.prepare_run(_configuration(gpu=False), actor_role="read")
    assert caught.value.category == "authorization_failure"
    summary = service.summary(actor_role="read")
    assert summary["status"] == "OK"
    assert summary["local_only"] is True
    assert summary["event_count"] == 0

