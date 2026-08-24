from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.laplace_core import LaplaceCore
from research_workspace.trajectory import (
    TrajectoryAuthorizationError,
    TrajectoryConflictError,
    TrajectoryCorruptionError,
    TrajectoryCrashInjected,
    TrajectoryEventType,
    TrajectoryIdentity,
    TrajectoryProvenance,
    TrajectorySchemaError,
    TrajectoryService,
    TrajectoryValidationError,
)


def _identity(owner: str = "owner-a") -> TrajectoryIdentity:
    return TrajectoryIdentity(
        owner_user_id=owner,
        project_id="project-a",
        session_id="session-a",
        task_id="task-a",
    )


def _provenance(source_kind: str = "system") -> TrajectoryProvenance:
    return TrajectoryProvenance(
        source_kind=source_kind,  # type: ignore[arg-type]
        source_id="source-a",
        actor_id="actor-a",
    )


def _append(
    service: TrajectoryService,
    identity: TrajectoryIdentity,
    key: str,
    event_type: TrajectoryEventType,
    before: dict[str, object],
    after: dict[str, object],
) -> None:
    service.append(
        identity,
        event_type=event_type,
        idempotency_key=key,
        payload={"key": key},
        state_before=before,
        state_after=after,
        provenance=_provenance(),
    )


def test_typed_events_replay_exactly_across_restart_and_checkpoint(tmp_path: Path) -> None:
    identity = _identity()
    service = TrajectoryService(tmp_path / "trajectory")
    _append(
        service,
        identity,
        "start",
        TrajectoryEventType.TASK_STARTED,
        {},
        {"status": "running", "step": 0},
    )
    _append(
        service,
        identity,
        "tool",
        TrajectoryEventType.TOOL_ACTION,
        {"status": "running", "step": 0},
        {"status": "running", "step": 1, "tool_result_sha256": "a" * 64},
    )
    _append(
        service,
        identity,
        "verify",
        TrajectoryEventType.VERIFICATION,
        {"status": "running", "step": 1, "tool_result_sha256": "a" * 64},
        {"status": "complete", "step": 1, "verified": True},
    )

    before_checkpoint = service.replay(identity)
    assert before_checkpoint.state == {"status": "complete", "step": 1, "verified": True}
    service.checkpoint(identity)
    restarted = TrajectoryService(tmp_path / "trajectory").replay(identity)

    assert restarted.state == before_checkpoint.state
    assert restarted.checkpoint_used is True
    assert restarted.checkpoint_recovered is False
    assert len(restarted.events) == 3
    assert [event.event_type for event in restarted.events] == [
        TrajectoryEventType.TASK_STARTED,
        TrajectoryEventType.TOOL_ACTION,
        TrajectoryEventType.VERIFICATION,
    ]


def test_resume_and_cancellation_are_replayable_lifecycle_events(tmp_path: Path) -> None:
    identity = _identity()
    root = tmp_path / "trajectory"
    service = TrajectoryService(root)
    _append(service, identity, "start", TrajectoryEventType.TASK_STARTED, {}, {"status": "running"})
    _append(
        service,
        identity,
        "resume",
        TrajectoryEventType.TASK_RESUMED,
        {"status": "running"},
        {"status": "running", "resumed": 1},
    )
    _append(
        service,
        identity,
        "cancel",
        TrajectoryEventType.TASK_CANCELLED,
        {"status": "running", "resumed": 1},
        {"status": "cancelled", "cancel_reason": "user_requested"},
    )
    service.checkpoint(identity)
    replay = TrajectoryService(root).replay(identity)
    assert replay.state == {"status": "cancelled", "cancel_reason": "user_requested"}
    assert [event.event_type for event in replay.events] == [
        TrajectoryEventType.TASK_STARTED,
        TrajectoryEventType.TASK_RESUMED,
        TrajectoryEventType.TASK_CANCELLED,
    ]


def test_idempotency_conflict_duplicate_and_versioning_fail_closed(tmp_path: Path) -> None:
    service = TrajectoryService(tmp_path / "trajectory")
    identity = _identity()
    first = service.append(
        identity,
        event_type=TrajectoryEventType.TASK_STARTED,
        idempotency_key="same-key",
        payload={"intent": "start"},
        state_before={},
        state_after={"status": "running"},
        provenance=_provenance(),
    )
    duplicate = service.append(
        identity,
        event_type=TrajectoryEventType.TASK_STARTED,
        idempotency_key="same-key",
        payload={"intent": "start"},
        state_before={},
        state_after={"status": "running"},
        provenance=_provenance(),
    )
    assert duplicate == first
    with pytest.raises(TrajectoryConflictError, match="trajectory_idempotency_conflict"):
        service.append(
            identity,
            event_type=TrajectoryEventType.TASK_STARTED,
            idempotency_key="same-key",
            payload={"intent": "different"},
            state_before={},
            state_after={"status": "running"},
            provenance=_provenance(),
        )

    raw = json.loads(service.events_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    service.events_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(TrajectorySchemaError, match="trajectory_event_schema_unsupported"):
        service.replay(identity)


def test_partial_append_is_recovered_but_full_corruption_and_duplicate_fail_closed(
    tmp_path: Path,
) -> None:
    crash = True

    def hook(stage: str) -> None:
        if stage == "after_partial_event_write" and crash:
            raise TrajectoryCrashInjected("crash_during_append")

    root = tmp_path / "trajectory"
    service = TrajectoryService(root, failure_hook=hook)
    identity = _identity()
    with pytest.raises(TrajectoryCrashInjected):
        _append(service, identity, "crashed", TrajectoryEventType.TASK_STARTED, {}, {"step": 0})
    crash = False

    recovered = service.replay(identity)
    assert recovered.state == {}
    assert recovered.partial_event_recovered is True
    _append(service, identity, "start", TrajectoryEventType.TASK_STARTED, {}, {"step": 0})

    lines = service.events_path.read_text(encoding="utf-8").splitlines()
    service.events_path.write_text(lines[0] + "\n" + lines[0] + "\n", encoding="utf-8")
    with pytest.raises(TrajectoryCorruptionError, match="trajectory_event_sequence_invalid"):
        service.replay(identity)


def test_corrupt_checkpoint_and_event_checkpoint_gap_recover_from_events(tmp_path: Path) -> None:
    identity = _identity()
    root = tmp_path / "trajectory"
    service = TrajectoryService(root)
    _append(service, identity, "start", TrajectoryEventType.TASK_STARTED, {}, {"step": 0})
    service.checkpoint(identity)
    _append(service, identity, "finish", TrajectoryEventType.COMPLETION, {"step": 0}, {"step": 1})

    service.checkpoint_path.write_text("{not-json", encoding="utf-8")
    recovered = TrajectoryService(root).replay(identity)
    assert recovered.state == {"step": 1}
    assert recovered.checkpoint_used is False
    assert recovered.checkpoint_recovered is True

    crash = True

    def checkpoint_hook(stage: str) -> None:
        if stage == "after_event_before_checkpoint" and crash:
            raise TrajectoryCrashInjected("crash_between_event_and_checkpoint")

    service = TrajectoryService(root, failure_hook=checkpoint_hook)
    with pytest.raises(TrajectoryCrashInjected):
        service.checkpoint(identity)
    crash = False
    resumed = TrajectoryService(root).replay(identity)
    assert resumed.state == {"step": 1}
    assert len(resumed.events) == 2


def test_owner_project_session_binding_and_core_adapter(tmp_path: Path) -> None:
    identity = _identity()
    service = TrajectoryService(tmp_path / "trajectory")
    core = LaplaceCore(tmp_path, object(), object(), trajectory=service)  # type: ignore[arg-type]
    core.append_trajectory_event(
        identity,
        event_type=TrajectoryEventType.TASK_STARTED,
        idempotency_key="start",
        payload={"objective_sha256": "b" * 64},
        state_before={},
        state_after={"status": "running"},
        provenance=_provenance(),
    )
    assert core.replay_trajectory(identity).state == {"status": "running"}
    with pytest.raises(TrajectoryAuthorizationError, match="trajectory_owner_or_project_denied"):
        service.replay(_identity("owner-b"))

    with pytest.raises(TrajectoryValidationError, match="forbidden_trajectory_payload_field"):
        service.append(
            identity,
            event_type=TrajectoryEventType.TASK_RESUMED,
            idempotency_key="secret",
            payload={"prompt": "do not persist raw prompt"},
            state_before={"status": "running"},
            state_after={"status": "running"},
            provenance=_provenance(),
        )
