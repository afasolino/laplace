from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.execution_records import (
    AppendOnlyEventLog,
    ExecutionRecordError,
    LocalTraceRecorder,
    RunIdentity,
    RunIdentityConflict,
    RunIdentityStore,
    canonical_sha256,
)


def _identity(run_id: str = "run-1", *, config: str = "config") -> RunIdentity:
    return RunIdentity(
        run_id=run_id,
        task_id="sv_elastic_buffer2",
        arm_id="C",
        configuration_sha256=canonical_sha256(config),
        request_sha256=canonical_sha256("request"),
    )


def test_run_identity_resumes_and_returns_terminal_without_work(tmp_path: Path) -> None:
    project = RunIdentityStore.project_path(tmp_path, "run-1")
    store = RunIdentityStore(project)
    identity = _identity()

    assert store.initialize(identity)["status"] == "CREATED"
    assert store.initialize(identity)["status"] == "RESUME"
    store.write_terminal(identity, {"status": "COMPLETE", "model_calls": 1})
    terminal = store.initialize(identity)

    assert terminal["status"] == "IDEMPOTENT_TERMINAL"
    assert terminal["model_calls_executed"] == 0
    assert terminal["eda_runs_executed"] == 0


def test_incompatible_identity_has_structured_conflict(tmp_path: Path) -> None:
    store = RunIdentityStore(RunIdentityStore.project_path(tmp_path, "run-1"))
    store.initialize(_identity())

    with pytest.raises(RunIdentityConflict) as caught:
        store.initialize(_identity(config="changed"))

    assert caught.value.evidence["status"] == "run_identity_conflict"
    assert "existing_identity" in caught.value.evidence


def test_event_deduplication_sequence_and_truncated_recovery(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = AppendOnlyEventLog(
        path, run_id="run-1", task_id="task-1", arm_id="C"
    )
    first = log.append(
        attempt=0,
        event_type="retrieval",
        from_state="requirements",
        to_state="retrieval",
        source_state_fingerprint=None,
        payload={"snapshot": "a" * 64},
    )
    duplicate = log.append(
        attempt=0,
        event_type="retrieval",
        from_state="requirements",
        to_state="retrieval",
        source_state_fingerprint=None,
        payload={"snapshot": "a" * 64},
    )
    with path.open("ab") as handle:
        handle.write(b'{"truncated":')
    second = log.append(
        attempt=0,
        event_type="implementation",
        from_state="retrieval",
        to_state="implementation",
        source_state_fingerprint="b" * 64,
        payload={"call": 1},
    )

    assert first["event_sequence"] == 1
    assert duplicate["deduplicated"] is True
    assert second["event_sequence"] == 2
    assert len(log.read()) == 2
    assert list(tmp_path.glob("events.jsonl.truncated.*.bin"))


def test_trace_export_is_local_bounded_and_noop_supported(tmp_path: Path) -> None:
    recorder = LocalTraceRecorder(tmp_path / "trace.jsonl", trace_id="a" * 32)
    with recorder.span("verification", attributes={"run_id": "run-1"}):
        pass

    record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert record["trace_id"] == "a" * 32
    assert record["status"] == "OK"
    assert (tmp_path / "metrics.json").is_file()

    noop = LocalTraceRecorder(tmp_path / "noop.jsonl", enabled=False)
    with noop.span("requirements"):
        pass
    assert not (tmp_path / "noop.jsonl").exists()

    with pytest.raises(ExecutionRecordError):
        with recorder.span("review", attributes={"full_prompt": "secret"}):
            pass
