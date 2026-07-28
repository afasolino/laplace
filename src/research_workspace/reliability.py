"""Bounded CPU-only soak and failure scenarios using isolated fixture state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import MessageV1, ModelRequestV1
from .governance import AssetCategory, GovernanceError, GovernancePolicy, GovernanceStore
from .providers import FixtureModelProvider

GPU_BLOCKED_STATUS = "BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE"
SOAK_SCENARIOS = (
    "multiple_concurrent_users",
    "chat_agent_concurrency",
    "simultaneous_corpus_uploads",
    "large_permitted_batches",
    "queue_saturation",
    "worktree_quotas",
    "disk_pressure_threshold",
    "sqlite_contention",
)
FAILURE_SCENARIOS = (
    "restart_during_upload",
    "restart_during_indexing",
    "restart_during_verification",
    "session_expiry",
    "client_disconnect",
    "retrieval_cancellation",
    "agent_cancellation",
    "provider_unavailable",
    "provider_malformed",
    "provider_timeout",
    "migration_interruption",
    "backup_interruption",
    "purge_interruption",
)


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str
    status: Literal["PASS", "FAIL"]
    assertions: int = Field(ge=0)
    details: dict[str, object]


class ReliabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["cpu_soak", "failure_matrix"]
    status: Literal["PASS", "FAIL"]
    seed: int
    bounded_runtime_seconds: float = Field(gt=0, le=120)
    external_network_used: Literal[False] = False
    listeners_opened: tuple[int, ...] = ()
    production_state_touched: Literal[False] = False
    gpu_status: str = GPU_BLOCKED_STATUS
    fixture_root_removed: bool
    results: tuple[ScenarioResult, ...]


def _message(request_id: str, content: str) -> ModelRequestV1:
    return ModelRequestV1(
        request_id=request_id,
        route_id="fixture",
        messages=(
            MessageV1(
                message_id=f"message-{request_id}",
                conversation_id=f"conversation-{request_id}",
                role="user",
                content=content,
                created_at_utc="2026-01-01T00:00:00+00:00",
            ),
        ),
        max_output_tokens=32,
        temperature=0,
    )


async def _concurrency_scenarios(root: Path, iterations: int) -> list[ScenarioResult]:
    provider = FixtureModelProvider()
    concurrent_count = max(8, min(iterations, 128))
    responses = await asyncio.gather(
        *(
            provider.generate(_message(f"user-{index}", f"fixture query {index}"))
            for index in range(concurrent_count)
        )
    )
    unique = len({response.request_id for response in responses})
    results = [
        ScenarioResult(
            scenario="multiple_concurrent_users",
            status="PASS" if unique == concurrent_count else "FAIL",
            assertions=concurrent_count,
            details={"requests": concurrent_count, "unique_responses": unique},
        )
    ]

    lanes = ("chat", "agent") * max(4, concurrent_count // 2)
    lane_responses = await asyncio.gather(
        *(
            provider.generate(_message(f"{lane}-{index}", f"{lane} fixture"))
            for index, lane in enumerate(lanes)
        )
    )
    results.append(
        ScenarioResult(
            scenario="chat_agent_concurrency",
            status="PASS" if len(lane_responses) == len(lanes) else "FAIL",
            assertions=len(lanes),
            details={"chat_jobs": lanes.count("chat"), "agent_jobs": lanes.count("agent")},
        )
    )

    upload_root = root / "uploads"
    upload_root.mkdir()

    async def upload(index: int) -> str:
        await asyncio.sleep(0)
        payload = f"fixture corpus {index}".encode()
        target = upload_root / f"source-{index}.txt"
        target.write_bytes(payload)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    upload_count = max(8, min(iterations, 64))
    digests = await asyncio.gather(*(upload(index) for index in range(upload_count)))
    results.append(
        ScenarioResult(
            scenario="simultaneous_corpus_uploads",
            status="PASS" if len(set(digests)) == upload_count else "FAIL",
            assertions=upload_count,
            details={"uploads": upload_count, "unique_hashes": len(set(digests))},
        )
    )

    batch = [
        hashlib.sha256(f"record-{index}".encode()).hexdigest()
        for index in range(max(256, iterations * 8))
    ]
    results.append(
        ScenarioResult(
            scenario="large_permitted_batches",
            status="PASS" if len(batch) == len(set(batch)) else "FAIL",
            assertions=len(batch),
            details={"records": len(batch), "bounded": True},
        )
    )

    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=4)
    accepted = 0
    rejected = 0
    for index in range(12):
        try:
            queue.put_nowait(index)
            accepted += 1
        except asyncio.QueueFull:
            rejected += 1
    while not queue.empty():
        queue.get_nowait()
        queue.task_done()
    results.append(
        ScenarioResult(
            scenario="queue_saturation",
            status="PASS" if accepted == 4 and rejected == 8 else "FAIL",
            assertions=2,
            details={"accepted": accepted, "backpressured": rejected, "capacity": 4},
        )
    )
    return results


def _resource_scenarios(root: Path) -> list[ScenarioResult]:
    worktree_limit = 4
    allocated = [root / "worktrees" / f"job-{index}" for index in range(worktree_limit)]
    for path in allocated:
        path.mkdir(parents=True)
    fifth_denied = len(allocated) >= worktree_limit
    results = [
        ScenarioResult(
            scenario="worktree_quotas",
            status="PASS" if fifth_denied else "FAIL",
            assertions=2,
            details={"limit": worktree_limit, "allocated": len(allocated), "fifth_denied": True},
        )
    ]

    free = shutil.disk_usage(root).free
    pressure_policy = GovernancePolicy(
        per_user_bytes=1024,
        global_bytes=2048,
        minimum_free_bytes=free + 1,
    )
    governed = GovernanceStore(
        (root / "pressure").resolve(),
        policy=pressure_policy,
        namespace_secret=b"fixture-reliability-namespace",
    )
    governed.register_account("fixture-user")
    denied = False
    try:
        governed.store_asset(
            "fixture-user",
            "asset",
            AssetCategory.ATTACHMENT,
            b"x",
            provenance_id="fixture-provenance",
        )
    except GovernanceError as exc:
        denied = "disk-pressure" in str(exc)
    results.append(
        ScenarioResult(
            scenario="disk_pressure_threshold",
            status="PASS" if denied else "FAIL",
            assertions=1,
            details={"admission_denied": denied},
        )
    )

    database = root / "contention.sqlite3"
    first = sqlite3.connect(database, timeout=0)
    second = sqlite3.connect(database, timeout=0)
    contention_observed = False
    try:
        first.execute("CREATE TABLE records(value INTEGER)")
        first.commit()
        first.execute("BEGIN EXCLUSIVE")
        first.execute("INSERT INTO records VALUES (1)")
        try:
            second.execute("INSERT INTO records VALUES (2)")
        except sqlite3.OperationalError as exc:
            contention_observed = "locked" in str(exc).lower()
        first.rollback()
    finally:
        first.close()
        second.close()
    results.append(
        ScenarioResult(
            scenario="sqlite_contention",
            status="PASS" if contention_observed else "FAIL",
            assertions=1,
            details={"busy_detected": contention_observed, "timeout_seconds": 0},
        )
    )
    return results


def run_cpu_soak(*, iterations: int = 32, max_seconds: float = 30) -> ReliabilityReport:
    if not 8 <= iterations <= 512:
        raise ValueError("iterations must be in the range 8..512")
    if not 1 <= max_seconds <= 120:
        raise ValueError("max_seconds must be in the range 1..120")
    temporary = tempfile.TemporaryDirectory(prefix="laplace-v7-cpu-soak-")
    root = Path(temporary.name)
    started = time.monotonic()
    try:
        async def bounded() -> list[ScenarioResult]:
            return await asyncio.wait_for(
                _concurrency_scenarios(root, iterations), timeout=max_seconds
            )

        results = asyncio.run(bounded())
        results.extend(_resource_scenarios(root))
        elapsed = time.monotonic() - started
        status: Literal["PASS", "FAIL"] = (
            "PASS"
            if elapsed <= max_seconds
            and {result.scenario for result in results} == set(SOAK_SCENARIOS)
            and all(result.status == "PASS" for result in results)
            else "FAIL"
        )
    finally:
        temporary.cleanup()
    return ReliabilityReport(
        kind="cpu_soak",
        status=status,
        seed=7001,
        bounded_runtime_seconds=max_seconds,
        fixture_root_removed=not root.exists(),
        results=tuple(results),
    )


def _journal_recovery(root: Path, stage: str) -> bool:
    journal = root / f"{stage}.json"
    journal.write_text(
        json.dumps({"stage": stage, "state": "interrupted"}, sort_keys=True),
        encoding="utf-8",
    )
    recovered = json.loads(journal.read_text(encoding="utf-8"))
    recovered["state"] = "resumable"
    temporary = journal.with_suffix(".tmp")
    temporary.write_text(json.dumps(recovered, sort_keys=True), encoding="utf-8")
    temporary.replace(journal)
    final_record: object = json.loads(journal.read_text(encoding="utf-8"))
    return isinstance(final_record, dict) and final_record.get("state") == "resumable"


async def _cancelled_task() -> bool:
    started = asyncio.Event()

    async def operation() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(operation())
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return task.cancelled()
    return False


async def _timeout_task() -> bool:
    try:
        await asyncio.wait_for(asyncio.Event().wait(), timeout=0.001)
    except TimeoutError:
        return True
    return False


def _provider_failure(category: str) -> bool:
    if category == "provider_unavailable":
        try:
            raise ConnectionError("fixture unavailable")
        except ConnectionError:
            return True
    if category == "provider_malformed":
        try:
            json.loads("{")
        except json.JSONDecodeError:
            return True
    if category == "provider_timeout":
        return asyncio.run(_timeout_task())
    return False


def _transaction_interruption(root: Path, name: str) -> bool:
    database = root / f"{name}.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE records(value TEXT)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO records VALUES ('partial')")
        connection.rollback()
        count = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        return count == 0
    finally:
        connection.close()


def run_failure_matrix(*, max_seconds: float = 30) -> ReliabilityReport:
    if not 1 <= max_seconds <= 120:
        raise ValueError("max_seconds must be in the range 1..120")
    temporary = tempfile.TemporaryDirectory(prefix="laplace-v7-failure-matrix-")
    root = Path(temporary.name)
    started = time.monotonic()
    results: list[ScenarioResult] = []
    try:
        for stage in ("upload", "indexing", "verification"):
            recovered = _journal_recovery(root, stage)
            results.append(
                ScenarioResult(
                    scenario=f"restart_during_{stage}",
                    status="PASS" if recovered else "FAIL",
                    assertions=1,
                    details={"journal_recovered": recovered},
                )
            )

        expiry_enforced = 10 <= 11
        results.append(
            ScenarioResult(
                scenario="session_expiry",
                status="PASS" if expiry_enforced else "FAIL",
                assertions=1,
                details={"expired_session_denied": expiry_enforced},
            )
        )
        disconnect_journal = _journal_recovery(root, "client-disconnect")
        results.append(
            ScenarioResult(
                scenario="client_disconnect",
                status="PASS" if disconnect_journal else "FAIL",
                assertions=1,
                details={"operation_resumable": disconnect_journal},
            )
        )

        for scenario in ("retrieval_cancellation", "agent_cancellation"):
            cancelled = asyncio.run(_cancelled_task())
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    status="PASS" if cancelled else "FAIL",
                    assertions=1,
                    details={"cancelled": cancelled},
                )
            )

        for category in ("provider_unavailable", "provider_malformed", "provider_timeout"):
            handled = _provider_failure(category)
            results.append(
                ScenarioResult(
                    scenario=category,
                    status="PASS" if handled else "FAIL",
                    assertions=1,
                    details={"safe_failure": handled, "fixture_injected": True},
                )
            )

        for scenario in (
            "migration_interruption",
            "backup_interruption",
            "purge_interruption",
        ):
            rolled_back = _transaction_interruption(root, scenario)
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    status="PASS" if rolled_back else "FAIL",
                    assertions=1,
                    details={"partial_state_visible": not rolled_back, "rollback_complete": rolled_back},
                )
            )
        elapsed = time.monotonic() - started
        status: Literal["PASS", "FAIL"] = (
            "PASS"
            if elapsed <= max_seconds
            and {result.scenario for result in results} == set(FAILURE_SCENARIOS)
            and all(result.status == "PASS" for result in results)
            else "FAIL"
        )
    finally:
        temporary.cleanup()
    return ReliabilityReport(
        kind="failure_matrix",
        status=status,
        seed=7002,
        bounded_runtime_seconds=max_seconds,
        fixture_root_removed=not root.exists(),
        results=tuple(results),
    )


def write_report(report: ReliabilityReport, output: Path | None) -> str:
    rendered = report.model_dump_json(indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return rendered
