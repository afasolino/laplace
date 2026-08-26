from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.conversations import ConversationStore
from research_workspace.laplace_core import LaplaceCore
from research_workspace.operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.personal_corpus import PersonalCorpusStore
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.service_tiers import (
    LanePolicy,
    ModelLane,
    ModelRoute,
    ServiceTierError,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.user_capabilities import Capability, CapabilityTier, UserCapabilityStore

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "async-agent-owner-token-000000000000"
OTHER_TOKEN = "async-agent-other-token-000000000000"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _Backend:
    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> dict[str, object]:
        del messages, route, tools, request_id
        raise AssertionError("fixture repository service owns agent execution")


class _PersistentTurns:
    def __init__(self, sandboxes: AgentSandboxManager, *, resumable: bool = False) -> None:
        self.sandboxes = sandboxes
        self.resumable = resumable
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = True
        self.calls = 0
        self.tiered: TieredServingService | None = None

    def run_turn(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        session_id = str(kwargs["session_id"])
        user_id = str(kwargs["user_id"])
        self.sandboxes.start_task(
            session_id,
            user_id=user_id,
            lane="quality",
            sanitized_model_name="fixture-quality",
            instruction_digest="0" * 64,
        )
        self.started.set()
        if self.block and not self.release.wait(timeout=5):
            raise RuntimeError("test_turn_not_released")
        if (
            self.tiered is not None
            and self.tiered.agent_session_status(user_id=user_id, session_id=session_id)["status"]
            == "CANCELLED"
        ):
            raise ServiceTierError("agent_session_cancelled")
        result_id = f"res_{self.calls:032x}"
        self.sandboxes.record_result(
            session_id,
            user_id=user_id,
            command_count=0,
            verification_summary="INCOMPLETE:max_steps" if self.resumable else "PASSED:read_only",
            result_id=result_id,
            resumable=self.resumable,
        )
        return {
            "status": "INCOMPLETE" if self.resumable else "SUCCESS",
            "session_id": session_id,
            "repo_id": kwargs["repo_id"],
            "model_id": "fixture-quality",
            "effective_lane": "quality",
            "content": "resumable boundary" if self.resumable else "async completed",
            "changed_paths": [],
            "verification": None,
            "validation_history": [],
            "unresolved_failures": [],
            "evidence_refs": [],
            "result_id": result_id,
            "result_artifacts": {},
            "delivery_status": "BOUNDED",
            "worktree_release": {"action": "PRESERVED_FOR_CONTINUATION"},
        }

    def result_page(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def run(self, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("one-shot repository execution was not requested")

    def scheduler_status(self, **_kwargs: object) -> dict[str, object]:
        return {"status": "OK"}

    def task_status(self, **_kwargs: object) -> dict[str, object]:
        return {"status": "OK"}

    def cancel_queued(self, **_kwargs: object) -> dict[str, object]:
        return {"status": "CANCELLED"}

    def handoff_evidence(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {"patch": None}


def _app(tmp_path: Path, *, resumable: bool = False):
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.test")
    (repository / "component.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "component.py")
    _git(repository, "commit", "-qm", "base")
    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    authorizations.register("repo-one", repository)
    authorizations.grant("owner", "repo-one")
    users = UserCapabilityStore(tmp_path / "users.sqlite3")
    for user_id in ("owner", "other"):
        users.set_user(
            user_id,
            CapabilityTier.PLUS,
            capabilities=frozenset({Capability.CHAT, Capability.AGENT}),
        )
    routes = {
        lane: ModelRoute(
            lane=lane,
            model_id=f"fixture-{lane.value}",
            endpoint=f"http://127.0.0.1:{8200 + index}",
            priority=3 - index,
            context_limit=16_384,
            output_limit=2_048,
        )
        for index, lane in enumerate(ModelLane)
    }
    sandboxes = AgentSandboxManager(tmp_path / "worktrees", authorizations)
    tiered = TieredServingService(
        users=users,
        sandboxes=sandboxes,
        lane_policy=LanePolicy(routes),
        chat_backend=_Backend(),
        agent_backend=_Backend(),  # type: ignore[arg-type]
        audit_log=TierAuditLog(tmp_path / "tier-audit.jsonl"),
    )
    persistent = _PersistentTurns(sandboxes, resumable=resumable)
    persistent.tiered = tiered
    core = LaplaceCore(
        ROOT,
        PersonalCorpusStore(tmp_path / "corpus"),
        tiered,
        repository_agent_service=persistent,  # type: ignore[arg-type]
    )
    app = create_operator_app(
        OperatorService(ROOT, tmp_path / "operator"),
        OperatorAuth(
            {
                TOKEN: AuthCredential(
                    role="operate", user_id="owner", capability_tier=CapabilityTier.PLUS
                ),
                OTHER_TOKEN: AuthCredential(
                    role="operate", user_id="other", capability_tier=CapabilityTier.PLUS
                ),
            }
        ),
        settings=OperatorApiSettings(fixture_mode=True, bearer_api_enabled=True),
        tiered=tiered,
        core=core,
        conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
    )
    return app, persistent


async def _headers(client: httpx.AsyncClient) -> dict[str, str]:
    base = {"Authorization": f"Bearer {TOKEN}"}
    response = await client.post("/api/v1/session", headers=base)
    assert response.status_code == 200, response.text
    nonce = response.json()["csrf_token"]
    return {**base, "X-CSRF-Token": nonce}


async def _wait_for_terminal_events(
    client: httpx.AsyncClient, headers: dict[str, str], session_id: str
) -> str:
    for _ in range(100):
        response = await client.get(
            f"/api/v1/agent/sessions/{session_id}/events?after_sequence=0&once=true",
            headers=headers,
        )
        if any(name in response.text for name in ("TURN_COMPLETED", "TURN_YIELDED_RESUMABLE", "TURN_CANCELLED")):
            return response.text
        await asyncio.sleep(0.01)
    raise AssertionError("asynchronous turn did not become terminal")


async def _wait_for_start(turns: _PersistentTurns) -> None:
    for _ in range(100):
        if turns.started.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("asynchronous turn did not start")


@pytest.mark.anyio
async def test_async_turn_survives_request_completion_and_events_are_owner_scoped(tmp_path: Path) -> None:
    app, persistent = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as client:
        headers = await _headers(client)
        assert (await client.post("/api/v1/agent/sessions", headers=headers, json={"repo_id": "repo-one", "session_id": "agent-one"})).status_code == 200
        started = time.monotonic()
        submitted = await client.post(
            "/api/v1/agent/sessions/agent-one/messages/async",
            headers=headers,
            json={"turn_id": "turn-00000001", "lane": "quality", "instruction": "inspect", "domain": "python"},
        )
        assert submitted.status_code == 202, submitted.text
        assert time.monotonic() - started < 0.5
        await _wait_for_start(persistent)
        assert persistent.calls == 1

    # Submission has returned and its caller is gone; a fresh connection must
    # see the same durable turn without scheduling a second executor.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as resumed:
        replay = await resumed.post(
            "/api/v1/agent/sessions/agent-one/messages/async",
            headers=headers,
            json={"turn_id": "turn-00000001", "lane": "quality", "instruction": "inspect", "domain": "python"},
        )
        assert replay.status_code == 202
        assert replay.json()["idempotent_replay"] is True
        other = await resumed.get(
            "/api/v1/agent/sessions/agent-one/events?once=true",
            headers={"Authorization": f"Bearer {OTHER_TOKEN}"},
        )
        assert other.status_code in {403, 404}
        persistent.release.set()
        events = await _wait_for_terminal_events(resumed, headers, "agent-one")
        assert "TURN_SUBMITTED" in events
        assert "TURN_STARTED" in events
        assert "TURN_COMPLETED" in events
        transcript = await resumed.get(
            "/api/v1/agent/sessions/agent-one/messages", headers=headers
        )
        messages = transcript.json()["conversation"]["messages"]
        assert messages[-1]["content"] == "async completed"
        assert messages[-1]["metadata"]["turn_id"] == "turn-00000001"


@pytest.mark.anyio
async def test_async_turn_cancellation_and_resumable_yield_are_durable(tmp_path: Path) -> None:
    app, persistent = _app(tmp_path, resumable=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as client:
        headers = await _headers(client)
        assert (await client.post("/api/v1/agent/sessions", headers=headers, json={"repo_id": "repo-one", "session_id": "agent-one"})).status_code == 200
        submitted = await client.post(
            "/api/v1/agent/sessions/agent-one/messages/async",
            headers=headers,
            json={"turn_id": "turn-00000002", "lane": "quality", "instruction": "inspect", "domain": "python"},
        )
        assert submitted.status_code == 202
        await _wait_for_start(persistent)
        persistent.release.set()
        events = await _wait_for_terminal_events(client, headers, "agent-one")
        assert "TURN_YIELDED_RESUMABLE" in events
        transcript = await client.get("/api/v1/agent/sessions/agent-one/messages", headers=headers)
        assert transcript.json()["conversation"]["messages"][-1]["content"] == "resumable boundary"

    cancelled_app, cancelled = _app(tmp_path / "cancelled")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cancelled_app), base_url="http://127.0.0.1:8765"
    ) as client:
        headers = await _headers(client)
        assert (await client.post("/api/v1/agent/sessions", headers=headers, json={"repo_id": "repo-one", "session_id": "agent-one"})).status_code == 200
        assert (await client.post(
            "/api/v1/agent/sessions/agent-one/messages/async",
            headers=headers,
            json={"turn_id": "turn-00000003", "lane": "quality", "instruction": "inspect", "domain": "python"},
        )).status_code == 202
        await _wait_for_start(cancelled)
        cancelled_response = await client.post("/api/v1/agent/sessions/agent-one/cancel", headers=headers)
        assert cancelled_response.status_code == 200
        assert cancelled_response.json()["status"] == "CANCELLATION_REQUESTED"
        cancelled.release.set()
        events = await _wait_for_terminal_events(client, headers, "agent-one")
        assert "TURN_CANCELLED" in events


@pytest.mark.anyio
async def test_async_turn_id_conflict_is_rejected(tmp_path: Path) -> None:
    app, persistent = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        headers = await _headers(client)
        created = await client.post(
            "/api/v1/agent/sessions",
            headers=headers,
            json={"repo_id": "repo-one", "session_id": "agent-one"},
        )
        assert created.status_code == 200, created.text

        original = {
            "turn_id": "turn-00000004",
            "lane": "quality",
            "instruction": "inspect",
            "domain": "python",
        }
        submitted = await client.post(
            "/api/v1/agent/sessions/agent-one/messages/async",
            headers=headers,
            json=original,
        )
        assert submitted.status_code == 202, submitted.text

        conflict = await client.post(
            "/api/v1/agent/sessions/agent-one/messages/async",
            headers=headers,
            json={**original, "instruction": "different instruction"},
        )
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"] == "agent_turn_id_conflict"

        persistent.release.set()
        await _wait_for_terminal_events(client, headers, "agent-one")
