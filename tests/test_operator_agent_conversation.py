from __future__ import annotations

import json
import base64
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.conversations import ConversationStore
from research_workspace.laplace_core import LaplaceCore
from research_workspace.manager_control import ManagerPlan
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
    TierAuditLog,
    TieredServingService,
)
from research_workspace.user_capabilities import (
    Capability,
    CapabilityTier,
    UserCapabilityStore,
)

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "agent-conversation-token-000000000000"
OTHER_TOKEN = "agent-conversation-other-000000000000"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _FinishBackend:
    def __init__(self) -> None:
        self.calls: list[list[Mapping[str, str]]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> dict[str, object]:
        del route, tools, request_id
        self.calls.append(list(messages))
        return {
            "content": json.dumps(
                {
                    "action": "finish",
                    "result": f"bounded turn {len(self.calls)} complete",
                }
            ),
            "finish_reason": "stop",
        }


class _UnusedAgentBackend:
    def run(self, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("legacy agent backend must not handle Operator conversations")


class _PersistentService:
    def __init__(self, sandboxes: AgentSandboxManager) -> None:
        self.sandboxes = sandboxes
        self.calls: list[dict[str, object]] = []

    def run_turn(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        session_id = str(kwargs["session_id"])
        user_id = str(kwargs["user_id"])
        result_id = f"res_{len(self.calls):032x}"
        self.sandboxes.record_result(
            session_id,
            user_id=user_id,
            command_count=0,
            verification_summary="PASSED:read_only",
            result_id=result_id,
        )
        return {
            "status": "SUCCESS",
            "session_id": session_id,
            "repo_id": kwargs["repo_id"],
            "model_id": "fixture-quality",
            "effective_lane": "quality",
            "content": f"bounded turn {len(self.calls)} complete",
            "changed_paths": [],
            "verification": None,
            "validation_history": [],
            "unresolved_failures": [],
            "evidence_refs": [],
            "checkpoint_path": "/private/checkpoint.json",
            "handoff": {
                "patch": None,
                "patch_inline": False,
                "patch_chars": 0,
                "patch_sha256": "0" * 64,
                "patch_path": "/private/handoff.patch",
            },
            "result_id": result_id,
            "result_artifacts": {"handoff.patch": {"bytes": 0, "sha256": "0" * 64}},
            "delivery_status": "BOUNDED",
            "worktree_release": {"action": "PRESERVED_FOR_CONTINUATION"},
        }

    def result_page(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["repo_id"] == "repo-one"
        return {
            "status": "SUCCESS",
            "result_id": kwargs["result_id"],
            "repo_id": kwargs["repo_id"],
            "session_id": kwargs["session_id"],
            "artifact": kwargs["artifact"],
            "encoding": "base64",
            "offset": kwargs["offset"],
            "next_offset": None,
            "total_bytes": 0,
            "artifact_sha256": "0" * 64,
            "content_base64": base64.b64encode(b"").decode("ascii"),
        }

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


class _Manager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def plan(self, **kwargs: object) -> ManagerPlan:
        self.calls.append(dict(kwargs))
        return ManagerPlan(objective="Inspect", milestones=("Inspect", "Verify"))


@pytest.mark.anyio
async def test_operator_persistent_agent_messages_transcript_and_paged_result(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.test")
    (repository / "component.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "component.py")
    _git(repository, "commit", "-qm", "base")
    revision = _git(repository, "rev-parse", "HEAD")

    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    authorizations.register("repo-one", repository)
    authorizations.grant("owner", "repo-one", base_revision=revision)
    users = UserCapabilityStore(tmp_path / "users.sqlite3")
    users.set_user(
        "owner",
        CapabilityTier.PLUS,
        capabilities=frozenset({Capability.CHAT, Capability.AGENT}),
    )
    users.set_user(
        "other",
        CapabilityTier.PLUS,
        capabilities=frozenset({Capability.CHAT, Capability.AGENT}),
    )
    backend = _FinishBackend()
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
    tiered = TieredServingService(
        users=users,
        sandboxes=AgentSandboxManager(tmp_path / "worktrees", authorizations),
        lane_policy=LanePolicy(routes),
        chat_backend=backend,
        agent_backend=_UnusedAgentBackend(),  # type: ignore[arg-type]
        audit_log=TierAuditLog(tmp_path / "tier-audit.jsonl"),
    )
    persistent = _PersistentService(tiered.sandboxes)
    manager = _Manager()
    operator = OperatorService(ROOT, tmp_path / "operator")
    core = LaplaceCore(
        ROOT,
        PersonalCorpusStore(operator.state_root),
        tiered,
        repository_agent_service=persistent,  # type: ignore[arg-type]
        manager_provider=manager,
    )
    app = create_operator_app(
        operator,
        OperatorAuth(
            {
                TOKEN: AuthCredential(
                    role="operate",
                    user_id="owner",
                    capability_tier=CapabilityTier.PLUS,
                ),
                OTHER_TOKEN: AuthCredential(
                    role="operate",
                    user_id="other",
                    capability_tier=CapabilityTier.PLUS,
                ),
            }
        ),
        settings=OperatorApiSettings(fixture_mode=True, bearer_api_enabled=True),
        tiered=tiered,
        core=core,
        conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        nonce = (await client.post("/api/v1/session", headers=headers)).json()["csrf_token"]
        mutations = {**headers, "X-CSRF-Token": nonce}
        created = await client.post(
            "/api/v1/agent/sessions",
            headers=mutations,
            json={"repo_id": "repo-one", "session_id": "agent-one"},
        )
        assert created.status_code == 200

        results = []
        for instruction in ("Inspect component.py", "Explain the related detail"):
            response = await client.post(
                "/api/v1/agent/sessions/agent-one/messages",
                headers=mutations,
                json={
                    "lane": "quality",
                    "instruction": instruction,
                    "domain": "python",
                    "manager_complexity": {"architecture_sensitive": True},
                },
            )
            assert response.status_code == 200, response.text
            results.append(response.json())

        assert len(manager.calls) == 2
        assert results[0]["manager_control"]["decision"] == "plan"
        assert "Advisory manager plan" in str(persistent.calls[0]["instruction"])
        assert persistent.calls[0]["allow_mutation"] is False
        assert persistent.calls[0]["verification_argv"] is None

        assert [item["session_id"] for item in results] == ["agent-one", "agent-one"]
        assert [item["session_id"] for item in persistent.calls] == [
            "agent-one",
            "agent-one",
        ]
        assert "checkpoint_path" not in results[-1]
        assert "patch_path" not in results[-1]["handoff"]

        transcript = await client.get(
            "/api/v1/agent/sessions/agent-one/messages", headers=headers
        )
        assert transcript.status_code == 200
        conversation = transcript.json()["conversation"]
        assert conversation["repo_id"] == "repo-one"
        assert [item["role"] for item in conversation["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

        status = (
            await client.get("/api/v1/agent/sessions/agent-one/status", headers=headers)
        ).json()
        result_id = status["worktree_status"]["result_id"]
        page = await client.get(
            f"/api/v1/agent/sessions/agent-one/results/{result_id}",
            headers=headers,
            params={"artifact": "handoff.patch"},
        )
        assert page.status_code == 200
        assert page.json()["session_id"] == "agent-one"

        other_headers = {"Authorization": f"Bearer {OTHER_TOKEN}"}
        assert (
            await client.get(
                "/api/v1/agent/sessions/agent-one/messages", headers=other_headers
            )
        ).status_code in {403, 404}
