from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.personal_corpus import PersonalCorpusPolicy, PersonalCorpusStore
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


class _Chat:
    def complete(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": "Fixture answer.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
        }


class _Agent:
    def run(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": "Fixture patch.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "modified_paths": [],
        }


def _service(tmp_path: Path) -> tuple[TieredServingService, UserCapabilityStore]:
    users = UserCapabilityStore(tmp_path / "users.sqlite3")
    authorizations = RepositoryAuthorizationStore(tmp_path / "repositories.sqlite3")
    routes = {
        lane: ModelRoute(
            lane=lane,
            model_id=f"fixture-{lane.value}",
            endpoint=f"http://127.0.0.1:{8300 + index}",
            priority=index,
            context_limit=8192,
            output_limit=1024,
        )
        for index, lane in enumerate(ModelLane)
    }
    return (
        TieredServingService(
            users=users,
            sandboxes=AgentSandboxManager(tmp_path / "worktrees", authorizations),
            lane_policy=LanePolicy(routes=routes),
            chat_backend=_Chat(),
            agent_backend=_Agent(),
            audit_log=TierAuditLog(tmp_path / "tier-audit.jsonl"),
        ),
        users,
    )


@pytest.mark.anyio
async def test_personal_corpus_api_owner_isolation_manifest_index_and_retrieval(
    tmp_path: Path,
) -> None:
    tiered, users = _service(tmp_path)
    users.set_user("basic", CapabilityTier.BASIC)
    users.set_user("plus-one", CapabilityTier.PLUS)
    users.set_user("plus-two", CapabilityTier.PLUS)
    users.set_user("administrator", CapabilityTier.OPERATOR)
    tokens = {
        "basic-token-0000000000000000000000": AuthCredential(
            "read", "basic", CapabilityTier.BASIC
        ),
        "plus-one-token-000000000000000000": AuthCredential(
            "read", "plus-one", CapabilityTier.PLUS
        ),
        "plus-two-token-000000000000000000": AuthCredential(
            "read", "plus-two", CapabilityTier.PLUS
        ),
        "admin-token-000000000000000000000": AuthCredential(
            "admin", "administrator", CapabilityTier.OPERATOR
        ),
    }
    corpus_store = PersonalCorpusStore(
        tmp_path / "state",
        policy=PersonalCorpusPolicy(min_free_disk_bytes=1),
    )
    app = create_operator_app(
        OperatorService(ROOT, tmp_path / "operator"),
        OperatorAuth(tokens),
        settings=OperatorApiSettings(fixture_mode=True, bearer_api_enabled=True),
        tiered=tiered,
        personal_corpora=corpus_store,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        async def headers(token: str) -> dict[str, str]:
            authorization = {"Authorization": f"Bearer {token}"}
            session = await client.post("/api/v1/session", headers=authorization)
            return {
                **authorization,
                "X-CSRF-Token": str(session.json()["csrf_token"]),
            }

        basic = await headers("basic-token-0000000000000000000000")
        assert (
            await client.get("/api/v1/personal-corpora", headers=basic)
        ).status_code == 403

        owner = await headers("plus-one-token-000000000000000000")
        created = await client.post(
            "/api/v1/personal-corpora",
            headers=owner,
            json={"name": "API corpus"},
        )
        assert created.status_code == 200
        corpus_id = created.json()["corpus"]["corpus_id"]
        upload = await client.post(
            "/api/v1/personal-corpus/uploads",
            headers=owner,
            json={"corpus_id": corpus_id, "idempotency_key": "upload:api-fixture"},
        )
        upload_id = upload.json()["upload_id"]
        resumable = await client.get(
            "/api/v1/personal-corpus/uploads?state=STAGING",
            headers=owner,
        )
        assert resumable.status_code == 200
        assert resumable.json()["uploads"][0]["upload_id"] == upload_id
        staged = await client.post(
            f"/api/v1/personal-corpus/uploads/{upload_id}/files",
            headers=owner,
            files={
                "file": ("reference.md", b"API unique retrieval marker.\n", "text/markdown"),
                "relative_path": (None, "folder/reference.md"),
            },
        )
        assert staged.status_code == 200
        assert staged.json()["state"] == "ACCEPTED"
        indexed = await client.post(
            f"/api/v1/personal-corpus/uploads/{upload_id}/index",
            headers=owner,
            json={"idempotency_key": "index:api-fixture"},
        )
        assert indexed.status_code == 200
        searched = await client.post(
            f"/api/v1/personal-corpora/{corpus_id}/search-test",
            headers=owner,
            json={
                "query": "unique retrieval marker",
                "corpus_id": corpus_id,
                "limit": 8,
            },
        )
        assert searched.status_code == 200
        assert searched.json()["results"][0]["file"] == "folder/reference.md"
        chat = await client.post(
            "/api/v1/chat",
            headers=owner,
            json={
                "lane": "standard",
                "domain": "general",
                "retrieval_selection": "selected_personal",
                "personal_corpus_id": corpus_id,
                "messages": [
                    {"role": "user", "content": "What is the unique retrieval marker?"}
                ],
            },
        )
        assert chat.status_code == 200
        assert chat.json()["retrieval"]["retrieval_used"] is True
        citation = chat.json()["retrieval"]["personal"]["results"][0]["citation"]
        assert citation["file"] == "folder/reference.md"
        assert citation["chunk_id"].startswith("chk_")

        other = await headers("plus-two-token-000000000000000000")
        denied = await client.get(
            f"/api/v1/personal-corpora/{corpus_id}", headers=other
        )
        assert denied.status_code == 404

        administrator = await headers("admin-token-000000000000000000000")
        inventory = await client.get(
            "/api/v1/admin/personal-corpora", headers=administrator
        )
        assert inventory.status_code == 200
        sanitized = inventory.json()["corpora"][0]
        assert sanitized["source_count"] == 1
        assert sanitized["content_access"] == "DISABLED_BY_POLICY"
        assert "owner_user_id" not in sanitized
        assert "name" not in sanitized
        assert "hash" not in sanitized

        updated = await client.patch(
            "/api/v1/admin/users/administrator/capabilities",
            headers=administrator,
            json={
                "capabilities": [
                    Capability.CHAT.value,
                    Capability.AGENT.value,
                    Capability.OPERATOR.value,
                    Capability.ADMIN.value,
                    Capability.PERSONAL_CORPUS.value,
                ]
            },
        )
        assert updated.status_code == 200
        assert {"agent", "operator"} <= set(
            updated.json()["capability"]["capabilities"]
        )


@pytest.mark.anyio
async def test_domain_api_enumeration_and_unknown_domain_rejection(
    tmp_path: Path,
) -> None:
    tiered, users = _service(tmp_path)
    users.set_user("basic", CapabilityTier.BASIC)
    token = "basic-domain-token-0000000000000000"
    app = create_operator_app(
        OperatorService(ROOT, tmp_path / "operator"),
        OperatorAuth(
            {
                token: AuthCredential(
                    "read", "basic", CapabilityTier.BASIC
                )
            }
        ),
        settings=OperatorApiSettings(fixture_mode=True, bearer_api_enabled=True),
        tiered=tiered,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        authorization = {"Authorization": f"Bearer {token}"}
        session = await client.post("/api/v1/session", headers=authorization)
        headers = {
            **authorization,
            "X-CSRF-Token": session.json()["csrf_token"],
        }
        domains = await client.get("/api/v1/domains", headers=authorization)
        assert domains.status_code == 200
        assert domains.json()["default_domain_id"] == "general"
        rejected = await client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "lane": "standard",
                "domain": "not-implemented",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert rejected.status_code == 403
        assert rejected.json()["failure_category"] == "unknown_domain"
