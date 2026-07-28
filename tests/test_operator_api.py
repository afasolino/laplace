from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from research_workspace.operator_api import (
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.research_models import (
    ClaimAssertion,
    DiscoveredSource,
    FetchedSource,
    FixtureResearchAdapter,
)
from research_workspace.research_plane import DeepResearchService


ROOT = Path(__file__).resolve().parents[1]
TOKENS = {
    "read-token-000000000000000000000000": "read",
    "operate-token-000000000000000000000": "operate",
    "approve-token-000000000000000000000": "approve",
    "admin-token-00000000000000000000000": "admin",
}


class _ModelServers:
    def status(self) -> dict[str, object]:
        return {
            "status": "OBSERVED",
            "gpu_observation": {
                "status": "OBSERVED",
                "gpu": {
                    "name": "Fixture A6000",
                    "memory_free_mib": 49_000,
                    "utilization_percent": 0,
                },
            },
            "servers": [
                {
                    "profile": "phase2_main",
                    "port": 8102,
                    "expected_model_id": "fixture-main",
                    "model_path": "/private/model",
                    "endpoint_observation": {"status": "HEALTHY_EXACT_MODEL"},
                }
            ],
            "laplace_owned_processes": [],
        }

    def start(self) -> dict[str, object]:
        return {"status": "STARTED_HEALTHY_SERVERS"}

    def release_owned(self) -> dict[str, object]:
        return {"status": "RELEASED_LAPLACE_OWNED_SERVERS"}


def _fixture_research(state: Path) -> DeepResearchService:
    source = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://example.org/official",
            title="Fixture official source",
            backend="fixture",
            query="",
            source_type="official_documentation",
            license="CC-BY-4.0",
        ),
        content=b"Fixture evidence supports deterministic local research.",
        assertions=(
            ClaimAssertion(
                normalized_claim="The fixture supports deterministic local research.",
                claim_key="fixture-claim",
            ),
        ),
        retrieved_at="2026-07-27T00:00:00+00:00",
    )
    return DeepResearchService(
        state,
        {"fixture": FixtureResearchAdapter([source])},
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )


@pytest.fixture
def app(tmp_path: Path) -> object:
    operator = OperatorService(
        ROOT,
        tmp_path,
        model_servers=_ModelServers(),  # type: ignore[arg-type]
    )
    return create_operator_app(
        operator,
        OperatorAuth(TOKENS),
        settings=OperatorApiSettings(fixture_mode=True, bearer_api_enabled=True),
        research=_fixture_research(tmp_path),
    )


async def _session(
    client: httpx.AsyncClient, token: str
) -> tuple[dict[str, str], str]:
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.post("/api/v1/session", headers=headers)
    assert response.status_code == 200
    return headers, str(response.json()["csrf_token"])


def _run_configuration() -> dict[str, object]:
    return {
        "task_id": "fixture",
        "arm_id": "C",
        "model_route": "main+codev",
        "corpus_snapshot_sha256": "1" * 64,
        "skills_lock_sha256": "2" * 64,
        "smoke_profile": "fixture",
        "request_sha256": "3" * 64,
        "gpu_required": False,
    }


@pytest.mark.anyio
async def test_auth_csrf_roles_and_idempotent_run_mutation(app: object) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://127.0.0.1:8765",
    ) as client:
        assert (await client.get("/api/v1/dashboard")).status_code == 401
        read_headers, read_csrf = await _session(
            client, "read-token-000000000000000000000000"
        )
        denied = await client.post(
            "/api/v1/runs",
            headers={**read_headers, "X-CSRF-Token": read_csrf},
            json={"configuration": _run_configuration(), "run_id": "run-api"},
        )
        assert denied.status_code == 403
        operate_headers, operate_csrf = await _session(
            client, "operate-token-000000000000000000000"
        )
        missing_csrf = await client.post(
            "/api/v1/runs",
            headers=operate_headers,
            json={"configuration": _run_configuration(), "run_id": "run-api"},
        )
        assert missing_csrf.status_code == 403
        mutation_headers = {
            **operate_headers,
            "X-CSRF-Token": operate_csrf,
            "Origin": "http://127.0.0.1:8765",
        }
        first = await client.post(
            "/api/v1/runs",
            headers=mutation_headers,
            json={"configuration": _run_configuration(), "run_id": "run-api"},
        )
        second = await client.post(
            "/api/v1/runs",
            headers=mutation_headers,
            json={"configuration": _run_configuration(), "run_id": "run-api"},
        )
        assert first.json()["status"] == "PREPARED"
        assert second.json()["status"] == "IDEMPOTENT_EXISTING_RUN"
        bad_origin = await client.post(
            "/api/v1/runs",
            headers={
                **operate_headers,
                "X-CSRF-Token": operate_csrf,
                "Origin": "https://remote.example",
            },
            json={"configuration": _run_configuration(), "run_id": "other"},
        )
        assert bad_origin.status_code == 403


@pytest.mark.anyio
async def test_approval_workflow_sse_and_private_model_path_redaction(
    app: object,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://127.0.0.1:8765",
    ) as client:
        operate_headers, operate_csrf = await _session(
            client, "operate-token-000000000000000000000"
        )
        approval = await client.post(
            "/api/v1/approvals",
            headers={**operate_headers, "X-CSRF-Token": operate_csrf},
            json={
                "action": "START_MODEL_SERVERS",
                "entity_id": "phase3",
                "payload": {},
            },
        )
        assert approval.status_code == 200
        approval_id = approval.json()["approval_id"]
        approve_headers, approve_csrf = await _session(
            client, "approve-token-000000000000000000000"
        )
        decision = await client.post(
            f"/api/v1/approvals/{approval_id}/decision",
            headers={**approve_headers, "X-CSRF-Token": approve_csrf},
            json={"approve": True},
        )
        assert decision.json()["status"] == "APPROVED"
        events = await client.get(
            "/api/v1/events?once=true",
            headers=operate_headers,
        )
        assert events.status_code == 200
        assert "event: operator_event" in events.text
        assert "APPROVAL_APPROVED" in events.text
        model_status = await client.get(
            "/api/v1/model-servers/status", headers=operate_headers
        )
        assert "/private/model" not in model_status.text


@pytest.mark.anyio
async def test_research_report_and_artifact_security(app: object) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://127.0.0.1:8765",
    ) as client:
        headers, csrf = await _session(
            client, "operate-token-000000000000000000000"
        )
        mutation_headers = {**headers, "X-CSRF-Token": csrf}
        created = await client.post(
            "/api/v1/research/jobs",
            headers=mutation_headers,
            json={
                "research_job_id": "api-research",
                "job": {
                    "question": "What does the fixture support?",
                    "scope": "fixture",
                    "research_mode": "quick",
                    "search_backends": ["fixture"],
                    "source_policy": "primary_preferred",
                    "model_route": "deterministic",
                },
            },
        )
        assert created.status_code == 200
        completed = await client.post(
            "/api/v1/research/jobs/api-research/run",
            headers=mutation_headers,
        )
        assert completed.json()["status"] == "COMPLETE"
        report = await client.get(
            "/api/v1/research/jobs/api-research/report", headers=headers
        )
        assert report.status_code == 200
        assert report.json()["evidence_ledger"]["claims"][0]["supporting_source_ids"]

        traversal = await client.get(
            "/api/v1/artifacts",
            params={"path": "../../etc/passwd"},
            headers=headers,
        )
        assert traversal.status_code in {400, 404}


@pytest.mark.anyio
async def test_static_gui_is_responsive_pwa_shell(app: object) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://127.0.0.1:8765",
    ) as client:
        page = await client.get("/")
        assert page.status_code == 200
        assert 'id="chat"' in page.text
        assert 'id="research"' in page.text
        assert 'id="auth-dialog"' in page.text
        assert 'type="email"' in page.text
        assert "Access token" not in page.text
        assert "viewport" in page.text
        css = await client.get("/assets/app.css")
        assert "@media (max-width: 760px)" in css.text
        javascript = await client.get("/assets/app.js")
        assert "localStorage" not in javascript.text
        assert "sessionStorage" not in javascript.text
        assert ".innerHTML" not in javascript.text
        manifest = await client.get("/manifest.webmanifest")
        assert manifest.json()["display"] == "standalone"
        worker = await client.get("/sw.js")
        assert "cache.add" not in worker.text
        assert "caches.match" not in worker.text
        assert "caches.delete" in worker.text
        assert "default-src 'self'" in page.headers["content-security-policy"]
