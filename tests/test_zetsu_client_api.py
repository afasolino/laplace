from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from research_workspace.operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.service_tiers import LanePolicy, ModelLane, ModelRoute
from research_workspace.user_capabilities import Capability, CapabilityTier


ROOT = Path(__file__).resolve().parents[1]
TOKENS = {
    "owner-token-000000000000000000000000": AuthCredential(
        "read", "owner", CapabilityTier.PLUS
    ),
    "other-token-000000000000000000000000": AuthCredential(
        "read", "other", CapabilityTier.PLUS
    ),
}


class _Tiered:
    def __init__(self) -> None:
        self.lane_policy = LanePolicy(
            routes={
                ModelLane.QUALITY: ModelRoute(
                    ModelLane.QUALITY, "laplace-qwen38", "http://127.0.0.1:8206", 0, 32768, 4096
                ),
                ModelLane.STANDARD: ModelRoute(
                    ModelLane.STANDARD, "laplace-qwen38", "http://127.0.0.1:8206", 10, 16384, 2048
                ),
                ModelLane.ECONOMY: ModelRoute(
                    ModelLane.ECONOMY, "laplace-codev", "http://127.0.0.1:8103", 20, 8192, 2048
                ),
            },
            quality_reserved_slots=1,
            standard_capacity=4,
            economy_capacity=4,
        )

    def effective_capabilities(self, _user_id: str) -> frozenset[Capability]:
        return frozenset({Capability.CHAT, Capability.AGENT, Capability.PERSONAL_CORPUS})


@pytest.fixture
def app(tmp_path: Path) -> object:
    return create_operator_app(
        OperatorService(ROOT, tmp_path / "state"),
        OperatorAuth(TOKENS),
        settings=OperatorApiSettings(fixture_mode=True, bearer_api_enabled=True),
        tiered=_Tiered(),  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_zetsu_mcp_requires_normal_bearer_and_runs_real_retrieval(app: object) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://127.0.0.1:8765",
    ) as client:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}},
        }
        assert (await client.post("/mcp", json=request)).status_code == 401
        headers = {"Authorization": "Bearer owner-token-000000000000000000000000"}
        initialized = await client.post("/mcp", headers=headers, json=request)
        assert initialized.status_code == 200
        assert initialized.json()["result"]["serverInfo"]["name"] == "Laplace Zetsu"

        listed = await client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert {item["name"] for item in listed.json()["result"]["tools"]} == {
            "search",
            "get_evidence",
            "project_context",
            "experiment_context",
            "delegate",
            "rtl_task",
            "verify",
        }
        searched = await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "connectivity probe", "max_chars": 512},
                },
            },
        )
        result = searched.json()["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["grounded"] is False
        bad_origin = await client.post(
            "/mcp",
            headers={**headers, "Origin": "https://attacker.invalid"},
            json=request,
        )
        assert bad_origin.status_code == 403


@pytest.mark.anyio
async def test_client_pair_queue_result_and_cross_user_isolation(app: object) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://127.0.0.1:8765",
    ) as client:
        owner = {"Authorization": "Bearer owner-token-000000000000000000000000"}
        other = {"Authorization": "Bearer other-token-000000000000000000000000"}
        paired = await client.post(
            "/api/v1/client/devices/pair",
            headers=owner,
            json={"name": "pc", "capabilities": {"tools": {"python": True}}},
        )
        assert paired.status_code == 200
        device_id = paired.json()["device_id"]
        assert (await client.get("/api/v1/client/devices", headers=other)).json()["devices"] == []
        denied = await client.post(
            f"/api/v1/client/devices/{device_id}/operations",
            headers=other,
            json={
                "workspace_id": "ws-" + "a" * 24,
                "action": "read",
                "arguments": {"path": "a.txt"},
            },
        )
        assert denied.status_code == 404

        created = await client.post(
            f"/api/v1/client/devices/{device_id}/operations",
            headers=owner,
            json={
                "workspace_id": "ws-" + "a" * 24,
                "action": "read",
                "arguments": {"path": "a.txt"},
            },
        )
        operation_id = created.json()["operation_id"]
        claimed = await client.get(
            f"/api/v1/client/devices/{device_id}/operations/next", headers=owner
        )
        assert claimed.json()["operation"]["state"] == "CLAIMED"
        completed = await client.post(
            f"/api/v1/client/devices/{device_id}/operations/{operation_id}/result",
            headers=owner,
            json={"result": {"content": "ok"}, "failed": False},
        )
        assert completed.json()["state"] == "COMPLETE"
