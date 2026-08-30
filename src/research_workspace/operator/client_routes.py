"""Owner-scoped Operator routes for paired client devices."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI

from ..client_service import ClientDeviceStore
from .auth import AuthPrincipal
from .request_models import (
    ClientHeartbeatRequest,
    ClientOperationRequest,
    ClientPairRequest,
    ClientResultRequest,
)

JsonObject = dict[str, object]
PrincipalDependency = Callable[..., Awaitable[AuthPrincipal]]


def register_client_routes(
    app: FastAPI,
    *,
    client_devices: ClientDeviceStore,
    agent_principal: PrincipalDependency,
    client_mutation_principal: PrincipalDependency,
) -> None:
    """Register client routes while retaining facade-owned authorization policy."""

    @app.post("/api/v1/client/devices/pair")
    async def pair_client_device(
        body: ClientPairRequest,
        authenticated: AuthPrincipal = Depends(client_mutation_principal),
    ) -> JsonObject:
        return client_devices.pair(
            authenticated.user_id,
            name=body.name,
            capabilities=body.capabilities,
            device_id=body.device_id,
        )

    @app.get("/api/v1/client/devices")
    async def list_client_devices(
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        return {"devices": client_devices.list_devices(authenticated.user_id)}

    @app.delete("/api/v1/client/devices/{device_id}")
    async def revoke_client_device(
        device_id: str,
        authenticated: AuthPrincipal = Depends(client_mutation_principal),
    ) -> JsonObject:
        return client_devices.revoke(authenticated.user_id, device_id)

    @app.post("/api/v1/client/devices/{device_id}/heartbeat")
    async def client_heartbeat(
        device_id: str,
        body: ClientHeartbeatRequest,
        authenticated: AuthPrincipal = Depends(client_mutation_principal),
    ) -> JsonObject:
        return client_devices.heartbeat(
            authenticated.user_id,
            device_id,
            body.capabilities,
        )

    @app.post("/api/v1/client/devices/{device_id}/operations")
    async def create_client_operation(
        device_id: str,
        body: ClientOperationRequest,
        authenticated: AuthPrincipal = Depends(client_mutation_principal),
    ) -> JsonObject:
        return client_devices.enqueue(
            authenticated.user_id,
            device_id,
            workspace_id=body.workspace_id,
            action=body.action,
            arguments=body.arguments,
        )

    @app.get("/api/v1/client/devices/{device_id}/operations/next")
    async def next_client_operation(
        device_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        operation = client_devices.claim(authenticated.user_id, device_id)
        return {"operation": operation}

    @app.post("/api/v1/client/devices/{device_id}/operations/{operation_id}/result")
    async def complete_client_operation(
        device_id: str,
        operation_id: str,
        body: ClientResultRequest,
        authenticated: AuthPrincipal = Depends(client_mutation_principal),
    ) -> JsonObject:
        return client_devices.complete(
            authenticated.user_id,
            device_id,
            operation_id,
            result=body.result,
            failed=body.failed,
        )

    @app.get("/api/v1/client/operations/{operation_id}")
    async def get_client_operation(
        operation_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        return client_devices.get_operation(authenticated.user_id, operation_id)

    @app.post("/api/v1/client/operations/{operation_id}/cancel")
    async def cancel_client_operation(
        operation_id: str,
        authenticated: AuthPrincipal = Depends(client_mutation_principal),
    ) -> JsonObject:
        return client_devices.cancel(authenticated.user_id, operation_id)
