"""Run, approval, and model-server routes for the Operator transport."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Query

from ..operator_service import OperatorService, RunExecutor
from .auth import AuthPrincipal
from .request_models import (
    ApprovalDecisionRequest,
    ApprovalRequest,
    ModelServerActionRequest,
    RunPrepareRequest,
    StartRunRequest,
)

JsonObject = dict[str, object]
PrincipalDependency = Callable[..., Awaitable[AuthPrincipal]]
ModelStatus = Callable[[str], JsonObject]


def register_run_routes(
    app: FastAPI,
    *,
    operator: OperatorService,
    run_executor: RunExecutor | None,
    operator_principal: PrincipalDependency,
    mutation_principal: PrincipalDependency,
    model_admin_principal: PrincipalDependency,
    model_admin_mutation_principal: PrincipalDependency,
    sanitized_model_status: ModelStatus,
) -> None:
    """Register run routes while retaining facade-owned authorization policy."""

    @app.post("/api/v1/runs")
    async def prepare_run(
        body: RunPrepareRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> JsonObject:
        return operator.prepare_run(
            body.configuration,
            actor_role=authenticated.role,
            run_id=body.run_id,
        )

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(
        run_id: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> JsonObject:
        return operator.get_run(run_id, actor_role=authenticated.role)

    @app.post("/api/v1/runs/{run_id}/start")
    async def start_run(
        run_id: str,
        body: StartRunRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> JsonObject:
        return operator.start_run(
            run_id,
            approval_id=body.approval_id,
            actor_role=authenticated.role,
            executor=run_executor,
        )

    @app.post("/api/v1/approvals")
    async def request_approval(
        body: ApprovalRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> JsonObject:
        return operator.request_approval(
            body.action,
            body.entity_id,
            body.payload,
            actor_role=authenticated.role,
        )

    @app.get("/api/v1/approvals")
    async def list_approvals(
        authenticated: AuthPrincipal = Depends(operator_principal),
        state: str | None = Query(default=None),
    ) -> JsonObject:
        return {
            "status": "OK",
            "approvals": operator.approvals(actor_role=authenticated.role, state=state),
        }

    @app.post("/api/v1/approvals/{approval_id}/decision")
    async def decide_approval(
        approval_id: str,
        body: ApprovalDecisionRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> JsonObject:
        return operator.decide_approval(
            approval_id,
            approve=body.approve,
            actor_role=authenticated.role,
        )

    @app.post("/api/v1/model-servers/action")
    async def model_server_action(
        body: ModelServerActionRequest,
        authenticated: AuthPrincipal = Depends(model_admin_mutation_principal),
    ) -> JsonObject:
        return operator.model_server_action(
            body.action,
            approval_id=body.approval_id,
            actor_role=authenticated.role,
        )

    @app.get("/api/v1/model-servers/status")
    async def model_server_status(
        authenticated: AuthPrincipal = Depends(model_admin_principal),
    ) -> JsonObject:
        return sanitized_model_status(authenticated.role)
