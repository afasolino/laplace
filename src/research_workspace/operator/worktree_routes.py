"""Worktree and operator-inventory routes for the local Operator Plane."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Response

from ..operator_service import OperatorService
from ..personal_corpus import PersonalCorpusStore
from ..service_tiers import TieredServingService
from .auth import AuthPrincipal
from .request_models import WorktreeDiscardRequest, WorktreeExportRequest

JsonObject = dict[str, object]
PrincipalDependency = Callable[..., Awaitable[AuthPrincipal]]
TieredService = Callable[[], TieredServingService]


def register_worktree_routes(
    app: FastAPI,
    *,
    operator: OperatorService,
    corpus_store: PersonalCorpusStore,
    require_tiered: TieredService,
    agent_principal: PrincipalDependency,
    agent_mutation_principal: PrincipalDependency,
    operator_principal: PrincipalDependency,
    mutation_principal: PrincipalDependency,
) -> None:
    """Register worktree routes with facade-owned authorization decisions."""

    @app.get("/api/v1/worktrees")
    async def list_worktrees(
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "worktrees": require_tiered().sandboxes.list_mine(authenticated.user_id),
        }

    @app.get("/api/v1/worktrees/{session_id}")
    async def inspect_worktree(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "worktree": require_tiered().sandboxes.inspect(
                session_id, user_id=authenticated.user_id
            ),
        }

    @app.get("/api/v1/worktrees/{session_id}/history")
    async def worktree_history(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "events": require_tiered().sandboxes.history(session_id, user_id=authenticated.user_id),
        }

    @app.post("/api/v1/worktrees/{session_id}/resume")
    async def resume_worktree(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_mutation_principal),
    ) -> JsonObject:
        return {
            "status": "RESUMED",
            "worktree": require_tiered().sandboxes.resume(
                session_id, user_id=authenticated.user_id
            ),
        }

    @app.post("/api/v1/worktrees/{session_id}/close")
    async def close_worktree(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_mutation_principal),
    ) -> JsonObject:
        return require_tiered().sandboxes.close_if_clean(session_id, user_id=authenticated.user_id)

    @app.post("/api/v1/worktrees/{session_id}/discard")
    async def discard_worktree(
        session_id: str,
        body: WorktreeDiscardRequest,
        authenticated: AuthPrincipal = Depends(agent_mutation_principal),
    ) -> JsonObject:
        return require_tiered().sandboxes.discard(
            session_id,
            user_id=authenticated.user_id,
            confirmation=body.confirmation,
        )

    @app.post("/api/v1/worktrees/{session_id}/export")
    async def request_worktree_export(
        session_id: str,
        body: WorktreeExportRequest,
        authenticated: AuthPrincipal = Depends(agent_mutation_principal),
    ) -> JsonObject:
        return require_tiered().sandboxes.request_export(
            session_id,
            user_id=authenticated.user_id,
            promotion=body.promotion,
        )

    @app.get("/api/v1/worktrees/{session_id}/patch")
    async def download_worktree_patch(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> Response:
        content = require_tiered().sandboxes.patch(session_id, user_id=authenticated.user_id)
        return Response(
            content,
            media_type="text/x-diff",
            headers={
                "Content-Disposition": (f'attachment; filename="{session_id}.patch"'),
                "X-Content-SHA256": hashlib.sha256(content).hexdigest(),
            },
        )

    @app.get("/api/v1/admin/worktrees")
    async def operator_worktree_inventory(
        _authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "worktrees": require_tiered().sandboxes.operator_inventory(),
        }

    @app.get("/api/v1/admin/personal-corpora")
    async def operator_personal_corpus_inventory(
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> JsonObject:
        inventory = corpus_store.sanitized_inventory()
        operator.record_action(
            actor_role=authenticated.role,
            action="PERSONAL_CORPUS_SANITIZED_INVENTORY_VIEWED",
            entity_type="personal_corpus_inventory",
            entity_id="all",
            payload={"corpus_count": len(inventory), "content_access": False},
        )
        return {
            "status": "OK",
            "content_access": "DISABLED_BY_POLICY",
            "corpora": inventory,
        }

    @app.get("/api/v1/admin/worktrees/{session_id}")
    async def operator_inspect_worktree(
        session_id: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "worktree": require_tiered().sandboxes.inspect(
                session_id,
                user_id=authenticated.user_id,
                operator=True,
            ),
        }

    @app.post("/api/v1/admin/worktrees/{session_id}/discard")
    async def operator_discard_worktree(
        session_id: str,
        body: WorktreeDiscardRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> JsonObject:
        result = require_tiered().sandboxes.discard(
            session_id,
            user_id=authenticated.user_id,
            confirmation=body.confirmation,
            operator=True,
        )
        operator.record_action(
            actor_role=authenticated.role,
            action="WORKTREE_FORCE_DISCARD",
            entity_type="worktree",
            entity_id=session_id,
            payload={"status": result["status"]},
        )
        return result
