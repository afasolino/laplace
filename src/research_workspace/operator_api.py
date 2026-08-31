"""Authenticated versioned HTTP adapter for the local Operator Plane."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Literal, Mapping, TypeAlias
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from .artifact_registry import ArtifactRegistry, ArtifactRegistryError
from .auth_registry import AuthRegistryError
from .auth_sessions import (
    AuthSessionError,
    NewSession,
    RegisteredEmailAuth,
)
from .conversations import ConversationError, ConversationStore
from .client_service import ClientDeviceStore, ClientServiceError
from .contracts import ProviderCapabilitiesV1, ProviderV1, RouteV1
from .domain_registry import (
    DEFAULT_DOMAIN_REGISTRY,
    DomainRegistry,
    DomainRegistryError,
)
from .operator_service import (
    OperatorService,
    OperatorServiceError,
    RunExecutor,
)
from .operator.auth import AuthCredential, AuthPrincipal, OperatorAuth  # noqa: F401
from .operator.agent_requests import (  # noqa: F401
    AgentAsyncRunRequest,
    AgentRunRequest,
    AgentTaskComplexityRequest,
)
from .operator.request_models import (  # noqa: F401
    ActivationRequest,
    AgentSessionRequest,
    ApprovalDecisionRequest,
    ApprovalRequest,
    CapabilitySetRequest,
    ClientHeartbeatRequest,
    ClientOperationRequest,
    ClientPairRequest,
    ClientResultRequest,
    ConversationCreateRequest,
    ConversationUpdateRequest,
    CorpusCreateRequest,
    CorpusSearchRequest,
    CorpusUpdateRequest,
    IndexUploadRequest,
    LoginRequest,
    ModelServerActionRequest,
    PasswordChangeRequest,
    RepositoryGrantRequest,
    RepositoryRegistrationRequest,
    ResearchCreateRequest,
    RunPrepareRequest,
    ServingProfileActionRequest,
    StartRunRequest,
    TierChatMessage,
    TierChatRequest,
    TierUserRequest,
    UploadCreateRequest,
)
from .operator.settings import OperatorApiSettings
from .operator.responses import agent_result_content, public_agent_result
from .operator.research_payloads import research_summaries, sanitize_research_payload
from .operator.artifacts import safe_artifact_path
from .operator.auth_routes import register_auth_routes
from .operator.client_routes import register_client_routes
from .operator.json_utils import canonical_json_bytes, sorted_json_text
from .operator.run_routes import register_run_routes
from .operator.static_routes import register_static_routes
from .operator.worktree_routes import register_worktree_routes
from .laplace_core import LaplaceCore
from .memory import MemoryService, SQLiteMemoryBackend
from .personal_corpus import (
    CorpusError,
    PersonalCorpusStore,
    RetrievalSelection,
)
from .request_state import RequestStateError, RequestStateStore
from .rules import RuleService, SQLiteRuleBackend
from .trajectory import TrajectoryService
from .agent_sandbox import AgentSandboxError, AgentSessionBinding, AgentToolPolicy
from .research_admission import ResearchAdmissionError, ResearchAdmissionStore
from .research_plane import DeepResearchService, ResearchPlaneError
from .research_web_adapters import supported_web_adapter_names
from .repository_authorization import RepositoryAuthorizationError
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .task_labels import derive_task_label
from .serving_profile_runtime import (
    ServingProfileOperator,
    ServingRuntimeError,
)
from .user_capabilities import (
    Capability,
    CapabilityTier,
    UserCapabilityError,
    default_capabilities,
)
from .versioning import version_record
from .zetsu_mcp import ZetsuError, ZetsuMcpDispatcher, ZetsuService

JsonObject: TypeAlias = dict[str, object]
LOGGER = logging.getLogger("laplace.operator")


def _agent_async_request_sha256(body: AgentAsyncRunRequest) -> str:
    """Bind one durable turn ID to the exact normalized API request."""

    payload = {
        "instruction": body.instruction,
        "lane": body.lane,
        "domain": body.domain,
        "retrieval_selection": body.retrieval_selection,
        "personal_corpus_id": body.personal_corpus_id,
        "max_steps": body.max_steps,
        "max_chars": body.max_chars,
        "verification_argv": body.verification_argv,
        "allow_mutation": body.allow_mutation,
        "wait_timeout_seconds": body.wait_timeout_seconds,
        "manager_complexity": (
            body.manager_complexity.model_dump() if body.manager_complexity is not None else None
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def create_operator_app(
    operator: OperatorService,
    auth: OperatorAuth,
    *,
    settings: OperatorApiSettings = OperatorApiSettings(),
    research: DeepResearchService | None = None,
    run_executor: RunExecutor | None = None,
    tiered: TieredServingService | None = None,
    core: LaplaceCore | None = None,
    zetsu_enabled: bool = True,
    memory: MemoryService | None = None,
    rules: RuleService | None = None,
    trajectory: TrajectoryService | None = None,
    serving_profile_operator: ServingProfileOperator | None = None,
    registered_auth: RegisteredEmailAuth | None = None,
    conversation_store: ConversationStore | None = None,
    artifact_registry: ArtifactRegistry | None = None,
    research_admission: ResearchAdmissionStore | None = None,
    personal_corpora: PersonalCorpusStore | None = None,
    domain_registry: DomainRegistry = DEFAULT_DOMAIN_REGISTRY,
    request_states: RequestStateStore | None = None,
) -> FastAPI:
    """Create the localhost GUI/API application without changing execution semantics."""

    if settings.bind_host not in {"127.0.0.1", "localhost", "::1"} and auth is None:
        raise ValueError("authentication is mandatory for non-loopback binding")
    web_root = Path(__file__).with_name("operator_web")
    corpus_store = personal_corpora or PersonalCorpusStore(operator.state_root)
    progress_store = request_states or RequestStateStore(
        operator.state_root / "requests/request_states.sqlite3"
    )
    client_devices = ClientDeviceStore(operator.state_root / "client/client_devices.sqlite3")
    memory_service = memory or MemoryService(
        SQLiteMemoryBackend(operator.state_root / "memory/memory.sqlite3")
    )
    rules_service = rules or RuleService(
        SQLiteRuleBackend(operator.state_root / "rules/rules.sqlite3")
    )
    trajectory_service = trajectory or TrajectoryService(operator.state_root / "trajectory")
    shared_core = (
        core
        if core is not None
        else (
            LaplaceCore(
                operator.repository_root,
                corpus_store,
                tiered,
                memory=memory_service,
                rules=rules_service,
                trajectory=trajectory_service,
            )
            if tiered is not None
            else None
        )
    )
    zetsu_service = (
        ZetsuService(operator.repository_root, corpus_store, tiered, core=shared_core)
        if tiered is not None and zetsu_enabled
        else None
    )
    zetsu_dispatcher = ZetsuMcpDispatcher(zetsu_service) if zetsu_service is not None else None
    app = FastAPI(
        title="Laplace Research and Operator Plane",
        version="1.1",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )
    agent_turn_tasks: dict[tuple[str, str], tuple[str, asyncio.Task[None]]] = {}
    agent_turn_lock = asyncio.Lock()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-CSRF-Token",
            "Authorization",
            "X-Request-ID",
            "MCP-Protocol-Version",
            "Mcp-Method",
            "Mcp-Name",
        ],
        expose_headers=["X-Trace-Id", "X-Content-SHA256", "MCP-Protocol-Version"],
        max_age=600,
    )

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = request.headers.get("x-request-id") or f"trace-{uuid.uuid4().hex}"
        request.state.trace_id = trace_id
        started = time.monotonic()
        host = urlsplit(f"//{request.headers.get('host', '')}").hostname or ""
        response: Response | None = None
        if host not in settings.allowed_hosts:
            response = JSONResponse(
                status_code=400,
                content={"status": "ERROR", "failure_category": "host_not_allowed"},
                headers={"X-Trace-Id": trace_id},
            )
        forwarded = any(
            name in request.headers
            for name in ("x-forwarded-for", "x-forwarded-host", "x-forwarded-proto")
        )
        transport_client = request.client.host if request.client is not None else ""
        if response is None and forwarded and transport_client not in settings.trusted_proxies:
            response = JSONResponse(
                status_code=400,
                content={
                    "status": "ERROR",
                    "failure_category": "untrusted_forwarded_headers",
                },
                headers={"X-Trace-Id": trace_id},
            )
        request.state.client_ip = transport_client
        if response is None and forwarded:
            forwarded_for = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
            try:
                request.state.client_ip = str(ipaddress.ip_address(forwarded_for))
            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content={
                        "status": "ERROR",
                        "failure_category": "invalid_forwarded_client",
                    },
                    headers={"X-Trace-Id": trace_id},
                )
        if (
            response is None
            and settings.deployment_mode == "reverse-proxy"
            and request.url.path in {"/api/v1/auth/login", "/api/v1/auth/activate"}
        ):
            forwarded_host = urlsplit(f"//{request.headers.get('x-forwarded-host', '')}").hostname
            if (
                request.headers.get("x-forwarded-proto") != "https"
                or forwarded_host not in settings.allowed_hosts
            ):
                response = JSONResponse(
                    status_code=400,
                    content={
                        "status": "ERROR",
                        "failure_category": "trusted_https_proxy_required",
                    },
                    headers={"X-Trace-Id": trace_id},
                )
        length = request.headers.get("content-length")
        if response is None and length is not None:
            try:
                too_large = int(length) > settings.maximum_request_bytes
            except ValueError:
                too_large = True
            if too_large:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "status": "ERROR",
                        "failure_category": "request_body_too_large",
                    },
                    headers={"X-Trace-Id": trace_id},
                )
        if response is None:
            response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'; "
            "require-trusted-types-for 'script'; trusted-types laplace-markdown"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = (
            "no-store"
            if request.url.path.startswith(("/api/", "/auth/"))
            else "no-cache, no-store, must-revalidate"
        )
        response.headers["X-Trace-Id"] = trace_id
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if settings.secure_cookie:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        LOGGER.info(
            json.dumps(
                {
                    "event": "HTTP_REQUEST",
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return response

    async def principal(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> AuthPrincipal:
        cookie = request.cookies.get("laplace_session")
        if cookie is not None and registered_auth is not None:
            try:
                record = registered_auth.sessions.resolve(cookie, registered_auth.registry)
            except AuthSessionError as exc:
                raise HTTPException(status_code=401, detail=exc.category) from exc
            return AuthPrincipal(
                role=record.role,
                user_id=record.user_id,
                capability_tier=record.capability_tier,
                credential_sha256=record.session_hash,
                email=record.email,
                display_name=record.display_name,
                default_lane=record.default_lane,
                auth_method="session",
                session_identifier=cookie,
            )
        if not settings.bearer_api_enabled:
            raise HTTPException(status_code=401, detail="authentication_required")
        return auth.authenticate(authorization)

    def _effective_capabilities(authenticated: AuthPrincipal) -> frozenset[Capability]:
        if tiered is not None:
            try:
                return tiered.effective_capabilities(authenticated.user_id)
            except UserCapabilityError:
                if authenticated.auth_method == "session":
                    raise HTTPException(status_code=401, detail="credential_capability_changed")
        return default_capabilities(authenticated.capability_tier)

    def _require_named(authenticated: AuthPrincipal, required: Capability) -> AuthPrincipal:
        if required not in _effective_capabilities(authenticated):
            raise HTTPException(
                status_code=403,
                detail=f"{required.value}_capability_required",
            )
        return authenticated

    async def operator_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.OPERATOR)

    async def admin_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.ADMIN)

    async def personal_corpus_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.PERSONAL_CORPUS)

    async def research_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.RESEARCH)

    async def chat_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.CHAT)

    async def agent_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.AGENT)

    async def model_admin_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        return _require_named(authenticated, Capability.MODEL_ADMIN)

    async def mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(operator_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def tier_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def research_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        _require_named(authenticated, Capability.RESEARCH)
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def corpus_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def chat_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(chat_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def agent_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(agent_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def client_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(agent_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        if authenticated.auth_method == "session":
            _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def admin_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(admin_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def repository_admin_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        _require_named(authenticated, Capability.REPOSITORY_ADMIN)
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def model_admin_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        _require_named(authenticated, Capability.MODEL_ADMIN)
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        _validate_csrf(authenticated, x_csrf_token)
        return authenticated

    def _validate_csrf(
        authenticated: AuthPrincipal,
        supplied: str | None,
    ) -> None:
        if authenticated.auth_method == "session":
            if registered_auth is None or authenticated.session_identifier is None:
                raise HTTPException(status_code=401, detail="authentication_required")
            try:
                registered_auth.sessions.validate_csrf(
                    authenticated.session_identifier,
                    supplied,
                )
            except AuthSessionError as exc:
                raise HTTPException(status_code=403, detail=exc.category) from exc
        else:
            auth.validate_csrf(authenticated, supplied)

    @app.exception_handler(OperatorServiceError)
    async def operator_error(_request: Request, exc: OperatorServiceError) -> JSONResponse:
        status_code = (
            403
            if exc.category
            in {
                "authorization_failure",
                "approval_required",
            }
            else 409
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ERROR",
                "failure_category": exc.category,
                "evidence": exc.evidence,
            },
        )

    @app.exception_handler(ResearchPlaneError)
    async def research_error(_request: Request, exc: ResearchPlaneError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "status": "ERROR",
                "failure_category": exc.category,
                "evidence": exc.evidence,
            },
        )

    @app.exception_handler(ServiceTierError)
    @app.exception_handler(UserCapabilityError)
    @app.exception_handler(RepositoryAuthorizationError)
    @app.exception_handler(AgentSandboxError)
    @app.exception_handler(ServingRuntimeError)
    @app.exception_handler(DomainRegistryError)
    async def tier_error(
        request: Request,
        exc: (
            AgentSandboxError
            | ServiceTierError
            | UserCapabilityError
            | RepositoryAuthorizationError
            | ServingRuntimeError
            | DomainRegistryError
        ),
    ) -> JSONResponse:
        category = getattr(exc, "category", str(exc))
        evidence = getattr(exc, "evidence", {})
        LOGGER.warning(
            json.dumps(
                {
                    "event": "POLICY_REQUEST_REJECTED",
                    "trace_id": str(getattr(request.state, "trace_id", "unavailable")),
                    "method": request.method,
                    "path": request.url.path,
                    "failure_category": category,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return JSONResponse(
            status_code=403,
            content={
                "status": "ERROR",
                "failure_category": category,
                "evidence": evidence,
            },
        )

    @app.exception_handler(CorpusError)
    @app.exception_handler(RequestStateError)
    async def corpus_or_request_error(
        _request: Request, exc: CorpusError | RequestStateError
    ) -> JSONResponse:
        category = exc.category
        if category.endswith("_not_found"):
            status_code = 404
        elif "quota" in category or category in {
            "disk_pressure",
            "request_body_too_large",
        }:
            status_code = 413
        elif category in {
            "corpus_not_found",
            "source_not_found",
            "upload_not_found",
        }:
            status_code = 404
        else:
            status_code = 409
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ERROR",
                "failure_category": category,
                "evidence": getattr(exc, "evidence", {}),
            },
        )

    @app.exception_handler(ClientServiceError)
    async def client_service_error(_request: Request, exc: ClientServiceError) -> JSONResponse:
        status_code = 404 if exc.category.endswith("not_found") else 409
        return JSONResponse(
            status_code=status_code,
            content={"status": "ERROR", "failure_category": exc.category},
        )

    @app.exception_handler(ConversationError)
    @app.exception_handler(ArtifactRegistryError)
    @app.exception_handler(ResearchAdmissionError)
    async def isolated_resource_error(
        _request: Request,
        exc: ConversationError | ArtifactRegistryError | ResearchAdmissionError,
    ) -> JSONResponse:
        category = exc.category
        status_code = 404 if category.endswith("not_found") else 409
        if category in {
            "artifact_integrity_failure",
            "capacity_guardrail",
            "research_job_cancelled",
        }:
            status_code = 409
        return JSONResponse(
            status_code=status_code,
            content={"status": "ERROR", "failure_category": category},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.error(
            json.dumps(
                {
                    "event": "UNEXPECTED_ERROR",
                    "trace_id": str(getattr(request.state, "trace_id", "unavailable")),
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "failure_category": "internal_error",
                "trace_id": str(getattr(request.state, "trace_id", "unavailable")),
            },
        )

    def _origin_allowed(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")

    def _set_session_cookie(response: Response, session_value: str) -> None:
        response.set_cookie(
            key="laplace_session",
            value=session_value,
            max_age=None,
            httponly=True,
            secure=settings.secure_cookie,
            samesite="strict",
            path="/",
        )

    def _session_response(session_value: NewSession) -> JsonObject:
        user = (
            registered_auth.registry.require_user(session_value.record.user_id)
            if registered_auth
            else None
        )
        return {
            "status": "AUTHENTICATED",
            "account": session_value.record.public_account(),
            "role": session_value.record.role,
            "capability_tier": session_value.record.capability_tier.value,
            "default_lane": session_value.record.default_lane,
            "must_change_password": user.must_change_password if user is not None else False,
            "csrf_token": session_value.csrf_token,
            "development_http": settings.development_http,
            "deployment_mode": settings.deployment_mode,
        }

    register_static_routes(app, web_root=web_root, pwa_enabled=settings.pwa_enabled)

    @app.get("/api/v1/health")
    async def health() -> dict[str, object]:
        return {
            "status": "OK",
            "local_only": settings.bind_host in {"127.0.0.1", "localhost", "::1"},
            "api_version": "v1",
            "fixture_mode": settings.fixture_mode,
            "development_http": settings.development_http,
            "deployment_mode": settings.deployment_mode,
            "personal_corpus": corpus_store.health(),
        }

    @app.get("/api/v1/readiness")
    async def readiness() -> JSONResponse:
        reasons: list[str] = []
        if registered_auth is None:
            reasons.append("registered_email_auth_unavailable")
        else:
            try:
                registered_auth.registry.snapshot
                registered_auth.sessions.active_count()
            except (AuthRegistryError, AuthSessionError, OSError, sqlite3.Error) as exc:
                reasons.append(f"authentication_state:{type(exc).__name__}")
        if tiered is None:
            reasons.append("lane_routing_unavailable")
        elif not settings.fixture_mode:
            unique_routes = {
                (route.endpoint, route.model_id): route
                for route in tiered.lane_policy.routes.values()
                if tiered.lane_policy.codev_enabled or route.lane is not ModelLane.ECONOMY
            }

            async def endpoint_failure(
                endpoint: str,
                model_id: str,
                lane: str,
            ) -> str | None:
                parsed = urlsplit(endpoint)
                if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
                    return f"model_endpoint_non_local:{lane}"
                writer: asyncio.StreamWriter | None = None
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(
                            parsed.hostname,
                            parsed.port or 80,
                        ),
                        timeout=2,
                    )
                    assert writer is not None
                    base_path = parsed.path.rstrip("/")
                    path = (
                        base_path + "/models"
                        if base_path.endswith("/v1")
                        else base_path + "/v1/models"
                    )
                    writer.write(
                        (
                            f"GET {path} HTTP/1.1\r\n"
                            f"Host: {parsed.hostname}:{parsed.port or 80}\r\n"
                            "Accept: application/json\r\n"
                            "Connection: close\r\n\r\n"
                        ).encode("ascii")
                    )
                    await asyncio.wait_for(writer.drain(), timeout=2)
                    header = await asyncio.wait_for(
                        reader.readuntil(b"\r\n\r\n"),
                        timeout=2,
                    )
                    status_line = header.split(b"\r\n", 1)[0].split()
                    if len(status_line) < 2 or status_line[1] != b"200":
                        return f"model_endpoint_unavailable:{lane}"
                    length_match = re.search(
                        rb"(?im)^content-length:\s*(\d+)\s*$",
                        header,
                    )
                    if length_match is not None:
                        length = int(length_match.group(1))
                        if length > 2_000_000:
                            return f"model_endpoint_invalid_response:{lane}"
                        body = await asyncio.wait_for(
                            reader.readexactly(length),
                            timeout=2,
                        )
                    else:
                        body = await asyncio.wait_for(
                            reader.read(2_000_001),
                            timeout=2,
                        )
                        if len(body) > 2_000_000:
                            return f"model_endpoint_invalid_response:{lane}"
                    raw: object = json.loads(body)
                except (OSError, asyncio.TimeoutError, json.JSONDecodeError):
                    return f"model_endpoint_unavailable:{lane}"
                finally:
                    if writer is not None:
                        writer.close()
                        try:
                            await writer.wait_closed()
                        except OSError:
                            pass
                data = raw.get("data") if isinstance(raw, dict) else None
                served = (
                    {
                        str(item["id"])
                        for item in data
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    }
                    if isinstance(data, list)
                    else set()
                )
                return None if model_id in served else f"model_identity_mismatch:{lane}"

            endpoint_checks = await asyncio.gather(
                *(
                    endpoint_failure(
                        route.endpoint,
                        route.model_id,
                        route.lane.value,
                    )
                    for route in unique_routes.values()
                )
            )
            reasons.extend(reason for reason in endpoint_checks if reason is not None)
        for current in (
            operator.state_root,
            operator.state_root / "auth",
            operator.state_root / "conversations",
        ):
            try:
                current.mkdir(parents=True, exist_ok=True)
                if not os.access(current, os.W_OK):
                    reasons.append(f"state_not_writable:{current.name}")
            except OSError:
                reasons.append(f"state_unavailable:{current.name}")
        corpus_health = corpus_store.health()
        if corpus_health.get("status") != "READY":
            reasons.append("personal_corpus_store_degraded")
        return JSONResponse(
            status_code=200 if not reasons else 503,
            content={
                "status": "READY" if not reasons else "DEGRADED",
                "reasons": reasons,
                "fixture_mode": settings.fixture_mode,
                "model_endpoints_required": tiered is not None,
                "codev": (
                    "intentionally_disabled"
                    if tiered is not None and not tiered.lane_policy.codev_enabled
                    else (
                        "required_but_failed"
                        if tiered is not None
                        and any(reason.endswith(":economy") for reason in reasons)
                        else "healthy"
                    )
                ),
                "personal_corpus": corpus_health,
            },
        )

    @app.get("/api/v1/version")
    async def version() -> dict[str, object]:
        build = version_record(operator.repository_root)
        return {
            "status": "OK",
            "application": "Laplace",
            "application_version": build["application_version"],
            "api_version": "v1",
            "git_revision": build["git_revision"],
            "build_identity": build["build_identity"],
            "deployment_mode": settings.deployment_mode,
        }

    @app.get("/api/v1/zetsu/status")
    async def zetsu_status(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> JsonObject:
        if zetsu_service is None:
            raise HTTPException(status_code=503, detail="zetsu_unavailable")
        return zetsu_service.status(authenticated.user_id)

    @app.post("/mcp")
    async def zetsu_mcp(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        mcp_protocol_version: str | None = Header(default=None, alias="MCP-Protocol-Version"),
    ) -> Response:
        """Authenticated stateless Streamable HTTP MCP endpoint for Zetsu."""

        if authenticated.auth_method != "bearer":
            raise HTTPException(status_code=401, detail="mcp_bearer_authentication_required")
        _origin_allowed(request)
        if zetsu_dispatcher is None:
            raise HTTPException(status_code=503, detail="zetsu_unavailable")
        if "application/json" not in request.headers.get("content-type", ""):
            raise HTTPException(status_code=415, detail="mcp_json_required")
        try:
            raw: object = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="mcp_invalid_json") from exc
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="mcp_batch_not_supported")
        try:
            result = zetsu_dispatcher.dispatch(
                authenticated.user_id,
                raw,
                protocol_header=mcp_protocol_version,
            )
        except ZetsuError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": raw.get("id"),
                    "error": {"code": -32602, "message": exc.category},
                },
            )
        if result.payload is None:
            return Response(status_code=result.status_code)
        return JSONResponse(
            status_code=result.status_code,
            content=result.payload,
            headers={"MCP-Protocol-Version": (mcp_protocol_version or "2025-11-25")},
        )

    register_client_routes(
        app,
        client_devices=client_devices,
        agent_principal=agent_principal,
        client_mutation_principal=client_mutation_principal,
    )

    register_auth_routes(
        app,
        settings=settings,
        registered_auth=registered_auth,
        tier_mutation_principal=tier_mutation_principal,
        origin_allowed=_origin_allowed,
        set_session_cookie=_set_session_cookie,
        session_response=_session_response,
    )

    @app.post("/api/v1/session")
    async def session(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        if authenticated.auth_method == "session" and registered_auth is not None:
            if authenticated.session_identifier is None:
                raise HTTPException(status_code=401, detail="authentication_required")
            csrf_token = registered_auth.sessions.rotate_csrf(authenticated.session_identifier)
        else:
            csrf_token = auth.issue_csrf(authenticated)
        return {
            "status": "AUTHENTICATED",
            "role": authenticated.role,
            "user_id": authenticated.user_id,
            "email": authenticated.email,
            "display_name": authenticated.display_name,
            "capability_tier": authenticated.capability_tier.value,
            "default_lane": authenticated.default_lane,
            "model_lanes": [lane.value for lane in ModelLane],
            "csrf_token": csrf_token,
        }

    def sanitized_model_status(role: str) -> dict[str, object]:
        status = operator.model_server_action("status", approval_id=None, actor_role=role)
        if role == "admin":
            return status
        sanitized = dict(status)
        servers = status.get("servers")
        if isinstance(servers, list):
            sanitized["servers"] = [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"model_path", "command_line"}
                }
                if isinstance(item, dict)
                else item
                for item in servers
            ]
        return sanitized

    @app.get("/api/v1/dashboard")
    async def dashboard(
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        summary = operator.summary(actor_role=authenticated.role)
        try:
            model_status: object = sanitized_model_status(authenticated.role)
        except Exception as exc:
            model_status = {
                "status": "UNAVAILABLE",
                "error_type": type(exc).__name__,
            }
        result: dict[str, object] = {
            **summary,
            "model_servers": model_status,
            "research_jobs": (
                research_summaries(research.layout.research_jobs) if research is not None else []
            ),
            "warnings": (
                ["Insecure non-loopback HTTP override is active"]
                if settings.allow_insecure_lan_http
                else []
            ),
            "deployment": {
                "mode": settings.deployment_mode,
                "development_http": settings.development_http,
                "allowed_hosts": list(settings.allowed_hosts),
                "allowed_origins": list(settings.allowed_origins),
            },
        }
        if tiered is not None:
            result["queue_guardrails"] = tiered.scheduler.snapshot()
            inventory = tiered.sandboxes.authorizations.operator_inventory()
            result["repositories"] = (
                inventory
                if authenticated.role == "admin"
                else [
                    {key: value for key, value in item.items() if key != "canonical_root"}
                    for item in inventory
                ]
            )
        if registered_auth is not None:
            result["users"] = [
                {"user_id": user.user_id, **user.public()}
                for user in registered_auth.registry.snapshot.users_by_id.values()
            ]
            result["registry_revision"] = registered_auth.registry.snapshot.revision
            result["active_browser_sessions"] = registered_auth.sessions.active_count()
        if artifact_registry is not None:
            result["provenance"] = {
                "status": "AVAILABLE",
                "registered_artifacts": len(artifact_registry.compact_operator_export()),
            }
        return result

    register_run_routes(
        app,
        operator=operator,
        run_executor=run_executor,
        operator_principal=operator_principal,
        mutation_principal=mutation_principal,
        model_admin_principal=model_admin_principal,
        model_admin_mutation_principal=model_admin_mutation_principal,
        sanitized_model_status=sanitized_model_status,
    )

    @app.post("/api/v1/research/jobs")
    async def create_research(
        body: ResearchCreateRequest,
        authenticated: AuthPrincipal = Depends(research_mutation_principal),
    ) -> dict[str, object]:
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        selected_domain = domain_registry.require(body.domain, surface="research")
        result = research.create(body.job, job_id=body.research_job_id)
        result["domain"] = selected_domain.domain_id
        result["domain_routing"] = {
            "eligible_model_routes": list(selected_domain.eligible_model_routes),
            "eligible_verification_tools": list(selected_domain.eligible_verification_tools),
        }
        job_id = str(result["research_job_id"])
        if research_admission is not None:
            result["admission"] = research_admission.create(
                authenticated.user_id,
                job_id,
            )
        operator.record_action(
            actor_role=authenticated.role,
            action="RESEARCH_JOB_CREATED",
            entity_type="research_job",
            entity_id=job_id,
            payload={
                "request_sha256": hashlib.sha256(
                    body.job.model_dump_json().encode("utf-8")
                ).hexdigest(),
                "domain": selected_domain.domain_id,
            },
        )
        return result

    @app.post("/api/v1/research/jobs/{job_id}/run")
    async def run_research(
        job_id: str,
        authenticated: AuthPrincipal = Depends(research_mutation_principal),
    ) -> dict[str, object]:
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        if research_admission is not None:
            research_admission.begin(authenticated.user_id, job_id)
        try:
            result = (
                research.run(job_id)
                if settings.fixture_mode
                else await run_in_threadpool(research.run, job_id)
            )
        except Exception:
            if research_admission is not None:
                research_admission.finish(
                    authenticated.user_id,
                    job_id,
                    failed=True,
                )
            raise
        if research_admission is not None:
            research_admission.finish(authenticated.user_id, job_id)
            result["admission"] = research_admission.status(
                authenticated.user_id,
                job_id,
            )
        operator.record_action(
            actor_role=authenticated.role,
            action="RESEARCH_JOB_RUN",
            entity_type="research_job",
            entity_id=job_id,
            payload={"status": result.get("status")},
        )
        if result.get("status") == "COMPLETE":
            operator.notifier.send(
                "RESEARCH_REPORT_READY",
                {"job_id": job_id, "status": "COMPLETE"},
            )
        return result

    @app.get("/api/v1/research/jobs/{job_id}")
    async def get_research(
        job_id: str,
        authenticated: AuthPrincipal = Depends(research_principal),
    ) -> dict[str, object]:
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        admission = (
            research_admission.status(authenticated.user_id, job_id)
            if research_admission is not None
            else None
        )
        result = research.get(job_id).model_dump(mode="json")
        if admission is not None:
            result["admission"] = admission
        sanitized = sanitize_research_payload(result)
        if not isinstance(sanitized, dict):
            raise HTTPException(status_code=500, detail="research_record_invalid")
        return sanitized

    @app.get("/api/v1/research/jobs/{job_id}/report")
    async def get_research_report(
        job_id: str,
        authenticated: AuthPrincipal = Depends(research_principal),
    ) -> dict[str, object]:
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        if research_admission is not None:
            research_admission.status(authenticated.user_id, job_id)
        root = research.layout.research_jobs / job_id
        job = research.get(job_id)
        if job.status != "COMPLETE":
            raise HTTPException(status_code=409, detail="research_job_not_complete")
        return {
            "status": "OK",
            "job": sanitize_research_payload(job.model_dump(mode="json")),
            "report_markdown": (root / "report.md").read_text(encoding="utf-8"),
            "evidence_ledger": sanitize_research_payload(
                json.loads((root / "evidence_ledger.json").read_text(encoding="utf-8"))
            ),
            "claim_source_graph": sanitize_research_payload(
                json.loads((root / "claim_source_graph.json").read_text(encoding="utf-8"))
            ),
        }

    @app.post("/api/v1/research/jobs/{job_id}/cancel")
    async def cancel_research(
        job_id: str,
        authenticated: AuthPrincipal = Depends(research_mutation_principal),
    ) -> dict[str, object]:
        if research_admission is None:
            raise HTTPException(status_code=503, detail="research_admission_unavailable")
        return {
            "status": "CANCELLED",
            "admission": research_admission.cancel(authenticated.user_id, job_id),
        }

    @app.post("/api/v1/research/jobs/{job_id}/export")
    async def export_research_report(
        job_id: str,
        request: Request,
        authenticated: AuthPrincipal = Depends(research_mutation_principal),
    ) -> dict[str, object]:
        if research is None or artifact_registry is None:
            raise HTTPException(status_code=503, detail="artifact_export_unavailable")
        if research_admission is not None:
            research_admission.status(authenticated.user_id, job_id)
        job = research.get(job_id)
        if job.status != "COMPLETE":
            raise HTTPException(status_code=409, detail="research_job_not_complete")
        content = (research.layout.research_jobs / job_id / "report.md").read_bytes()
        source_fingerprint = hashlib.sha256(
            (research.layout.research_jobs / job_id / "claim_source_graph.json").read_bytes()
        ).hexdigest()
        record = artifact_registry.create(
            owner_user_id=authenticated.user_id,
            content=content,
            relative_path=f"research/{job_id}/report.md",
            source_state_fingerprint=source_fingerprint,
            generator_model_route=job.model_route,
            capability_tier=authenticated.capability_tier.value,
            trace_id=str(request.state.trace_id),
            run_id=job_id,
        )
        return {
            "status": "EXPORTED",
            "artifact": record.normal(),
            "download_url": f"/api/v1/generated-artifacts/{record.artifact_id}/download",
        }

    @app.get("/api/v1/events")
    async def events(
        authenticated: AuthPrincipal = Depends(operator_principal),
        after_sequence: int = Query(default=0, ge=0),
        once: bool = Query(default=False),
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            cursor = after_sequence
            iterations = 1 if once else 240
            for _ in range(iterations):
                records = operator.events(
                    actor_role=authenticated.role,
                    after_sequence=cursor,
                    limit=200,
                )
                if records:
                    for record in records:
                        sequence = record.get("sequence")
                        if not isinstance(sequence, int):
                            continue
                        cursor = sequence
                        yield (
                            "event: operator_event\n"
                            f"id: {cursor}\n"
                            f"data: {sorted_json_text(record)}\n\n"
                        )
                elif once:
                    yield "event: heartbeat\ndata: {}\n\n"
                if not once:
                    await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/artifacts")
    async def artifact_metadata(
        path: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        artifact = safe_artifact_path(
            path,
            state_root=operator.state_root,
            repository_root=operator.repository_root,
            allow_sensitive=authenticated.role == "admin",
        )
        data = artifact.read_bytes()
        preview: str | None = None
        if artifact.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".log", ".sv"}:
            preview = data[:100_000].decode("utf-8", errors="replace")
        return {
            "status": "OK",
            "name": artifact.name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "preview": preview,
            "truncated": len(data) > 100_000,
        }

    @app.get("/api/v1/artifacts/download")
    async def artifact_download(
        path: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> FileResponse:
        artifact = safe_artifact_path(
            path,
            state_root=operator.state_root,
            repository_root=operator.repository_root,
            allow_sensitive=authenticated.role == "admin",
        )
        return FileResponse(
            artifact,
            filename=artifact.name,
            media_type="application/octet-stream",
        )

    @app.get("/api/v1/generated-artifacts/{artifact_id}/download")
    async def generated_artifact_download(
        artifact_id: str,
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        repo_id: str | None = Query(default=None),
    ) -> Response:
        if artifact_registry is None:
            raise HTTPException(status_code=503, detail="artifact_registry_unavailable")
        record = artifact_registry.require(
            artifact_id,
            owner_user_id=authenticated.user_id,
            repo_id=repo_id,
        )
        content = artifact_registry.read(
            artifact_id,
            owner_user_id=authenticated.user_id,
            repo_id=repo_id,
            capability_tier=authenticated.capability_tier.value,
            trace_id=str(request.state.trace_id),
        )
        filename = Path(record.relative_path).name
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Content-SHA256": record.content_sha256,
            },
        )

    @app.get("/api/v1/admin/artifact-provenance")
    async def artifact_provenance(
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        if artifact_registry is None:
            raise HTTPException(status_code=503, detail="artifact_registry_unavailable")
        return {
            "status": "OK",
            "artifacts": artifact_registry.compact_operator_export(),
        }

    @app.get("/api/v1/runs/compare/{left_run_id}/{right_run_id}")
    async def compare_runs(
        left_run_id: str,
        right_run_id: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        left = operator.get_run(left_run_id, actor_role=authenticated.role)
        right = operator.get_run(right_run_id, actor_role=authenticated.role)
        fields = (
            "configuration_sha256",
            "request_sha256",
            "arm_id",
            "state",
            "gpu_required",
        )
        return {
            "status": "OK",
            "left_run_id": left_run_id,
            "right_run_id": right_run_id,
            "comparison": {
                field: {
                    "left": left.get(field),
                    "right": right.get(field),
                    "equal": left.get(field) == right.get(field),
                }
                for field in fields
            },
        }

    @app.get("/api/v1/corpora")
    async def corpora(
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        del authenticated
        governed = (
            research.layout.governed_corpus
            if research is not None
            else operator.state_root / "stores/governed_corpus"
        )
        snapshots = [
            {
                "domain": current.parent.name,
                "current": json.loads(current.read_text(encoding="utf-8")),
            }
            for current in sorted(governed.glob("*/current.json"))
        ]
        return {"status": "OK", "governed_snapshots": snapshots}

    def require_tiered() -> TieredServingService:
        if tiered is None:
            raise HTTPException(status_code=503, detail="tiered_serving_unavailable")
        return tiered

    def require_core() -> LaplaceCore:
        if shared_core is None:
            raise HTTPException(status_code=503, detail="laplace_core_unavailable")
        return shared_core

    @app.get("/api/v1/help")
    async def role_aware_help(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        service = require_tiered()
        capability = service.capability(authenticated.user_id)
        effective = service.effective_capabilities(authenticated.user_id)
        functions: list[dict[str, str]] = [
            {
                "name": "Chat",
                "description": "Private local chat with no dormant tool schemas.",
            },
            {
                "name": "Model lanes",
                "description": (
                    "Quality never downgrades; standard and economy can escalate once "
                    "after a deterministic validation failure."
                ),
            },
            {
                "name": "Privacy and provenance",
                "description": (
                    "Conversations and artifacts are owner-isolated; generated files keep "
                    "clean names while internal provenance is stored separately."
                ),
            },
            {
                "name": "Stop and cancel",
                "description": "Stop generation or cancel queued work from its active panel.",
            },
        ]
        if effective == default_capabilities(CapabilityTier.BASIC):
            functions.append(
                {
                    "name": "Basic capability",
                    "description": (
                        "Chat only: no repositories, tools, file mutation, agents, "
                        "shell, Git actions, or background work."
                    ),
                }
            )
        if Capability.AGENT in effective:
            functions.extend(
                [
                    {
                        "name": "Agent capability",
                        "description": "Explicitly authorized repository work.",
                    },
                    {
                        "name": "Repository-bound agent",
                        "description": (
                            "Agent work is restricted to an operator-authorized repository "
                            "and a dedicated isolated worktree."
                        ),
                    },
                ]
            )
        if Capability.PERSONAL_CORPUS in effective:
            functions.append(
                {
                    "name": "My corpus",
                    "description": (
                        "Owner-private folder upload, validation, deterministic "
                        "indexing, retrieval snapshots, and deletion."
                    ),
                }
            )
        if Capability.RESEARCH in effective:
            functions.append(
                {
                    "name": "Deep Research",
                    "description": (
                        "Governed knowledge and RAG sources, visible citations and "
                        "conflicts, bounded queueing, export, and cancellation."
                    ),
                }
            )
        if Capability.OPERATOR in effective:
            functions.extend(
                [
                    {
                        "name": "Operator controls",
                        "description": (
                            "Inspect users, repositories, queues, approvals, models, GPU, "
                            "audit events, and readiness without exposing secrets."
                        ),
                    },
                    {
                        "name": "Serving lifecycle",
                        "description": "Start or stop only explicitly owned local serving profiles.",
                    },
                    {
                        "name": "Approvals and guardrails",
                        "description": (
                            "Risk-bearing actions require approval; concurrency limits "
                            "and Quality reservation remain visible."
                        ),
                    },
                ]
            )
        return {
            "status": "OK",
            "capability_tier": capability.value,
            "capabilities": sorted(item.value for item in effective),
            "role": authenticated.role,
            "functions": functions,
            "guardrails": service.scheduler.snapshot(),
            "privacy": {
                "local_inference": True,
                "repository_isolation": True,
                "artifact_provenance": artifact_registry is not None,
                "retention": "Operator-configured external state",
            },
        }

    @app.get("/api/v1/about")
    async def about(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        service = require_tiered()
        build = version_record(operator.repository_root)
        return {
            "status": "OK",
            "application_version": build["application_version"],
            "git_revision": build["git_revision"],
            "build_identity": build["build_identity"],
            "api_version": "v1",
            "capability_tier": authenticated.capability_tier.value,
            "role": authenticated.role,
            "model_lanes": {
                lane.value: {
                    "display_name": service.lane_policy.routes[lane].model_id,
                    "context_limit": service.lane_policy.routes[lane].context_limit,
                    "output_limit": service.lane_policy.routes[lane].output_limit,
                }
                for lane in ModelLane
            },
            "guardrails": service.scheduler.snapshot(),
            "remote_access_mode": settings.deployment_mode,
            "health": "OK",
            "documentation": [
                "USER_GUIDE.md",
                "QUICKSTART.md",
                "ADMIN_GUIDE.md",
                "REMOTE_ACCESS.md",
                "PERSONAL_CORPUS.md",
                "AGENT_WORKTREES.md",
                "STORAGE_AND_RETENTION.md",
            ],
        }

    @app.get("/api/v1/providers")
    async def providers(
        _authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        """Return frontend-safe provider capabilities and logical routes."""

        if tiered is None:
            return {
                "schema_version": 1,
                "status": "UNAVAILABLE",
                "providers": [],
                "routes": [],
                "endpoint_details_exposed": False,
            }
        provider_type: Literal["fixture", "vllm"] = "fixture" if settings.fixture_mode else "vllm"
        public_providers: list[dict[str, object]] = []
        public_routes: list[dict[str, object]] = []
        for lane in ModelLane:
            configured = tiered.lane_policy.routes[lane]
            parsed_endpoint = urlsplit(configured.endpoint)
            endpoint_host = parsed_endpoint.hostname or ""
            endpoint_origin = (
                f"{parsed_endpoint.scheme}://"
                f"{'[' + endpoint_host + ']' if ':' in endpoint_host else endpoint_host}"
                f":{parsed_endpoint.port}"
                if parsed_endpoint.port is not None
                else f"{parsed_endpoint.scheme}://{endpoint_host}"
            )
            provider_id = f"{provider_type}-{lane.value}"
            provider = ProviderV1(
                provider_id=provider_id,
                display_name=(
                    f"{lane.value.title()} deterministic fixture"
                    if settings.fixture_mode
                    else f"{lane.value.title()} configured local provider"
                ),
                provider_type=provider_type,
                endpoint=("fixture://in-memory" if settings.fixture_mode else endpoint_origin),
                lifecycle="fixture" if settings.fixture_mode else "unowned",
                context_limit=configured.context_limit,
                output_limit=configured.output_limit,
                capabilities=ProviderCapabilitiesV1(
                    streaming=not settings.fixture_mode,
                    tools=True,
                    structured_output=True,
                    embeddings=False,
                    thinking_control=False,
                    requires_gpu=not settings.fixture_mode,
                    supports_cpu=settings.fixture_mode,
                    can_start=False,
                    can_stop=False,
                ),
            )
            route = RouteV1(
                route_id=f"{lane.value}-route",
                display_name=f"{lane.value.title()} — {configured.model_id}",
                provider_id=provider_id,
                model_id=configured.model_id,
                lane=lane.value,
                enabled=True,
            )
            public_providers.append(provider.public_summary())
            public_routes.append(route.model_dump(mode="json"))
        return {
            "schema_version": 1,
            "status": "OK",
            "providers": public_providers,
            "routes": public_routes,
            "endpoint_details_exposed": False,
        }

    @app.get("/api/v1/domains")
    async def domains(
        surface: Literal["chat", "agent", "research"] | None = Query(default=None),
        _authenticated: AuthPrincipal = Depends(principal),
    ) -> JsonObject:
        return domain_registry.public(surface=surface)

    @app.get("/api/v1/personal-corpus/policy")
    async def personal_corpus_policy(
        _authenticated: AuthPrincipal = Depends(principal),
    ) -> JsonObject:
        return corpus_store.policy.public()

    @app.get("/api/v1/capabilities")
    @app.get("/api/v1/tier/capabilities")
    async def tier_capabilities(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        service = require_tiered()
        effective = service.capability(authenticated.user_id)
        if effective is not authenticated.capability_tier:
            raise HTTPException(status_code=401, detail="credential_capability_changed")
        capabilities = service.effective_capabilities(authenticated.user_id)
        result: dict[str, object] = {
            "status": "OK",
            "display_name": authenticated.display_name or authenticated.user_id,
            "capability_tier": effective.value,
            "capability_schema_version": 2,
            "capabilities": sorted(item.value for item in capabilities),
            "role": authenticated.role,
            "default_lane": authenticated.default_lane,
            "chat_enabled": Capability.CHAT in capabilities,
            "agent_enabled": Capability.AGENT in capabilities,
            "research_enabled": Capability.RESEARCH in capabilities,
            "operator_enabled": Capability.OPERATOR in capabilities,
            "admin_enabled": Capability.ADMIN in capabilities,
            "personal_corpus_enabled": Capability.PERSONAL_CORPUS in capabilities,
            "shared_corpus_ingest_enabled": (Capability.SHARED_CORPUS_INGEST in capabilities),
            "repository_admin_enabled": Capability.REPOSITORY_ADMIN in capabilities,
            "model_admin_enabled": Capability.MODEL_ADMIN in capabilities,
            "model_lanes": [lane.value for lane in ModelLane],
            "lane_axis_independent": True,
            "routes": {
                lane.value: {
                    "model_id": service.lane_policy.routes[lane].model_id,
                    "priority": service.lane_policy.routes[lane].priority,
                }
                for lane in ModelLane
            },
            "queue": service.scheduler.snapshot(),
            "effective_limits": {
                "quality_reserved_slots": service.lane_policy.quality_reserved_slots,
                "standard_capacity": service.lane_policy.standard_capacity,
                "economy_capacity": service.lane_policy.economy_capacity,
                "agent_network_enabled": False,
            },
            "domain_registry": domain_registry.public(),
        }
        if Capability.AGENT in capabilities:
            authorized = service.sandboxes.authorizations.authorized_for_user(authenticated.user_id)
            if registered_auth is not None:
                registry_user = registered_auth.registry.require_user(authenticated.user_id)
                allowed = set(registry_user.authorized_repo_ids)
                authorized = [item for item in authorized if item.get("repo_id") in allowed]
            result["authorized_repositories"] = authorized
            result["worktree_quota"] = {
                "per_user": service.sandboxes.per_user_quota,
                "global": service.sandboxes.global_quota,
            }
        if authenticated.auth_method == "bearer":
            # Backwards-compatible non-browser API clients may need their configured ID.
            result["user_id"] = authenticated.user_id
        return result

    @app.post("/api/v1/chat")
    async def tier_chat(
        body: TierChatRequest,
        request: Request,
        authenticated: AuthPrincipal = Depends(chat_mutation_principal),
    ) -> dict[str, object]:
        require_tiered()
        _require_named(authenticated, Capability.CHAT)
        domain_registry.require(body.domain, surface="chat")
        progress_id = body.request_id or f"ui-chat-{uuid.uuid4().hex}"
        progress_store.create(
            progress_id,
            owner_user_id=authenticated.user_id,
            request_type="chat",
            trace_id=str(request.state.trace_id),
            requested_lane=body.lane,
        )
        messages = [message.model_dump() for message in body.messages]
        retrieval: JsonObject = {
            "selection": body.retrieval_selection,
            "retrieval_used": False,
            "personal": None,
            "shared": {
                "requested": body.retrieval_selection in {"shared", "both"},
                "retrieval_used": False,
            },
        }
        if body.retrieval_selection in {"personal", "both", "selected_personal"}:
            _require_named(authenticated, Capability.PERSONAL_CORPUS)
            if (
                body.retrieval_selection == RetrievalSelection.SELECTED_PERSONAL.value
                and body.personal_corpus_id is None
            ):
                raise CorpusError("selected_personal_corpus_required")
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="RETRIEVING",
                retrieval=retrieval,
            )
            retrieval_query = next(
                (message.content for message in reversed(body.messages) if message.role == "user"),
                "",
            )
            personal = require_core().retrieve(
                authenticated.user_id,
                retrieval_query,
                corpus_id=(
                    body.personal_corpus_id
                    if body.retrieval_selection == RetrievalSelection.SELECTED_PERSONAL.value
                    else None
                ),
                limit=8,
            )
            retrieval["personal"] = personal
            retrieval["retrieval_used"] = bool(personal["retrieval_used"])
            raw_personal_results = personal.get("results")
            if isinstance(raw_personal_results, list) and raw_personal_results:
                context_parts = [
                    (
                        f"[{item['chunk_id']}] file={item['file']} "
                        f"page={item['page'] or 'n/a'} "
                        f"section={item['section'] or 'n/a'}\n{item['text']}"
                    )
                    for item in raw_personal_results
                    if isinstance(item, dict)
                ]
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": (
                            "Use the following owner-authorized read-only personal "
                            "corpus excerpts only when relevant. Cite file, page or "
                            "section when available, and chunk ID. Do not claim any "
                            "other source.\n\n" + "\n\n".join(context_parts)
                        )[:40_000],
                    },
                )
        elif body.personal_corpus_id is not None:
            raise CorpusError("personal_corpus_id_without_selection")
        conversation_id = body.conversation_id
        if conversation_store is not None:
            if conversation_id is None:
                latest = body.messages[-1].content.strip().replace("\n", " ")
                conversation = conversation_store.create(
                    authenticated.user_id,
                    title=latest[:80] or "New conversation",
                )
                conversation_id = conversation.conversation_id
            else:
                conversation_store.require(authenticated.user_id, conversation_id)
            latest_user = next(
                (message.content for message in reversed(body.messages) if message.role == "user"),
                None,
            )
            if latest_user is not None:
                conversation_store.append_message(
                    authenticated.user_id,
                    conversation_id,
                    role="user",
                    content=latest_user,
                    metadata={"requested_lane": body.lane, "domain": body.domain},
                )
        try:
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="PREPARING_CONTEXT",
                retrieval=retrieval,
            )
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="QUEUED",
            )
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="ADMITTED",
            )
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="GENERATING",
            )
            if settings.fixture_mode:
                result = require_core().chat(
                    user_id=authenticated.user_id,
                    lane=ModelLane(body.lane),
                    messages=messages,
                    domain=body.domain,
                    session_id=body.session_id or conversation_id,
                )
            else:
                result = await run_in_threadpool(
                    require_core().chat,
                    user_id=authenticated.user_id,
                    lane=ModelLane(body.lane),
                    messages=messages,
                    domain=body.domain,
                    session_id=body.session_id or conversation_id,
                )
            queue_value = result.get("queue_position")
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="VALIDATING_OUTPUT",
                queue_position=(queue_value if isinstance(queue_value, int) else None),
                effective_lane=(
                    str(result["effective_lane"])
                    if result.get("effective_lane") is not None
                    else None
                ),
                model_name=(
                    str(result["model_id"]) if result.get("model_id") is not None else None
                ),
                retrieval=retrieval,
            )
        except Exception:
            progress_store.transition(
                progress_id,
                owner_user_id=authenticated.user_id,
                state="FAILED",
                retrieval=retrieval,
            )
            raise
        if conversation_store is not None and conversation_id is not None:
            envelope = result.get("response")
            content = envelope.get("content") if isinstance(envelope, dict) else None
            if isinstance(content, str) and content:
                message = conversation_store.append_message(
                    authenticated.user_id,
                    conversation_id,
                    role="assistant",
                    content=content,
                    metadata={
                        key: result.get(key)
                        for key in (
                            "request_id",
                            "trace_id",
                            "requested_lane",
                            "effective_lane",
                            "model_id",
                            "queue_wait_seconds",
                            "context_limit",
                            "output_limit",
                            "escalation",
                        )
                    },
                )
                result["conversation_message_id"] = message["message_id"]
            result["conversation_id"] = conversation_id
        result["client_request_id"] = progress_id
        result["retrieval"] = retrieval
        progress_store.transition(
            progress_id,
            owner_user_id=authenticated.user_id,
            state="COMPLETE",
            effective_lane=str(result.get("effective_lane") or ""),
            model_name=str(result.get("model_id") or ""),
            retrieval=retrieval,
        )
        return result

    @app.get("/api/v1/requests/{request_id}")
    async def request_status(
        request_id: str,
        authenticated: AuthPrincipal = Depends(principal),
    ) -> JsonObject:
        return progress_store.get(authenticated.user_id, request_id)

    @app.post("/api/v1/requests/{request_id}/cancel")
    async def cancel_request(
        request_id: str,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> JsonObject:
        return progress_store.cancel(authenticated.user_id, request_id)

    @app.get("/api/v1/conversations")
    async def list_conversations(
        authenticated: AuthPrincipal = Depends(chat_principal),
        include_archived: bool = Query(default=True),
    ) -> dict[str, object]:
        if conversation_store is None:
            raise HTTPException(status_code=503, detail="conversation_store_unavailable")
        return {
            "status": "OK",
            "conversations": conversation_store.list(
                authenticated.user_id,
                include_archived=include_archived,
            ),
        }

    @app.post("/api/v1/conversations")
    async def create_conversation(
        body: ConversationCreateRequest,
        authenticated: AuthPrincipal = Depends(chat_mutation_principal),
    ) -> dict[str, object]:
        if conversation_store is None:
            raise HTTPException(status_code=503, detail="conversation_store_unavailable")
        return {
            "status": "CREATED",
            "conversation": conversation_store.create(
                authenticated.user_id,
                title=body.title,
            ).public(),
        }

    @app.get("/api/v1/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str,
        authenticated: AuthPrincipal = Depends(chat_principal),
    ) -> dict[str, object]:
        if conversation_store is None:
            raise HTTPException(status_code=503, detail="conversation_store_unavailable")
        return {
            "status": "OK",
            "conversation": conversation_store.get_with_messages(
                authenticated.user_id,
                conversation_id,
            ),
        }

    @app.patch("/api/v1/conversations/{conversation_id}")
    async def update_conversation(
        conversation_id: str,
        body: ConversationUpdateRequest,
        authenticated: AuthPrincipal = Depends(chat_mutation_principal),
    ) -> dict[str, object]:
        if conversation_store is None:
            raise HTTPException(status_code=503, detail="conversation_store_unavailable")
        updated = conversation_store.update(
            authenticated.user_id,
            conversation_id,
            title=body.title,
            archived=body.archived,
            draft=body.draft,
        )
        return {"status": "UPDATED", "conversation": updated.public()}

    @app.delete("/api/v1/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str,
        authenticated: AuthPrincipal = Depends(chat_mutation_principal),
    ) -> dict[str, object]:
        if conversation_store is None:
            raise HTTPException(status_code=503, detail="conversation_store_unavailable")
        conversation_store.delete(authenticated.user_id, conversation_id)
        return {"status": "DELETED"}

    @app.get("/api/v1/personal-corpora")
    async def list_personal_corpora(
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
        include_archived: bool = Query(default=True),
    ) -> JsonObject:
        return {
            "status": "OK",
            "corpora": corpus_store.list_corpora(
                authenticated.user_id, include_archived=include_archived
            ),
        }

    @app.post("/api/v1/personal-corpora")
    async def create_personal_corpus(
        body: CorpusCreateRequest,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return {
            "status": "CREATED",
            "corpus": corpus_store.create_corpus(authenticated.user_id, body.name),
        }

    @app.get("/api/v1/personal-corpora/{corpus_id}")
    async def get_personal_corpus(
        corpus_id: str,
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "corpus": corpus_store.require_corpus(authenticated.user_id, corpus_id),
            "sources": corpus_store.list_sources(authenticated.user_id, corpus_id),
        }

    @app.patch("/api/v1/personal-corpora/{corpus_id}")
    async def update_personal_corpus(
        corpus_id: str,
        body: CorpusUpdateRequest,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return {
            "status": "UPDATED",
            "corpus": corpus_store.update_corpus(
                authenticated.user_id,
                corpus_id,
                name=body.name,
                archived=body.archived,
            ),
        }

    @app.delete("/api/v1/personal-corpora/{corpus_id}")
    async def delete_personal_corpus(
        corpus_id: str,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return corpus_store.delete_corpus(authenticated.user_id, corpus_id)

    @app.post("/api/v1/personal-corpus/uploads")
    async def create_folder_upload(
        body: UploadCreateRequest,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return corpus_store.create_upload(
            authenticated.user_id,
            body.corpus_id,
            idempotency_key=body.idempotency_key,
        )

    @app.get("/api/v1/personal-corpus/uploads")
    async def list_folder_uploads(
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
        state: Literal["STAGING", "CANCELLED", "INDEXED"] = Query(default="STAGING"),
    ) -> JsonObject:
        return {
            "status": "OK",
            "uploads": corpus_store.list_uploads(authenticated.user_id, state=state),
        }

    @app.get("/api/v1/personal-corpus/uploads/{upload_id}")
    async def folder_upload_manifest(
        upload_id: str,
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
    ) -> JsonObject:
        return corpus_store.upload_manifest(authenticated.user_id, upload_id)

    async def _upload_form(
        request: Request,
        *,
        zip_fallback: bool,
    ) -> tuple[str, bytes, str]:
        form = await request.form(
            max_files=1,
            max_fields=2,
            max_part_size=corpus_store.policy.max_file_bytes,
        )
        expected = {"file"} if zip_fallback else {"file", "relative_path"}
        if set(form) != expected:
            raise CorpusError("invalid_multipart_fields")
        uploaded = form.get("file")
        if not isinstance(uploaded, UploadFile):
            raise CorpusError("upload_file_required")
        content = await uploaded.read(corpus_store.policy.max_file_bytes + 1)
        if len(content) > corpus_store.policy.max_file_bytes:
            raise CorpusError("file_size_quota")
        relative_path = (
            uploaded.filename or "folder.zip"
            if zip_fallback
            else str(form.get("relative_path") or "")
        )
        return relative_path, content, uploaded.content_type or ""

    @app.post("/api/v1/personal-corpus/uploads/{upload_id}/files")
    async def upload_folder_file(
        upload_id: str,
        request: Request,
        _authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        relative_path, content, media_type = await _upload_form(request, zip_fallback=False)
        return corpus_store.stage_file(
            _authenticated.user_id,
            upload_id,
            logical_path=relative_path,
            content=content,
            client_mime=media_type,
        )

    @app.post("/api/v1/personal-corpus/uploads/{upload_id}/zip")
    async def upload_folder_zip(
        upload_id: str,
        request: Request,
        _authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        _name, content, _media_type = await _upload_form(request, zip_fallback=True)
        return corpus_store.stage_zip_fallback(_authenticated.user_id, upload_id, content=content)

    @app.post("/api/v1/personal-corpus/uploads/{upload_id}/cancel")
    async def cancel_folder_upload(
        upload_id: str,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return corpus_store.cancel_upload(authenticated.user_id, upload_id)

    @app.post("/api/v1/personal-corpus/uploads/{upload_id}/index")
    async def index_folder_upload(
        upload_id: str,
        body: IndexUploadRequest,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        if settings.fixture_mode:
            return corpus_store.index_upload(
                authenticated.user_id,
                upload_id,
                idempotency_key=body.idempotency_key,
            )
        return await run_in_threadpool(
            corpus_store.index_upload,
            authenticated.user_id,
            upload_id,
            idempotency_key=body.idempotency_key,
        )

    @app.post("/api/v1/personal-corpora/{corpus_id}/search-test")
    async def personal_corpus_search_test(
        corpus_id: str,
        body: CorpusSearchRequest,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        if body.corpus_id is not None and body.corpus_id != corpus_id:
            raise CorpusError("conflicting_corpus_id")
        return corpus_store.search(
            authenticated.user_id,
            body.query,
            corpus_id=corpus_id,
            limit=body.limit,
        )

    @app.post("/api/v1/personal-corpora/{corpus_id}/reindex")
    async def reindex_personal_corpus(
        corpus_id: str,
        body: IndexUploadRequest,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        if settings.fixture_mode:
            return corpus_store.reindex_corpus(
                authenticated.user_id,
                corpus_id,
                idempotency_key=body.idempotency_key,
            )
        return await run_in_threadpool(
            corpus_store.reindex_corpus,
            authenticated.user_id,
            corpus_id,
            idempotency_key=body.idempotency_key,
        )

    @app.delete("/api/v1/personal-corpora/{corpus_id}/sources/{source_id}")
    async def delete_personal_source(
        corpus_id: str,
        source_id: str,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return corpus_store.delete_source(authenticated.user_id, corpus_id, source_id)

    @app.get("/api/v1/personal-corpora/{corpus_id}/sources/{source_id}/download")
    async def download_personal_source(
        corpus_id: str,
        source_id: str,
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
    ) -> Response:
        name, media_type, content = corpus_store.source_content(
            authenticated.user_id, corpus_id, source_id
        )
        safe_name = PurePosixPath(name).name.replace('"', "")
        return Response(
            content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}"',
                "X-Content-SHA256": hashlib.sha256(content).hexdigest(),
            },
        )

    @app.post("/api/v1/worktrees")
    @app.post("/api/v1/agent/sessions")
    async def create_agent_session(
        body: AgentSessionRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        if registered_auth is not None and authenticated.auth_method == "session":
            user = registered_auth.registry.require_user(authenticated.user_id)
            if body.repo_id not in user.authorized_repo_ids:
                raise HTTPException(
                    status_code=403,
                    detail="repository_not_authorized",
                )
        tool_policy = AgentToolPolicy(
            policy_id=f"api-{body.session_id}",
            allowed_tools=tuple(body.allowed_tools),
            network_enabled=False,
            max_commands=body.max_commands,
            max_wall_seconds=body.max_wall_seconds,
        )
        if settings.fixture_mode:
            result = service.create_agent_session(
                user_id=authenticated.user_id,
                repo_id=body.repo_id,
                session_id=body.session_id,
                tool_policy=tool_policy,
                task_title=body.task_title,
                idempotency_key=body.idempotency_key,
            )
        else:
            result = await run_in_threadpool(
                service.create_agent_session,
                user_id=authenticated.user_id,
                repo_id=body.repo_id,
                session_id=body.session_id,
                tool_policy=tool_policy,
                task_title=body.task_title,
                idempotency_key=body.idempotency_key,
            )
        binding = result.get("binding")
        if isinstance(binding, dict):
            result["binding"] = {
                "session_id": binding.get("session_id"),
                "repo_id": binding.get("repo_id"),
                "logical_repository_name": binding.get("repo_id"),
                "base_revision": binding.get("base_revision"),
                "grant_revision": binding.get("grant_revision"),
                "tool_policy": binding.get("tool_policy"),
                "worktree_status": "ACTIVE_ISOLATED",
                "network_policy": "network-denied-v1",
            }
        if conversation_store is not None:
            conversation_store.bind_agent_session(
                authenticated.user_id,
                body.session_id,
                repo_id=body.repo_id,
                title=body.task_title,
            )
        return result

    def _prepare_agent_turn(
        *,
        session_id: str,
        body: AgentRunRequest,
        authenticated: AuthPrincipal,
    ) -> tuple[AgentSessionBinding, str, JsonObject]:
        """Apply the identical policy/retrieval preflight for both turn routes."""

        service = require_tiered()
        _require_named(authenticated, Capability.AGENT)
        instruction = body.instruction
        retrieval: JsonObject = {
            "selection": body.retrieval_selection,
            "retrieval_used": False,
            "personal": None,
            "shared": {
                "requested": body.retrieval_selection in {"shared", "both"},
                "retrieval_used": False,
            },
            "repository_write_policy": "personal_corpus_copy_denied",
        }
        if body.retrieval_selection in {"personal", "both", "selected_personal"}:
            _require_named(authenticated, Capability.PERSONAL_CORPUS)
            if (
                body.retrieval_selection == RetrievalSelection.SELECTED_PERSONAL.value
                and body.personal_corpus_id is None
            ):
                raise CorpusError("selected_personal_corpus_required")
            personal = require_core().retrieve(
                authenticated.user_id,
                body.instruction,
                corpus_id=(
                    body.personal_corpus_id
                    if body.retrieval_selection == RetrievalSelection.SELECTED_PERSONAL.value
                    else None
                ),
                limit=8,
            )
            retrieval["personal"] = personal
            retrieval["retrieval_used"] = bool(personal["retrieval_used"])
            raw_results = personal.get("results")
            if isinstance(raw_results, list) and raw_results:
                excerpts = [
                    (
                        f"[{item['chunk_id']}] file={item['file']} "
                        f"page={item['page'] or 'n/a'} "
                        f"section={item['section'] or 'n/a'}\n{item['text']}"
                    )
                    for item in raw_results
                    if isinstance(item, dict)
                ]
                instruction = (
                    "Owner-authorized personal-corpus reference excerpts follow. "
                    "They are read-only context: do not copy, reproduce, or write "
                    "their content into the repository. Cite logical file/page/"
                    "section/chunk identifiers in the explanation when used.\n\n"
                    + "\n\n".join(excerpts)
                    + "\n\nUSER TASK:\n"
                    + body.instruction
                )[:100_000]
        elif body.personal_corpus_id is not None:
            raise CorpusError("personal_corpus_id_without_selection")
        binding = service.sandboxes.require_active(
            session_id,
            user_id=authenticated.user_id,
        )
        return binding, instruction, retrieval

    async def _execute_prepared_agent_turn(
        *,
        session_id: str,
        body: AgentRunRequest,
        authenticated: AuthPrincipal,
        repo_id: str,
        instruction: str,
        retrieval: JsonObject,
        background: bool = False,
    ) -> JsonObject:
        if zetsu_service is None:
            raise HTTPException(
                status_code=503,
                detail="repository_agent_conversation_unavailable",
            )
        zetsu_service.repository_agent_service()
        if settings.fixture_mode and not background:
            result = require_core().repository_agent_turn(
                user_id=authenticated.user_id,
                repo_id=repo_id,
                session_id=session_id,
                lane=ModelLane(body.lane),
                instruction=instruction,
                max_steps=body.max_steps,
                max_chars=body.max_chars,
                verification_argv=body.verification_argv,
                allow_mutation=body.allow_mutation,
                wait_timeout_seconds=body.wait_timeout_seconds,
                task_label=derive_task_label(body.instruction),
                task_complexity=(
                    body.manager_complexity.as_task_complexity()
                    if body.manager_complexity is not None
                    else None
                ),
            )
        else:
            result = await run_in_threadpool(
                require_core().repository_agent_turn,
                user_id=authenticated.user_id,
                repo_id=repo_id,
                session_id=session_id,
                lane=ModelLane(body.lane),
                instruction=instruction,
                max_steps=body.max_steps,
                max_chars=body.max_chars,
                verification_argv=body.verification_argv,
                allow_mutation=body.allow_mutation,
                wait_timeout_seconds=body.wait_timeout_seconds,
                task_label=derive_task_label(body.instruction),
                task_complexity=(
                    body.manager_complexity.as_task_complexity()
                    if body.manager_complexity is not None
                    else None
                ),
            )
        result["retrieval"] = retrieval
        return result

    def _persist_agent_turn_result(
        *,
        session_id: str,
        authenticated: AuthPrincipal,
        result: JsonObject,
        turn_id: str | None = None,
    ) -> JsonObject:
        if conversation_store is not None:
            try:
                metadata = {
                    key: result.get(key)
                    for key in (
                        "status",
                        "result_id",
                        "delivery_status",
                        "verification_status",
                        "changed_paths",
                        "task_label",
                    )
                }
                if turn_id is not None:
                    metadata["turn_id"] = turn_id
                message = conversation_store.append_agent_message(
                    authenticated.user_id,
                    session_id,
                    role="assistant",
                    content=agent_result_content(result),
                    metadata=metadata,
                )
                result["agent_conversation_message_id"] = message["message_id"]
            except Exception:
                LOGGER.exception(
                    "agent conversation result persistence failed",
                    extra={"session_id": session_id},
                )
                result["agent_conversation_persistence"] = "FAILED_AFTER_DURABLE_RESULT"
        return public_agent_result(result)

    @app.post("/api/v1/agent/sessions/{session_id}/run")
    @app.post("/api/v1/agent/sessions/{session_id}/messages")
    async def run_agent_session(
        session_id: str,
        body: AgentRunRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        binding, instruction, retrieval = _prepare_agent_turn(
            session_id=session_id,
            body=body,
            authenticated=authenticated,
        )
        if conversation_store is not None:
            conversation_store.append_agent_message(
                authenticated.user_id,
                session_id,
                role="user",
                content=body.instruction,
                metadata={
                    "lane": body.lane,
                    "domain": body.domain,
                    "retrieval_selection": body.retrieval_selection,
                    "task_label": derive_task_label(body.instruction),
                },
            )
        result = await _execute_prepared_agent_turn(
            session_id=session_id,
            body=body,
            authenticated=authenticated,
            repo_id=binding.repo_id,
            instruction=instruction,
            retrieval=retrieval,
        )
        return _persist_agent_turn_result(
            session_id=session_id,
            authenticated=authenticated,
            result=result,
        )

    @app.post("/api/v1/agent/sessions/{session_id}/messages/async", status_code=202)
    async def submit_agent_session(
        session_id: str,
        body: AgentAsyncRunRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> JsonObject:
        if conversation_store is None:
            raise HTTPException(status_code=503, detail="agent_conversation_store_unavailable")
        if zetsu_service is None:
            raise HTTPException(
                status_code=503,
                detail="repository_agent_conversation_unavailable",
            )
        _require_named(authenticated, Capability.AGENT)
        request_sha256 = _agent_async_request_sha256(body)

        def replay_response(existing: JsonObject) -> JsonObject:
            submitted = existing.get("submitted_message")
            metadata = submitted.get("metadata") if isinstance(submitted, Mapping) else None
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("request_sha256") != request_sha256
            ):
                raise HTTPException(status_code=409, detail="agent_turn_id_conflict")
            return {
                "status": str(existing["status"]),
                "session_id": session_id,
                "turn_id": body.turn_id,
                "event_cursor": 0,
                "idempotent_replay": True,
            }

        existing = conversation_store.agent_turn(
            authenticated.user_id,
            session_id,
            body.turn_id,
        )
        if existing is not None:
            return replay_response(existing)

        binding, instruction, retrieval = _prepare_agent_turn(
            session_id=session_id,
            body=body,
            authenticated=authenticated,
        )
        key = (authenticated.user_id, session_id)
        async with agent_turn_lock:
            # Recheck under the per-session admission lock.  A client retry can
            # arrive between the first durable lookup and this critical section;
            # it must replay the original turn rather than append another user
            # message or schedule a second execution.
            existing = conversation_store.agent_turn(
                authenticated.user_id,
                session_id,
                body.turn_id,
            )
            if existing is not None:
                return replay_response(existing)
            active = agent_turn_tasks.get(key)
            if active is not None and not active[1].done() and active[0] != body.turn_id:
                raise HTTPException(status_code=409, detail="agent_session_turn_in_progress")
            task_label = derive_task_label(body.instruction)
            conversation_store.append_agent_message(
                authenticated.user_id,
                session_id,
                role="user",
                content=body.instruction,
                metadata={
                    "turn_id": body.turn_id,
                    "request_sha256": request_sha256,
                    "lane": body.lane,
                    "domain": body.domain,
                    "retrieval_selection": body.retrieval_selection,
                    "task_label": task_label,
                },
            )
            submitted = require_tiered().sandboxes.record_progress(
                session_id,
                user_id=authenticated.user_id,
                event="TURN_SUBMITTED",
                details={"turn_id": body.turn_id, "lane": body.lane, "task_label": task_label},
            )

            async def run_submitted_turn() -> None:
                try:
                    require_tiered().sandboxes.record_progress(
                        session_id,
                        user_id=authenticated.user_id,
                        event="TURN_STARTED",
                        details={"turn_id": body.turn_id, "task_label": task_label},
                    )
                    result = await _execute_prepared_agent_turn(
                        session_id=session_id,
                        body=body,
                        authenticated=authenticated,
                        repo_id=binding.repo_id,
                        instruction=instruction,
                        retrieval=retrieval,
                        background=True,
                    )
                    _persist_agent_turn_result(
                        session_id=session_id,
                        authenticated=authenticated,
                        result=result,
                        turn_id=body.turn_id,
                    )
                    status = str(result.get("status", "UNKNOWN"))
                    require_tiered().sandboxes.record_progress(
                        session_id,
                        user_id=authenticated.user_id,
                        event=(
                            "TURN_YIELDED_RESUMABLE"
                            if status == "INCOMPLETE"
                            else "TURN_COMPLETED"
                        ),
                        details={
                            "turn_id": body.turn_id,
                            "status": status,
                            "result_id": result.get("result_id"),
                            "task_label": task_label,
                        },
                    )
                except Exception as exc:
                    category = str(getattr(exc, "category", type(exc).__name__))
                    event = (
                        "TURN_CANCELLED"
                        if category == "agent_session_cancelled"
                        else "TURN_FAILED"
                    )
                    try:
                        conversation_store.append_agent_message(
                            authenticated.user_id,
                            session_id,
                            role="assistant",
                            content=f"Agent turn ended: {category}",
                            metadata={
                                "turn_id": body.turn_id,
                                "status": category,
                                "task_label": task_label,
                            },
                        )
                    except Exception:
                        LOGGER.exception(
                            "asynchronous agent turn transcript persistence failed",
                            extra={"session_id": session_id},
                        )
                    try:
                        require_tiered().sandboxes.record_progress(
                            session_id,
                            user_id=authenticated.user_id,
                            event=event,
                            details={
                                "turn_id": body.turn_id,
                                "category": category,
                                "task_label": task_label,
                            },
                        )
                    except AgentSandboxError:
                        # The coordinator may have safely released a clean
                        # terminal worktree after recording its own TASK_* event.
                        # The durable conversation message above remains the
                        # UI-level terminal record.
                        pass
                    except Exception:
                        LOGGER.exception(
                            "asynchronous agent turn event persistence failed",
                            extra={"session_id": session_id},
                        )
                finally:
                    async with agent_turn_lock:
                        active = agent_turn_tasks.get(key)
                        if active is not None and active[0] == body.turn_id:
                            agent_turn_tasks.pop(key, None)

            agent_turn_tasks[key] = (body.turn_id, asyncio.create_task(run_submitted_turn()))
        return {
            "status": "SUBMITTED",
            "session_id": session_id,
            "turn_id": body.turn_id,
            "event_cursor": submitted["sequence"],
        }

    @app.get("/api/v1/agent/sessions/{session_id}/messages")
    async def agent_session_messages(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
        limit: int = Query(default=200, ge=1, le=200),
    ) -> JsonObject:
        service = require_tiered()
        service.sandboxes.inspect(
            session_id,
            user_id=authenticated.user_id,
        )
        if conversation_store is None:
            raise HTTPException(
                status_code=503,
                detail="agent_conversation_store_unavailable",
            )
        return {
            "status": "OK",
            "conversation": conversation_store.get_agent_conversation(
                authenticated.user_id,
                session_id,
                limit=limit,
            ),
        }

    @app.get("/api/v1/agent/sessions/{session_id}/events")
    async def agent_session_events(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
        after_sequence: int = Query(default=0, ge=0),
        once: bool = Query(default=False),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        cursor = after_sequence
        if last_event_id is not None:
            try:
                parsed = int(last_event_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="agent_event_cursor_invalid") from exc
            if parsed < 0:
                raise HTTPException(status_code=400, detail="agent_event_cursor_invalid")
            cursor = max(cursor, parsed)
        service = require_tiered()
        service.sandboxes.inspect(session_id, user_id=authenticated.user_id)

        async def stream() -> AsyncIterator[str]:
            nonlocal cursor
            iterations = 1 if once else 240
            for _ in range(iterations):
                records = service.sandboxes.events(
                    session_id,
                    user_id=authenticated.user_id,
                    after_sequence=cursor,
                    limit=200,
                )
                if records:
                    for record in records:
                        sequence = record.get("sequence")
                        if not isinstance(sequence, int):
                            continue
                        cursor = sequence
                        yield (
                            "event: agent_event\n"
                            f"id: {cursor}\n"
                            f"data: {sorted_json_text(record)}\n\n"
                        )
                elif once:
                    yield "event: heartbeat\ndata: {}\n\n"
                if not once:
                    await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/agent/sessions/{session_id}/status")
    async def agent_session_status(
        session_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        if settings.fixture_mode:
            result = service.agent_session_status(
                user_id=authenticated.user_id,
                session_id=session_id,
            )
        else:
            result = await run_in_threadpool(
                service.agent_session_status,
                user_id=authenticated.user_id,
                session_id=session_id,
            )
        worktree = result.get("worktree_status")
        if isinstance(worktree, dict):
            result["worktree_status"] = {
                key: value for key, value in worktree.items() if key != "worktree_root"
            }
        return result

    @app.get("/api/v1/agent/sessions/{session_id}/results/{result_id}")
    async def agent_result_page(
        session_id: str,
        result_id: str,
        authenticated: AuthPrincipal = Depends(agent_principal),
        artifact: str = Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"),
        offset: int = Query(default=0, ge=0),
        max_bytes: int = Query(default=24_000, ge=1, le=65_536),
    ) -> JsonObject:
        binding = require_tiered().sandboxes.inspect(
            session_id,
            user_id=authenticated.user_id,
        )
        if zetsu_service is None:
            raise HTTPException(
                status_code=503,
                detail="repository_agent_conversation_unavailable",
            )
        zetsu_service.repository_agent_service()
        return require_core().repository_result_page(
            user_id=authenticated.user_id,
            repo_id=str(binding["repo_id"]),
            session_id=session_id,
            result_id=result_id,
            artifact=artifact,
            offset=offset,
            max_bytes=max_bytes,
        )

    @app.post("/api/v1/worktrees/{session_id}/cancel")
    @app.post("/api/v1/agent/sessions/{session_id}/cancel")
    async def cancel_agent_session(
        session_id: str,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        if settings.fixture_mode:
            result = service.cancel_agent_session(
                user_id=authenticated.user_id,
                session_id=session_id,
            )
        else:
            result = await run_in_threadpool(
                service.cancel_agent_session,
                user_id=authenticated.user_id,
                session_id=session_id,
            )
        result.pop("worktree_root", None)
        return result

    register_worktree_routes(
        app,
        operator=operator,
        corpus_store=corpus_store,
        require_tiered=require_tiered,
        agent_principal=agent_principal,
        agent_mutation_principal=agent_mutation_principal,
        operator_principal=operator_principal,
        mutation_principal=mutation_principal,
    )

    @app.post("/api/v1/admin/tier/users")
    async def set_tier_user(
        body: TierUserRequest,
        authenticated: AuthPrincipal = Depends(admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        capability = require_tiered().users.set_user(
            body.user_id, CapabilityTier(body.tier), enabled=body.enabled
        )
        if registered_auth is not None:
            registered_auth.registry.update_user(
                body.user_id,
                capability_tier=CapabilityTier(body.tier),
                enabled=body.enabled,
                capabilities=default_capabilities(CapabilityTier(body.tier)),
            )
            registered_auth.sessions.revoke_user(body.user_id)
        operator.record_action(
            actor_role=authenticated.role,
            action="USER_CAPABILITY_SET",
            entity_type="user",
            entity_id=body.user_id,
            payload={
                "tier": body.tier,
                "enabled": body.enabled,
                "revision": capability.revision,
            },
        )
        return {"status": "UPDATED", "capability": capability.public()}

    @app.patch("/api/v1/admin/users/{user_id}/capabilities")
    async def set_user_capabilities(
        user_id: str,
        body: CapabilitySetRequest,
        authenticated: AuthPrincipal = Depends(admin_mutation_principal),
    ) -> JsonObject:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        values = frozenset(Capability(item) for item in body.capabilities)
        service = require_tiered()
        current = service.users.get(user_id)
        capability = service.users.set_user(
            user_id,
            current.tier,
            enabled=current.enabled if body.enabled is None else body.enabled,
            capabilities=values,
        )
        revoked = 0
        if registered_auth is not None:
            registered_auth.registry.update_user(
                user_id,
                capabilities=values,
                **({"enabled": body.enabled} if body.enabled is not None else {}),
            )
            revoked = registered_auth.sessions.revoke_user(user_id)
            registered_auth.audit.append(
                "CAPABILITY_CHANGE",
                outcome="SUCCESS",
                user_id=user_id,
                reason=f"sessions_revoked:{revoked}",
            )
        operator.record_action(
            actor_role=authenticated.role,
            action="USER_CAPABILITIES_SET",
            entity_type="user",
            entity_id=user_id,
            payload={
                "capabilities": sorted(item.value for item in values),
                "enabled": capability.enabled,
                "revision": capability.revision,
                "sessions_revoked": revoked,
            },
        )
        return {
            "status": "UPDATED",
            "capability": capability.public(),
            "sessions_revoked": revoked,
        }

    @app.get("/api/v1/admin/users")
    async def registered_users(
        authenticated: AuthPrincipal = Depends(admin_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        return {
            "status": "OK",
            "registry_revision": registered_auth.registry.snapshot.revision,
            "users": [
                {"user_id": user.user_id, **user.public()}
                for user in sorted(
                    registered_auth.registry.snapshot.users_by_id.values(),
                    key=lambda item: item.normalized_email,
                )
            ],
        }

    @app.post("/api/v1/admin/registry/reload")
    async def reload_registry(
        request: Request,
        authenticated: AuthPrincipal = Depends(admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        before = registered_auth.registry.snapshot
        success, error, after = registered_auth.registry.try_reload()
        if not success:
            registered_auth.audit.append(
                "REGISTRY_RELOAD",
                outcome="DENIED",
                user_id=authenticated.user_id,
                reason=error,
                trace_id=str(request.state.trace_id),
            )
            return {
                "status": "REJECTED_LAST_VALID_RETAINED",
                "failure_category": error,
                "registry_revision": before.revision,
            }
        changed_users = {
            user_id
            for user_id in set(before.users_by_id) | set(after.users_by_id)
            if before.users_by_id.get(user_id) != after.users_by_id.get(user_id)
        }
        for user_id in changed_users:
            registered_auth.sessions.revoke_user(user_id)
            current = after.users_by_id.get(user_id)
            if current is not None:
                require_tiered().users.set_user(
                    user_id,
                    current.capability_tier,
                    enabled=current.enabled,
                    capabilities=current.effective_capabilities,
                )
        registered_auth.audit.append(
            "REGISTRY_RELOAD",
            outcome="SUCCESS",
            user_id=authenticated.user_id,
            reason=f"changed_users:{len(changed_users)}",
            trace_id=str(request.state.trace_id),
        )
        return {
            "status": "RELOADED",
            "registry_revision": after.revision,
            "revoked_user_count": len(changed_users),
        }

    @app.post("/api/v1/admin/users/{user_id}/sessions/revoke")
    async def revoke_user_sessions(
        user_id: str,
        request: Request,
        authenticated: AuthPrincipal = Depends(admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        registered_auth.registry.require_user(user_id)
        count = registered_auth.sessions.revoke_user(user_id)
        registered_auth.audit.append(
            "SESSION_REVOCATION",
            outcome="SUCCESS",
            user_id=user_id,
            reason=f"operator:{authenticated.user_id}",
            trace_id=str(request.state.trace_id),
        )
        return {"status": "SESSIONS_REVOKED", "count": count}

    @app.post("/api/v1/admin/repositories")
    async def register_repository(
        body: RepositoryRegistrationRequest,
        authenticated: AuthPrincipal = Depends(repository_admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        repository = require_tiered().sandboxes.authorizations.register(
            body.repo_id, Path(body.canonical_root)
        )
        operator.record_action(
            actor_role=authenticated.role,
            action="REPOSITORY_REGISTERED",
            entity_type="repository",
            entity_id=body.repo_id,
            payload={"canonical_root": str(repository.canonical_root)},
        )
        return {
            "status": "REGISTERED",
            "repository": {
                **asdict(repository),
                "canonical_root": str(repository.canonical_root),
            },
        }

    @app.post("/api/v1/admin/repository-grants")
    async def grant_repository(
        body: RepositoryGrantRequest,
        authenticated: AuthPrincipal = Depends(repository_admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        grant = require_tiered().sandboxes.authorizations.grant(
            body.user_id,
            body.repo_id,
            base_revision=body.base_revision,
        )
        if registered_auth is not None:
            user = registered_auth.registry.require_user(body.user_id)
            repositories = tuple(sorted(set(user.authorized_repo_ids) | {body.repo_id}))
            registered_auth.registry.update_user(
                body.user_id,
                authorized_repo_ids=repositories,
            )
            registered_auth.sessions.revoke_user(body.user_id)
        operator.record_action(
            actor_role=authenticated.role,
            action="REPOSITORY_GRANTED",
            entity_type="repository_grant",
            entity_id=f"{body.user_id}:{body.repo_id}",
            payload={
                "base_revision": grant.base_revision,
                "revision": grant.revision,
            },
        )
        return {
            "status": "GRANTED",
            "user_id": grant.user_id,
            "repo_id": grant.repository.repo_id,
            "base_revision": grant.base_revision,
            "revision": grant.revision,
        }

    @app.post("/api/v1/admin/repository-grants/revoke")
    async def revoke_repository(
        body: RepositoryGrantRequest,
        authenticated: AuthPrincipal = Depends(repository_admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        grant = require_tiered().sandboxes.authorizations.revoke(body.user_id, body.repo_id)
        if registered_auth is not None:
            user = registered_auth.registry.require_user(body.user_id)
            registered_auth.registry.update_user(
                body.user_id,
                authorized_repo_ids=tuple(
                    item for item in user.authorized_repo_ids if item != body.repo_id
                ),
            )
            registered_auth.sessions.revoke_user(body.user_id)
        operator.record_action(
            actor_role=authenticated.role,
            action="REPOSITORY_REVOKED",
            entity_type="repository_grant",
            entity_id=f"{body.user_id}:{body.repo_id}",
            payload={"revision": grant.revision},
        )
        return {
            "status": "REVOKED",
            "user_id": grant.user_id,
            "repo_id": grant.repository.repo_id,
            "revision": grant.revision,
        }

    @app.get("/api/v1/serving-profiles/status")
    async def serving_profile_status(
        authenticated: AuthPrincipal = Depends(model_admin_principal),
    ) -> dict[str, object]:
        if serving_profile_operator is None:
            raise HTTPException(status_code=503, detail="serving_profile_runtime_unavailable")
        return serving_profile_operator.status()

    @app.post("/api/v1/serving-profiles/action")
    async def serving_profile_action(
        body: ServingProfileActionRequest,
        authenticated: AuthPrincipal = Depends(model_admin_mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        if serving_profile_operator is None:
            raise HTTPException(status_code=503, detail="serving_profile_runtime_unavailable")
        if body.action == "start":
            if body.profile_id is None:
                raise HTTPException(status_code=422, detail="profile_id_required")
            result = serving_profile_operator.start(body.profile_id)
            entity_id = body.profile_id
            action = "SERVING_PROFILE_STARTED"
        else:
            result = serving_profile_operator.stop()
            entity_id = str(result.get("profile_id", "owned_profile"))
            action = "SERVING_PROFILE_STOPPED"
        operator.record_action(
            actor_role=authenticated.role,
            action=action,
            entity_type="serving_profile",
            entity_id=entity_id,
            payload={"status": result.get("status")},
        )
        return result

    @app.get("/api/v1/diagnostics")
    async def diagnostics(
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        return {
            "status": "OK",
            "role": authenticated.role,
            "bind_host": settings.bind_host,
            "allowed_origins": list(settings.allowed_origins),
            "authentication_enabled": True,
            "csrf_enabled": True,
            "pwa_enabled": settings.pwa_enabled,
            "research_backends_supported": [
                "local_governed_corpus",
                "local_uploaded_documents",
                *supported_web_adapter_names(),
            ],
            "searxng": "OPTIONAL_NOT_PROBED",
            "chromadb": "NOT_INSTALLED_NOT_REQUIRED",
            "notifications": "DISABLED_BY_DEFAULT",
        }

    return app
