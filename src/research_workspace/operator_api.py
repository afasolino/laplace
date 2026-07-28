"""Authenticated versioned HTTP adapter for the local Operator Plane."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Literal, Mapping, TypeAlias
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
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
from .personal_corpus import (
    CorpusError,
    PersonalCorpusStore,
    RetrievalSelection,
)
from .request_state import RequestStateError, RequestStateStore
from .agent_sandbox import AgentSandboxError, AgentToolPolicy
from .research_models import ResearchJobRequest
from .research_admission import ResearchAdmissionError, ResearchAdmissionStore
from .research_plane import DeepResearchService, ResearchPlaneError
from .research_web_adapters import supported_web_adapter_names
from .repository_authorization import RepositoryAuthorizationError
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
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

JsonObject: TypeAlias = dict[str, object]
LOGGER = logging.getLogger("laplace.operator")


class RunPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    configuration: dict[str, object]
    run_id: str | None = Field(default=None, max_length=160)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str = Field(min_length=1, max_length=80)
    entity_id: str = Field(min_length=1, max_length=320)
    payload: dict[str, object] = Field(default_factory=dict)


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approve: bool


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approval_id: str | None = Field(default=None, max_length=160)


class ModelServerActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    approval_id: str | None = Field(default=None, max_length=160)


class ResearchCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    job: ResearchJobRequest
    research_job_id: str | None = Field(default=None, max_length=160)
    domain: str = Field(default="general", min_length=1, max_length=80)


class TierChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: str
    content: str = Field(min_length=1, max_length=100_000)


class TierChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lane: Literal["quality", "standard", "economy"]
    domain: str = Field(default="general", min_length=1, max_length=80)
    session_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
    )
    conversation_id: str | None = Field(
        default=None, pattern=r"^conv-[a-f0-9]{32}$"
    )
    messages: list[TierChatMessage] = Field(min_length=1, max_length=200)
    request_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,159}$"
    )
    retrieval_selection: Literal[
        "none", "personal", "shared", "both", "selected_personal"
    ] = "none"
    personal_corpus_id: str | None = Field(
        default=None, pattern=r"^pc_[a-f0-9]{32}$"
    )


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1_024)


class ActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    email: str = Field(min_length=3, max_length=320)
    activation_code: str = Field(min_length=1, max_length=1_024)
    new_password: str = Field(min_length=12, max_length=1_024)


class PasswordChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    current_password: str = Field(min_length=1, max_length=1_024)
    new_password: str = Field(min_length=12, max_length=1_024)


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="New conversation", max_length=160)


class ConversationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str | None = Field(default=None, max_length=160)
    archived: bool | None = None
    draft: str | None = Field(default=None, max_length=100_000)


class AgentSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["read_file", "apply_patch", "run_validation"],
        min_length=1,
        max_length=20,
    )
    max_commands: int = Field(default=100, ge=1, le=1_000)
    max_wall_seconds: int = Field(default=1_800, ge=1, le=14_400)
    task_title: str = Field(default="New Agent task", min_length=1, max_length=200)
    idempotency_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$"
    )


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lane: Literal["quality", "standard", "economy"]
    instruction: str = Field(min_length=1, max_length=100_000)
    domain: str = Field(min_length=1, max_length=80)
    retrieval_selection: Literal[
        "none", "personal", "shared", "both", "selected_personal"
    ] = "none"
    personal_corpus_id: str | None = Field(
        default=None, pattern=r"^pc_[a-f0-9]{32}$"
    )


class TierUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    tier: Literal["basic", "plus", "operator"]
    enabled: bool = True


class CapabilitySetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    capabilities: list[
        Literal[
            "chat",
            "agent",
            "research",
            "operator",
            "admin",
            "personal_corpus",
            "shared_corpus_ingest",
            "repository_admin",
            "model_admin",
        ]
    ] = Field(max_length=9)
    enabled: bool | None = None


class CorpusCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=160)


class CorpusUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str | None = Field(default=None, min_length=1, max_length=160)
    archived: bool | None = None


class UploadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    corpus_id: str = Field(pattern=r"^pc_[a-f0-9]{32}$")
    idempotency_key: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$"
    )


class IndexUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    idempotency_key: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$"
    )


class CorpusSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=4_000)
    corpus_id: str | None = Field(default=None, pattern=r"^pc_[a-f0-9]{32}$")
    limit: int = Field(default=8, ge=1, le=50)


class WorktreeDiscardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confirmation: str = Field(min_length=9, max_length=180)


class WorktreeExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    promotion: bool = False


class RepositoryRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    canonical_root: str = Field(min_length=1, max_length=4_096)


class RepositoryGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    base_revision: str = Field(default="HEAD", min_length=1, max_length=200)


class ServingProfileActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "stop"]
    profile_id: str | None = Field(
        default=None, pattern=r"^P[0-9]+(?:_[a-z0-9_]+)?$"
    )


@dataclass(frozen=True)
class AuthCredential:
    role: str
    user_id: str
    capability_tier: CapabilityTier


@dataclass(frozen=True)
class AuthPrincipal:
    role: str
    user_id: str
    capability_tier: CapabilityTier
    credential_sha256: str
    email: str | None = None
    display_name: str | None = None
    default_lane: str = "standard"
    auth_method: Literal["bearer", "session"] = "bearer"
    session_identifier: str | None = None


class OperatorAuth:
    """Bearer-role mapping with in-memory, credential-bound CSRF nonces."""

    def __init__(self, token_roles: Mapping[str, str | AuthCredential]) -> None:
        allowed = {"read", "operate", "approve", "admin"}
        credentials: dict[str, AuthCredential] = {}
        for token, value in token_roles.items():
            binding = (
                AuthCredential(
                    role=value,
                    user_id=f"operator-{value}",
                    capability_tier=CapabilityTier.OPERATOR,
                )
                if isinstance(value, str)
                else value
            )
            if binding.role not in allowed:
                raise ValueError("invalid Operator Plane role")
            credentials[token] = binding
        if any(binding.role not in allowed for binding in credentials.values()):
            raise ValueError("invalid Operator Plane role")
        if any(len(token) < 24 for token in token_roles):
            raise ValueError("Operator Plane tokens must contain at least 24 characters")
        self._credentials = {
            hashlib.sha256(token.encode("utf-8")).hexdigest(): binding
            for token, binding in credentials.items()
        }
        self._csrf: dict[str, str] = {}

    def authenticate(self, authorization: str | None) -> AuthPrincipal:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="authentication_required")
        token = authorization.removeprefix("Bearer ")
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        matched_digest: str | None = None
        binding: AuthCredential | None = None
        for expected, expected_binding in self._credentials.items():
            if hmac.compare_digest(digest, expected):
                matched_digest = expected
                binding = expected_binding
        if matched_digest is None or binding is None:
            raise HTTPException(status_code=401, detail="authentication_failed")
        return AuthPrincipal(
            role=binding.role,
            user_id=binding.user_id,
            capability_tier=binding.capability_tier,
            credential_sha256=matched_digest,
        )

    def issue_csrf(self, principal: AuthPrincipal) -> str:
        nonce = secrets.token_urlsafe(32)
        self._csrf[principal.credential_sha256] = nonce
        return nonce

    def validate_csrf(self, principal: AuthPrincipal, supplied: str | None) -> None:
        expected = self._csrf.get(principal.credential_sha256)
        if expected is None or supplied is None or not hmac.compare_digest(expected, supplied):
            raise HTTPException(status_code=403, detail="csrf_validation_failed")


@dataclass(frozen=True)
class OperatorApiSettings:
    bind_host: str = "127.0.0.1"
    port: int = 8765
    deployment_mode: Literal["local", "ssh-tunnel", "reverse-proxy"] = "local"
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    )
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    trusted_proxies: tuple[str, ...] = ("127.0.0.1", "::1")
    external_url: str | None = None
    allow_insecure_lan_http: bool = False
    bearer_api_enabled: bool = False
    pwa_enabled: bool = True
    fixture_mode: bool = False
    maximum_request_bytes: int = 70_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65_535:
            raise ValueError("invalid Operator Plane port")
        if not self.allowed_origins:
            raise ValueError("at least one allowed origin is required")
        if any(
            not origin.startswith(("http://", "https://")) or "*" in origin
            for origin in self.allowed_origins
        ):
            raise ValueError("Operator Plane origins must be explicit HTTP origins")
        if not self.allowed_hosts or any("*" in host or "/" in host for host in self.allowed_hosts):
            raise ValueError("Operator Plane hosts must be explicit")
        loopback = self.bind_host in {"127.0.0.1", "localhost", "::1"}
        if not loopback and not self.allow_insecure_lan_http:
            raise ValueError(
                "non-loopback direct binding requires --allow-insecure-lan-http"
            )
        if self.deployment_mode in {"local", "ssh-tunnel", "reverse-proxy"} and not loopback:
            if not self.allow_insecure_lan_http:
                raise ValueError("production and tunnel modes require loopback binding")
        if self.deployment_mode == "reverse-proxy":
            if self.external_url is None:
                raise ValueError("reverse-proxy mode requires an external URL")
            external = urlsplit(self.external_url)
            if external.scheme != "https" or not external.hostname:
                raise ValueError("reverse-proxy external URL must use HTTPS")
            if self.external_url.rstrip("/") not in {
                origin.rstrip("/") for origin in self.allowed_origins
            }:
                raise ValueError("external URL must be an explicit allowed origin")
        if self.maximum_request_bytes < 1_024:
            raise ValueError("maximum request body is too small")

    @property
    def secure_cookie(self) -> bool:
        return self.deployment_mode == "reverse-proxy"

    @property
    def development_http(self) -> bool:
        return self.deployment_mode in {"local", "ssh-tunnel"} and self.bind_host in {
            "127.0.0.1",
            "localhost",
            "::1",
        }


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def create_operator_app(
    operator: OperatorService,
    auth: OperatorAuth,
    *,
    settings: OperatorApiSettings = OperatorApiSettings(),
    research: DeepResearchService | None = None,
    run_executor: RunExecutor | None = None,
    tiered: TieredServingService | None = None,
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
    app = FastAPI(
        title="Laplace Research and Operator Plane",
        version="1.1",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token", "Authorization", "X-Request-ID"],
        expose_headers=["X-Trace-Id", "X-Content-SHA256"],
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
        if (
            response is None
            and forwarded
            and transport_client not in settings.trusted_proxies
        ):
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
            forwarded_host = urlsplit(
                f"//{request.headers.get('x-forwarded-host', '')}"
            ).hostname
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
                record = registered_auth.sessions.resolve(
                    cookie, registered_auth.registry
                )
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
                    raise HTTPException(
                        status_code=401, detail="credential_capability_changed"
                    )
        return default_capabilities(authenticated.capability_tier)

    def _require_named(
        authenticated: AuthPrincipal, required: Capability
    ) -> AuthPrincipal:
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
    async def operator_error(
        _request: Request, exc: OperatorServiceError
    ) -> JSONResponse:
        status_code = 403 if exc.category in {
            "authorization_failure",
            "approval_required",
        } else 409
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ERROR",
                "failure_category": exc.category,
                "evidence": exc.evidence,
            },
        )

    @app.exception_handler(ResearchPlaneError)
    async def research_error(
        _request: Request, exc: ResearchPlaneError
    ) -> JSONResponse:
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
                    "trace_id": str(
                        getattr(request.state, "trace_id", "unavailable")
                    ),
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
        user = registered_auth.registry.require_user(session_value.record.user_id) if registered_auth else None
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

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((web_root / "index.html").read_text(encoding="utf-8"))

    @app.get("/assets/{asset_name}")
    async def asset(asset_name: str) -> Response:
        allowed = {
            "app.css": "text/css",
            "app.js": "text/javascript",
            "favicon.svg": "image/svg+xml",
        }
        if asset_name not in allowed:
            raise HTTPException(status_code=404)
        return Response(
            (web_root / asset_name).read_bytes(),
            media_type=allowed[asset_name],
        )

    @app.get("/manifest.webmanifest")
    async def manifest() -> Response:
        if not settings.pwa_enabled:
            raise HTTPException(status_code=404)
        return Response(
            (web_root / "manifest.webmanifest").read_bytes(),
            media_type="application/manifest+json",
        )

    @app.get("/sw.js")
    async def service_worker() -> Response:
        if not settings.pwa_enabled:
            raise HTTPException(status_code=404)
        return Response(
            (web_root / "sw.js").read_bytes(),
            media_type="text/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )

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
            }

            async def endpoint_failure(
                endpoint: str,
                model_id: str,
                lane: str,
            ) -> str | None:
                parsed = urlsplit(endpoint)
                if (
                    parsed.scheme != "http"
                    or parsed.hostname not in {"127.0.0.1", "localhost"}
                ):
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
                    path = parsed.path.rstrip("/") + "/models"
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
                served = {
                    str(item["id"])
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                } if isinstance(data, list) else set()
                return (
                    None
                    if model_id in served
                    else f"model_identity_mismatch:{lane}"
                )

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
            reasons.extend(
                reason for reason in endpoint_checks if reason is not None
            )
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
                "model_endpoints_required": tiered is not None,
                "personal_corpus": corpus_health,
            },
        )

    @app.get("/api/v1/version")
    async def version() -> dict[str, object]:
        return {
            "status": "OK",
            "application": "Laplace",
            "application_version": "0.1.0",
            "api_version": "v1",
            "git_revision": _git_revision(operator.repository_root),
            "deployment_mode": settings.deployment_mode,
        }

    @app.post("/api/v1/auth/login")
    async def login(body: LoginRequest, request: Request, response: Response) -> JsonObject:
        _origin_allowed(request)
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        started = time.monotonic()
        try:
            session_value = registered_auth.login(
                body.email,
                body.password,
                client_ip=str(getattr(request.state, "client_ip", "unknown")),
                trace_id=str(request.state.trace_id),
            )
        except AuthSessionError as exc:
            headers = (
                {"Retry-After": str(max(1, int(exc.retry_after_seconds)))}
                if exc.retry_after_seconds is not None
                else None
            )
            raise HTTPException(
                status_code=429 if exc.category == "authentication_rate_limited" else 401,
                detail="authentication_failed",
                headers=headers,
            ) from exc
        if time.monotonic() - started > 15:
            registered_auth.sessions.revoke(session_value.identifier)
            raise HTTPException(status_code=408, detail="authentication_timeout")
        _set_session_cookie(response, session_value.identifier)
        return _session_response(session_value)

    @app.post("/api/v1/auth/activate")
    async def activate(
        body: ActivationRequest,
        request: Request,
        response: Response,
    ) -> JsonObject:
        _origin_allowed(request)
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        try:
            session_value = registered_auth.activate(
                body.email,
                body.activation_code,
                body.new_password,
                client_ip=str(getattr(request.state, "client_ip", "unknown")),
                trace_id=str(request.state.trace_id),
            )
        except (AuthSessionError, AuthRegistryError) as exc:
            raise HTTPException(status_code=401, detail="authentication_failed") from exc
        _set_session_cookie(response, session_value.identifier)
        return _session_response(session_value)

    @app.get("/api/v1/auth/session")
    async def browser_session(
        request: Request,
    ) -> JsonObject:
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        identifier = request.cookies.get("laplace_session")
        if identifier is None:
            return {
                "status": "SIGNED_OUT",
                "development_http": settings.development_http,
                "deployment_mode": settings.deployment_mode,
            }
        try:
            record = registered_auth.sessions.resolve(
                identifier,
                registered_auth.registry,
            )
            csrf = registered_auth.sessions.rotate_csrf(identifier)
            user = registered_auth.registry.require_user(record.user_id)
        except (AuthSessionError, AuthRegistryError) as exc:
            raise HTTPException(status_code=401, detail="authentication_required") from exc
        return {
            "status": "AUTHENTICATED",
            "account": user.public(),
            "role": user.role,
            "capability_tier": user.capability_tier.value,
            "default_lane": user.default_lane,
            "csrf_token": csrf,
            "development_http": settings.development_http,
            "deployment_mode": settings.deployment_mode,
        }

    @app.post("/api/v1/auth/logout")
    async def logout(
        response: Response,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> JsonObject:
        if authenticated.auth_method == "session" and registered_auth is not None:
            if authenticated.session_identifier is not None:
                registered_auth.sessions.revoke(authenticated.session_identifier)
            registered_auth.audit.append(
                "LOGOUT",
                outcome="SUCCESS",
                user_id=authenticated.user_id,
            )
        response.delete_cookie(
            "laplace_session",
            path="/",
            secure=settings.secure_cookie,
            httponly=True,
            samesite="strict",
        )
        return {"status": "SIGNED_OUT"}

    @app.post("/api/v1/auth/logout-all")
    async def logout_all(
        response: Response,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> JsonObject:
        if registered_auth is None:
            raise HTTPException(status_code=503, detail="registered_email_auth_unavailable")
        count = registered_auth.sessions.revoke_user(authenticated.user_id)
        registered_auth.audit.append(
            "LOGOUT_ALL",
            outcome="SUCCESS",
            user_id=authenticated.user_id,
        )
        response.delete_cookie(
            "laplace_session",
            path="/",
            secure=settings.secure_cookie,
            httponly=True,
            samesite="strict",
        )
        return {"status": "SESSIONS_REVOKED", "count": count}

    @app.post("/api/v1/auth/change-password")
    async def change_password(
        body: PasswordChangeRequest,
        response: Response,
        request: Request,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> JsonObject:
        if (
            registered_auth is None
            or authenticated.auth_method != "session"
            or authenticated.session_identifier is None
        ):
            raise HTTPException(status_code=401, detail="browser_session_required")
        record = registered_auth.sessions.resolve(
            authenticated.session_identifier,
            registered_auth.registry,
        )
        try:
            session_value = registered_auth.change_password(
                record,
                body.current_password,
                body.new_password,
                trace_id=str(request.state.trace_id),
            )
        except (AuthSessionError, AuthRegistryError) as exc:
            raise HTTPException(status_code=401, detail="authentication_failed") from exc
        _set_session_cookie(response, session_value.identifier)
        return _session_response(session_value)

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
            csrf_token = registered_auth.sessions.rotate_csrf(
                authenticated.session_identifier
            )
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
        status = operator.model_server_action(
            "status", approval_id=None, actor_role=role
        )
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
                _research_summaries(research.layout.research_jobs)
                if research is not None
                else []
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

    @app.post("/api/v1/runs")
    async def prepare_run(
        body: RunPrepareRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        return operator.prepare_run(
            body.configuration,
            actor_role=authenticated.role,
            run_id=body.run_id,
        )

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(
        run_id: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        return operator.get_run(run_id, actor_role=authenticated.role)

    @app.post("/api/v1/runs/{run_id}/start")
    async def start_run(
        run_id: str,
        body: StartRunRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
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
    ) -> dict[str, object]:
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
    ) -> dict[str, object]:
        return {
            "status": "OK",
            "approvals": operator.approvals(
                actor_role=authenticated.role, state=state
            ),
        }

    @app.post("/api/v1/approvals/{approval_id}/decision")
    async def decide_approval(
        approval_id: str,
        body: ApprovalDecisionRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        return operator.decide_approval(
            approval_id,
            approve=body.approve,
            actor_role=authenticated.role,
        )

    @app.post("/api/v1/model-servers/action")
    async def model_server_action(
        body: ModelServerActionRequest,
        authenticated: AuthPrincipal = Depends(model_admin_mutation_principal),
    ) -> dict[str, object]:
        return operator.model_server_action(
            body.action,
            approval_id=body.approval_id,
            actor_role=authenticated.role,
        )

    @app.get("/api/v1/model-servers/status")
    async def model_server_status(
        authenticated: AuthPrincipal = Depends(model_admin_principal),
    ) -> dict[str, object]:
        return sanitized_model_status(authenticated.role)

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
            "eligible_verification_tools": list(
                selected_domain.eligible_verification_tools
            ),
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
        sanitized = _sanitize_research_payload(result)
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
            "job": _sanitize_research_payload(job.model_dump(mode="json")),
            "report_markdown": (root / "report.md").read_text(encoding="utf-8"),
            "evidence_ledger": _sanitize_research_payload(
                json.loads(
                    (root / "evidence_ledger.json").read_text(encoding="utf-8")
                )
            ),
            "claim_source_graph": _sanitize_research_payload(
                json.loads(
                    (root / "claim_source_graph.json").read_text(encoding="utf-8")
                )
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
            (
                research.layout.research_jobs / job_id / "claim_source_graph.json"
            ).read_bytes()
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
                            f"data: {json.dumps(record, sort_keys=True)}\n\n"
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
        artifact = _safe_artifact_path(
            path,
            operator,
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
        artifact = _safe_artifact_path(
            path,
            operator,
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
        return {
            "status": "OK",
            "application_version": "0.1.0",
            "git_revision": _git_revision(operator.repository_root),
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
            "shared_corpus_ingest_enabled": (
                Capability.SHARED_CORPUS_INGEST in capabilities
            ),
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
            authorized = service.sandboxes.authorizations.authorized_for_user(
                authenticated.user_id
            )
            if registered_auth is not None:
                registry_user = registered_auth.registry.require_user(
                    authenticated.user_id
                )
                allowed = set(registry_user.authorized_repo_ids)
                authorized = [
                    item for item in authorized if item.get("repo_id") in allowed
                ]
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
        service = require_tiered()
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
                (
                    message.content
                    for message in reversed(body.messages)
                    if message.role == "user"
                ),
                "",
            )
            personal = corpus_store.search(
                authenticated.user_id,
                retrieval_query,
                corpus_id=(
                    body.personal_corpus_id
                    if body.retrieval_selection
                    == RetrievalSelection.SELECTED_PERSONAL.value
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
                (
                    message.content
                    for message in reversed(body.messages)
                    if message.role == "user"
                ),
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
                result = service.chat(
                    user_id=authenticated.user_id,
                    lane=ModelLane(body.lane),
                    messages=messages,
                    domain=body.domain,
                    session_id=body.session_id or conversation_id,
                )
            else:
                result = await run_in_threadpool(
                    service.chat,
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
                queue_position=(
                    queue_value
                    if isinstance(queue_value, int)
                    else None
                ),
                effective_lane=(
                    str(result["effective_lane"])
                    if result.get("effective_lane") is not None
                    else None
                ),
                model_name=(
                    str(result["model_id"])
                    if result.get("model_id") is not None
                    else None
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
            "corpus": corpus_store.create_corpus(
                authenticated.user_id, body.name
            ),
        }

    @app.get("/api/v1/personal-corpora/{corpus_id}")
    async def get_personal_corpus(
        corpus_id: str,
        authenticated: AuthPrincipal = Depends(personal_corpus_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "corpus": corpus_store.require_corpus(
                authenticated.user_id, corpus_id
            ),
            "sources": corpus_store.list_sources(
                authenticated.user_id, corpus_id
            ),
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
        state: Literal["STAGING", "CANCELLED", "INDEXED"] = Query(
            default="STAGING"
        ),
    ) -> JsonObject:
        return {
            "status": "OK",
            "uploads": corpus_store.list_uploads(
                authenticated.user_id, state=state
            ),
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
        relative_path, content, media_type = await _upload_form(
            request, zip_fallback=False
        )
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
        _name, content, _media_type = await _upload_form(
            request, zip_fallback=True
        )
        return corpus_store.stage_zip_fallback(
            _authenticated.user_id, upload_id, content=content
        )

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

    @app.delete(
        "/api/v1/personal-corpora/{corpus_id}/sources/{source_id}"
    )
    async def delete_personal_source(
        corpus_id: str,
        source_id: str,
        authenticated: AuthPrincipal = Depends(corpus_mutation_principal),
    ) -> JsonObject:
        return corpus_store.delete_source(
            authenticated.user_id, corpus_id, source_id
        )

    @app.get(
        "/api/v1/personal-corpora/{corpus_id}/sources/{source_id}/download"
    )
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
        if registered_auth is not None:
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
        return result

    @app.post("/api/v1/agent/sessions/{session_id}/run")
    @app.post("/api/v1/agent/sessions/{session_id}/messages")
    async def run_agent_session(
        session_id: str,
        body: AgentRunRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
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
            personal = corpus_store.search(
                authenticated.user_id,
                body.instruction,
                corpus_id=(
                    body.personal_corpus_id
                    if body.retrieval_selection
                    == RetrievalSelection.SELECTED_PERSONAL.value
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
        if settings.fixture_mode:
            result = service.agent(
                user_id=authenticated.user_id,
                session_id=session_id,
                lane=ModelLane(body.lane),
                instruction=instruction,
                domain=body.domain,
            )
        else:
            result = await run_in_threadpool(
                service.agent,
                user_id=authenticated.user_id,
                session_id=session_id,
                lane=ModelLane(body.lane),
                instruction=instruction,
                domain=body.domain,
            )
        result["retrieval"] = retrieval
        return result

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
                key: value
                for key, value in worktree.items()
                if key != "worktree_root"
            }
        return result

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

    @app.get("/api/v1/worktrees")
    async def list_worktrees(
        authenticated: AuthPrincipal = Depends(agent_principal),
    ) -> JsonObject:
        return {
            "status": "OK",
            "worktrees": require_tiered().sandboxes.list_mine(
                authenticated.user_id
            ),
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
            "events": require_tiered().sandboxes.history(
                session_id, user_id=authenticated.user_id
            ),
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
        return require_tiered().sandboxes.close_if_clean(
            session_id, user_id=authenticated.user_id
        )

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
        content = require_tiered().sandboxes.patch(
            session_id, user_id=authenticated.user_id
        )
        return Response(
            content,
            media_type="text/x-diff",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{session_id}.patch"'
                ),
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
                **(
                    {"enabled": body.enabled}
                    if body.enabled is not None
                    else {}
                ),
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
        grant = require_tiered().sandboxes.authorizations.revoke(
            body.user_id, body.repo_id
        )
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


def _research_summaries(root: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if not root.is_dir():
        return summaries
    for path in sorted(root.glob("*/job.json"), reverse=True)[:20]:
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            summaries.append(
                {
                    key: value.get(key)
                    for key in (
                        "research_job_id",
                        "question",
                        "research_mode",
                        "status",
                        "current_stage",
                        "created_at",
                    )
                }
            )
    return summaries


def _sanitize_research_payload(value: object) -> object:
    """Remove server paths from research records while preserving cited evidence."""

    if isinstance(value, list):
        return [_sanitize_research_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, object] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        if key in {"local_snapshot_path", "report_path", "evidence_ledger_path"}:
            continue
        if key == "canonical_url" and isinstance(item, str) and item.startswith("file:"):
            sanitized[key] = "local-document://authorized-source"
            continue
        sanitized[key] = _sanitize_research_payload(item)
    return sanitized


def _git_revision(repository_root: Path) -> str:
    """Return a sanitized immutable revision without surfacing Git stderr."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root.resolve()), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    revision = completed.stdout.strip().lower()
    if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision):
        return revision
    return "unavailable"


def _safe_artifact_path(
    relative_path: str,
    operator: OperatorService,
    *,
    allow_sensitive: bool,
) -> Path:
    if "\x00" in relative_path or Path(relative_path).is_absolute():
        raise HTTPException(status_code=400, detail="invalid_artifact_path")
    if relative_path.startswith("outputs/"):
        candidate = (operator.repository_root / relative_path).resolve()
    else:
        candidate = (operator.state_root / relative_path).resolve()
    roots = (
        operator.state_root.resolve(),
        (operator.repository_root / "outputs").resolve(),
    )
    if not any(_is_within(candidate, root) for root in roots) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="artifact_not_found")
    lowered = {part.lower() for part in candidate.parts}
    forbidden = {"held_out", "held-out", "secrets", "credentials", "prompts"}
    if not allow_sensitive and lowered.intersection(forbidden):
        raise HTTPException(status_code=403, detail="artifact_access_forbidden")
    allowed_suffixes = {
        ".json",
        ".jsonl",
        ".md",
        ".html",
        ".txt",
        ".log",
        ".sv",
        ".v",
        ".zip",
        ".gz",
        ".tar",
        ".bib",
    }
    if candidate.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=403, detail="artifact_type_forbidden")
    if candidate.stat().st_size > 256_000_000:
        raise HTTPException(status_code=413, detail="artifact_too_large")
    return candidate
