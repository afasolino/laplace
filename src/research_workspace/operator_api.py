"""Authenticated versioned HTTP adapter for the local Operator Plane."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AsyncIterator, Literal, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .operator_service import (
    OperatorService,
    OperatorServiceError,
    RunExecutor,
)
from .agent_sandbox import AgentSandboxError, AgentToolPolicy
from .research_models import ResearchJobRequest
from .research_plane import DeepResearchService, ResearchPlaneError
from .research_web_adapters import supported_web_adapter_names
from .repository_authorization import RepositoryAuthorizationError
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .serving_profile_runtime import (
    ServingProfileOperator,
    ServingRuntimeError,
)
from .user_capabilities import CapabilityTier, UserCapabilityError


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
    messages: list[TierChatMessage] = Field(min_length=1, max_length=200)


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


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lane: Literal["quality", "standard", "economy"]
    instruction: str = Field(min_length=1, max_length=100_000)
    domain: str = Field(min_length=1, max_length=80)


class TierUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    tier: Literal["basic", "plus", "operator"]
    enabled: bool = True


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


class OperatorAuth:
    """Bearer-role mapping with in-memory, credential-bound CSRF nonces."""

    def __init__(self, token_roles: Mapping[str, str | AuthCredential]) -> None:
        if not token_roles:
            raise ValueError("at least one Operator Plane token is required")
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
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    )
    pwa_enabled: bool = True
    fixture_mode: bool = False

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
) -> FastAPI:
    """Create the localhost GUI/API application without changing execution semantics."""

    if settings.bind_host not in {"127.0.0.1", "localhost", "::1"} and auth is None:
        raise ValueError("authentication is mandatory for non-loopback binding")
    web_root = Path(__file__).with_name("operator_web")
    app = FastAPI(
        title="Laplace Research and Operator Plane",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api/") else "no-cache"
        )
        return response

    async def principal(
        authorization: str | None = Header(default=None),
    ) -> AuthPrincipal:
        return auth.authenticate(authorization)

    async def operator_principal(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> AuthPrincipal:
        if authenticated.capability_tier is not CapabilityTier.OPERATOR:
            raise HTTPException(status_code=403, detail="operator_capability_required")
        return authenticated

    async def mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(operator_principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        auth.validate_csrf(authenticated, x_csrf_token)
        return authenticated

    async def tier_mutation_principal(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
        x_csrf_token: str | None = Header(default=None),
    ) -> AuthPrincipal:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        auth.validate_csrf(authenticated, x_csrf_token)
        return authenticated

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
    async def tier_error(
        _request: Request,
        exc: (
            AgentSandboxError
            | ServiceTierError
            | UserCapabilityError
            | RepositoryAuthorizationError
            | ServingRuntimeError
        ),
    ) -> JSONResponse:
        category = getattr(exc, "category", str(exc))
        evidence = getattr(exc, "evidence", {})
        return JSONResponse(
            status_code=403,
            content={
                "status": "ERROR",
                "failure_category": category,
                "evidence": evidence,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((web_root / "index.html").read_text(encoding="utf-8"))

    @app.get("/assets/{asset_name}")
    async def asset(asset_name: str) -> Response:
        allowed = {
            "app.css": "text/css",
            "app.js": "text/javascript",
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
        }

    @app.post("/api/v1/session")
    async def session(
        request: Request,
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        return {
            "status": "AUTHENTICATED",
            "role": authenticated.role,
            "user_id": authenticated.user_id,
            "capability_tier": authenticated.capability_tier.value,
            "model_lanes": [lane.value for lane in ModelLane],
            "csrf_token": auth.issue_csrf(authenticated),
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
        return {
            **summary,
            "model_servers": model_status,
            "research_jobs": (
                _research_summaries(research.layout.research_jobs)
                if research is not None
                else []
            ),
            "warnings": [],
        }

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
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        return operator.model_server_action(
            body.action,
            approval_id=body.approval_id,
            actor_role=authenticated.role,
        )

    @app.get("/api/v1/model-servers/status")
    async def model_server_status(
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        return sanitized_model_status(authenticated.role)

    @app.post("/api/v1/research/jobs")
    async def create_research(
        body: ResearchCreateRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        result = research.create(body.job, job_id=body.research_job_id)
        job_id = str(result["research_job_id"])
        operator.record_action(
            actor_role=authenticated.role,
            action="RESEARCH_JOB_CREATED",
            entity_type="research_job",
            entity_id=job_id,
            payload={
                "request_sha256": hashlib.sha256(
                    body.job.model_dump_json().encode("utf-8")
                ).hexdigest()
            },
        )
        return result

    @app.post("/api/v1/research/jobs/{job_id}/run")
    async def run_research(
        job_id: str,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        result = research.run(job_id)
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
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        del authenticated
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        return research.get(job_id).model_dump(mode="json")

    @app.get("/api/v1/research/jobs/{job_id}/report")
    async def get_research_report(
        job_id: str,
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        del authenticated
        if research is None:
            raise HTTPException(status_code=503, detail="research_plane_unavailable")
        root = research.layout.research_jobs / job_id
        job = research.get(job_id)
        if job.status != "COMPLETE":
            raise HTTPException(status_code=409, detail="research_job_not_complete")
        return {
            "status": "OK",
            "job": job.model_dump(mode="json"),
            "report_markdown": (root / "report.md").read_text(encoding="utf-8"),
            "evidence_ledger": json.loads(
                (root / "evidence_ledger.json").read_text(encoding="utf-8")
            ),
            "claim_source_graph": json.loads(
                (root / "claim_source_graph.json").read_text(encoding="utf-8")
            ),
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

    @app.get("/api/v1/tier/capabilities")
    async def tier_capabilities(
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        service = require_tiered()
        effective = service.capability(authenticated.user_id)
        if effective is not authenticated.capability_tier:
            raise HTTPException(status_code=401, detail="credential_capability_changed")
        return {
            "status": "OK",
            "user_id": authenticated.user_id,
            "capability_tier": effective.value,
            "chat_enabled": effective in {CapabilityTier.BASIC, CapabilityTier.PLUS},
            "agent_enabled": effective is CapabilityTier.PLUS,
            "operator_enabled": effective is CapabilityTier.OPERATOR,
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
        }

    @app.post("/api/v1/chat")
    async def tier_chat(
        body: TierChatRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        messages = [message.model_dump() for message in body.messages]
        if settings.fixture_mode:
            return service.chat(
                user_id=authenticated.user_id,
                lane=ModelLane(body.lane),
                messages=messages,
                domain=body.domain,
                session_id=body.session_id,
            )
        return await run_in_threadpool(
            service.chat,
            user_id=authenticated.user_id,
            lane=ModelLane(body.lane),
            messages=messages,
            domain=body.domain,
            session_id=body.session_id,
        )

    @app.post("/api/v1/agent/sessions")
    async def create_agent_session(
        body: AgentSessionRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        tool_policy = AgentToolPolicy(
            policy_id=f"api-{body.session_id}",
            allowed_tools=tuple(body.allowed_tools),
            network_enabled=False,
            max_commands=body.max_commands,
            max_wall_seconds=body.max_wall_seconds,
        )
        if settings.fixture_mode:
            return service.create_agent_session(
                user_id=authenticated.user_id,
                repo_id=body.repo_id,
                session_id=body.session_id,
                tool_policy=tool_policy,
            )
        return await run_in_threadpool(
            service.create_agent_session,
            user_id=authenticated.user_id,
            repo_id=body.repo_id,
            session_id=body.session_id,
            tool_policy=tool_policy,
        )

    @app.post("/api/v1/agent/sessions/{session_id}/run")
    @app.post("/api/v1/agent/sessions/{session_id}/messages")
    async def run_agent_session(
        session_id: str,
        body: AgentRunRequest,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        if settings.fixture_mode:
            return service.agent(
                user_id=authenticated.user_id,
                session_id=session_id,
                lane=ModelLane(body.lane),
                instruction=body.instruction,
                domain=body.domain,
            )
        return await run_in_threadpool(
            service.agent,
            user_id=authenticated.user_id,
            session_id=session_id,
            lane=ModelLane(body.lane),
            instruction=body.instruction,
            domain=body.domain,
        )

    @app.get("/api/v1/agent/sessions/{session_id}/status")
    async def agent_session_status(
        session_id: str,
        authenticated: AuthPrincipal = Depends(principal),
    ) -> dict[str, object]:
        service = require_tiered()
        if settings.fixture_mode:
            return service.agent_session_status(
                user_id=authenticated.user_id,
                session_id=session_id,
            )
        return await run_in_threadpool(
            service.agent_session_status,
            user_id=authenticated.user_id,
            session_id=session_id,
        )

    @app.post("/api/v1/agent/sessions/{session_id}/cancel")
    async def cancel_agent_session(
        session_id: str,
        authenticated: AuthPrincipal = Depends(tier_mutation_principal),
    ) -> dict[str, object]:
        service = require_tiered()
        if settings.fixture_mode:
            return service.cancel_agent_session(
                user_id=authenticated.user_id,
                session_id=session_id,
            )
        return await run_in_threadpool(
            service.cancel_agent_session,
            user_id=authenticated.user_id,
            session_id=session_id,
        )

    @app.post("/api/v1/admin/tier/users")
    async def set_tier_user(
        body: TierUserRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        capability = require_tiered().users.set_user(
            body.user_id, CapabilityTier(body.tier), enabled=body.enabled
        )
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
        return {"status": "UPDATED", "capability": asdict(capability)}

    @app.post("/api/v1/admin/repositories")
    async def register_repository(
        body: RepositoryRegistrationRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
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
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        grant = require_tiered().sandboxes.authorizations.grant(
            body.user_id,
            body.repo_id,
            base_revision=body.base_revision,
        )
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
        authenticated: AuthPrincipal = Depends(mutation_principal),
    ) -> dict[str, object]:
        if authenticated.role != "admin":
            raise HTTPException(status_code=403, detail="admin_role_required")
        grant = require_tiered().sandboxes.authorizations.revoke(
            body.user_id, body.repo_id
        )
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
        authenticated: AuthPrincipal = Depends(operator_principal),
    ) -> dict[str, object]:
        if serving_profile_operator is None:
            raise HTTPException(status_code=503, detail="serving_profile_runtime_unavailable")
        return serving_profile_operator.status()

    @app.post("/api/v1/serving-profiles/action")
    async def serving_profile_action(
        body: ServingProfileActionRequest,
        authenticated: AuthPrincipal = Depends(mutation_principal),
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
