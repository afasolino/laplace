"""Browser-session authentication routes for the Operator transport."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response

from ..auth_registry import AuthRegistryError
from ..auth_sessions import AuthSessionError, NewSession, RegisteredEmailAuth
from .auth import AuthPrincipal
from .request_models import ActivationRequest, LoginRequest, PasswordChangeRequest
from .settings import OperatorApiSettings

JsonObject = dict[str, object]
PrincipalDependency = Callable[..., Awaitable[AuthPrincipal]]
OriginGuard = Callable[[Request], None]
SessionCookieWriter = Callable[[Response, str], None]
SessionResponse = Callable[[NewSession], JsonObject]


def register_auth_routes(
    app: FastAPI,
    *,
    settings: OperatorApiSettings,
    registered_auth: RegisteredEmailAuth | None,
    tier_mutation_principal: PrincipalDependency,
    origin_allowed: OriginGuard,
    set_session_cookie: SessionCookieWriter,
    session_response: SessionResponse,
) -> None:
    """Register browser-auth routes without moving principal/CSRF policy."""

    @app.post("/api/v1/auth/login")
    async def login(body: LoginRequest, request: Request, response: Response) -> JsonObject:
        origin_allowed(request)
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
        set_session_cookie(response, session_value.identifier)
        return session_response(session_value)

    @app.post("/api/v1/auth/activate")
    async def activate(
        body: ActivationRequest,
        request: Request,
        response: Response,
    ) -> JsonObject:
        origin_allowed(request)
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
        set_session_cookie(response, session_value.identifier)
        return session_response(session_value)

    @app.get("/api/v1/auth/session")
    async def browser_session(request: Request) -> JsonObject:
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
            record = registered_auth.sessions.resolve(identifier, registered_auth.registry)
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
            registered_auth.audit.append("LOGOUT", outcome="SUCCESS", user_id=authenticated.user_id)
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
        registered_auth.audit.append("LOGOUT_ALL", outcome="SUCCESS", user_id=authenticated.user_id)
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
        set_session_cookie(response, session_value.identifier)
        return session_response(session_value)
