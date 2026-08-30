"""Credential and CSRF primitives for the Operator HTTP transport."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Literal, Mapping

from fastapi import HTTPException

from ..user_capabilities import CapabilityTier


@dataclass(frozen=True)
class AuthCredential:
    """Credential binding supplied when constructing the local Operator app."""

    role: str
    user_id: str
    capability_tier: CapabilityTier


@dataclass(frozen=True)
class AuthPrincipal:
    """Authenticated local user context passed to route dependencies."""

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
