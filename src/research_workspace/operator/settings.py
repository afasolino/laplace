"""Validated deployment settings for the Operator HTTP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


@dataclass(frozen=True)
class OperatorApiSettings:
    bind_host: str = "127.0.0.1"
    port: int = 8765
    deployment_mode: Literal["local", "ssh-tunnel", "reverse-proxy"] = "local"
    allowed_origins: tuple[str, ...] = ("http://127.0.0.1:8765", "http://localhost:8765")
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    trusted_proxies: tuple[str, ...] = ("127.0.0.1", "::1")
    external_url: str | None = None
    allow_insecure_lan_http: bool = False
    bearer_api_enabled: bool = False
    pwa_enabled: bool = True
    fixture_mode: bool = False
    codev_enabled: bool = True
    maximum_request_bytes: int = 70_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65_535:
            raise ValueError("invalid Operator Plane port")
        if not self.allowed_origins:
            raise ValueError("at least one allowed origin is required")
        if any(not origin.startswith(("http://", "https://")) or "*" in origin for origin in self.allowed_origins):
            raise ValueError("Operator Plane origins must be explicit HTTP origins")
        if not self.allowed_hosts or any("*" in host or "/" in host for host in self.allowed_hosts):
            raise ValueError("Operator Plane hosts must be explicit")
        loopback = self.bind_host in {"127.0.0.1", "localhost", "::1"}
        if not loopback and not self.allow_insecure_lan_http:
            raise ValueError("non-loopback direct binding requires --allow-insecure-lan-http")
        if self.deployment_mode == "reverse-proxy":
            if self.external_url is None:
                raise ValueError("reverse-proxy mode requires an external URL")
            external = urlsplit(self.external_url)
            if external.scheme != "https" or not external.hostname:
                raise ValueError("reverse-proxy external URL must use HTTPS")
            if self.external_url.rstrip("/") not in {origin.rstrip("/") for origin in self.allowed_origins}:
                raise ValueError("external URL must be an explicit allowed origin")
        if self.maximum_request_bytes < 1_024:
            raise ValueError("maximum request body is too small")

    @property
    def secure_cookie(self) -> bool:
        return self.deployment_mode == "reverse-proxy"

    @property
    def development_http(self) -> bool:
        return self.deployment_mode in {"local", "ssh-tunnel"} and self.bind_host in {"127.0.0.1", "localhost", "::1"}
