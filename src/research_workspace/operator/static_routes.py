"""Static Operator GUI and PWA transport routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response


def register_static_routes(app: FastAPI, *, web_root: Path, pwa_enabled: bool) -> None:
    """Register the compatibility-preserving static Operator routes."""

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
        if not pwa_enabled:
            raise HTTPException(status_code=404)
        return Response(
            (web_root / "manifest.webmanifest").read_bytes(),
            media_type="application/manifest+json",
        )

    @app.get("/sw.js")
    async def service_worker() -> Response:
        if not pwa_enabled:
            raise HTTPException(status_code=404)
        return Response(
            (web_root / "sw.js").read_bytes(),
            media_type="text/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )
