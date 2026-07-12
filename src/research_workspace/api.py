from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .core import ensure_layout, load_settings
from .documents import ALLOWED, ingest
from .library import ingest_library
from .online import search_ieee, search_scholarly
from .projects import init_project, list_projects, project_summary
from .retrieval import evidence_packet, search
from .ui import offline_dashboard


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    limit: int = 6
    document_class: str | None = None
    collection: str | None = None
    author: str | None = None
    year: int | None = None
    doi: str | None = None
    availability: str | None = None
    source_kind: str | None = None


class ProjectInitRequest(BaseModel):
    name: str
    update: bool = False


def create_app(root: Path | None = None, database: Path | None = None) -> FastAPI:
    if root is None or database is None:
        settings = load_settings()
        ensure_layout(settings)
        root, database = settings.root, settings.database
    app = FastAPI(title="Local Research Workspace", docs_url="/docs")

    @app.get("/")
    def home() -> dict[str, object]:
        return {
            "name": "Local Research Workspace",
            "pages": [
                "status",
                "collections",
                "search",
                "extraction",
                "analysis",
                "drafting",
                "benchmarks",
            ],
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return offline_dashboard()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "bind": "127.0.0.1"}

    @app.get("/projects")
    def projects() -> dict[str, object]:
        return {"projects": list_projects()}

    @app.post("/projects/init")
    def project_init(request: ProjectInitRequest) -> dict[str, object]:
        paths = init_project(request.name, update=request.update)
        return {"status": "CREATED", "project": str(paths.root)}

    @app.get("/projects/{name}")
    def project_show(name: str) -> dict[str, object]:
        return project_summary(name)

    @app.post("/projects/{name}/library/ingest")
    def project_library_ingest(name: str, collection: str = "MyWorks") -> dict[str, object]:
        return ingest_library(name, collection=collection)

    @app.get("/search/scholarly")
    def scholarly_search(
        query: str,
        providers: str = "crossref,openalex,arxiv",
        limit: int = 10,
        offline: bool = False,
    ) -> dict[str, object]:
        return search_scholarly(
            query,
            providers=[item.strip() for item in providers.split(",") if item.strip()],
            limit=limit,
            offline=offline,
        )

    @app.get("/search/ieee")
    def ieee_search(query: str, limit: int = 10) -> dict[str, object]:
        response = search_ieee(query, limit=limit)
        return {
            "provider": response.provider,
            "status": response.status,
            "query": response.query,
            "error": response.error,
            "results": [item.__dict__ for item in response.results],
        }

    @app.post("/search")
    def do_search(request: SearchRequest) -> dict[str, Any]:
        if request.limit < 1 or request.limit > 20:
            raise HTTPException(400, "limit must be 1..20")
        evidence = search(
            database,
            request.query,
            request.mode,
            request.limit,
            request.document_class,
            None,
            request.collection,
            request.author,
            request.year,
            request.doi,
            request.availability,
            request.source_kind,
        )
        return evidence_packet(request.query, evidence)

    @app.post("/ingest")
    async def do_ingest(
        file: UploadFile = File(...), document_class: str = "project_document"
    ) -> dict[str, Any]:
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(415, "unsupported file extension")
        if file.content_type in {
            "application/x-msdownload",
            "application/x-sh",
            "text/x-shellscript",
        }:
            raise HTTPException(415, "unsafe MIME type")
        target = root / "data" / "cache" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(413, "file too large")
        target.write_bytes(content)
        try:
            return ingest(target, root, database, document_class)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
