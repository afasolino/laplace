from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .chat import ChatEngine, ChatProject, ConversationStore
from .core import ensure_layout, load_settings
from .documents import ALLOWED, ingest
from .engineering import (
    EngineeringError,
    LocalToolRunner,
    ReferenceLibrary,
    resolve_shared_reference_root,
)
from .library import ingest_library
from .online import search_ieee, search_scholarly
from .projects import init_project, list_projects, project_summary
from .retrieval import evidence_packet, search
from .ui import offline_dashboard, project_page


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=12_000)
    mode: str = "hybrid"
    limit: int = Field(default=6, ge=1, le=20)
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


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="New chat", max_length=160)
    collections: list[str] = Field(default_factory=lambda: ["MyWorks"], max_length=8)
    mode: Literal["ASK", "SEARCH", "WRITE", "RESEARCH", "GENERAL"] = "ASK"


class ConversationPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=160)
    archived: bool | None = None


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=12_000)
    collections: list[str] | None = Field(default=None, max_length=8)
    mode: Literal["ASK", "SEARCH", "WRITE", "RESEARCH", "GENERAL"] = "ASK"


class AttachmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_once: bool = True
    ingest: bool = False


class SettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=50_000)


class EngineeringReferenceRequest(BaseModel):
    """Strict, project-scoped reference operations exposed over localhost."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["initialize", "status", "sync", "select", "ingest", "verify"] = "status"
    topics: list[str] = Field(default_factory=list, max_length=32)
    reference_id: str | None = Field(default=None, max_length=120)


def _project_info(root: Path, config: dict[str, Any], database: Path) -> ChatProject:
    project = config.get("project", {}) if isinstance(config, dict) else {}
    models = config.get("models", {}) if isinstance(config, dict) else {}
    return ChatProject(
        root=root,
        database=database,
        project_id=str(project.get("project_id") or f"legacy:{root.resolve()}"),
        name=str(project.get("name") or root.name),
        model=str(models.get("main_text") or os.getenv("RW_MODEL") or "qwen3:4b"),
        endpoint=str(
            models.get("endpoint") or os.getenv("RW_MODEL_ENDPOINT") or "http://127.0.0.1:11434"
        ),
        context_tokens=min(8192, int(models.get("context_tokens", 8192))),
    )


def _load_project_config(root: Path) -> dict[str, Any]:
    path = root / ".laplace" / "project.yaml"
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _counts(database: Path) -> dict[str, int]:
    if not database.exists():
        return {"documents": 0, "chunks": 0}
    try:
        with sqlite3.connect(database) as conn:
            docs = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return {"documents": docs, "chunks": chunks}
    except sqlite3.Error:
        return {"documents": 0, "chunks": 0}


def create_app(root: Path | None = None, database: Path | None = None) -> FastAPI:
    if root is None or database is None:
        settings = load_settings()
        ensure_layout(settings)
        root, database = settings.root, settings.database
    assert root is not None and database is not None
    root = root.resolve()
    config = _load_project_config(root)
    project = _project_info(
        root, config, database if config == {} else root / "Data" / "Metadata" / "laplace.db"
    )
    index_database = database if config == {} else root / "Data" / "Metadata" / "workspace.db"
    store = ConversationStore(project)
    engine = ChatEngine(project, store)
    app = FastAPI(title="Laplace Local Research Workspace", docs_url="/docs")

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return project_page("Chat", project.name, "/chat")

    @app.get("/chat", response_class=HTMLResponse)
    def chat() -> HTMLResponse:
        return project_page("Chat", project.name, "/chat")

    @app.get("/library", response_class=HTMLResponse)
    def library_page() -> HTMLResponse:
        return project_page("Library", project.name, "/library")

    @app.get("/research", response_class=HTMLResponse)
    def research_page() -> HTMLResponse:
        return project_page("Research", project.name, "/research")

    @app.get("/downloads", response_class=HTMLResponse)
    def downloads_page() -> HTMLResponse:
        return project_page("Downloads", project.name, "/downloads")

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page() -> HTMLResponse:
        return project_page("Settings", project.name, "/settings")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return offline_dashboard(project.name)

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

    @app.get("/api/project/current")
    def current_project() -> dict[str, Any]:
        return {
            "name": project.name,
            "project_id": project.project_id,
            "root": str(project.root),
            "model": project.model,
            "endpoint": project.endpoint,
            "context_tokens": project.context_tokens,
            "local_only": True,
        }

    @app.get("/api/project/collections")
    def project_collections() -> dict[str, Any]:
        from .projects import COLLECTIONS

        counts: dict[str, int] = {item: 0 for item in COLLECTIONS}
        if index_database.exists():
            try:
                with sqlite3.connect(index_database) as conn:
                    rows = conn.execute("SELECT metadata FROM documents").fetchall()
                for row in rows:
                    try:
                        metadata = json.loads(row[0] or "{}")
                    except json.JSONDecodeError:
                        metadata = {}
                    collection = metadata.get("collection")
                    if collection in counts:
                        counts[collection] += 1
            except sqlite3.Error:
                pass
        return {
            "collections": [
                {"name": item, "selected": item in ["MyWorks"], "documents": counts[item]}
                for item in COLLECTIONS
            ]
        }

    @app.get("/api/project/status")
    def project_status() -> dict[str, Any]:
        counts = _counts(index_database)
        downloads = (
            len(list((root / "Data" / "Downloads").rglob("*.pdf")))
            if (root / "Data" / "Downloads").is_dir()
            else 0
        )
        return {
            "project": project.name,
            "project_id": project.project_id,
            **counts,
            "downloads": downloads,
            "model": project.model,
            "embedding_model": str(
                config.get("models", {}).get("embedding", "qwen3-embedding:0.6b")
            ),
            "bind": "127.0.0.1",
            "ollama": project.endpoint,
        }

    @app.post("/api/engineering/references/{domain}")
    def engineering_references(
        domain: Literal["python", "systemverilog"], request: EngineeringReferenceRequest
    ) -> dict[str, object]:
        shared = resolve_shared_reference_root()
        library = (
            ReferenceLibrary(shared, domain, shared=True)
            if shared is not None
            else ReferenceLibrary(root, domain)
        )
        try:
            if request.action == "initialize":
                catalog = (
                    Path(__file__).resolve().parents[2]
                    / "codex_a6000"
                    / "reference_sources"
                    / f"{domain}_sources.yaml"
                )
                return library.initialize(catalog)
            if request.action == "status":
                return library.status()
            if request.action == "sync":
                return library.synchronize()
            if request.action == "select":
                return library.select(request.topics)
            if request.action == "ingest":
                return library.ingest(None if library.shared else index_database)
            return library.verify(request.reference_id)
        except EngineeringError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/engineering/python-quality")
    def engineering_python_quality() -> dict[str, object]:
        try:
            return LocalToolRunner(
                Path(__file__).resolve().parents[2], root / "Outputs" / "AgentTeam" / "tool_logs"
            ).run_python_quality_gates()
        except EngineeringError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/project/settings")
    def project_settings() -> dict[str, Any]:
        settings: dict[str, str] = {}
        for name in ("instructions.md", "writing_style.yaml", "retrieval.yaml", "providers.yaml"):
            path = root / "Config" / name
            settings[name] = path.read_text(encoding="utf-8") if path.is_file() else ""
        return {"settings": settings, "secrets_included": False}

    @app.patch("/api/project/settings/{name}")
    def update_project_settings(name: str, request: SettingsUpdateRequest) -> dict[str, Any]:
        allowed = {"instructions.md", "writing_style.yaml", "retrieval.yaml", "providers.yaml"}
        if name not in allowed:
            raise HTTPException(400, "settings file is not editable through this endpoint")
        if name.endswith(".yaml"):
            try:
                value = yaml.safe_load(request.content)
            except yaml.YAMLError as exc:
                raise HTTPException(400, f"invalid YAML: {exc}") from exc
            if value is not None and not isinstance(value, dict):
                raise HTTPException(400, "settings YAML must contain an object")
        target = root / "Config" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_name(
                f"{target.name}.{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.bak"
            )
            backup.write_bytes(target.read_bytes())
        target.write_text(request.content, encoding="utf-8")
        return {
            "status": "UPDATED",
            "name": name,
            "backup": str(backup) if "backup" in locals() else None,
            "secrets_included": False,
        }

    @app.get("/api/chat/conversations")
    def conversations(
        include_archived: bool = False, q: str | None = Query(default=None, max_length=200)
    ) -> dict[str, Any]:
        values = store.list(include_archived)
        if q:
            values = [item for item in values if q.lower() in item.title.lower()]
        return {"conversations": [item.model_dump() for item in values]}

    @app.post("/api/chat/conversations")
    def create_conversation(request: ConversationCreateRequest) -> dict[str, Any]:
        return store.create(request.title, request.collections, request.mode).model_dump()

    @app.get("/api/chat/conversations/{conversation_id}")
    def conversation_detail(conversation_id: str) -> dict[str, Any]:
        try:
            return store.detail(conversation_id).model_dump()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.patch("/api/chat/conversations/{conversation_id}")
    def patch_conversation(
        conversation_id: str, request: ConversationPatchRequest
    ) -> dict[str, Any]:
        try:
            result = store.summary(conversation_id)
            if request.title is not None:
                result = store.rename(conversation_id, request.title)
            if request.archived is not None:
                result = store.archive(conversation_id, request.archived)
            return result.model_dump()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/chat/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, confirm: bool = False) -> dict[str, str]:
        if not confirm:
            raise HTTPException(
                400, "confirm=true is required; documents and indexes are not deleted"
            )
        try:
            store.delete(conversation_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "DELETED", "conversation_id": conversation_id}

    @app.post("/api/chat/conversations/{conversation_id}/messages")
    def post_message(
        conversation_id: str, request: ChatMessageRequest, stream: bool = False
    ) -> Any:
        try:
            store.summary(conversation_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if stream:

            def next_event(iterator: Iterator[dict[str, Any]]) -> dict[str, Any] | None:
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            async def events() -> AsyncIterator[str]:
                iterator = iter(
                    engine.stream(
                        conversation_id,
                        request.content,
                        collections=request.collections,
                        mode=request.mode,
                    )
                )
                try:
                    while True:
                        event = await asyncio.to_thread(next_event, iterator)
                        if event is None:
                            return
                        yield _sse(event)
                except Exception as exc:
                    yield _sse({"type": "message_failed", "error": str(exc)})

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return engine.answer(
                conversation_id, request.content, collections=request.collections, mode=request.mode
            ).model_dump()
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/chat/conversations/{conversation_id}/stop")
    def stop_message(conversation_id: str) -> dict[str, Any]:
        return {
            "status": "STOP_REQUESTED" if engine.stop(conversation_id) else "NOT_GENERATING",
            "conversation_id": conversation_id,
        }

    @app.post("/api/chat/conversations/{conversation_id}/regenerate")
    def regenerate(conversation_id: str) -> dict[str, Any]:
        try:
            detail = store.detail(conversation_id)
            user_messages = [item for item in detail.messages if item.get("role") == "user"]
            if not user_messages:
                raise HTTPException(400, "no user message to regenerate")
            return engine.answer(
                conversation_id,
                str(user_messages[-1]["content"]),
                collections=detail.collections,
                mode=detail.mode,
            ).model_dump()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/chat/messages/{message_id}/evidence")
    def message_evidence(message_id: str) -> dict[str, Any]:
        try:
            return store.evidence(message_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/chat/messages/{message_id}/audit")
    def message_audit(message_id: str) -> dict[str, Any]:
        try:
            return store.audit(message_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/chat/messages/{message_id}/source/{citation_id}")
    def open_source(message_id: str, citation_id: int) -> FileResponse:
        try:
            audit = store.audit(message_id)
            evidence = audit.get("citations", audit.get("evidence", []))
            if not isinstance(evidence, list) or citation_id < 1 or citation_id > len(evidence):
                raise HTTPException(404, "citation not found")
            item = evidence[citation_id - 1]
            source_value = item.get("absolute_source_path") or item.get("source_path")
            if not source_value:
                raise HTTPException(404, "source path unavailable")
            source = Path(str(source_value)).resolve()
            library_root = (
                Path(str(config.get("library", {}).get("root", ""))).expanduser().resolve()
                if config.get("library", {}).get("root")
                else None
            )
            allowed_roots = [root.resolve()]
            if library_root is not None:
                allowed_roots.append(library_root)
            if (
                not any(source == allowed or allowed in source.parents for allowed in allowed_roots)
                or not source.is_file()
            ):
                raise HTTPException(
                    403, "source is outside the active project and configured Library"
                )
            return FileResponse(
                source,
                media_type="application/pdf" if source.suffix.lower() == ".pdf" else None,
                filename=source.name,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/chat/conversations/{conversation_id}/export")
    def export_conversation(conversation_id: str) -> FileResponse:
        target = root / "Outputs" / "Conversations" / f"{conversation_id}.json"
        try:
            store.export(conversation_id, target)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(
            target, media_type="application/json", filename=f"{conversation_id}.json"
        )

    @app.post("/api/chat/conversations/{conversation_id}/attachments")
    async def attachment(
        conversation_id: str, file: UploadFile = File(...), request: AttachmentRequest | None = None
    ) -> dict[str, Any]:
        try:
            store.summary(conversation_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        name = Path(file.filename or "attachment").name
        suffix = Path(name).suffix.lower()
        allowed = {".pdf", ".txt", ".md", ".csv", ".json"}
        if suffix not in allowed or not name or name in {".", ".."}:
            raise HTTPException(415, "unsupported attachment extension")
        content = await file.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(413, "attachment too large")
        if suffix == ".pdf" and not content.startswith(b"%PDF-"):
            raise HTTPException(415, "PDF signature is invalid")
        target = root / "Data" / "Cache" / "ChatAttachments" / f"{uuid.uuid4().hex}_{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {
            "status": "STAGED",
            "filename": name,
            "path": str(target),
            "use_once": True,
            "ingest": bool(request and request.ingest),
        }

    @app.get("/api/search/scholarly")
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

    @app.get("/search/scholarly")
    def legacy_scholarly_search(
        query: str,
        providers: str = "crossref,openalex,arxiv",
        limit: int = 10,
        offline: bool = False,
    ) -> dict[str, object]:
        return scholarly_search(query, providers, limit, offline)

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
        evidence = search(
            index_database,
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
        content = await file.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(413, "file too large")
        target = root / "data" / "cache" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        try:
            return ingest(target, root, index_database, document_class)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
