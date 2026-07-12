"""Project-local conversations, grounded response normalization, and streaming."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, cast

from pydantic import BaseModel, ConfigDict, Field

from .real_benchmark import _local_endpoint
from .retrieval import Evidence, evidence_packet, search, validate_citations


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ChatCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: int = Field(ge=1)
    filename: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_id: str
    quoted_evidence: str
    availability: str
    source_class: str
    score: float
    source_path: str | None = None
    doi: str | None = None


class ChatRetrieval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    collections: list[str] = Field(default_factory=list)
    candidate_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    mode: str = "hybrid"


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    conversation_id: str
    role: str = "assistant"
    content: str
    status: str
    citations: list[ChatCitation] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    created_at: str
    model: str
    retrieval: ChatRetrieval
    run_id: str | None = None


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    project_id: str
    collections: list[str]
    mode: str
    archived: bool = False


class ConversationDetail(ConversationSummary):
    messages: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(frozen=True)
class ChatProject:
    root: Path
    database: Path
    project_id: str
    name: str
    model: str = "qwen3:4b"
    endpoint: str = "http://127.0.0.1:11434"
    context_tokens: int = 8192


def _db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, project_id TEXT NOT NULL, collections TEXT NOT NULL,
            mode TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', archived INTEGER NOT NULL DEFAULT 0,
            deleted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, status TEXT NOT NULL, citations TEXT NOT NULL,
            unsupported_claims TEXT NOT NULL, fallback_used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, model TEXT NOT NULL, retrieval TEXT NOT NULL,
            run_id TEXT, interrupted INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        );
        CREATE TABLE IF NOT EXISTS chat_audits (
            message_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id, created_at);
        """
    )
    conn.commit()
    return conn


class ConversationStore:
    """SQLite-backed conversation store scoped to one active project."""

    def __init__(self, project: ChatProject) -> None:
        self.project = project
        self.path = project.database
        _db(self.path).close()

    def _check(self, conn: sqlite3.Connection, conversation_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id=? AND project_id=? AND deleted=0",
            (conversation_id, self.project.project_id),
        ).fetchone()
        if row is None:
            raise KeyError("conversation not found in the active project")
        return cast(sqlite3.Row, row)

    def create(self, title: str = "New chat", collections: list[str] | None = None, mode: str = "ASK") -> ConversationSummary:
        conversation_id = str(uuid.uuid4())
        stamp = _now()
        chosen = collections or ["MyWorks"]
        with _db(self.path) as conn:
            conn.execute(
                "INSERT INTO conversations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (conversation_id, title[:160] or "New chat", stamp, stamp, self.project.project_id, json.dumps(chosen), mode, "", 0, 0),
            )
            conn.commit()
        return self.summary(conversation_id)

    def summary(self, conversation_id: str) -> ConversationSummary:
        with _db(self.path) as conn:
            row = self._check(conn, conversation_id)
        return ConversationSummary(
            conversation_id=row["id"], title=row["title"], created_at=row["created_at"],
            updated_at=row["updated_at"], project_id=row["project_id"],
            collections=json.loads(row["collections"]), mode=row["mode"], archived=bool(row["archived"]),
        )

    def list(self, include_archived: bool = False) -> list[ConversationSummary]:
        where = "AND archived=0" if not include_archived else ""
        with _db(self.path) as conn:
            rows = conn.execute(
                f"SELECT * FROM conversations WHERE project_id=? AND deleted=0 {where} ORDER BY updated_at DESC",
                (self.project.project_id,),
            ).fetchall()
        return [
            ConversationSummary(
                conversation_id=row["id"], title=row["title"], created_at=row["created_at"],
                updated_at=row["updated_at"], project_id=row["project_id"],
                collections=json.loads(row["collections"]), mode=row["mode"], archived=bool(row["archived"]),
            )
            for row in rows
        ]

    def detail(self, conversation_id: str) -> ConversationDetail:
        summary = self.summary(conversation_id)
        with _db(self.path) as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE conversation_id=? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "message_id": row["id"], "role": row["role"], "content": row["content"],
                    "status": row["status"], "citations": json.loads(row["citations"]),
                    "unsupported_claims": json.loads(row["unsupported_claims"]),
                    "fallback_used": bool(row["fallback_used"]), "created_at": row["created_at"],
                    "model": row["model"], "retrieval": json.loads(row["retrieval"]),
                    "run_id": row["run_id"], "interrupted": bool(row["interrupted"]),
                }
            )
        return ConversationDetail(**summary.model_dump(), messages=messages)

    def rename(self, conversation_id: str, title: str) -> ConversationSummary:
        with _db(self.path) as conn:
            self._check(conn, conversation_id)
            conn.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title[:160] or "New chat", _now(), conversation_id))
            conn.commit()
        return self.summary(conversation_id)

    def archive(self, conversation_id: str, archived: bool = True) -> ConversationSummary:
        with _db(self.path) as conn:
            self._check(conn, conversation_id)
            conn.execute("UPDATE conversations SET archived=?,updated_at=? WHERE id=?", (int(archived), _now(), conversation_id))
            conn.commit()
        return self.summary(conversation_id)

    def delete(self, conversation_id: str) -> None:
        with _db(self.path) as conn:
            self._check(conn, conversation_id)
            conn.execute("UPDATE conversations SET deleted=1,updated_at=? WHERE id=?", (_now(), conversation_id))
            conn.commit()

    def append_user(self, conversation_id: str, content: str) -> str:
        message_id = str(uuid.uuid4())
        stamp = _now()
        with _db(self.path) as conn:
            self._check(conn, conversation_id)
            conn.execute(
                "INSERT INTO chat_messages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (message_id, conversation_id, "user", content, "USER", "[]", "[]", 0, stamp, self.project.model, "{}", None, 0),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))
            conn.commit()
        return message_id

    def append_assistant(self, response: ChatResponse, audit: dict[str, Any]) -> None:
        audit_payload = {**audit, "citations": [item.model_dump() for item in response.citations]}
        with _db(self.path) as conn:
            self._check(conn, response.conversation_id)
            conn.execute(
                "INSERT INTO chat_messages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (response.message_id, response.conversation_id, "assistant", response.content, response.status,
                 json.dumps([item.model_dump() for item in response.citations], ensure_ascii=False),
                 json.dumps(response.unsupported_claims, ensure_ascii=False), int(response.fallback_used),
                 response.created_at, response.model, response.retrieval.model_dump_json(), response.run_id, int(response.status == "INTERRUPTED")),
            )
            conn.execute("INSERT OR REPLACE INTO chat_audits VALUES(?,?,?)", (response.message_id, json.dumps(audit_payload, ensure_ascii=False), _now()))
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), response.conversation_id))
            conn.commit()

    def audit(self, message_id: str) -> dict[str, Any]:
        with _db(self.path) as conn:
            row = conn.execute(
                "SELECT a.payload FROM chat_audits a JOIN chat_messages m ON m.id=a.message_id JOIN conversations c ON c.id=m.conversation_id WHERE a.message_id=? AND c.project_id=?",
                (message_id, self.project.project_id),
            ).fetchone()
        if row is None:
            raise KeyError("message audit not found")
        value = json.loads(row[0])
        return cast(dict[str, Any], value) if isinstance(value, dict) else {}

    def evidence(self, message_id: str) -> dict[str, Any]:
        audit = self.audit(message_id)
        return {"message_id": message_id, "evidence": audit.get("evidence", []), "retrieval": audit.get("retrieval", {})}

    def export(self, conversation_id: str, target: Path) -> Path:
        detail = self.detail(conversation_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(detail.model_dump_json(indent=2), encoding="utf-8")
        return target


def _extract_json(value: str) -> Any:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                return _extract_json(parsed)
            return parsed
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _citation(item: Evidence, citation_id: int) -> ChatCitation:
    return ChatCitation(
        citation_id=citation_id, filename=item.filename, title=item.title, page=item.page,
        section=item.section, chunk_id=item.chunk_id, quoted_evidence=item.text[:700].strip(),
        availability=item.availability, source_class=item.document_class, score=item.score,
        source_path=item.source_path, doi=item.doi,
    )


def _map_citations(raw: Any, evidence: list[Evidence]) -> list[ChatCitation]:
    if not isinstance(raw, list):
        return []
    mapped: list[ChatCitation] = []
    for position, value in enumerate(raw, 1):
        number: int | None = None
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and value.strip().isdigit():
            number = int(value.strip())
        elif isinstance(value, dict):
            candidate = value.get("citation_id", value.get("id", value.get("index")))
            if isinstance(candidate, int) or (isinstance(candidate, str) and candidate.isdigit()):
                number = int(candidate)
            if number is None:
                for index, item in enumerate(evidence, 1):
                    if value.get("filename") == item.filename and value.get("page") == item.page and value.get("chunk_id") == item.chunk_id:
                        number = index
                        break
        if number is not None and 1 <= number <= len(evidence):
            item = evidence[number - 1]
            if all(existing.chunk_id != item.chunk_id for existing in mapped):
                mapped.append(_citation(item, len(mapped) + 1))
        elif position <= len(evidence) and isinstance(value, dict) and value.get("chunk_id") == evidence[position - 1].chunk_id:
            mapped.append(_citation(evidence[position - 1], len(mapped) + 1))
    return mapped


def normalize_response(
    output: str,
    *,
    conversation_id: str,
    query: str,
    evidence: list[Evidence],
    model: str,
    collections: list[str],
    candidate_count: int,
    mode: str = "hybrid",
    run_id: str | None = None,
) -> tuple[ChatResponse, dict[str, Any]]:
    parsed = _extract_json(output)
    attempted = parsed if parsed is not None else {"content": output, "citations": []}
    if isinstance(attempted, dict) and isinstance(attempted.get("answer"), str):
        nested = _extract_json(attempted["answer"])
        if isinstance(nested, dict):
            attempted = {**attempted, **nested}
    if not isinstance(attempted, dict):
        attempted = {"content": str(attempted), "citations": []}
    content = str(attempted.get("content", attempted.get("answer", attempted.get("draft", output))))
    content = content.strip()
    if _extract_json(content) is not None and isinstance(_extract_json(content), dict):
        nested = _extract_json(content)
        if isinstance(nested, dict):
            content = str(nested.get("content", nested.get("answer", content))).strip()
    citations = _map_citations(attempted.get("citations", attempted.get("sources", [])), evidence)
    model_valid = bool(citations) and validate_citations(
        evidence_packet(query, evidence),
        [{"filename": item.filename, "page": item.page, "chunk_id": item.chunk_id} for item in citations],
    )
    unsupported = attempted.get("unsupported_claims", attempted.get("unsupported", []))
    unsupported_claims = [str(item) for item in unsupported] if isinstance(unsupported, list) else []
    fallback = False
    if evidence and (not model_valid or not citations):
        fallback = True
        citations = [_citation(item, index) for index, item in enumerate(evidence[: min(6, len(evidence))], 1)]
        lines = ["The retrieved local evidence supports the following:", ""]
        for item in citations[:2]:
            lines.append(f"- {item.quoted_evidence.rstrip()} [{item.citation_id}]")
        content = "\n".join(lines)
        status = "GROUNDED_EXTRACTIVE_FALLBACK"
    elif evidence and model_valid:
        status = "GROUNDED"
    else:
        status = "NOT_GROUNDED"
        content = content or "No indexed evidence was retrieved for this request."
        if evidence and not citations:
            unsupported_claims.append("Model citations did not resolve to retrieved evidence.")
    response = ChatResponse(
        message_id=str(uuid.uuid4()), conversation_id=conversation_id, content=content,
        status=status, citations=citations, unsupported_claims=unsupported_claims,
        fallback_used=fallback, created_at=_now(), model=model,
        retrieval=ChatRetrieval(query=query, collections=collections, candidate_count=candidate_count, evidence_count=len(evidence), mode=mode),
        run_id=run_id,
    )
    audit = {"raw_model_output": output, "parsed_model_output": parsed, "evidence": [item.__dict__ for item in evidence], "retrieval": response.retrieval.model_dump(), "model_citation_valid": model_valid, "fallback_used": fallback}
    return response, audit


def _prompt(query: str, evidence: list[Evidence], history: list[dict[str, Any]], instructions: str = "") -> str:
    compact = [
        {"citation_id": index, "filename": item.filename, "page": item.page, "section": item.section, "chunk_id": item.chunk_id, "text": item.text}
        for index, item in enumerate(evidence, 1)
    ]
    recent = [{"role": item.get("role"), "content": str(item.get("content", ""))[:1800]} for item in history[-6:]]
    return (
        "Return ONLY JSON with keys content, citations, unsupported_claims. content must be readable Markdown, not JSON. "
        "Citations must be integer citation_id values copied from the supplied evidence. Never invent provenance. "
        f"Project instructions: {instructions[:3000]}\nConversation context: {json.dumps(recent, ensure_ascii=False)}\n"
        f"User request: {query}\nEvidence: {json.dumps(compact, ensure_ascii=False)}"
    )


def _project_instructions(project: ChatProject) -> tuple[str, list[str]]:
    parts: list[str] = []
    used: list[str] = []
    for name in ("instructions.md", "writing_style.yaml", "retrieval.yaml"):
        path = project.root / "Config" / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace")[:4000])
            used.append(str(path))
    return "\n\n".join(parts), used


def generate_stream(endpoint: str, model: str, prompt: str, *, context_tokens: int, stop: threading.Event, json_mode: bool = True) -> Iterator[dict[str, Any]]:
    _local_endpoint(endpoint)
    payload = json.dumps({"model": model, "prompt": prompt, "stream": True, "think": False, "keep_alive": -1, **({"format": "json"} if json_mode else {}), "options": {"num_ctx": context_tokens, "num_predict": 360, "temperature": 0}}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                if stop.is_set():
                    return
                if not raw_line.strip():
                    continue
                item = json.loads(raw_line)
                yield item
                if item.get("done"):
                    return
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local Ollama generation failed: {exc}") from exc


class ChatEngine:
    def __init__(self, project: ChatProject, store: ConversationStore) -> None:
        self.project = project
        self.store = store
        self._active: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def stop(self, conversation_id: str) -> bool:
        with self._lock:
            event = self._active.get(conversation_id)
            if event is None:
                return False
            event.set()
            return True

    def answer(self, conversation_id: str, query: str, *, collections: list[str] | None = None, mode: str = "ASK", run_id: str | None = None) -> ChatResponse:
        with self._lock:
            if self._active:
                raise RuntimeError("another local generation is active; wait or press Stop")
        self.store.append_user(conversation_id, query)
        detail = self.store.detail(conversation_id)
        selected = collections or detail.collections
        document_class = "user_work" if selected == ["MyWorks"] else None
        evidence = search(self.project.database if self.project.database.name == "workspace.db" else self.project.database.parent / "workspace.db", query, limit=6, document_class=document_class)
        if not evidence:
            response, audit = normalize_response("", conversation_id=conversation_id, query=query, evidence=[], model=self.project.model, collections=selected, candidate_count=0, mode="hybrid", run_id=run_id)
            self.store.append_assistant(response, audit)
            return response
        instructions, instruction_files = _project_instructions(self.project)
        prompt = _prompt(query, evidence, detail.messages, instructions)
        stop = threading.Event()
        with self._lock:
            if self._active:
                raise RuntimeError("another local generation is active; wait or press Stop")
            self._active[conversation_id] = stop
        try:
            output: list[str] = []
            for item in generate_stream(self.project.endpoint, self.project.model, prompt, context_tokens=self.project.context_tokens, stop=stop):
                output.append(str(item.get("response", "")))
            text = "".join(output)
            response, audit = normalize_response(text, conversation_id=conversation_id, query=query, evidence=evidence, model=self.project.model, collections=selected, candidate_count=len(evidence), mode="hybrid", run_id=run_id)
            audit["instruction_files"] = instruction_files
            self.store.append_assistant(response, audit)
            return response
        finally:
            with self._lock:
                self._active.pop(conversation_id, None)

    def stream(self, conversation_id: str, query: str, *, collections: list[str] | None = None, mode: str = "ASK", run_id: str | None = None) -> Iterator[dict[str, Any]]:
        with self._lock:
            if self._active:
                yield {"type": "message_failed", "error": "another local generation is active; wait or press Stop"}
                return
        self.store.append_user(conversation_id, query)
        detail = self.store.detail(conversation_id)
        selected = collections or detail.collections
        evidence = search(self.project.database if self.project.database.name == "workspace.db" else self.project.database.parent / "workspace.db", query, limit=6, document_class="user_work" if selected == ["MyWorks"] else None)
        yield {"type": "message_started", "conversation_id": conversation_id}
        yield {"type": "retrieval_started", "query": query}
        yield {"type": "retrieval_result", "count": len(evidence), "collections": selected}
        if not evidence:
            response, audit = normalize_response("", conversation_id=conversation_id, query=query, evidence=[], model=self.project.model, collections=selected, candidate_count=0, mode="hybrid", run_id=run_id)
            self.store.append_assistant(response, audit)
            yield {"type": "validation_started"}
            yield {"type": "message_completed", "message": response.model_dump()}
            return
        stop = threading.Event()
        with self._lock:
            self._active[conversation_id] = stop
        output: list[str] = []
        try:
            yield {"type": "status", "message": "Generating..."}
            instructions, instruction_files = _project_instructions(self.project)
            prompt = _prompt(query, evidence, detail.messages, instructions)
            for item in generate_stream(self.project.endpoint, self.project.model, prompt, context_tokens=self.project.context_tokens, stop=stop):
                piece = str(item.get("response", ""))
                if piece:
                    output.append(piece)
                    yield {"type": "token", "text": piece}
            if stop.is_set():
                interrupted = ChatResponse(
                    message_id=str(uuid.uuid4()), conversation_id=conversation_id, content="Generation stopped by the user.", status="INTERRUPTED",
                    citations=[], unsupported_claims=[], fallback_used=False, created_at=_now(), model=self.project.model,
                    retrieval=ChatRetrieval(query=query, collections=selected, candidate_count=len(evidence), evidence_count=len(evidence)), run_id=run_id,
                )
                self.store.append_assistant(interrupted, {"raw_model_output": "".join(output), "interrupted": True, "evidence": [item.__dict__ for item in evidence], "retrieval": interrupted.retrieval.model_dump()})
                yield {"type": "generation_stopped"}
                return
            yield {"type": "validation_started"}
            response, audit = normalize_response("".join(output), conversation_id=conversation_id, query=query, evidence=evidence, model=self.project.model, collections=selected, candidate_count=len(evidence), run_id=run_id)
            audit["instruction_files"] = instruction_files
            self.store.append_assistant(response, audit)
            yield {"type": "citation_update", "citations": [item.model_dump() for item in response.citations]}
            yield {"type": "message_completed", "message": response.model_dump()}
        except Exception as exc:
            yield {"type": "message_failed", "error": str(exc)}
        finally:
            with self._lock:
                self._active.pop(conversation_id, None)
