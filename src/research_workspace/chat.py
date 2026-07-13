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
    evidence_id: str | None = None
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
    revision_id: str = ""
    state: str = "COMPLETED"
    grounding_status: str = "UNVERIFIED"
    citation_status: str = "UNKNOWN"
    fallback_of_message_id: str | None = None
    completed_at: str | None = None
    sequence: int | None = None


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
    # Conversation databases created by earlier Laplace builds are upgraded in
    # place.  These columns make assistant revisions immutable and allow the UI
    # to distinguish a rejected draft from its grounded fallback.
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
    additions = {
        "revision_id": "TEXT NOT NULL DEFAULT ''",
        "state": "TEXT NOT NULL DEFAULT 'COMPLETED'",
        "grounding_status": "TEXT NOT NULL DEFAULT 'UNVERIFIED'",
        "citation_status": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
        "fallback_of_message_id": "TEXT",
        "completed_at": "TEXT",
        "sequence": "INTEGER",
    }
    for name, declaration in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE chat_messages ADD COLUMN {name} {declaration}")
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
                    "revision_id": row["revision_id"] or "", "state": row["state"] or row["status"],
                    "grounding_status": row["grounding_status"] or "UNVERIFIED",
                    "citation_status": row["citation_status"] or "UNKNOWN",
                    "fallback_of_message_id": row["fallback_of_message_id"],
                    "completed_at": row["completed_at"], "sequence": row["sequence"],
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
                "INSERT INTO chat_messages (id,conversation_id,role,content,status,citations,unsupported_claims,fallback_used,created_at,model,retrieval,run_id,interrupted,revision_id,state,grounding_status,citation_status,fallback_of_message_id,completed_at,sequence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (message_id, conversation_id, "user", content, "USER", "[]", "[]", 0, stamp, self.project.model, "{}", None, 0,
                 message_id, "USER", "NOT_APPLICABLE", "NOT_APPLICABLE", None, stamp, None),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))
            conn.commit()
        return message_id

    def append_assistant(self, response: ChatResponse, audit: dict[str, Any]) -> None:
        audit_payload = {**audit, "citations": [item.model_dump() for item in response.citations]}
        with _db(self.path) as conn:
            self._check(conn, response.conversation_id)
            conn.execute(
                "INSERT INTO chat_messages (id,conversation_id,role,content,status,citations,unsupported_claims,fallback_used,created_at,model,retrieval,run_id,interrupted,revision_id,state,grounding_status,citation_status,fallback_of_message_id,completed_at,sequence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (response.message_id, response.conversation_id, "assistant", response.content, response.status,
                 json.dumps([item.model_dump() for item in response.citations], ensure_ascii=False),
                 json.dumps(response.unsupported_claims, ensure_ascii=False), int(response.fallback_used),
                 response.created_at, response.model, response.retrieval.model_dump_json(), response.run_id, int(response.status == "INTERRUPTED"),
                 response.revision_id or response.message_id, response.state, response.grounding_status,
                 response.citation_status, response.fallback_of_message_id, response.completed_at, response.sequence),
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


def _citation(item: Evidence, citation_id: int, evidence_id: str | None = None) -> ChatCitation:
    return ChatCitation(
        citation_id=citation_id, evidence_id=evidence_id or f"E{citation_id}", filename=item.filename, title=item.title, page=item.page,
        section=item.section, chunk_id=item.chunk_id, quoted_evidence=item.text[:700].strip(),
        availability=item.availability, source_class=item.document_class, score=item.score,
        source_path=item.source_path, doi=item.doi,
    )


def _evidence_records(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {"evidence_id": f"E{index}", "filename": item.filename, "page": item.page,
         "section": item.section, "chunk_id": item.chunk_id, "text": item.text}
        for index, item in enumerate(evidence, 1)
    ]


def _citation_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\[?E?(\d+)\]?", value.strip(), flags=re.I)
        if match:
            return int(match.group(1))
    return None


def _map_citations(raw: Any, evidence: list[Evidence], content: str = "") -> list[ChatCitation]:
    values: list[Any] = raw if isinstance(raw, list) else []
    # Models sometimes put only [E1][E2] markers in Markdown.  Recover those
    # markers without treating arbitrary numbers in prose as citations.
    if not values:
        values = [f"E{number}" for number in re.findall(r"\[E(\d+)\]", content, flags=re.I)]
    mapped: list[ChatCitation] = []
    for position, value in enumerate(values, 1):
        number: int | None = None
        evidence_id: str | None = None
        if isinstance(value, (int, str)):
            number = _citation_number(value)
            if isinstance(value, str) and re.fullmatch(r"\[?E\d+\]?", value.strip(), flags=re.I):
                evidence_id = value.strip("[]").upper()
        elif isinstance(value, dict):
            evidence_id_value = value.get("evidence_id")
            if isinstance(evidence_id_value, str):
                evidence_id = evidence_id_value.strip("[]").upper()
                number = _citation_number(evidence_id)
            if number is None:
                number = _citation_number(value.get("citation_id", value.get("id", value.get("index"))))
            if number is None:
                for index, item in enumerate(evidence, 1):
                    if value.get("filename") == item.filename and value.get("page") == item.page and value.get("chunk_id") == item.chunk_id:
                        number = index
                        break
        if number is not None and 1 <= number <= len(evidence):
            item = evidence[number - 1]
            if all(existing.chunk_id != item.chunk_id for existing in mapped):
                mapped.append(_citation(item, len(mapped) + 1, evidence_id or f"E{number}"))
        elif position <= len(evidence) and isinstance(value, dict) and value.get("chunk_id") == evidence[position - 1].chunk_id:
            mapped.append(_citation(evidence[position - 1], len(mapped) + 1))
    return mapped


def _fallback_content(query: str, evidence: list[Evidence]) -> str:
    """Build a concise answer-first fallback from retrieved snippets only."""
    if not evidence:
        return "No indexed evidence was retrieved for this request."
    lowered = query.lower()
    memory_query = any(term in lowered for term in ("compute-in-memory", "compute in memory", "cim", "in-memory", "in memory"))
    selected: list[Evidence] = []
    seen_docs: set[str] = set()
    for item in evidence:
        doc_key = item.filename.lower()
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        selected.append(item)
        if len(selected) >= 3:
            break
    sentences: list[str] = []
    for item in selected:
        cleaned = re.sub(r"\s+", " ", item.text).strip()
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
        if memory_query:
            parts = [part for part in parts if re.search(r"(?i)\b(cim|compute[- ]in[- ]memory|sram|mac|in[- ]memory|memory[- ]centric|processing[- ]in[- ]memory|bitcell|word line|bit line|time[- ]domain)\b", part)] or parts
        sentences.extend([part[:240].rstrip() for part in parts[:2]])
    if not sentences:
        sentences = [re.sub(r"\s+", " ", selected[0].text).strip()]
    markers = " ".join(f"[E{index}]" for index in range(1, len(selected) + 1))
    lead = "The retrieved evidence indicates: "
    if memory_query:
        lead = "For compute-in-memory, the retrieved evidence indicates: "
    return (lead + " ".join(sentences[:4]).strip() + (f" {markers}" if markers else ""))[:1100]


def normalize_revisions(
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
) -> tuple[ChatResponse, ChatResponse | None, dict[str, Any]]:
    """Normalize a model result into immutable candidate and optional fallback revisions."""
    parsed = _extract_json(output)
    attempted = parsed if parsed is not None else {"content": output, "citations": []}
    if isinstance(attempted, dict) and isinstance(attempted.get("answer"), str):
        nested = _extract_json(attempted["answer"])
        if isinstance(nested, dict):
            attempted = {**attempted, **nested}
    if not isinstance(attempted, dict):
        attempted = {"content": str(attempted), "citations": []}
    content = str(attempted.get("content", attempted.get("answer", attempted.get("draft", output)))).strip()
    nested_content = _extract_json(content)
    if isinstance(nested_content, dict):
        content = str(nested_content.get("content", nested_content.get("answer", content))).strip()
    citations = _map_citations(attempted.get("citations", attempted.get("sources", [])), evidence, content)
    model_valid = bool(citations) and validate_citations(
        evidence_packet(query, evidence),
        [{"filename": item.filename, "page": item.page, "chunk_id": item.chunk_id} for item in citations],
    )
    unsupported = attempted.get("unsupported_claims", attempted.get("unsupported", []))
    unsupported_claims = [str(item) for item in unsupported] if isinstance(unsupported, list) else []
    retrieval = ChatRetrieval(query=query, collections=collections, candidate_count=candidate_count, evidence_count=len(evidence), mode=mode)
    candidate_id = str(uuid.uuid4())
    candidate = ChatResponse(
        message_id=candidate_id, conversation_id=conversation_id,
        content=content or ("No indexed evidence was retrieved for this request." if not evidence else "No model draft was returned."),
        status="GROUNDED" if model_valid else ("CITATION_REJECTED" if evidence else "NOT_GROUNDED"),
        citations=citations, unsupported_claims=unsupported_claims, fallback_used=False, created_at=_now(), model=model,
        retrieval=retrieval, run_id=run_id, revision_id=str(uuid.uuid4()),
        state="GROUNDED" if model_valid else ("CITATION_REJECTED" if evidence else "NOT_GROUNDED"),
        grounding_status="GROUNDED" if model_valid else "UNVERIFIED",
        citation_status="VALID" if model_valid else ("REJECTED" if evidence else "NOT_APPLICABLE"), completed_at=_now(), sequence=1,
    )
    fallback: ChatResponse | None = None
    if evidence and not model_valid:
        fallback_citations = [_citation(item, index, f"E{index}") for index, item in enumerate(evidence[: min(6, len(evidence))], 1)]
        fallback = ChatResponse(
            message_id=str(uuid.uuid4()), conversation_id=conversation_id, content=_fallback_content(query, evidence),
            status="GROUNDED_EXTRACTIVE_FALLBACK", citations=fallback_citations,
            unsupported_claims=[], fallback_used=True, created_at=_now(), model=model, retrieval=retrieval, run_id=run_id,
            revision_id=str(uuid.uuid4()), state="GROUNDED_FALLBACK", grounding_status="GROUNDED", citation_status="VALID",
            fallback_of_message_id=candidate.message_id, completed_at=_now(), sequence=2,
        )
    audit = {
        "raw_model_output": output, "parsed_model_output": parsed, "evidence": [item.__dict__ for item in evidence],
        "evidence_records": _evidence_records(evidence), "retrieval": retrieval.model_dump(), "model_citation_valid": model_valid,
        "fallback_used": fallback is not None, "candidate_message_id": candidate.message_id,
        "fallback_message_id": fallback.message_id if fallback else None,
    }
    return candidate, fallback, audit


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
    candidate, fallback, audit = normalize_revisions(output, conversation_id=conversation_id, query=query, evidence=evidence,
        model=model, collections=collections, candidate_count=candidate_count, mode=mode, run_id=run_id)
    return fallback or candidate, audit


def _prompt(query: str, evidence: list[Evidence], history: list[dict[str, Any]], instructions: str = "") -> str:
    compact = _evidence_records(evidence)
    recent = [{"role": item.get("role"), "content": str(item.get("content", ""))[:1800]} for item in history[-6:]]
    return (
        "Return ONLY JSON with keys content, citations, unsupported_claims. content must be readable Markdown, not JSON. "
        "Citations must be compact evidence IDs such as E1 or [E1] copied from the supplied evidence. You may cite the same evidence ID for multiple claims. Never invent provenance. "
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

    def _retrieve(self, query: str, selected: list[str]) -> list[Evidence]:
        database = self.project.database if self.project.database.name == "workspace.db" else self.project.database.parent / "workspace.db"
        document_class = "user_work" if selected == ["MyWorks"] else None
        memory_query = any(term in query.lower() for term in ("compute-in-memory", "compute in memory", "cim", "in-memory", "in memory"))
        if not memory_query:
            return search(database, query, limit=6, document_class=document_class)
        expanded = query + " compute-in-memory computing-in-memory SRAM-CIM CIM macro in-memory MAC processing-in-memory memory-centric inference bitcell word-line bit-line time-domain"
        candidates = search(database, expanded, limit=12, document_class=document_class)
        positive = re.compile(r"(?i)\b(sram|cim|compute[- ]in[- ]memory|in[- ]memory|mac|bitcell|word[- ]line|bit[- ]line|analog accumulation|time[- ]domain|processing[- ]in[- ]memory)\b")
        weak = re.compile(r"(?i)\b(feature map|buffer|memory allocation|storage reuse|cache|overwrite scheduling)\b")
        ranked = sorted(candidates, key=lambda item: item.score + 0.08 * len(positive.findall(item.text)) - 0.10 * len(weak.findall(item.text)), reverse=True)
        result: list[Evidence] = []
        seen_docs: set[str] = set()
        for item in ranked:
            key = item.filename.lower()
            if key in seen_docs and len(result) < 4:
                continue
            seen_docs.add(key)
            result.append(item)
            if len(result) >= 6:
                break
        return result

    def answer(self, conversation_id: str, query: str, *, collections: list[str] | None = None, mode: str = "ASK", run_id: str | None = None) -> ChatResponse:
        with self._lock:
            if self._active:
                raise RuntimeError("another local generation is active; wait or press Stop")
        self.store.append_user(conversation_id, query)
        detail = self.store.detail(conversation_id)
        selected = collections or detail.collections
        evidence = self._retrieve(query, selected)
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
            candidate, fallback, audit = normalize_revisions(text, conversation_id=conversation_id, query=query, evidence=evidence, model=self.project.model, collections=selected, candidate_count=len(evidence), mode="hybrid", run_id=run_id)
            audit["instruction_files"] = instruction_files
            self.store.append_assistant(candidate, audit)
            if fallback is not None:
                self.store.append_assistant(fallback, audit)
                return fallback
            return candidate
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
        evidence = self._retrieve(query, selected)
        candidate_id = str(uuid.uuid4())
        revision_id = str(uuid.uuid4())
        yield {"type": "message_started", "conversation_id": conversation_id, "message_id": candidate_id, "revision_id": revision_id, "sequence": 1}
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
                    yield {"type": "token", "text": piece, "message_id": candidate_id, "revision_id": revision_id, "sequence": 1}
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
            candidate, fallback, audit = normalize_revisions("".join(output), conversation_id=conversation_id, query=query, evidence=evidence, model=self.project.model, collections=selected, candidate_count=len(evidence), run_id=run_id)
            candidate = candidate.model_copy(update={"message_id": candidate_id, "revision_id": revision_id, "sequence": 1})
            if fallback is not None:
                fallback = fallback.model_copy(update={"sequence": 2, "fallback_of_message_id": candidate.message_id})
            audit["instruction_files"] = instruction_files
            yield {"type": "citation_validation_completed", "valid": fallback is None, "message_id": candidate.message_id, "revision_id": candidate.revision_id, "sequence": 1}
            if fallback is not None:
                self.store.append_assistant(candidate, audit)
                yield {"type": "message_rejected", "message": candidate.model_dump(), "reason": "citation_validation_failed"}
                yield {"type": "fallback_started", "message_id": fallback.message_id, "revision_id": fallback.revision_id, "sequence": 2}
                for piece_start in range(0, len(fallback.content), 160):
                    yield {"type": "fallback_token", "text": fallback.content[piece_start:piece_start + 160], "message_id": fallback.message_id, "revision_id": fallback.revision_id, "sequence": 2}
                self.store.append_assistant(fallback, audit)
                yield {"type": "fallback_completed", "message": fallback.model_dump()}
            else:
                self.store.append_assistant(candidate, audit)
                yield {"type": "citation_update", "citations": [item.model_dump() for item in candidate.citations], "message_id": candidate.message_id, "revision_id": candidate.revision_id}
                yield {"type": "message_completed", "message": candidate.model_dump()}
        except Exception as exc:
            yield {"type": "message_failed", "error": str(exc)}
        finally:
            with self._lock:
                self._active.pop(conversation_id, None)
