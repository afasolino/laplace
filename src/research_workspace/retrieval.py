from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


def embed(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[\w.-]+", text.lower()):
        index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@dataclass(frozen=True)
class Evidence:
    filename: str
    page: int | None
    section: str | None
    chunk_id: str
    text: str
    score: float
    document_class: str
    source_path: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    collection: str | None = None
    availability: str = "COMPLETE_LOCAL_PDF"
    source_kind: str = "local"


def search(
    database: Path,
    query: str,
    mode: str = "hybrid",
    limit: int = 6,
    document_class: str | None = None,
    filename: str | None = None,
    collection: str | None = None,
    author: str | None = None,
    year: int | None = None,
    doi: str | None = None,
    availability: str | None = None,
    source_kind: str | None = None,
) -> list[Evidence]:
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT, class TEXT, metadata TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT, page_start INTEGER, page_end INTEGER, section TEXT, text TEXT)"
    )
    sql = "SELECT d.filename,d.class,c.page_start,c.section,c.id,c.text,d.metadata FROM chunks c JOIN documents d ON d.id=c.document_id"
    clauses, params = [], []
    if document_class:
        clauses.append("d.class=?")
        params.append(document_class)
    if filename:
        clauses.append("d.filename=?")
        params.append(filename)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    q_tokens = set(re.findall(r"[\w.-]+", query.lower()))
    qvec = embed(query)
    found = []
    for fname, cls, page, section, cid, text, metadata_text in rows:
        try:
            metadata = json.loads(metadata_text or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if collection and metadata.get("collection") != collection:
            continue
        if author and not any(
            author.lower() in str(value).lower() for value in metadata.get("authors", [])
        ):
            continue
        if year is not None and metadata.get("year") != year:
            continue
        if doi and metadata.get("doi") != doi:
            continue
        if availability and metadata.get("availability") != availability:
            continue
        if source_kind and metadata.get("source_kind", "local") != source_kind:
            continue
        tokens = set(re.findall(r"[\w.-]+", text.lower()))
        lexical = len(q_tokens & tokens) / max(1, len(q_tokens))
        vector = embed(text)
        semantic = sum(a * b for a, b in zip(qvec, vector))
        score = (
            lexical
            if mode == "keyword"
            else semantic
            if mode == "semantic"
            else 0.55 * semantic + 0.45 * lexical
        )
        if score > 0:
            found.append(
                Evidence(
                    fname,
                    page,
                    section,
                    cid,
                    text,
                    score,
                    cls,
                    metadata.get("absolute_source_path"),
                    metadata.get("title"),
                    metadata.get("authors"),
                    metadata.get("year"),
                    metadata.get("doi"),
                    metadata.get("collection"),
                    metadata.get("availability", "COMPLETE_LOCAL_PDF"),
                    metadata.get("source_kind", "local"),
                )
            )
    unique: dict[str, Evidence] = {}
    for item in sorted(found, key=lambda x: x.score, reverse=True):
        fingerprint = hashlib.sha256(item.text.encode()).hexdigest()
        unique.setdefault(fingerprint, item)
    return list(unique.values())[:limit]


def evidence_packet(query: str, evidence: list[Evidence]) -> dict[str, object]:
    return {
        "query": query,
        "grounded": bool(evidence),
        "evidence": [asdict(e) for e in evidence],
        "uncertainty": [] if evidence else ["No valid evidence chunks retrieved"],
        "missing_evidence": [] if evidence else ["Indexed source evidence"],
    }


def validate_citations(packet: dict[str, object], citations: list[dict[str, object]]) -> bool:
    evidence = cast(list[dict[str, object]], packet.get("evidence", []))
    valid = {(e["filename"], e["page"], e["chunk_id"]) for e in evidence}
    return bool(valid) and all(
        (c.get("filename"), c.get("page"), c.get("chunk_id")) in valid for c in citations
    )
