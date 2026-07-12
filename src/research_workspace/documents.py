from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

ALLOWED = {".pdf", ".docx", ".md", ".txt", ".html", ".htm", ".csv", ".json", ".log", ".out", ".rpt"}
DOCUMENT_CLASSES = {
    "user_work",
    "external_literature",
    "technical_documentation",
    "project_document",
    "experiment_record",
}


@dataclass(frozen=True)
class Chunk:
    document_id: str
    sha256: str
    filename: str
    document_class: str
    page_start: int | None
    page_end: int | None
    section: str | None
    chunk_id: str
    text: str


def _db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT, class TEXT, metadata TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT, page_start INTEGER, page_end INTEGER, section TEXT, text TEXT)"
    )
    return conn


def _safe_text(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError("DOCX support requires the docx optional dependency") from exc
        return [(None, "\n".join(p.text for p in Document(path).paragraphs))]
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    elif suffix == ".json":
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    return [(None, text)]


def _chunks(text: str, size: int = 2600, overlap: int = 400) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return
    position = 0
    while position < len(normalized):
        end = min(len(normalized), position + size)
        if end < len(normalized):
            split = normalized.rfind(" ", position, end)
            end = split if split > position else end
        yield normalized[position:end]
        if end == len(normalized):
            break
        position = max(position + 1, end - overlap)


def ingest(path: Path, root: Path, database: Path, document_class: str) -> dict[str, object]:
    source = path.resolve()
    if not source.is_file() or source.suffix.lower() not in ALLOWED:
        raise ValueError("Unsupported or missing input file")
    if document_class not in DOCUMENT_CLASSES:
        raise ValueError(f"Unsupported document class: {document_class}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    document_id = digest[:20]
    conn = _db(database)
    if conn.execute("SELECT id FROM documents WHERE sha256=?", (digest,)).fetchone():
        conn.close()
        return {"status": "duplicate", "document_id": document_id, "sha256": digest}
    stored = root / "data" / "documents" / digest / source.name
    derived = root / "data" / "parsed" / f"{document_id}.json"
    try:
        pages = _safe_text(source)
        if not pages or all(not text.strip() for _, text in pages):
            raise ValueError("No usable native text; OCR required")
        all_chunks: list[Chunk] = []
        for page, text in pages:
            for index, value in enumerate(_chunks(text)):
                cid = f"{document_id}:p{page or 0}:c{index}"
                all_chunks.append(
                    Chunk(
                        document_id,
                        digest,
                        source.name,
                        document_class,
                        page,
                        page,
                        None,
                        cid,
                        value,
                    )
                )
        stored.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stored)
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_text(json.dumps([asdict(c) for c in all_chunks], indent=2), encoding="utf-8")
        conn.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?)",
            (
                document_id,
                digest,
                source.name,
                document_class,
                json.dumps(
                    {
                        "absolute_source_path": str(source),
                        "title": source.stem,
                        "source_class": document_class,
                        "availability": "COMPLETE_LOCAL_PDF"
                        if source.suffix.lower() == ".pdf"
                        else "COMPLETE_LOCAL_DOCUMENT",
                        "source_kind": "local",
                    }
                ),
            ),
        )
        conn.executemany(
            "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
            [
                (c.chunk_id, c.document_id, c.page_start, c.page_end, c.section, c.text)
                for c in all_chunks
            ],
        )
        conn.commit()
        return {
            "status": "ingested",
            "document_id": document_id,
            "sha256": digest,
            "chunks": len(all_chunks),
            "pages": len(pages),
        }
    except Exception as exc:
        quarantine = root / "data" / "quarantine" / f"{digest}.json"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        quarantine.write_text(
            json.dumps({"source": str(source), "sha256": digest, "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        conn.close()


def delete_document(database: Path, root: Path, document_id: str) -> None:
    conn = _db(database)
    row = conn.execute("SELECT sha256 FROM documents WHERE id=?", (document_id,)).fetchone()
    if row:
        conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        conn.commit()
        shutil.rmtree(root / "data" / "documents" / row[0], ignore_errors=True)
        (root / "data" / "parsed" / f"{document_id}.json").unlink(missing_ok=True)
    conn.close()
