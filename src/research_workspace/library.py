from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .documents import Chunk, _chunks, _db, _safe_text
from .projects import ProjectPaths, formalscience_root, load_project


COLLECTION_MAP: dict[str, dict[str, str]] = {
    "MyWorks": {
        "source_class": "user_work",
        "style_eligible": "true",
        "ownership_scope": "user_supplied_work",
    },
    "MyTopics": {"source_class": "external_literature", "relevance_scope": "primary_topics"},
    "OtherTopics": {"source_class": "external_literature", "relevance_scope": "secondary_topics"},
    "Documentations": {"source_class": "technical_documentation"},
    "ExpRes": {"source_class": "experimental_record"},
}
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _metadata(path: Path, collection: str) -> dict[str, Any]:
    reader = PdfReader(path)
    info: Any = reader.metadata or {}
    first_pages = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    title = str(getattr(info, "title", "") or "").strip() or next(
        (line.strip() for line in first_pages.splitlines() if len(line.strip()) > 10), None
    )
    author = str(getattr(info, "author", "") or "").strip() or None
    doi_match = DOI_RE.search(first_pages)
    year_match = YEAR_RE.search(first_pages) or YEAR_RE.search(
        str(getattr(info, "subject", "") or "")
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "filename": path.name,
        "absolute_source_path": str(path.resolve()),
        "relative_library_path": str(path.resolve().relative_to(formalscience_root() / "Library")),
        "sha256": digest,
        "document_id": digest[:20],
        "title": title,
        "authors": [author] if author else [],
        "year": int(year_match.group(1)) if year_match else None,
        "doi": doi_match.group(0).rstrip(".,;)") if doi_match else None,
        "publication": str(getattr(info, "subject", "") or "").strip() or None,
        "page_count": len(reader.pages),
        "source_class": COLLECTION_MAP[collection]["source_class"],
        "collection": collection,
        "collection_metadata": COLLECTION_MAP[collection],
        "availability": "COMPLETE_LOCAL_PDF",
        "source_kind": "local",
        "parser": "pypdf",
        "ingested_at": datetime.now(UTC).isoformat(),
    }


def _existing_metadata(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT metadata FROM documents").fetchall()
    values = []
    for row in rows:
        try:
            value = json.loads(row[0] or "{}")
        except json.JSONDecodeError:
            value = {}
        if isinstance(value, dict):
            values.append(value)
    return values


def ingest_library(
    project: str,
    *,
    collection: str = "MyWorks",
    root: Path | None = None,
    source_dir: Path | None = None,
    project_paths_override: ProjectPaths | None = None,
) -> dict[str, Any]:
    if collection not in COLLECTION_MAP:
        raise ValueError(f"Unsupported Library collection: {collection}")
    if project_paths_override is not None:
        paths = project_paths_override
    else:
        paths, _ = load_project(project, root=root)
    formal = formalscience_root().resolve()
    library_dir = (source_dir or (formal / "Library" / collection)).resolve()
    library_root = (formal / "Library").resolve()
    if library_root not in library_dir.parents and library_dir != library_root:
        raise ValueError("Library source directory must remain under the configured Library root")
    if not library_dir.is_dir():
        raise FileNotFoundError(library_dir)
    if not library_dir.is_dir():
        raise FileNotFoundError(library_dir)
    database = paths.data / "Metadata" / "workspace.db"
    conn = _db(database)
    existing = _existing_metadata(conn)
    report: dict[str, Any] = {
        "project": project,
        "collection": collection,
        "source_directory": str(library_dir),
        "files": [],
        "counts": {"discovered": 0, "ingested": 0, "unchanged": 0, "duplicates": 0, "failed": 0},
    }
    for source in sorted(library_dir.rglob("*.pdf")):
        report["counts"]["discovered"] += 1
        try:
            metadata = _metadata(source, collection)
            same_hash = next(
                (item for item in existing if item.get("sha256") == metadata["sha256"]), None
            )
            same_doi = next(
                (
                    item
                    for item in existing
                    if metadata.get("doi") and item.get("doi") == metadata.get("doi")
                ),
                None,
            )
            same_title = next(
                (
                    item
                    for item in existing
                    if metadata.get("title") and item.get("title") == metadata.get("title")
                ),
                None,
            )
            if same_hash or same_doi or same_title:
                report["counts"]["unchanged" if same_hash else "duplicates"] += 1
                report["files"].append(
                    {"status": "unchanged" if same_hash else "duplicate", **metadata}
                )
                continue
            reader = PdfReader(source)
            chunks: list[Chunk] = []
            for page_number, page in enumerate(reader.pages, 1):
                for index, text in enumerate(_chunks(page.extract_text() or "")):
                    chunks.append(
                        Chunk(
                            metadata["document_id"],
                            metadata["sha256"],
                            source.name,
                            metadata["source_class"],
                            page_number,
                            page_number,
                            None,
                            f"{metadata['document_id']}:p{page_number}:c{index}",
                            text,
                        )
                    )
            if not chunks:
                raise ValueError("No usable native text; OCR required")
            parsed = paths.data / "Parsed" / f"{metadata['document_id']}.json"
            parsed.write_text(
                json.dumps([asdict(chunk) for chunk in chunks], indent=2), encoding="utf-8"
            )
            conn.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?)",
                (
                    metadata["document_id"],
                    metadata["sha256"],
                    source.name,
                    metadata["source_class"],
                    json.dumps(metadata),
                ),
            )
            conn.executemany(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
                [
                    (c.chunk_id, c.document_id, c.page_start, c.page_end, c.section, c.text)
                    for c in chunks
                ],
            )
            conn.commit()
            existing.append(metadata)
            report["counts"]["ingested"] += 1
            report["files"].append({"status": "ingested", "chunks": len(chunks), **metadata})
        except Exception as exc:
            report["counts"]["failed"] += 1
            report["files"].append(
                {
                    "status": "failed",
                    "filename": source.name,
                    "error": str(exc),
                    "absolute_source_path": str(source.resolve()),
                }
            )
    conn.close()
    report_path = paths.outputs / "Reports" / f"library_ingestion_{collection}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report_path": str(report_path), **report}


def ingest_downloads(project: str, *, root: Path | None = None) -> dict[str, Any]:
    paths, _ = load_project(project, root=root)
    database = paths.data / "Metadata" / "workspace.db"
    conn = _db(database)
    existing = {row[0] for row in conn.execute("SELECT sha256 FROM documents").fetchall()}
    sources = list((paths.data / "Downloads" / "OpenAccess").glob("*.pdf")) + list(
        (paths.data / "Downloads" / "IEEE" / "Downloaded").glob("*.pdf")
    )
    report: dict[str, Any] = {
        "project": project,
        "files": [],
        "counts": {"discovered": len(sources), "ingested": 0, "duplicates": 0, "failed": 0},
    }
    for source in sorted(sources):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in existing:
            report["counts"]["duplicates"] += 1
            report["files"].append(
                {"status": "duplicate", "filename": source.name, "sha256": digest}
            )
            continue
        try:
            pages = _safe_text(source)
            document_id = digest[:20]
            chunks: list[Chunk] = []
            for page, text in pages:
                for index, value in enumerate(_chunks(text)):
                    chunks.append(
                        Chunk(
                            document_id,
                            digest,
                            source.name,
                            "external_literature",
                            page,
                            page,
                            None,
                            f"{document_id}:p{page or 0}:c{index}",
                            value,
                        )
                    )
            if not chunks:
                raise ValueError("No usable native text; OCR required")
            parsed = paths.data / "Parsed" / f"{document_id}.json"
            parsed.write_text(
                json.dumps([asdict(chunk) for chunk in chunks], indent=2), encoding="utf-8"
            )
            sidecar = source.with_suffix(".json")
            source_metadata = (
                json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            )
            metadata = {
                "filename": source.name,
                "absolute_source_path": str(source.resolve()),
                "sha256": digest,
                "document_id": document_id,
                "title": source_metadata.get("title") or source.stem,
                "doi": source_metadata.get("doi"),
                "page_count": len(pages),
                "source_class": "external_literature",
                "availability": "COMPLETE_DOWNLOADED_PDF",
                "source_kind": "online",
                "access_type": source_metadata.get("access_type"),
                "ingested_at": datetime.now(UTC).isoformat(),
            }
            conn.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?)",
                (document_id, digest, source.name, "external_literature", json.dumps(metadata)),
            )
            conn.executemany(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section,
                        chunk.text,
                    )
                    for chunk in chunks
                ],
            )
            conn.commit()
            existing.add(digest)
            report["counts"]["ingested"] += 1
            report["files"].append({"status": "ingested", "chunks": len(chunks), **metadata})
        except Exception as exc:
            report["counts"]["failed"] += 1
            report["files"].append(
                {"status": "failed", "filename": source.name, "error": str(exc), "sha256": digest}
            )
    conn.close()
    target = paths.outputs / "Reports" / "download_ingestion.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report_path": str(target), **report}
