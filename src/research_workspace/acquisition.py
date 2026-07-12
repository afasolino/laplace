from __future__ import annotations

import hashlib
import json
import re
import shutil
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .online import SearchResult, _blocked_host
from .projects import load_project
from .projects import formalscience_root

try:
    import certifi as _certifi
except ImportError:
    _certifi = None  # type: ignore[assignment]


def _safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:180]
    return value or "download.pdf"


def download_open_access(
    project: str,
    candidate: SearchResult | dict[str, Any],
    *,
    root: Path | None = None,
    max_bytes: int = 50_000_000,
) -> dict[str, Any]:
    paths, _ = load_project(project, root=root)
    record = asdict(candidate) if isinstance(candidate, SearchResult) else dict(candidate)
    if record.get("open_access") is not True:
        raise PermissionError("The candidate is not explicitly identified as open access")
    url = str(record.get("pdf_url") or "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or _blocked_host(parsed.hostname):
        raise ValueError("Open-access acquisition requires a public HTTPS PDF URL")
    target_dir = paths.data / "Downloads" / "OpenAccess"
    target_dir.mkdir(parents=True, exist_ok=True)
    title = record.get("title") or record.get("doi") or "download"
    target = target_dir / _safe_filename(str(title) + ".pdf")
    metadata_path = target.with_suffix(".json")
    request = urllib.request.Request(url, headers={"User-Agent": "FormalScience/0.1"})
    digest = hashlib.sha256()
    temp_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_dir, prefix=".download-", suffix=".tmp", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            context = (
                ssl.create_default_context(cafile=_certifi.where())
                if _certifi is not None
                else ssl.create_default_context()
            )
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"application/pdf", "application/octet-stream"}:
                    raise ValueError(f"invalid PDF MIME type: {content_type}")
                first = response.read(5)
                if first != b"%PDF-":
                    raise ValueError("download does not begin with PDF magic bytes")
                temp.write(first)
                digest.update(first)
                total = len(first)
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("download exceeds maximum size")
                    temp.write(chunk)
                    digest.update(chunk)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    assert temp_path is not None
    if target.exists():
        existing_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        temp_path.unlink(missing_ok=True)
        if existing_hash == digest.hexdigest():
            return {"status": "duplicate", "path": str(target), "sha256": existing_hash}
        raise FileExistsError(f"Refusing overwrite: {target}")
    temp_path.replace(target)
    metadata = {
        "title": record.get("title"),
        "doi": record.get("doi"),
        "source_provider": record.get("provider"),
        "source_url": url,
        "access_type": "OPEN_ACCESS_EXPLICIT",
        "timestamp": datetime.now(UTC).isoformat(),
        "sha256": digest.hexdigest(),
        "project": project,
        "approval_state": "AUTOMATIC_OPEN_ACCESS",
        "ingestion_state": "PENDING",
        "bytes": total,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "downloaded",
        "path": str(target),
        "metadata": str(metadata_path),
        "sha256": digest.hexdigest(),
        "bytes": total,
    }


def promote_download(
    project: str,
    filename: str,
    collection: str,
    *,
    topic: str | None = None,
    confirm: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    if not confirm:
        raise PermissionError("Promotion is a destructive move and requires explicit --confirm")
    if collection not in {"MyTopics", "OtherTopics", "Documentations"}:
        raise ValueError("Promotion collection must be MyTopics, OtherTopics, or Documentations")
    paths, _ = load_project(project, root=root)
    source_candidates = list((paths.data / "Downloads" / "OpenAccess").glob(filename)) + list(
        (paths.data / "Downloads" / "IEEE" / "Downloaded").glob(filename)
    )
    if len(source_candidates) != 1:
        raise FileNotFoundError(
            "Promotion source must identify exactly one project-local downloaded file"
        )
    source = source_candidates[0]
    destination_dir = (
        formalscience_root() / "Library" / collection / (_safe_filename(topic) if topic else "")
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        raise FileExistsError(f"Refusing overwrite: {destination}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.move(str(source), str(destination))
    audit = paths.outputs / "Reports" / "promotion_audit.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "project": project,
        "source": str(source),
        "destination": str(destination),
        "collection": collection,
        "sha256": digest,
        "confirmed": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    audit.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"status": "PROMOTED", **record, "audit": str(audit)}
