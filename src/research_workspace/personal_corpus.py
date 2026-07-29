"""Owner-isolated personal corpus ingestion, indexing, and retrieval.

The implementation deliberately uses deterministic local parsers and SQLite. Model
output is never used to decide whether a file is safe or how it is chunked.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import mimetypes
import multiprocessing
import os
import re
import secrets
import shutil
import sqlite3
import stat
import unicodedata
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Iterable, TypeAlias, cast
from multiprocessing.connection import Connection

from defusedxml import ElementTree
from pypdf import PdfReader

try:
    import resource
except ImportError:  # pragma: no cover - exercised by the Windows CI matrix
    resource = None  # type: ignore[assignment]

JsonObject: TypeAlias = dict[str, object]

POLICY_SCHEMA_VERSION = 1
CHUNKING_VERSION = "line-chunks-v1-3000-300"
DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".md", ".markdown", ".txt", ".json", ".jsonl", ".log"}
)
ENGINEERING_EXTENSIONS = frozenset(
    {".py", ".pyi", ".v", ".vh", ".sv", ".svh"}
)
EXTENSIONLESS_BASENAMES = frozenset({"Makefile", "Dockerfile", "CMakeLists.txt"})
IGNORED_COMPONENTS = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
    }
)
_TEMP_PREFIXES = ("~$", ".~lock.")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)
_TEXT_MIME = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-python",
        "text/x-verilog",
        "text/x-systemverilog",
        "application/json",
        "application/x-ndjson",
        "application/octet-stream",
        "",
    }
)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "api_token",
        re.compile(
            r"(?i)\b(?:api[_-]?key|[A-Za-z0-9_]*(?:token|secret))"
            r"\s*[:=]\s*[^\s]{12,}"
        ),
    ),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
)


class RetrievalSelection(StrEnum):
    NONE = "none"
    PERSONAL = "personal"
    SHARED = "shared"
    BOTH = "both"
    SELECTED_PERSONAL = "selected_personal"


class CorpusError(RuntimeError):
    """A corpus operation failed a validation, ownership, or lifecycle rule."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class PersonalCorpusPolicy:
    schema_version: int = POLICY_SCHEMA_VERSION
    policy_id: str = "personal-corpus-upload-v1"
    document_extensions: tuple[str, ...] = tuple(sorted(DOCUMENT_EXTENSIONS))
    engineering_extensions: tuple[str, ...] = tuple(sorted(ENGINEERING_EXTENSIONS))
    extensionless_basenames: tuple[str, ...] = tuple(sorted(EXTENSIONLESS_BASENAMES))
    max_files_per_batch: int = 2_000
    max_file_bytes: int = 64 * 1024 * 1024
    max_batch_bytes: int = 512 * 1024 * 1024
    max_extracted_text_bytes: int = 512 * 1024 * 1024
    max_user_bytes: int = 5 * 1024 * 1024 * 1024
    max_path_bytes: int = 1_024
    max_filename_bytes: int = 255
    max_zip_entries: int = 2_000
    max_zip_ratio: int = 100
    min_free_disk_bytes: int = 512 * 1024 * 1024
    parser_timeout_seconds: int = 60
    chunk_target_characters: int = 3_000
    chunk_overlap_characters: int = 300
    soft_delete_days: int = 30

    def public(self) -> JsonObject:
        value = asdict(self)
        value["accepted_extensions"] = sorted(
            {*self.document_extensions, *self.engineering_extensions}
        )
        value["zip_fallback_only"] = True
        value["ocr_enabled"] = False
        value["source_support"] = {
            ".py": "Python Agent and retrieval",
            ".pyi": "Python reference and retrieval",
            ".v": "Verilog/SystemVerilog Agent and retrieval",
            ".vh": "Verilog/SystemVerilog reference and retrieval",
            ".sv": "SystemVerilog Agent and retrieval",
            ".svh": "SystemVerilog reference and retrieval",
            "other_allowed_text": "retrieval only",
        }
        return value


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    media_type: str
    replacement_count: int
    page_spans: tuple[tuple[int, int, int], ...] = ()
    section: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise CorpusError(f"invalid_{label}")
    return value


def _atomic_private_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_logical_path(raw: str, policy: PersonalCorpusPolicy) -> str:
    if "\x00" in raw or "\\" in raw:
        raise CorpusError("invalid_upload_path")
    normalized = unicodedata.normalize("NFC", raw.strip())
    if normalized != raw.strip():
        # Normalization is permitted, but collisions are rejected by the session store.
        raw = normalized
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise CorpusError("upload_path_traversal")
    if len(normalized.encode("utf-8")) > policy.max_path_bytes:
        raise CorpusError("upload_path_too_long")
    for component in path.parts:
        if (
            not component
            or len(component.encode("utf-8")) > policy.max_filename_bytes
            or component in IGNORED_COMPONENTS
            or component.startswith(_TEMP_PREFIXES)
            or component.upper().split(".", 1)[0] in _RESERVED_NAMES
        ):
            raise CorpusError("upload_path_rejected", {"component": component[:80]})
        if component.startswith("."):
            raise CorpusError("hidden_metadata_rejected")
    return path.as_posix()


def _support_label(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        return "python_reference"
    if suffix in {".v", ".vh", ".sv", ".svh"}:
        return "systemverilog_reference"
    return "retrieval_only"


def _validate_extension(path: str) -> None:
    logical = PurePosixPath(path)
    suffix = logical.suffix.lower()
    if logical.name in EXTENSIONLESS_BASENAMES:
        return
    if suffix not in DOCUMENT_EXTENSIONS | ENGINEERING_EXTENSIONS:
        if suffix in {".zip"}:
            raise CorpusError("zip_fallback_requires_controlled_endpoint")
        raise CorpusError("unsupported_extension", {"extension": suffix or "(none)"})


def _validate_magic(path: str, content: bytes, client_mime: str) -> str:
    if not content:
        raise CorpusError("empty_file")
    if content.startswith((b"\x7fELF", b"MZ")):
        raise CorpusError("executable_rejected")
    suffix = PurePosixPath(path).suffix.lower()
    supplied = client_mime.split(";", 1)[0].strip().lower()
    guessed = (mimetypes.guess_type(path)[0] or "").lower()
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise CorpusError("magic_mismatch")
        if supplied and supplied not in {"application/pdf", "application/octet-stream"}:
            raise CorpusError("mime_mismatch")
        return "application/pdf"
    if suffix == ".docx":
        if not content.startswith(b"PK"):
            raise CorpusError("magic_mismatch")
        if supplied and supplied not in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
            "application/octet-stream",
        }:
            raise CorpusError("mime_mismatch")
        _inspect_docx(content)
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if content.startswith(b"PK") or zipfile.is_zipfile(io.BytesIO(content)):
        raise CorpusError("archive_rejected")
    if b"\x00" in content[:65_536]:
        raise CorpusError("binary_content_rejected")
    if supplied not in _TEXT_MIME and not supplied.startswith("text/"):
        raise CorpusError("mime_mismatch", {"client_mime": supplied})
    if guessed and guessed not in _TEXT_MIME and not guessed.startswith("text/"):
        raise CorpusError("mime_mismatch", {"guessed_mime": guessed})
    if suffix == ".json":
        return "application/json"
    if suffix == ".jsonl":
        return "application/x-ndjson"
    return supplied or guessed or "text/plain"


def _inspect_docx(content: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                raise CorpusError("invalid_docx")
            lowered = {name.lower() for name in names}
            if any(
                name.endswith(("vbaproject.bin", ".exe", ".dll", ".js", ".vbs"))
                for name in lowered
            ):
                raise CorpusError("macro_or_embedded_executable_rejected")
            for name in names:
                if name.endswith(".rels"):
                    data = archive.read(name)
                    if b'TargetMode="External"' in data or b"TargetMode='External'" in data:
                        raise CorpusError("docx_external_relationship_rejected")
    except zipfile.BadZipFile as exc:
        raise CorpusError("invalid_docx") from exc


def _decode_text(content: bytes) -> tuple[str, int]:
    try:
        return content.decode("utf-8"), 0
    except UnicodeDecodeError:
        decoded = content.decode("utf-8", errors="replace")
        replacements = decoded.count("\ufffd")
        if replacements > max(32, len(decoded) // 50):
            # Latin-1 is deterministic and lossless for bytes, but explicitly reported.
            return content.decode("latin-1"), len(content)
        return decoded, replacements


def _extract(path: str, content: bytes) -> ExtractedDocument:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise CorpusError("encrypted_pdf_rejected")
            pages: list[str] = []
            spans: list[tuple[int, int, int]] = []
            cursor = 0
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append(text)
                spans.append((cursor, cursor + len(text), page_number))
                cursor += len(text) + 2
            return ExtractedDocument(
                text="\n\n".join(pages),
                media_type="application/pdf",
                replacement_count=0,
                page_spans=tuple(spans),
            )
        except CorpusError:
            raise
        except Exception as exc:
            raise CorpusError(
                "pdf_parser_rejected", {"error_type": type(exc).__name__}
            ) from exc
    if suffix == ".docx":
        _inspect_docx(content)
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
            pieces = [
                str(node.text)
                for node in root.iter()
                if node.tag.endswith("}t") and node.text
            ]
            return ExtractedDocument(
                text="\n".join(pieces),
                media_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                replacement_count=0,
            )
        except CorpusError:
            raise
        except Exception as exc:
            raise CorpusError(
                "docx_parser_rejected", {"error_type": type(exc).__name__}
            ) from exc
    text, replacements = _decode_text(content)
    if suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise CorpusError("invalid_json", {"line": exc.lineno}) from exc
    if suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError("invalid_jsonl", {"line": line_number}) from exc
    return ExtractedDocument(
        text=text.replace("\r\n", "\n").replace("\r", "\n"),
        media_type=(
            "application/json"
            if suffix == ".json"
            else ("application/x-ndjson" if suffix == ".jsonl" else "text/plain")
        ),
        replacement_count=replacements,
    )


def _extract_worker(
    connection: Connection,
    path: str,
    content: bytes,
    memory_limit_bytes: int,
) -> None:
    try:
        if resource is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_limit_bytes, memory_limit_bytes),
            )
        extracted = _extract(path, content)
        connection.send(("ok", extracted))
    except CorpusError as exc:
        connection.send(("corpus_error", exc.category, exc.evidence))
    except BaseException as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _extract_bounded(
    path: str, content: bytes, policy: PersonalCorpusPolicy
) -> ExtractedDocument:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_extract_worker,
        args=(child, path, content, max(256 * 1024 * 1024, policy.max_file_bytes * 4)),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(policy.parser_timeout_seconds):
            process.terminate()
            process.join(timeout=5)
            raise CorpusError("parser_timeout")
        value: object = parent.recv()
    except EOFError as exc:
        raise CorpusError("parser_worker_failed") from exc
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
    if not isinstance(value, tuple) or not value:
        raise CorpusError("parser_worker_failed")
    if value[0] == "corpus_error":
        evidence = value[2] if len(value) > 2 and isinstance(value[2], dict) else {}
        raise CorpusError(str(value[1]), evidence)
    if value[0] == "error":
        raise CorpusError("parser_worker_failed", {"error_type": str(value[1])})
    extracted = value[1]
    if not isinstance(extracted, ExtractedDocument):
        raise CorpusError("parser_worker_failed")
    return extracted


def _secret_warnings(content: bytes) -> list[str]:
    sample = content[:2_000_000].decode("utf-8", errors="replace")
    return [name for name, pattern in _SECRET_PATTERNS if pattern.search(sample)]


class PersonalCorpusStore:
    """Crash-safe owner-scoped corpus store rooted outside Git."""

    def __init__(
        self,
        state_root: Path,
        *,
        policy: PersonalCorpusPolicy = PersonalCorpusPolicy(),
    ) -> None:
        self.root = (state_root.resolve() / "personal_corpora")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.policy = policy
        self.database = self.root / "registry.sqlite3"
        self.audit_path = self.root / "provenance.jsonl"
        self.owner_key_path = self.root / "owner_hmac.key"
        if not self.owner_key_path.exists():
            _atomic_private_bytes(self.owner_key_path, secrets.token_bytes(32))
        self._owner_key = self.owner_key_path.read_bytes()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS corpus_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version INTEGER NOT NULL,
                    chunking_version TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS corpora (
                    corpus_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    owner_pseudonym TEXT NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    deleted_at_utc TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS corpora_owner_name_active
                    ON corpora(owner_user_id, name)
                    WHERE state != 'DELETED';
                CREATE TABLE IF NOT EXISTS upload_sessions (
                    upload_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    corpus_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    UNIQUE(owner_user_id, idempotency_key),
                    FOREIGN KEY(corpus_id) REFERENCES corpora(corpus_id)
                );
                CREATE TABLE IF NOT EXISTS upload_files (
                    upload_id TEXT NOT NULL,
                    logical_path TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT,
                    size_bytes INTEGER NOT NULL,
                    content_sha256 TEXT,
                    media_type TEXT,
                    support_label TEXT,
                    warnings_json TEXT NOT NULL,
                    staging_name TEXT,
                    PRIMARY KEY(upload_id, logical_path),
                    UNIQUE(upload_id, normalized_key),
                    FOREIGN KEY(upload_id) REFERENCES upload_sessions(upload_id)
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    corpus_id TEXT NOT NULL,
                    logical_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    support_label TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    extracted_sha256 TEXT,
                    replacement_count INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    storage_name TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    deleted_at_utc TEXT,
                    UNIQUE(owner_user_id, corpus_id, content_sha256),
                    FOREIGN KEY(corpus_id) REFERENCES corpora(corpus_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    corpus_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    snapshot_revision INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    page_number INTEGER,
                    section TEXT,
                    text TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK(active IN (0,1)),
                    FOREIGN KEY(source_id) REFERENCES sources(source_id)
                );
                CREATE INDEX IF NOT EXISTS chunks_owner_corpus_active
                    ON chunks(owner_user_id, corpus_id, active);
                CREATE TABLE IF NOT EXISTS idempotency (
                    owner_user_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY(owner_user_id, operation, idempotency_key)
                );
                """
            )
            connection.execute(
                """
                INSERT INTO corpus_schema (
                    singleton, schema_version, chunking_version, updated_at_utc
                ) VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    chunking_version=excluded.chunking_version,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (POLICY_SCHEMA_VERSION, CHUNKING_VERSION, _now()),
            )
        os.chmod(self.database, 0o600)

    def _pseudonym(self, owner_user_id: str) -> str:
        _identifier(owner_user_id, label="user_id")
        return "u_" + hmac.new(
            self._owner_key, owner_user_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:32]

    def _owner_root(self, owner_user_id: str) -> Path:
        value = self.root / "owners" / self._pseudonym(owner_user_id)
        value.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(value, 0o700)
        return value

    def _event(
        self,
        event: str,
        *,
        owner_user_id: str,
        corpus_id: str | None = None,
        source_id: str | None = None,
        outcome: str = "SUCCESS",
        details: JsonObject | None = None,
    ) -> None:
        value = {
            "timestamp_utc": _now(),
            "event": event,
            "outcome": outcome,
            "owner_pseudonym": self._pseudonym(owner_user_id),
            "corpus_id": corpus_id,
            "source_id": source_id,
            "details": details or {},
        }
        descriptor = os.open(
            self.audit_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

    def create_corpus(self, owner_user_id: str, name: str) -> JsonObject:
        _identifier(owner_user_id, label="user_id")
        normalized_name = unicodedata.normalize("NFC", name.strip())
        if not normalized_name or len(normalized_name) > 160:
            raise CorpusError("invalid_corpus_name")
        corpus_id = f"pc_{uuid.uuid4().hex}"
        now = _now()
        pseudonym = self._pseudonym(owner_user_id)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO corpora (
                        corpus_id, owner_user_id, owner_pseudonym, name, state,
                        revision, created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, 'EMPTY', 1, ?, ?)
                    """,
                    (
                        corpus_id,
                        owner_user_id,
                        pseudonym,
                        normalized_name,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise CorpusError("corpus_name_exists") from exc
        corpus_root = self._owner_root(owner_user_id) / corpus_id
        (corpus_root / "sources").mkdir(parents=True, mode=0o700)
        (corpus_root / "staging").mkdir(parents=True, mode=0o700)
        self._event("CORPUS_CREATE", owner_user_id=owner_user_id, corpus_id=corpus_id)
        return self.require_corpus(owner_user_id, corpus_id)

    def require_corpus(
        self, owner_user_id: str, corpus_id: str, *, include_deleted: bool = False
    ) -> JsonObject:
        _identifier(owner_user_id, label="user_id")
        if not re.fullmatch(r"pc_[a-f0-9]{32}", corpus_id):
            raise CorpusError("corpus_not_found")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM corpora WHERE corpus_id=? AND owner_user_id=?",
                (corpus_id, owner_user_id),
            ).fetchone()
        if row is None or (str(row["state"]) == "DELETED" and not include_deleted):
            raise CorpusError("corpus_not_found")
        return self._corpus_public(row)

    def list_corpora(
        self, owner_user_id: str, *, include_archived: bool = True
    ) -> list[JsonObject]:
        _identifier(owner_user_id, label="user_id")
        query = (
            """
            SELECT * FROM corpora
            WHERE owner_user_id=? AND state != 'DELETED'
            ORDER BY updated_at_utc DESC
            """
            if include_archived
            else """
            SELECT * FROM corpora
            WHERE owner_user_id=? AND state NOT IN ('DELETED', 'ARCHIVED')
            ORDER BY updated_at_utc DESC
            """
        )
        with self._connect() as connection:
            rows = connection.execute(query, (owner_user_id,)).fetchall()
        return [self._corpus_public(row) for row in rows]

    def sanitized_inventory(self) -> list[JsonObject]:
        """Return aggregate corpus records without owner IDs, names, paths, or hashes."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.corpus_id, c.owner_pseudonym, c.state, c.revision,
                       c.created_at_utc, c.updated_at_utc,
                       COUNT(s.source_id) AS source_count,
                       COALESCE(SUM(s.size_bytes), 0) AS size_bytes
                FROM corpora AS c
                LEFT JOIN sources AS s
                  ON s.corpus_id=c.corpus_id
                 AND s.owner_user_id=c.owner_user_id
                 AND s.state != 'DELETED'
                WHERE c.state != 'DELETED'
                GROUP BY c.corpus_id, c.owner_pseudonym, c.state, c.revision,
                         c.created_at_utc, c.updated_at_utc
                ORDER BY c.updated_at_utc DESC
                """
            ).fetchall()
        return [
            {
                "corpus_id": str(row["corpus_id"]),
                "owner_pseudonym": str(row["owner_pseudonym"]),
                "state": str(row["state"]),
                "revision": int(row["revision"]),
                "source_count": int(row["source_count"]),
                "size_bytes": int(row["size_bytes"]),
                "created_at_utc": str(row["created_at_utc"]),
                "updated_at_utc": str(row["updated_at_utc"]),
                "content_access": "DISABLED_BY_POLICY",
            }
            for row in rows
        ]

    def _corpus_public(self, row: sqlite3.Row) -> JsonObject:
        with self._connect() as connection:
            counts = connection.execute(
                """
                SELECT COUNT(*) AS source_count,
                       COALESCE(SUM(size_bytes), 0) AS size_bytes
                FROM sources
                WHERE corpus_id=? AND owner_user_id=? AND state != 'DELETED'
                """,
                (str(row["corpus_id"]), str(row["owner_user_id"])),
            ).fetchone()
        return {
            "corpus_id": str(row["corpus_id"]),
            "name": str(row["name"]),
            "state": str(row["state"]),
            "revision": int(row["revision"]),
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
            "source_count": int(counts["source_count"]) if counts else 0,
            "size_bytes": int(counts["size_bytes"]) if counts else 0,
            "owner": "You",
            "storage_class": "private external state",
            "retention": f"soft delete {self.policy.soft_delete_days} days",
        }

    def update_corpus(
        self,
        owner_user_id: str,
        corpus_id: str,
        *,
        name: str | None = None,
        archived: bool | None = None,
    ) -> JsonObject:
        current = self.require_corpus(owner_user_id, corpus_id)
        effective_name = str(current["name"]) if name is None else unicodedata.normalize(
            "NFC", name.strip()
        )
        if not effective_name or len(effective_name) > 160:
            raise CorpusError("invalid_corpus_name")
        state = str(current["state"])
        if archived is True:
            state = "ARCHIVED"
        elif archived is False and state == "ARCHIVED":
            state = "INDEXED" if int(str(current["source_count"])) else "EMPTY"
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE corpora SET name=?, state=?, revision=revision+1,
                        updated_at_utc=?
                    WHERE corpus_id=? AND owner_user_id=?
                    """,
                    (effective_name, state, _now(), corpus_id, owner_user_id),
                )
        except sqlite3.IntegrityError as exc:
            raise CorpusError("corpus_name_exists") from exc
        self._event("CORPUS_UPDATE", owner_user_id=owner_user_id, corpus_id=corpus_id)
        return self.require_corpus(owner_user_id, corpus_id)

    def delete_corpus(self, owner_user_id: str, corpus_id: str) -> JsonObject:
        self.require_corpus(owner_user_id, corpus_id)
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE chunks SET active=0
                WHERE owner_user_id=? AND corpus_id=?
                """,
                (owner_user_id, corpus_id),
            )
            connection.execute(
                """
                UPDATE sources SET state='DELETED', deleted_at_utc=?,
                    updated_at_utc=?
                WHERE owner_user_id=? AND corpus_id=?
                """,
                (now, now, owner_user_id, corpus_id),
            )
            connection.execute(
                """
                UPDATE corpora SET state='DELETED', deleted_at_utc=?,
                    updated_at_utc=?, revision=revision+1
                WHERE owner_user_id=? AND corpus_id=?
                """,
                (now, now, owner_user_id, corpus_id),
            )
        self._event("CORPUS_DELETE", owner_user_id=owner_user_id, corpus_id=corpus_id)
        return {"status": "SOFT_DELETED", "corpus_id": corpus_id}

    def create_upload(
        self,
        owner_user_id: str,
        corpus_id: str,
        *,
        idempotency_key: str,
    ) -> JsonObject:
        self.require_corpus(owner_user_id, corpus_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}", idempotency_key):
            raise CorpusError("invalid_idempotency_key")
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT upload_id, state FROM upload_sessions
                WHERE owner_user_id=? AND idempotency_key=?
                """,
                (owner_user_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return self.upload_manifest(
                    owner_user_id, str(existing["upload_id"])
                )
            upload_id = f"up_{uuid.uuid4().hex}"
            now = _now()
            connection.execute(
                """
                INSERT INTO upload_sessions (
                    upload_id, owner_user_id, corpus_id, state,
                    idempotency_key, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, 'STAGING', ?, ?, ?)
                """,
                (upload_id, owner_user_id, corpus_id, idempotency_key, now, now),
            )
        staging = self._owner_root(owner_user_id) / corpus_id / "staging" / upload_id
        staging.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._event(
            "UPLOAD_CREATE",
            owner_user_id=owner_user_id,
            corpus_id=corpus_id,
            details={"upload_id": upload_id},
        )
        return self.upload_manifest(owner_user_id, upload_id)

    def _upload_row(self, owner_user_id: str, upload_id: str) -> sqlite3.Row:
        if not re.fullmatch(r"up_[a-f0-9]{32}", upload_id):
            raise CorpusError("upload_not_found")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM upload_sessions
                WHERE upload_id=? AND owner_user_id=?
                """,
                (upload_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise CorpusError("upload_not_found")
        return cast(sqlite3.Row, row)

    def stage_file(
        self,
        owner_user_id: str,
        upload_id: str,
        *,
        logical_path: str,
        content: bytes,
        client_mime: str = "",
    ) -> JsonObject:
        upload = self._upload_row(owner_user_id, upload_id)
        if str(upload["state"]) != "STAGING":
            raise CorpusError("upload_not_staging")
        try:
            normalized = _normalized_logical_path(logical_path, self.policy)
            _validate_extension(normalized)
            if len(content) > self.policy.max_file_bytes:
                raise CorpusError("file_size_quota")
            media_type = _validate_magic(normalized, content, client_mime)
            if PurePosixPath(normalized).suffix.lower() in {
                ".pdf",
                ".docx",
                ".json",
                ".jsonl",
            }:
                _extract_bounded(normalized, content, self.policy)
            state = "ACCEPTED"
            reason: str | None = None
            digest = hashlib.sha256(content).hexdigest()
            warnings = _secret_warnings(content)
        except CorpusError as exc:
            normalized = unicodedata.normalize("NFC", logical_path.strip())[:1_024]
            state = "REJECTED"
            reason = exc.category
            media_type = ""
            digest = None
            warnings = []
        normalized_key = unicodedata.normalize("NFKC", normalized).casefold()
        staging_name: str | None = None
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT logical_path, state, reason, size_bytes, content_sha256,
                       media_type, support_label, warnings_json
                FROM upload_files
                WHERE upload_id=? AND normalized_key=?
                """,
                (upload_id, normalized_key),
            ).fetchone()
            if existing is not None:
                if (
                    state == "ACCEPTED"
                    and str(existing["state"]) == "ACCEPTED"
                    and str(existing["logical_path"]) == normalized
                    and str(existing["content_sha256"]) == digest
                    and int(existing["size_bytes"]) == len(content)
                ):
                    existing_warnings = json.loads(str(existing["warnings_json"]))
                    self._event(
                        "UPLOAD_FILE_RESUME",
                        owner_user_id=owner_user_id,
                        corpus_id=str(upload["corpus_id"]),
                        details={"logical_path": normalized, "idempotent": True},
                    )
                    return {
                        "logical_path": normalized,
                        "state": "ACCEPTED",
                        "reason": None,
                        "size_bytes": len(content),
                        "content_sha256": digest,
                        "media_type": str(existing["media_type"]),
                        "support_label": str(existing["support_label"]),
                        "warnings": existing_warnings,
                        "idempotent": True,
                    }
                raise CorpusError("duplicate_or_normalization_collision")
            counts = connection.execute(
                """
                SELECT COUNT(*) AS count, COALESCE(SUM(size_bytes), 0) AS size
                FROM upload_files WHERE upload_id=?
                """,
                (upload_id,),
            ).fetchone()
            if counts and int(counts["count"]) >= self.policy.max_files_per_batch:
                raise CorpusError("file_count_quota")
            if counts and int(counts["size"]) + len(content) > self.policy.max_batch_bytes:
                raise CorpusError("batch_size_quota")
            if state == "ACCEPTED":
                self._require_disk_space(len(content))
                self._require_owner_quota(connection, owner_user_id, len(content))
                staging_name = f"{uuid.uuid4().hex}.source"
                target = (
                    self._owner_root(owner_user_id)
                    / str(upload["corpus_id"])
                    / "staging"
                    / upload_id
                    / staging_name
                )
                _atomic_private_bytes(target, content)
            try:
                connection.execute(
                    """
                    INSERT INTO upload_files (
                        upload_id, logical_path, normalized_key, state, reason,
                        size_bytes, content_sha256, media_type, support_label,
                        warnings_json, staging_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_id,
                        normalized,
                        normalized_key,
                        state,
                        reason,
                        len(content),
                        digest,
                        media_type,
                        _support_label(normalized),
                        json.dumps(warnings, separators=(",", ":")),
                        staging_name,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if staging_name:
                    (
                        self._owner_root(owner_user_id)
                        / str(upload["corpus_id"])
                        / "staging"
                        / upload_id
                        / staging_name
                    ).unlink(missing_ok=True)
                raise CorpusError("duplicate_or_normalization_collision") from exc
            connection.execute(
                """
                UPDATE upload_sessions SET updated_at_utc=?
                WHERE upload_id=?
                """,
                (_now(), upload_id),
            )
        self._event(
            "UPLOAD_FILE_VALIDATE",
            owner_user_id=owner_user_id,
            corpus_id=str(upload["corpus_id"]),
            outcome=state,
            details={"logical_path": normalized, "reason": reason},
        )
        return {
            "logical_path": normalized,
            "state": state,
            "reason": reason,
            "size_bytes": len(content),
            "content_sha256": digest,
            "media_type": media_type,
            "support_label": _support_label(normalized),
            "warnings": warnings,
        }

    def stage_zip_fallback(
        self,
        owner_user_id: str,
        upload_id: str,
        *,
        content: bytes,
    ) -> JsonObject:
        if len(content) > self.policy.max_batch_bytes:
            raise CorpusError("batch_size_quota")
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise CorpusError("invalid_zip_fallback") from exc
        with archive:
            entries = archive.infolist()
            if len(entries) > self.policy.max_zip_entries:
                raise CorpusError("zip_entry_count_quota")
            total = 0
            for entry in entries:
                mode = entry.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode) or file_type not in {
                    0,
                    stat.S_IFREG,
                    stat.S_IFDIR,
                }:
                    raise CorpusError("zip_special_entry_rejected")
                if entry.is_dir():
                    continue
                normalized = _normalized_logical_path(entry.filename, self.policy)
                if PurePosixPath(normalized).suffix.lower() in {".zip", ".7z", ".tar", ".gz"}:
                    raise CorpusError("nested_archive_rejected")
                total += entry.file_size
                if total > self.policy.max_batch_bytes:
                    raise CorpusError("zip_expanded_size_quota")
                if (
                    entry.compress_size == 0
                    and entry.file_size > 0
                    or entry.compress_size > 0
                    and entry.file_size / entry.compress_size > self.policy.max_zip_ratio
                ):
                    raise CorpusError("zip_decompression_ratio")
            accepted = 0
            rejected = 0
            for entry in entries:
                if entry.is_dir():
                    continue
                with archive.open(entry, "r") as handle:
                    data = handle.read(self.policy.max_file_bytes + 1)
                result = self.stage_file(
                    owner_user_id,
                    upload_id,
                    logical_path=entry.filename,
                    content=data,
                    client_mime="",
                )
                if result["state"] == "ACCEPTED":
                    accepted += 1
                else:
                    rejected += 1
        return {
            "status": "ZIP_INSPECTED",
            "accepted": accepted,
            "rejected": rejected,
            "manifest": self.upload_manifest(owner_user_id, upload_id),
        }

    def _require_disk_space(self, incoming: int) -> None:
        usage = shutil.disk_usage(self.root)
        if usage.free - incoming < self.policy.min_free_disk_bytes:
            raise CorpusError("disk_pressure")

    def _require_owner_quota(
        self, connection: sqlite3.Connection, owner_user_id: str, incoming: int
    ) -> None:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(size_bytes), 0) AS size
            FROM sources WHERE owner_user_id=? AND state != 'DELETED'
            """,
            (owner_user_id,),
        ).fetchone()
        if int(row["size"]) + incoming > self.policy.max_user_bytes:
            raise CorpusError("user_storage_quota")

    def upload_manifest(self, owner_user_id: str, upload_id: str) -> JsonObject:
        upload = self._upload_row(owner_user_id, upload_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT logical_path, state, reason, size_bytes, content_sha256,
                       media_type, support_label, warnings_json
                FROM upload_files WHERE upload_id=? ORDER BY logical_path
                """,
                (upload_id,),
            ).fetchall()
        files = [
            {
                "logical_path": str(row["logical_path"]),
                "state": str(row["state"]),
                "reason": row["reason"],
                "size_bytes": int(row["size_bytes"]),
                "content_sha256": row["content_sha256"],
                "media_type": row["media_type"],
                "support_label": row["support_label"],
                "warnings": json.loads(str(row["warnings_json"])),
            }
            for row in rows
        ]
        return {
            "upload_id": upload_id,
            "corpus_id": str(upload["corpus_id"]),
            "state": str(upload["state"]),
            "accepted_count": sum(item["state"] == "ACCEPTED" for item in files),
            "rejected_count": sum(item["state"] == "REJECTED" for item in files),
            "files": files,
            "requires_index_confirmation": str(upload["state"]) == "STAGING",
        }

    def list_uploads(
        self, owner_user_id: str, *, state: str = "STAGING"
    ) -> list[JsonObject]:
        _identifier(owner_user_id, label="user_id")
        if state not in {"STAGING", "CANCELLED", "INDEXED"}:
            raise CorpusError("invalid_upload_state")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT upload_id FROM upload_sessions
                WHERE owner_user_id=? AND state=?
                ORDER BY updated_at_utc DESC LIMIT 20
                """,
                (owner_user_id, state),
            ).fetchall()
        return [
            self.upload_manifest(owner_user_id, str(row["upload_id"]))
            for row in rows
        ]

    def cancel_upload(self, owner_user_id: str, upload_id: str) -> JsonObject:
        upload = self._upload_row(owner_user_id, upload_id)
        if str(upload["state"]) in {"INDEXED", "CANCELLED"}:
            return {"status": str(upload["state"]), "upload_id": upload_id}
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE upload_sessions SET state='CANCELLED', updated_at_utc=?
                WHERE upload_id=? AND owner_user_id=?
                """,
                (_now(), upload_id, owner_user_id),
            )
        staging = (
            self._owner_root(owner_user_id)
            / str(upload["corpus_id"])
            / "staging"
            / upload_id
        )
        shutil.rmtree(staging, ignore_errors=True)
        self._event(
            "UPLOAD_CANCEL",
            owner_user_id=owner_user_id,
            corpus_id=str(upload["corpus_id"]),
            details={"upload_id": upload_id},
        )
        return {"status": "CANCELLED", "upload_id": upload_id}

    def index_upload(
        self,
        owner_user_id: str,
        upload_id: str,
        *,
        idempotency_key: str,
    ) -> JsonObject:
        cached = self._idempotent(owner_user_id, "index_upload", idempotency_key)
        if cached is not None:
            return cached
        upload = self._upload_row(owner_user_id, upload_id)
        if str(upload["state"]) == "CANCELLED":
            raise CorpusError("upload_cancelled")
        if str(upload["state"]) == "INDEXED":
            result = self.upload_manifest(owner_user_id, upload_id)
            self._remember(owner_user_id, "index_upload", idempotency_key, result)
            return result
        corpus_id = str(upload["corpus_id"])
        corpus = self.require_corpus(owner_user_id, corpus_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM upload_files
                WHERE upload_id=? AND state='ACCEPTED'
                ORDER BY logical_path
                """,
                (upload_id,),
            ).fetchall()
        if not rows:
            raise CorpusError("no_accepted_files")
        extracted_total = 0
        promoted: list[tuple[str, Path]] = []
        indexed = 0
        deduplicated = 0
        try:
            for row in rows:
                staging = (
                    self._owner_root(owner_user_id)
                    / corpus_id
                    / "staging"
                    / upload_id
                    / str(row["staging_name"])
                )
                content = staging.read_bytes()
                extracted = _extract_bounded(
                    str(row["logical_path"]), content, self.policy
                )
                extracted_bytes = extracted.text.encode("utf-8")
                extracted_total += len(extracted_bytes)
                if extracted_total > self.policy.max_extracted_text_bytes:
                    raise CorpusError("extracted_text_quota")
                digest = str(row["content_sha256"])
                with self._connect() as connection:
                    duplicate = connection.execute(
                        """
                        SELECT source_id FROM sources
                        WHERE owner_user_id=? AND corpus_id=? AND content_sha256=?
                          AND state != 'DELETED'
                        """,
                        (owner_user_id, corpus_id, digest),
                    ).fetchone()
                if duplicate is not None:
                    deduplicated += 1
                    staging.unlink(missing_ok=True)
                    continue
                source_id = f"src_{uuid.uuid4().hex}"
                storage_name = f"{source_id}.source"
                target = self._owner_root(owner_user_id) / corpus_id / "sources" / storage_name
                os.replace(staging, target)
                os.chmod(target, 0o600)
                promoted.append((source_id, target))
                extracted_hash = hashlib.sha256(extracted_bytes).hexdigest()
                chunks = list(
                    self._chunks(
                        extracted,
                        owner_user_id=owner_user_id,
                        corpus_id=corpus_id,
                        source_id=source_id,
                        snapshot_revision=int(str(corpus["revision"])) + 1,
                    )
                )
                now = _now()
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        """
                        INSERT INTO sources (
                            source_id, owner_user_id, corpus_id, logical_path,
                            media_type, support_label, size_bytes, content_sha256,
                            extracted_sha256, replacement_count, state, storage_name,
                            created_at_utc, updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'INDEXED', ?, ?, ?)
                        """,
                        (
                            source_id,
                            owner_user_id,
                            corpus_id,
                            str(row["logical_path"]),
                            str(row["media_type"]),
                            str(row["support_label"]),
                            int(row["size_bytes"]),
                            digest,
                            extracted_hash,
                            extracted.replacement_count,
                            storage_name,
                            now,
                            now,
                        ),
                    )
                    connection.executemany(
                        """
                        INSERT INTO chunks (
                            chunk_id, owner_user_id, corpus_id, source_id,
                            snapshot_revision, ordinal, page_number, section,
                            text, text_sha256, active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        chunks,
                    )
                indexed += 1
                self._event(
                    "SOURCE_INDEX",
                    owner_user_id=owner_user_id,
                    corpus_id=corpus_id,
                    source_id=source_id,
                    details={
                        "logical_path": str(row["logical_path"]),
                        "chunk_count": len(chunks),
                        "content_sha256": digest,
                        "extracted_sha256": extracted_hash,
                    },
                )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE upload_sessions SET state='INDEXED', updated_at_utc=?
                    WHERE upload_id=? AND owner_user_id=?
                    """,
                    (_now(), upload_id, owner_user_id),
                )
                connection.execute(
                    """
                    UPDATE corpora SET state='INDEXED', revision=revision+1,
                        updated_at_utc=?
                    WHERE corpus_id=? AND owner_user_id=?
                    """,
                    (_now(), corpus_id, owner_user_id),
                )
        except Exception:
            # Files already inserted are valid indexed sources. Any promoted file whose
            # transaction failed is returned to quarantine for safe retry.
            with self._connect() as connection:
                for source_id, target in promoted:
                    exists = connection.execute(
                        "SELECT 1 FROM sources WHERE source_id=?", (source_id,)
                    ).fetchone()
                    if exists is None and target.exists():
                        quarantine = (
                            self._owner_root(owner_user_id)
                            / corpus_id
                            / "staging"
                            / upload_id
                            / f"recovered-{target.name}"
                        )
                        os.replace(target, quarantine)
            raise
        staging_root = (
            self._owner_root(owner_user_id) / corpus_id / "staging" / upload_id
        )
        shutil.rmtree(staging_root, ignore_errors=True)
        result = {
            "status": "INDEXED",
            "upload_id": upload_id,
            "corpus_id": corpus_id,
            "indexed_sources": indexed,
            "deduplicated_sources": deduplicated,
            "snapshot_revision": int(str(corpus["revision"])) + 1,
            "chunking_version": CHUNKING_VERSION,
        }
        self._remember(owner_user_id, "index_upload", idempotency_key, result)
        return result

    def _chunks(
        self,
        extracted: ExtractedDocument,
        *,
        owner_user_id: str,
        corpus_id: str,
        source_id: str,
        snapshot_revision: int,
    ) -> Iterable[tuple[object, ...]]:
        text = extracted.text
        if not text:
            return
        target = self.policy.chunk_target_characters
        overlap = self.policy.chunk_overlap_characters
        cursor = 0
        ordinal = 0
        while cursor < len(text):
            end = min(len(text), cursor + target)
            if end < len(text):
                newline = text.rfind("\n", cursor + target // 2, end)
                if newline > cursor:
                    end = newline + 1
            value = text[cursor:end]
            page_number = next(
                (
                    page
                    for start, stop, page in extracted.page_spans
                    if start <= cursor < max(start + 1, stop)
                ),
                None,
            )
            chunk_id = f"chk_{hashlib.sha256(f'{source_id}:{ordinal}:{value}'.encode()).hexdigest()[:32]}"
            yield (
                chunk_id,
                owner_user_id,
                corpus_id,
                source_id,
                snapshot_revision,
                ordinal,
                page_number,
                extracted.section,
                value,
                hashlib.sha256(value.encode("utf-8")).hexdigest(),
            )
            ordinal += 1
            if end >= len(text):
                break
            cursor = max(cursor + 1, end - overlap)

    def list_sources(self, owner_user_id: str, corpus_id: str) -> list[JsonObject]:
        self.require_corpus(owner_user_id, corpus_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, logical_path, media_type, support_label,
                       size_bytes, content_sha256, extracted_sha256,
                       replacement_count, state, created_at_utc
                FROM sources
                WHERE owner_user_id=? AND corpus_id=? AND state != 'DELETED'
                ORDER BY logical_path
                """,
                (owner_user_id, corpus_id),
            ).fetchall()
        return [
            {
                "source_id": str(row["source_id"]),
                "name": str(row["logical_path"]),
                "type": str(row["media_type"]),
                "support_label": str(row["support_label"]),
                "size_bytes": int(row["size_bytes"]),
                "hash": str(row["content_sha256"]),
                "hash_short": str(row["content_sha256"])[:12],
                "extracted_sha256": row["extracted_sha256"],
                "replacement_count": int(row["replacement_count"]),
                "owner": "You",
                "storage_class": "private external state",
                "retention": f"soft delete {self.policy.soft_delete_days} days",
                "indexing_state": str(row["state"]),
                "created_at_utc": str(row["created_at_utc"]),
            }
            for row in rows
        ]

    def delete_source(
        self, owner_user_id: str, corpus_id: str, source_id: str
    ) -> JsonObject:
        self.require_corpus(owner_user_id, corpus_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_id FROM sources
                WHERE source_id=? AND owner_user_id=? AND corpus_id=?
                  AND state != 'DELETED'
                """,
                (source_id, owner_user_id, corpus_id),
            ).fetchone()
            if row is None:
                raise CorpusError("source_not_found")
            now = _now()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE chunks SET active=0 WHERE source_id=? AND owner_user_id=?",
                (source_id, owner_user_id),
            )
            connection.execute(
                """
                UPDATE sources SET state='DELETED', deleted_at_utc=?,
                    updated_at_utc=?
                WHERE source_id=? AND owner_user_id=?
                """,
                (now, now, source_id, owner_user_id),
            )
            connection.execute(
                """
                UPDATE corpora SET revision=revision+1, updated_at_utc=?
                WHERE corpus_id=? AND owner_user_id=?
                """,
                (now, corpus_id, owner_user_id),
            )
        self._event(
            "SOURCE_DELETE",
            owner_user_id=owner_user_id,
            corpus_id=corpus_id,
            source_id=source_id,
        )
        return {"status": "SOFT_DELETED", "source_id": source_id}

    def source_content(
        self, owner_user_id: str, corpus_id: str, source_id: str
    ) -> tuple[str, str, bytes]:
        self.require_corpus(owner_user_id, corpus_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT logical_path, media_type, storage_name
                FROM sources
                WHERE source_id=? AND owner_user_id=? AND corpus_id=?
                  AND state != 'DELETED'
                """,
                (source_id, owner_user_id, corpus_id),
            ).fetchone()
        if row is None:
            raise CorpusError("source_not_found")
        path = (
            self._owner_root(owner_user_id)
            / corpus_id
            / "sources"
            / str(row["storage_name"])
        )
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise CorpusError("source_content_unavailable") from exc
        return str(row["logical_path"]), str(row["media_type"]), content

    def reindex_corpus(
        self,
        owner_user_id: str,
        corpus_id: str,
        *,
        idempotency_key: str,
    ) -> JsonObject:
        cached = self._idempotent(owner_user_id, "reindex_corpus", idempotency_key)
        if cached is not None:
            return cached
        corpus = self.require_corpus(owner_user_id, corpus_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sources
                WHERE owner_user_id=? AND corpus_id=? AND state='INDEXED'
                ORDER BY source_id
                """,
                (owner_user_id, corpus_id),
            ).fetchall()
        revision = int(str(corpus["revision"])) + 1
        indexed_chunks = 0
        for row in rows:
            path = (
                self._owner_root(owner_user_id)
                / corpus_id
                / "sources"
                / str(row["storage_name"])
            )
            extracted = _extract_bounded(
                str(row["logical_path"]), path.read_bytes(), self.policy
            )
            chunks = list(
                self._chunks(
                    extracted,
                    owner_user_id=owner_user_id,
                    corpus_id=corpus_id,
                    source_id=str(row["source_id"]),
                    snapshot_revision=revision,
                )
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM chunks WHERE source_id=?",
                    (str(row["source_id"]),),
                )
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, owner_user_id, corpus_id, source_id,
                        snapshot_revision, ordinal, page_number, section,
                        text, text_sha256, active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    chunks,
                )
            indexed_chunks += len(chunks)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE corpora SET revision=?, updated_at_utc=?
                WHERE corpus_id=? AND owner_user_id=?
                """,
                (revision, _now(), corpus_id, owner_user_id),
            )
        result = {
            "status": "REINDEXED",
            "corpus_id": corpus_id,
            "snapshot_revision": revision,
            "source_count": len(rows),
            "chunk_count": indexed_chunks,
            "chunking_version": CHUNKING_VERSION,
        }
        self._remember(owner_user_id, "reindex_corpus", idempotency_key, result)
        self._event(
            "CORPUS_REINDEX",
            owner_user_id=owner_user_id,
            corpus_id=corpus_id,
            details={"source_count": len(rows), "chunk_count": indexed_chunks},
        )
        return result

    def search(
        self,
        owner_user_id: str,
        query: str,
        *,
        corpus_id: str | None = None,
        limit: int = 8,
    ) -> JsonObject:
        normalized_query = query.strip()
        if not normalized_query or len(normalized_query) > 4_000:
            raise CorpusError("invalid_search_query")
        if not 1 <= limit <= 50:
            raise CorpusError("invalid_search_limit")
        if corpus_id is not None:
            corpus = self.require_corpus(owner_user_id, corpus_id)
            if corpus["state"] != "INDEXED":
                raise CorpusError("corpus_not_indexed")
        terms = tuple(
            dict.fromkeys(re.findall(r"[A-Za-z0-9_]{2,}", normalized_query.lower()))
        )[:32]
        if not terms:
            raise CorpusError("invalid_search_query")
        sql = (
            """
            SELECT c.chunk_id, c.corpus_id, c.source_id, c.snapshot_revision,
                   c.ordinal, c.page_number, c.section, c.text,
                   s.logical_path, s.content_sha256
            FROM chunks AS c JOIN sources AS s ON s.source_id=c.source_id
            WHERE c.owner_user_id=? AND c.active=1 AND s.state='INDEXED'
            """
            if corpus_id is None
            else """
            SELECT c.chunk_id, c.corpus_id, c.source_id, c.snapshot_revision,
                   c.ordinal, c.page_number, c.section, c.text,
                   s.logical_path, s.content_sha256
            FROM chunks AS c JOIN sources AS s ON s.source_id=c.source_id
            WHERE c.owner_user_id=? AND c.active=1 AND s.state='INDEXED'
              AND c.corpus_id=?
            """
        )
        parameters: tuple[object, ...] = (
            (owner_user_id, corpus_id)
            if corpus_id is not None
            else (owner_user_id,)
        )
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        scored: list[tuple[int, sqlite3.Row]] = []
        for row in rows:
            lowered = str(row["text"]).lower()
            score = sum(lowered.count(term) for term in terms)
            if score:
                scored.append((score, row))
        scored.sort(
            key=lambda item: (
                -item[0],
                str(item[1]["logical_path"]),
                int(item[1]["ordinal"]),
            )
        )
        results = [
            {
                "corpus_id": str(row["corpus_id"]),
                "source_id": str(row["source_id"]),
                "source_title": str(row["logical_path"]),
                "file": str(row["logical_path"]),
                "page": row["page_number"],
                "section": row["section"],
                "chunk_id": str(row["chunk_id"]),
                "chunk_ordinal": int(row["ordinal"]),
                "snapshot_revision": int(row["snapshot_revision"]),
                "content_sha256": str(row["content_sha256"]),
                "score": score,
                "text": str(row["text"]),
                "citation": {
                    "file": str(row["logical_path"]),
                    "page": row["page_number"],
                    "section": row["section"],
                    "chunk_id": str(row["chunk_id"]),
                },
            }
            for score, row in scored[:limit]
        ]
        snapshots = sorted(
            {
                (str(item["corpus_id"]), int(item["snapshot_revision"]))
                for item in results
            }
        )
        self._event(
            "CORPUS_RETRIEVAL",
            owner_user_id=owner_user_id,
            corpus_id=corpus_id,
            details={"result_count": len(results), "query_sha256": hashlib.sha256(normalized_query.encode()).hexdigest()},
        )
        return {
            "retrieval_used": bool(results),
            "selection": (
                RetrievalSelection.SELECTED_PERSONAL.value
                if corpus_id is not None
                else RetrievalSelection.PERSONAL.value
            ),
            "snapshots": [
                {"corpus_id": item[0], "snapshot_revision": item[1]}
                for item in snapshots
            ],
            "results": results,
        }

    def purge_deleted(self, *, before_utc: str) -> JsonObject:
        """Purge soft-deleted content. Intended for an audited local maintenance job."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, owner_user_id, corpus_id, storage_name
                FROM sources
                WHERE state='DELETED' AND deleted_at_utc <= ?
                """,
                (before_utc,),
            ).fetchall()
            connection.execute("BEGIN IMMEDIATE")
            for row in rows:
                connection.execute(
                    "DELETE FROM chunks WHERE source_id=?", (str(row["source_id"]),)
                )
                connection.execute(
                    "DELETE FROM sources WHERE source_id=?", (str(row["source_id"]),)
                )
        for row in rows:
            target = (
                self._owner_root(str(row["owner_user_id"]))
                / str(row["corpus_id"])
                / "sources"
                / str(row["storage_name"])
            )
            target.unlink(missing_ok=True)
            self._event(
                "SOURCE_PURGE",
                owner_user_id=str(row["owner_user_id"]),
                corpus_id=str(row["corpus_id"]),
                source_id=str(row["source_id"]),
            )
        return {"status": "PURGED", "source_count": len(rows)}

    def health(self) -> JsonObject:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT schema_version, chunking_version FROM corpus_schema WHERE singleton=1"
                ).fetchone()
            usage = shutil.disk_usage(self.root)
            ready = (
                row is not None
                and int(row["schema_version"]) == POLICY_SCHEMA_VERSION
                and usage.free >= self.policy.min_free_disk_bytes
            )
            return {
                "status": "READY" if ready else "DEGRADED",
                "schema_version": int(row["schema_version"]) if row else None,
                "chunking_version": str(row["chunking_version"]) if row else None,
                "free_bytes": usage.free,
                "minimum_free_bytes": self.policy.min_free_disk_bytes,
            }
        except (OSError, sqlite3.Error) as exc:
            return {"status": "DEGRADED", "reason": type(exc).__name__}

    def _idempotent(
        self, owner_user_id: str, operation: str, key: str
    ) -> JsonObject | None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}", key):
            raise CorpusError("invalid_idempotency_key")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_json FROM idempotency
                WHERE owner_user_id=? AND operation=? AND idempotency_key=?
                """,
                (owner_user_id, operation, key),
            ).fetchone()
        if row is None:
            return None
        value: object = json.loads(str(row["response_json"]))
        if not isinstance(value, dict):
            raise CorpusError("invalid_idempotency_record")
        return dict(value)

    def _remember(
        self, owner_user_id: str, operation: str, key: str, response: JsonObject
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO idempotency (
                    owner_user_id, operation, idempotency_key,
                    response_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_user_id, operation, idempotency_key) DO NOTHING
                """,
                (
                    owner_user_id,
                    operation,
                    key,
                    json.dumps(response, sort_keys=True, separators=(",", ":")),
                    _now(),
                ),
            )
