"""Deterministic security validation helpers used by the v7 adversarial suite."""

from __future__ import annotations

import hashlib
import html
import hmac
import re
import stat
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlsplit

ARCHIVE_MAX_FILES: Final = 2_000
ARCHIVE_MAX_EXPANDED_BYTES: Final = 256 * 1024 * 1024
ARCHIVE_MAX_RATIO: Final = 200


class SecurityValidationError(ValueError):
    """Fail-closed security validation error with no private payload."""


def safe_log_text(value: str, *, limit: int = 512) -> str:
    """Return bounded single-line log text with credential-like values redacted."""

    collapsed = "".join(character if ord(character) >= 32 else " " for character in value)
    collapsed = re.sub(
        r"(?i)\b(password|token|secret|api[_-]?key)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        collapsed,
    )
    return collapsed[:limit]


def canonical_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    if not normalized or len(normalized) > 160 or any(ord(char) < 32 for char in normalized):
        raise SecurityValidationError("identifier is invalid")
    return normalized


class IdentifierRegistry:
    """Reject different originals that collapse to the same Unicode identifier."""

    def __init__(self) -> None:
        self._originals: dict[str, str] = {}

    def add(self, value: str) -> str:
        canonical = canonical_identifier(value)
        original = self._originals.get(canonical)
        if original is not None and original != value:
            raise SecurityValidationError("identifier normalization collision")
        self._originals[canonical] = value
        return canonical


def validate_markdown(markdown: str) -> str:
    if len(markdown.encode("utf-8")) > 4_000_000:
        raise SecurityValidationError("Markdown is too large")
    if re.search(r"<\s*/?\s*(?:script|iframe|object|embed|style|svg)\b", markdown, re.I):
        raise SecurityValidationError("unsafe Markdown HTML")
    for raw_target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)", markdown):
        target = html.unescape(raw_target).strip()
        parsed = urlsplit(target)
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            raise SecurityValidationError("unsafe Markdown link")
        if target.startswith(("//", "\\")) or "\x00" in target:
            raise SecurityValidationError("unsafe Markdown link")
    return markdown


def untrusted_source_envelope(source_id: str, content: str) -> dict[str, object]:
    """Represent source text as inert data, including prompt-injection-like text."""

    if not source_id or len(content.encode("utf-8")) > 8 * 1024 * 1024:
        raise SecurityValidationError("source is invalid")
    return {
        "source_id": source_id,
        "classification": "untrusted_document_content",
        "content": content,
        "executable": False,
        "instructions_authoritative": False,
    }


def _safe_archive_name(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name:
        raise SecurityValidationError("unsafe archive member")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SecurityValidationError("unsafe archive member")
    return path


def validate_zip_archive(path: Path, *, required_member: str | None = None) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > ARCHIVE_MAX_FILES:
                raise SecurityValidationError("archive member limit exceeded")
            expanded = sum(member.file_size for member in members)
            compressed = sum(max(1, member.compress_size) for member in members)
            if expanded > ARCHIVE_MAX_EXPANDED_BYTES:
                raise SecurityValidationError("archive expanded-size limit exceeded")
            if expanded > compressed * ARCHIVE_MAX_RATIO:
                raise SecurityValidationError("archive expansion ratio exceeded")
            names: list[str] = []
            for member in members:
                logical = _safe_archive_name(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise SecurityValidationError("archive links are forbidden")
                names.append(logical.as_posix())
    except (OSError, zipfile.BadZipFile) as exc:
        raise SecurityValidationError("archive is malformed") from exc
    if required_member is not None and required_member not in names:
        raise SecurityValidationError("archive required member is missing")
    return tuple(names)


def validate_document_upload(path: Path, media_type: str) -> None:
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise SecurityValidationError("document size or type is invalid")
    if media_type == "application/pdf":
        with path.open("rb") as handle:
            header = handle.read(5)
            handle.seek(max(0, path.stat().st_size - 1024))
            trailer = handle.read()
        if header != b"%PDF-" or b"%%EOF" not in trailer:
            raise SecurityValidationError("PDF is malformed")
        return
    if (
        media_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        validate_zip_archive(path, required_member="word/document.xml")
        return
    raise SecurityValidationError("document media type is not permitted")


def validate_repository_tree(root: Path, relative_path: str) -> tuple[int, int]:
    logical = PurePosixPath(relative_path)
    if (
        logical.is_absolute()
        or ".." in logical.parts
        or ".git" in logical.parts
        or not logical.parts
    ):
        raise SecurityValidationError("repository path is unsafe")
    base = root.resolve()
    target = base.joinpath(*logical.parts)
    before = target.lstat()
    resolved = target.resolve()
    if not resolved.is_relative_to(base):
        raise SecurityValidationError("repository path escaped root")
    if stat.S_ISLNK(before.st_mode) or before.st_nlink > 1:
        raise SecurityValidationError("repository links are forbidden")
    for parent in (target, *target.parents):
        if parent == base:
            break
        if (parent / ".git").exists():
            raise SecurityValidationError("nested repository is forbidden")
    after = target.lstat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise SecurityValidationError("repository entry changed during validation")
    return int(after.st_dev), int(after.st_ino)


def validate_browser_request(
    *,
    host: str,
    origin: str | None,
    expected_host: str,
    expected_origin: str,
    unsafe_method: bool,
    csrf_cookie: str | None,
    csrf_header: str | None,
) -> None:
    if host != expected_host:
        raise SecurityValidationError("Host validation failed")
    if origin is not None and origin != expected_origin:
        raise SecurityValidationError("Origin validation failed")
    if unsafe_method and (
        not csrf_cookie
        or not csrf_header
        or not hmac.compare_digest(
            hashlib.sha256(csrf_cookie.encode()).digest(),
            hashlib.sha256(csrf_header.encode()).digest(),
        )
    ):
        raise SecurityValidationError("CSRF validation failed")


def authorize_revision(
    *,
    authenticated_owner: str,
    resource_owner: str,
    current_revision: int,
    presented_revision: int,
    current_capabilities: frozenset[str],
    presented_capabilities: frozenset[str],
) -> None:
    if authenticated_owner != resource_owner:
        raise SecurityValidationError("resource not found")
    if current_revision != presented_revision:
        raise SecurityValidationError("authorization revision changed")
    if not presented_capabilities <= current_capabilities:
        raise SecurityValidationError("capability assertion is invalid")
    if presented_capabilities != current_capabilities:
        raise SecurityValidationError("capability downgrade requires reauthentication")


class ReplayGuard:
    """Idempotent replay guard: exact replay is accepted, mutation is rejected."""

    def __init__(self) -> None:
        self._digests: dict[tuple[str, int], str] = {}

    def accept(self, operation_id: str, sequence: int, payload: bytes) -> bool:
        if not operation_id or sequence < 0:
            raise SecurityValidationError("replay identity is invalid")
        key = (operation_id, sequence)
        digest = hashlib.sha256(payload).hexdigest()
        previous = self._digests.get(key)
        if previous is not None and previous != digest:
            raise SecurityValidationError("sync replay payload mismatch")
        self._digests[key] = digest
        return previous is not None


def validate_configuration_text(value: str) -> None:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise SecurityValidationError("configuration value contains control data")
    if len(value) > 512:
        raise SecurityValidationError("configuration value is too long")
