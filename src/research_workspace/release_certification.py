"""Safe deterministic evidence archive helpers for v7 CPU certification."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final

MAX_EVIDENCE_FILE_BYTES: Final = 64 * 1024 * 1024
FORBIDDEN_BUNDLE_PARTS: Final = frozenset(
    {
        ".git",
        ".env",
        "registries",
        "sessions",
        "user_uploads",
        "corpora",
        "worktrees",
        "credentials",
        "models",
    }
)
SECRET_PATTERNS: Final = (
    re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(rb"\bsk-[A-Za-z0-9]{32,}\b"),
)


class CertificationArchiveError(RuntimeError):
    """Certification evidence is incomplete or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> PurePosixPath:
    if "\x00" in name or "\\" in name:
        raise CertificationArchiveError("unsafe archive name")
    path = PurePosixPath(name)
    lowered = {part.casefold() for part in path.parts}
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or lowered & FORBIDDEN_BUNDLE_PARTS
    ):
        raise CertificationArchiveError("unsafe archive name")
    return path


def redact_private_paths(value: object, replacements: Mapping[str, str]) -> object:
    """Recursively redact known worktree/temp prefixes from evidence."""

    if isinstance(value, dict):
        return {
            str(key): redact_private_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_private_paths(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [redact_private_paths(item, replacements) for item in value]
    if isinstance(value, str):
        redacted = value
        for private, label in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if private:
                redacted = redacted.replace(private, label)
        return re.sub(r"/tmp/laplace-[^\s\"']+", "<temporary>", redacted)
    return value


def build_manifest(output: Path, names: Sequence[str]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for name in sorted(names):
        logical = _safe_name(name)
        path = output / logical.as_posix()
        if not path.is_file() or path.is_symlink():
            raise CertificationArchiveError(f"evidence missing or unsafe: {name}")
        size = path.stat().st_size
        if size > MAX_EVIDENCE_FILE_BYTES:
            raise CertificationArchiveError(f"evidence file too large: {name}")
        payload = path.read_bytes()
        if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
            raise CertificationArchiveError(f"secret-like material in evidence: {name}")
        entries.append(
            {
                "path": logical.as_posix(),
                "size_bytes": size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "files": entries,
        "hash_algorithm": "sha256",
        "unsafe_archive_paths": False,
        "secret_patterns_detected": False,
    }


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def create_archive(output: Path, names: Sequence[str], destination: Path) -> str:
    manifest_path = output / "manifest.json"
    manifest = build_manifest(output, names)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive_names = [*sorted(names), "manifest.json"]
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name in archive_names:
                    logical = _safe_name(name)
                    path = output / logical.as_posix()
                    payload = path.read_bytes()
                    tar.addfile(_tar_info(logical.as_posix(), len(payload)), io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, destination)
    verify_archive(destination)
    return sha256_file(destination)


def verify_archive(path: Path) -> dict[str, object]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names: list[str] = []
            payloads: dict[str, bytes] = {}
            for member in members:
                logical = _safe_name(member.name).as_posix()
                if not member.isfile() or member.issym() or member.islnk():
                    raise CertificationArchiveError("unsafe archive member type")
                if logical in payloads or member.size > MAX_EVIDENCE_FILE_BYTES:
                    raise CertificationArchiveError("duplicate or oversized archive member")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise CertificationArchiveError("archive member unreadable")
                payloads[logical] = extracted.read()
                names.append(logical)
    except (OSError, tarfile.TarError) as exc:
        raise CertificationArchiveError("certification archive malformed") from exc
    try:
        raw_manifest: object = json.loads(payloads["manifest.json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise CertificationArchiveError("certification manifest missing or malformed") from exc
    if not isinstance(raw_manifest, dict) or raw_manifest.get("status") != "PASS":
        raise CertificationArchiveError("certification manifest invalid")
    entries = raw_manifest.get("files")
    if not isinstance(entries, list):
        raise CertificationArchiveError("certification manifest entries invalid")
    expected_names = {"manifest.json"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CertificationArchiveError("certification manifest entry invalid")
        name = entry.get("path")
        if not isinstance(name, str):
            raise CertificationArchiveError("certification manifest path invalid")
        expected_names.add(name)
        payload = payloads.get(name)
        if (
            payload is None
            or len(payload) != entry.get("size_bytes")
            or hashlib.sha256(payload).hexdigest() != entry.get("sha256")
        ):
            raise CertificationArchiveError(f"certification hash mismatch: {name}")
    if set(names) != expected_names:
        raise CertificationArchiveError("archive and manifest inventory differ")
    return {
        "status": "PASS",
        "member_count": len(names),
        "manifest_hashes_verified": len(entries),
        "unsafe_archive_paths": False,
    }

