"""Durable, owner-bound Zetsu result artifacts with bounded byte-exact paging."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Mapping, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]

_RESULT_RE = re.compile(r"^res_[a-f0-9]{32}$")
_ARTIFACT_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MAX_PAGE_BYTES = 64 * 1024


class ZetsuResultError(RuntimeError):
    """A result artifact failed identity, authorization, or integrity validation."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _owner_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


class ZetsuResultStore:
    """Store exact task material outside disposable worktrees.

    Pages are base64-encoded byte ranges. This keeps UTF-8, JSON, binary patches,
    and structured verifier output reconstructable without guessing text boundaries.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ZetsuResultError("zetsu_result_root_invalid")
        if os.name != "nt":
            os.chmod(self.root, 0o700)

    @staticmethod
    def result_id(user_id: str, repo_id: str, session_id: str) -> str:
        material = json.dumps(
            [user_id, repo_id, session_id], separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return "res_" + hashlib.sha256(material).hexdigest()[:32]

    def _directory(self, result_id: str) -> Path:
        if _RESULT_RE.fullmatch(result_id) is None:
            raise ZetsuResultError("zetsu_result_id_invalid")
        target = self.root / result_id
        if target.exists() and (target.is_symlink() or not target.is_dir()):
            raise ZetsuResultError("zetsu_result_directory_invalid")
        return target

    @staticmethod
    def _artifact_name(name: str) -> str:
        if _ARTIFACT_RE.fullmatch(name) is None:
            raise ZetsuResultError("zetsu_result_artifact_invalid")
        return name

    @staticmethod
    def _copy_stream(source: BinaryIO, target: Path) -> JsonObject:
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                source.seek(0)
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    if not isinstance(block, bytes):
                        raise ZetsuResultError("zetsu_result_artifact_not_binary")
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o600)
        finally:
            temporary.unlink(missing_ok=True)
        details = target.stat()
        return {
            "bytes": size,
            "sha256": digest.hexdigest(),
            "device": details.st_dev,
            "inode": details.st_ino,
        }

    def stage_stream(self, session_id: str, name: str, source: BinaryIO) -> Path:
        """Persist a verifier stream before the task has reached a terminal result."""

        safe_name = self._artifact_name(name)
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        staging = self.root / ".staging" / digest
        staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        if staging.is_symlink() or staging.resolve() != staging:
            raise ZetsuResultError("zetsu_result_staging_invalid")
        target = staging / safe_name
        self._copy_stream(source, target)
        return target

    def persist(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        status: str,
        summary: str,
        artifacts: Mapping[str, Path | bytes | str],
    ) -> JsonObject:
        result_id = self.result_id(user_id, repo_id, session_id)
        directory = self._directory(result_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        records: JsonObject = {}
        for raw_name, source in sorted(artifacts.items()):
            name = self._artifact_name(raw_name)
            target = directory / name
            if isinstance(source, Path):
                try:
                    details = source.lstat()
                except OSError as exc:
                    raise ZetsuResultError("zetsu_result_source_unavailable") from exc
                if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                    raise ZetsuResultError("zetsu_result_source_invalid")
                with source.open("rb") as handle:
                    record = self._copy_stream(handle, target)
            else:
                raw = source if isinstance(source, bytes) else source.encode("utf-8")
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                    os.chmod(target, 0o600)
                finally:
                    temporary.unlink(missing_ok=True)
                details = target.stat()
                record = {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "device": details.st_dev,
                    "inode": details.st_ino,
                }
            records[name] = record
        manifest: JsonObject = {
            "schema_version": 1,
            "result_id": result_id,
            "owner_id_sha256": _owner_hash(user_id),
            "repo_id": repo_id,
            "session_id": session_id,
            "execution_status": status,
            "summary": summary[:1_000],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": records,
        }
        _atomic_json(directory / "manifest.json", manifest)
        return {
            "result_id": result_id,
            "execution_status": status,
            "summary": manifest["summary"],
            "artifacts": {
                name: {"bytes": value["bytes"], "sha256": value["sha256"]}
                for name, value in cast(dict[str, JsonObject], records).items()
            },
        }

    def _manifest(self, result_id: str) -> tuple[Path, JsonObject]:
        directory = self._directory(result_id)
        manifest_path = directory / "manifest.json"
        try:
            if manifest_path.is_symlink():
                raise ZetsuResultError("zetsu_result_manifest_invalid")
            raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ZetsuResultError("zetsu_result_not_found") from exc
        if not isinstance(raw, dict) or raw.get("result_id") != result_id:
            raise ZetsuResultError("zetsu_result_manifest_invalid")
        return directory, cast(JsonObject, raw)

    def page(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        result_id: str,
        artifact: str,
        offset: int,
        max_bytes: int,
    ) -> JsonObject:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ZetsuResultError("zetsu_result_offset_invalid")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 1 <= max_bytes <= _MAX_PAGE_BYTES
        ):
            raise ZetsuResultError("zetsu_result_page_size_invalid")
        directory, manifest = self._manifest(result_id)
        if (
            manifest.get("owner_id_sha256") != _owner_hash(user_id)
            or manifest.get("repo_id") != repo_id
            or manifest.get("session_id") != session_id
        ):
            raise ZetsuResultError("zetsu_result_not_found")
        name = self._artifact_name(artifact)
        artifacts = manifest.get("artifacts")
        record = artifacts.get(name) if isinstance(artifacts, dict) else None
        if not isinstance(record, dict):
            raise ZetsuResultError("zetsu_result_artifact_not_found")
        target = directory / name
        try:
            details = target.lstat()
        except OSError as exc:
            raise ZetsuResultError("zetsu_result_artifact_not_found") from exc
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or target.resolve() != target
            or details.st_dev != record.get("device")
            or details.st_ino != record.get("inode")
            or details.st_size != record.get("bytes")
        ):
            raise ZetsuResultError("zetsu_result_artifact_identity_changed")
        total = details.st_size
        if offset > total:
            raise ZetsuResultError("zetsu_result_offset_invalid")
        with target.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(max_bytes)
        next_offset = offset + len(content)
        return {
            "status": "SUCCESS",
            "result_id": result_id,
            "repo_id": repo_id,
            "session_id": session_id,
            "artifact": name,
            "encoding": "base64",
            "offset": offset,
            "next_offset": next_offset if next_offset < total else None,
            "total_bytes": total,
            "artifact_sha256": record.get("sha256"),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }

    def staging_artifacts(self, session_id: str) -> dict[str, Path]:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        staging = self.root / ".staging" / digest
        if not staging.is_dir() or staging.is_symlink() or staging.resolve() != staging:
            return {}
        return {
            item.name: item
            for item in sorted(staging.iterdir())
            if item.is_file() and not item.is_symlink() and _ARTIFACT_RE.fullmatch(item.name)
        }

    def clear_staging(self, session_id: str) -> None:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        staging = self.root / ".staging" / digest
        if staging.is_dir() and not staging.is_symlink() and staging.resolve() == staging:
            shutil.rmtree(staging)
