"""Typed, bounded repository tools for the local agent.

This module is an adapter-shaped ACI, not a shell.  Every operation is bound to
one already-authorized worktree/session, uses literal paths, caps returned
content, and rejects generic command or network input.  The existing
coordinator remains the authority for session grants and mutation accounting;
this class supplies a smaller structured tool surface for shadow comparison.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess  # nosec B404 - argv is allowlisted and shell=False
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

from .repository_authorization import RepositoryAuthorizationError, validate_workspace_path
from .repository_context import RepoMap, RepositoryContextService

JsonObject: TypeAlias = dict[str, object]
ACIPathOperation = Literal[
    "repo_map",
    "find_symbol",
    "find_references",
    "search_text",
    "read_region",
    "read_page",
    "inspect_diff",
    "edit_region",
    "create_text_file",
    "begin_file_write",
    "write_file_chunk",
    "finalize_file_write",
    "abort_file_write",
    "verify",
    "git_state",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ALLOWED_VERIFY_EXECUTABLES = frozenset({"pytest", "ruff", "mypy"})
_MAX_FILE_BYTES = 2_000_000
_MAX_READ_LINES = 400
_MAX_READ_CHARS = 32_000
_MAX_SEARCH_MATCHES = 80
_MAX_SEARCH_FILE_BYTES = 1_000_000
_MAX_DIFF_BYTES = 64_000
_MAX_RESULT_CHARS = 64_000
_MAX_TEXT_CHARS = 256_000
_MAX_WRITE_CHUNK_BYTES = 32_000
_MAX_CURSOR_CHARS = 2_048
_CURSOR_SCHEMA = 1


class BoundedACIError(RuntimeError):
    """A typed ACI operation failed closed."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BoundedACIError(f"invalid_{label}")
    return value


def _text(value: object, *, label: str, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise BoundedACIError(f"invalid_{label}")
    if not allow_empty and not value.strip():
        raise BoundedACIError(f"invalid_{label}")
    return value


def _integer(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise BoundedACIError(f"invalid_{label}")
    return value


def _sha256_hex(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BoundedACIError(f"invalid_{label}")
    return value


def _cursor_encode(value: JsonObject) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _cursor_decode(value: object) -> JsonObject:
    if not isinstance(value, str) or not value or len(value) > _MAX_CURSOR_CHARS:
        raise BoundedACIError("aci_read_cursor_invalid")
    try:
        padded = value + "=" * (-len(value) % 4)
        raw: object = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise BoundedACIError("aci_read_cursor_invalid") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "path",
        "snapshot_sha256",
        "next_line",
        "page_lines",
    }:
        raise BoundedACIError("aci_read_cursor_invalid")
    if raw["schema"] != _CURSOR_SCHEMA or not isinstance(raw["path"], str):
        raise BoundedACIError("aci_read_cursor_invalid")
    _sha256_hex(raw["snapshot_sha256"], label="read_cursor_snapshot")
    if (
        isinstance(raw["next_line"], bool)
        or not isinstance(raw["next_line"], int)
        or raw["next_line"] < 1
        or isinstance(raw["page_lines"], bool)
        or not isinstance(raw["page_lines"], int)
        or not 1 <= raw["page_lines"] <= _MAX_READ_LINES
    ):
        raise BoundedACIError("aci_read_cursor_invalid")
    return raw


class BoundedRepositoryACI:
    """Small structured repository interface with no generic shell action."""

    def __init__(
        self,
        worktree: Path,
        *,
        owner_user_id: str,
        session_id: str,
        allow_mutation: bool = False,
        required_verification_argv: Sequence[str] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        try:
            self.worktree = worktree.resolve(strict=True)
        except OSError as exc:
            raise BoundedACIError("aci_worktree_unavailable") from exc
        if not self.worktree.is_dir():
            raise BoundedACIError("aci_worktree_unavailable")
        self.owner_user_id = _identifier(owner_user_id, label="owner_user_id")
        self.session_id = _identifier(session_id, label="session_id")
        self.allow_mutation = allow_mutation
        self.required_verification_argv = (
            tuple(required_verification_argv) if required_verification_argv is not None else None
        )
        self.is_cancelled = is_cancelled or (lambda: False)
        self.repository_context = RepositoryContextService(self.worktree)
        self._write_transactions: dict[str, JsonObject] = {}

    def _envelope(self, result: JsonObject) -> JsonObject:
        return {
            "owner_user_id": self.owner_user_id,
            "session_id": self.session_id,
            **result,
        }

    def _target(self, value: object, *, label: str = "path") -> Path:
        path = _text(value, label=label, maximum=500)
        normalized = path.replace("\\", "/")
        if normalized == ".git" or normalized.startswith(".git/"):
            raise BoundedACIError("aci_git_metadata_forbidden")
        try:
            return validate_workspace_path(self.worktree, normalized)
        except RepositoryAuthorizationError as exc:
            raise BoundedACIError(f"aci_{exc.category}", exc.evidence) from exc

    @staticmethod
    def _read_text(target: Path, *, category: str) -> str:
        try:
            if not target.is_file() or target.stat().st_size > _MAX_FILE_BYTES:
                raise BoundedACIError(f"{category}_unavailable")
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise BoundedACIError(f"{category}_not_text") from exc

    def repo_map(
        self,
        *,
        query: str = "",
        focus_paths: Sequence[str] = (),
        token_budget: int = 1_000,
    ) -> JsonObject:
        query_value = _text(query, label="query", maximum=4_000, allow_empty=True)
        if len(focus_paths) > 32:
            raise BoundedACIError("aci_focus_paths_too_many")
        normalized_focus = tuple(self._target(path).relative_to(self.worktree).as_posix() for path in focus_paths)
        budget = _integer(token_budget, label="token_budget", minimum=128, maximum=12_000)
        try:
            value: RepoMap = self.repository_context.build_repo_map(
                query=query_value,
                focus_paths=normalized_focus,
                token_budget=budget,
            )
        except Exception as exc:
            if isinstance(exc, (BoundedACIError,)):
                raise
            raise BoundedACIError("aci_repo_map_failed") from exc
        return self._envelope(value.to_json())

    def find_symbol(self, name: str) -> JsonObject:
        query = _text(name, label="symbol_query", maximum=256)
        symbols = self.repository_context.find_symbol(query)
        return self._envelope(
            {
                "query": query,
                "symbols": [item.to_json() for item in symbols[:32]],
                "truncated": len(symbols) > 32,
            }
        )

    def find_references(self, name: str) -> JsonObject:
        query = _text(name, label="reference_query", maximum=256)
        references = self.repository_context.find_references(query)
        return self._envelope(
            {
                "query": query,
                "references": [item.to_json() for item in references[:64]],
                "truncated": len(references) > 64,
            }
        )

    def search_text(self, *, query: str, glob: str = "*") -> JsonObject:
        needle = _text(query, label="search_query", maximum=4_000)
        pattern = _text(glob, label="glob", maximum=200)
        if ".." in Path(pattern).parts:
            raise BoundedACIError("aci_glob_invalid")
        matches: list[JsonObject] = []
        for path in self.worktree.rglob(pattern):
            if len(matches) >= _MAX_SEARCH_MATCHES:
                break
            relative_path = path.relative_to(self.worktree)
            if ".git" in relative_path.parts:
                continue
            try:
                target = self._target(relative_path.as_posix())
                if not target.is_file() or target.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    continue
                content = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, BoundedACIError):
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if needle in line:
                    matches.append(
                        {
                            "path": relative_path.as_posix(),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        break
        return self._envelope(
            {
                "query": needle,
                "glob": pattern,
                "matches": matches,
                "truncated": len(matches) >= _MAX_SEARCH_MATCHES,
            }
        )

    def read_region(self, *, path: str, start_line: int, end_line: int) -> JsonObject:
        target = self._target(path)
        start = _integer(start_line, label="start_line", minimum=1, maximum=1_000_000)
        end = _integer(end_line, label="end_line", minimum=start, maximum=1_000_000)
        if end - start + 1 > _MAX_READ_LINES:
            raise BoundedACIError("aci_read_region_too_large")
        content = self._read_text(target, category="aci_read_region")
        lines = content.splitlines(keepends=True)
        selected = "".join(lines[start - 1 : end])
        if len(selected) > _MAX_READ_CHARS:
            raise BoundedACIError("aci_read_region_output_too_large")
        return self._envelope(
            {
                "path": target.relative_to(self.worktree).as_posix(),
                "start_line": start,
                "end_line": min(end, len(lines)),
                "content": selected,
                "sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
            }
        )

    def read_page(
        self,
        *,
        path: str,
        cursor: str | None = None,
        start_line: int = 1,
        max_lines: int = _MAX_READ_LINES,
    ) -> JsonObject:
        """Read a bounded page with a content-bound continuation cursor."""

        target = self._target(path)
        relative = target.relative_to(self.worktree).as_posix()
        page_lines = _integer(max_lines, label="page_lines", minimum=1, maximum=_MAX_READ_LINES)
        requested_start = _integer(start_line, label="start_line", minimum=1, maximum=1_000_000)
        cursor_value: JsonObject | None = None
        if cursor is not None:
            cursor_value = _cursor_decode(cursor)
            if cursor_value["path"] != relative or cursor_value["page_lines"] != page_lines:
                raise BoundedACIError("aci_read_cursor_invalid")
            if requested_start != 1:
                raise BoundedACIError("aci_read_cursor_invalid")
            requested_start = cast(int, cursor_value["next_line"])
        content = self._read_text(target, category="aci_read_page")
        snapshot_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if cursor_value is not None and cursor_value["snapshot_sha256"] != snapshot_sha256:
            raise BoundedACIError(
                "aci_read_cursor_stale",
                {"path": relative, "snapshot_sha256": snapshot_sha256},
            )
        lines = content.splitlines(keepends=True)
        selected = "".join(lines[requested_start - 1 : requested_start - 1 + page_lines])
        if len(selected.encode("utf-8")) > _MAX_READ_CHARS:
            raise BoundedACIError("aci_read_page_output_too_large")
        end_line = min(requested_start + page_lines - 1, len(lines))
        next_cursor = None
        if end_line < len(lines):
            next_cursor = _cursor_encode(
                {
                    "schema": _CURSOR_SCHEMA,
                    "path": relative,
                    "snapshot_sha256": snapshot_sha256,
                    "next_line": end_line + 1,
                    "page_lines": page_lines,
                }
            )
        return self._envelope(
            {
                "path": relative,
                "start_line": requested_start,
                "end_line": end_line,
                "content": selected,
                "content_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
                "snapshot_sha256": snapshot_sha256,
                "next_cursor": next_cursor,
                "total_lines": len(lines),
            }
        )

    def _run_git_capture(self, argv: Sequence[str], *, max_bytes: int) -> tuple[int, bytes, int]:
        if not argv or any("\x00" in item for item in argv):
            raise BoundedACIError("aci_git_arguments_invalid")
        with tempfile.TemporaryFile(mode="w+b") as output:
            try:
                process = subprocess.Popen(  # nosec B603 - fixed git executable, no shell
                    ["git", "-C", str(self.worktree), *argv],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                    start_new_session=os.name == "posix",
                )
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired as exc:
                    self._terminate(process)
                    raise BoundedACIError("aci_git_timeout") from exc
            except OSError as exc:
                raise BoundedACIError("aci_git_unavailable") from exc
            output.seek(0, os.SEEK_END)
            total = output.tell()
            output.seek(0)
            return returncode, output.read(max_bytes + 1), total

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            if process.poll() is None and os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
            process.wait(timeout=2)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass

    def git_state(self) -> JsonObject:
        head_code, head_raw, _ = self._run_git_capture(("rev-parse", "--verify", "HEAD"), max_bytes=256)
        status_code, status_raw, total = self._run_git_capture(
            ("status", "--porcelain=v1", "--untracked-files=all"), max_bytes=64_000
        )
        if head_code != 0 or status_code != 0:
            raise BoundedACIError("aci_git_state_unavailable")
        status_text = status_raw.decode("utf-8", errors="replace")
        changed = []
        for line in status_text.splitlines():
            if len(line) >= 4:
                path = line[3:]
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                changed.append(path.strip('"'))
        return self._envelope(
            {
                "head": head_raw.decode("utf-8", errors="replace").strip(),
                "changed_paths": sorted(dict.fromkeys(changed))[:256],
                "status_sha256": hashlib.sha256(status_raw).hexdigest(),
                "status_bytes": total,
                "status_truncated": total > 64_000,
            }
        )

    def inspect_diff(self, *, paths: Sequence[str] = ()) -> JsonObject:
        if len(paths) > 32:
            raise BoundedACIError("aci_diff_paths_too_many")
        normalized = [self._target(path).relative_to(self.worktree).as_posix() for path in paths]
        code, raw, total = self._run_git_capture(
            ("diff", "--no-ext-diff", "--binary", "HEAD", "--", *normalized),
            max_bytes=_MAX_DIFF_BYTES,
        )
        if code != 0:
            raise BoundedACIError("aci_diff_unavailable")
        bounded = raw[:_MAX_DIFF_BYTES]
        return self._envelope(
            {
                "paths": normalized,
                "diff": bounded.decode("utf-8", errors="replace"),
                "diff_sha256": hashlib.sha256(raw).hexdigest(),
                "diff_bytes": total,
                "truncated": total > _MAX_DIFF_BYTES,
            }
        )

    @staticmethod
    def _atomic_text(target: Path, content: str) -> None:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.aci.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def edit_region(self, *, path: str, old_text: str, new_text: str) -> JsonObject:
        if not self.allow_mutation:
            raise BoundedACIError("aci_mutation_not_allowed")
        target = self._target(path)
        old = _text(old_text, label="old_text", maximum=_MAX_TEXT_CHARS)
        new = _text(new_text, label="new_text", maximum=_MAX_TEXT_CHARS, allow_empty=True)
        original = self._read_text(target, category="aci_edit_region")
        if original.count(old) != 1:
            raise BoundedACIError("aci_edit_anchor_not_unique")
        replacement = original.replace(old, new, 1)
        if len(replacement) > _MAX_TEXT_CHARS:
            raise BoundedACIError("aci_edit_result_too_large")
        self._atomic_text(target, replacement)
        check, output, _ = self._run_git_capture(("diff", "--check"), max_bytes=2_000)
        if check != 0:
            self._atomic_text(target, original)
            raise BoundedACIError(
                "aci_edit_diff_check_failed",
                {"stderr": output.decode("utf-8", errors="replace")[-2_000:]},
            )
        return self._envelope(
            {
                "path": target.relative_to(self.worktree).as_posix(),
                "replacements": 1,
                "content_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
            }
        )

    def create_text_file(self, *, path: str, content: str) -> JsonObject:
        if not self.allow_mutation:
            raise BoundedACIError("aci_mutation_not_allowed")
        target = self._target(path)
        value = _text(content, label="content", maximum=_MAX_TEXT_CHARS, allow_empty=True)
        if target.exists() or not target.parent.is_dir():
            raise BoundedACIError("aci_create_target_invalid")
        try:
            self._atomic_text(target, value)
        except OSError as exc:
            raise BoundedACIError("aci_create_failed") from exc
        return self._envelope(
            {
                "path": target.relative_to(self.worktree).as_posix(),
                "content_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
        )

    def _write_transaction(self, transaction_id: str) -> JsonObject:
        identifier = _text(transaction_id, label="transaction_id", maximum=128)
        if _IDENTIFIER.fullmatch(identifier) is None:
            raise BoundedACIError("invalid_transaction_id")
        transaction = self._write_transactions.get(identifier)
        if transaction is None:
            raise BoundedACIError("aci_write_transaction_not_found")
        return transaction

    @staticmethod
    def _file_sha256(target: Path) -> str | None:
        try:
            details = target.lstat()
            if not target.is_file() or target.is_symlink() or details.st_size > _MAX_FILE_BYTES:
                raise BoundedACIError("aci_write_target_invalid")
            return hashlib.sha256(target.read_bytes()).hexdigest()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise BoundedACIError("aci_write_target_unavailable") from exc

    def _abort_if_cancelled(self, transaction_id: str) -> None:
        if self.is_cancelled():
            self.abort_file_write(transaction_id)
            raise BoundedACIError("aci_write_cancelled")

    def begin_file_write(
        self,
        *,
        path: str,
        expected_base_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> JsonObject:
        """Begin an ordered, same-directory staged write."""

        if not self.allow_mutation:
            raise BoundedACIError("aci_mutation_not_allowed")
        if self.is_cancelled():
            raise BoundedACIError("aci_write_cancelled")
        target = self._target(path)
        if not target.parent.is_dir() or (target.exists() and not target.is_file()):
            raise BoundedACIError("aci_write_target_invalid")
        if expected_base_sha256 is not None:
            expected_base_sha256 = _sha256_hex(
                expected_base_sha256, label="expected_base_sha256"
            )
        if expected_bytes is not None:
            expected_bytes = _integer(
                expected_bytes,
                label="expected_bytes",
                minimum=0,
                maximum=_MAX_FILE_BYTES,
            )
        current_sha256 = self._file_sha256(target)
        if expected_base_sha256 != current_sha256 and expected_base_sha256 is not None:
            raise BoundedACIError(
                "aci_write_base_changed",
                {"path": target.relative_to(self.worktree).as_posix()},
            )
        transaction_id = f"aciw_{uuid.uuid4().hex}"
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.aci.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise BoundedACIError("aci_write_begin_failed") from exc
        self._write_transactions[transaction_id] = {
            "transaction_id": transaction_id,
            "target": target,
            "relative_path": target.relative_to(self.worktree).as_posix(),
            "temporary": temporary,
            "base_sha256": current_sha256,
            "expected_bytes": expected_bytes,
            "next_sequence": 0,
            "next_offset": 0,
            "last_sequence": None,
            "last_offset": None,
            "last_bytes": 0,
            "last_sha256": None,
        }
        return self._envelope(
            {
                "transaction_id": transaction_id,
                "path": target.relative_to(self.worktree).as_posix(),
                "base_sha256": current_sha256,
                "next_sequence": 0,
                "next_offset": 0,
            }
        )

    def write_file_chunk(
        self,
        *,
        transaction_id: str,
        sequence: int,
        offset: int,
        content: str,
        chunk_sha256: str,
    ) -> JsonObject:
        """Append one ordered chunk; an exact retry of the last chunk is idempotent."""

        if not self.allow_mutation:
            raise BoundedACIError("aci_mutation_not_allowed")
        transaction = self._write_transaction(transaction_id)
        self._abort_if_cancelled(transaction_id)
        sequence = _integer(sequence, label="sequence", minimum=0, maximum=1_000_000)
        offset = _integer(offset, label="offset", minimum=0, maximum=_MAX_FILE_BYTES)
        value = _text(content, label="chunk", maximum=_MAX_TEXT_CHARS)
        raw = value.encode("utf-8")
        if len(raw) > _MAX_WRITE_CHUNK_BYTES:
            raise BoundedACIError("aci_write_chunk_too_large")
        chunk_sha256 = _sha256_hex(chunk_sha256, label="chunk_sha256")
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != chunk_sha256:
            raise BoundedACIError("aci_write_chunk_hash_mismatch")
        if sequence == transaction["last_sequence"] and offset == transaction["last_offset"]:
            if chunk_sha256 == transaction["last_sha256"] and len(raw) == transaction["last_bytes"]:
                return self._envelope(
                    {
                        "transaction_id": transaction_id,
                        "sequence": sequence,
                        "offset": offset,
                        "bytes": len(raw),
                        "next_sequence": transaction["next_sequence"],
                        "next_offset": transaction["next_offset"],
                        "idempotent_retry": True,
                    }
                )
            raise BoundedACIError("aci_write_duplicate_chunk_conflict")
        if sequence != transaction["next_sequence"] or offset != transaction["next_offset"]:
            raise BoundedACIError("aci_write_chunk_out_of_order")
        expected_bytes = cast(int | None, transaction["expected_bytes"])
        if expected_bytes is not None and offset + len(raw) > expected_bytes:
            raise BoundedACIError("aci_write_expected_size_exceeded")
        temporary = transaction["temporary"]
        if not isinstance(temporary, Path) or temporary.is_symlink():
            raise BoundedACIError("aci_write_staging_invalid")
        try:
            with temporary.open("ab") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise BoundedACIError("aci_write_chunk_failed") from exc
        transaction["last_sequence"] = sequence
        transaction["last_offset"] = offset
        transaction["last_bytes"] = len(raw)
        transaction["last_sha256"] = chunk_sha256
        transaction["next_sequence"] = sequence + 1
        transaction["next_offset"] = offset + len(raw)
        return self._envelope(
            {
                "transaction_id": transaction_id,
                "sequence": sequence,
                "offset": offset,
                "bytes": len(raw),
                "next_sequence": transaction["next_sequence"],
                "next_offset": transaction["next_offset"],
                "idempotent_retry": False,
            }
        )

    def finalize_file_write(self, *, transaction_id: str, content_sha256: str) -> JsonObject:
        """Verify staged bytes and atomically publish the complete file."""

        if not self.allow_mutation:
            raise BoundedACIError("aci_mutation_not_allowed")
        transaction = self._write_transaction(transaction_id)
        self._abort_if_cancelled(transaction_id)
        content_sha256 = _sha256_hex(content_sha256, label="content_sha256")
        expected_bytes = transaction["expected_bytes"]
        next_offset = cast(int, transaction["next_offset"])
        if expected_bytes is not None and next_offset != expected_bytes:
            raise BoundedACIError("aci_write_chunks_incomplete")
        temporary = transaction["temporary"]
        if not isinstance(temporary, Path) or temporary.is_symlink():
            raise BoundedACIError("aci_write_staging_invalid")
        try:
            staged = temporary.read_bytes()
        except OSError as exc:
            raise BoundedACIError("aci_write_staging_unavailable") from exc
        if len(staged) != next_offset or hashlib.sha256(staged).hexdigest() != content_sha256:
            raise BoundedACIError("aci_write_finalize_hash_mismatch")
        target = transaction["target"]
        if not isinstance(target, Path) or self._file_sha256(target) != transaction["base_sha256"]:
            raise BoundedACIError("aci_write_target_drift")
        try:
            validated_target = self._target(str(transaction["relative_path"]))
            if validated_target != target or validated_target.is_symlink():
                raise BoundedACIError("aci_write_target_invalid")
            os.replace(temporary, target)
        except OSError as exc:
            raise BoundedACIError("aci_write_finalize_failed") from exc
        self._write_transactions.pop(transaction_id, None)
        return self._envelope(
            {
                "transaction_id": transaction_id,
                "path": str(transaction["relative_path"]),
                "bytes": len(staged),
                "content_sha256": content_sha256,
                "finalized": True,
            }
        )

    def abort_file_write(self, transaction_id: str) -> JsonObject:
        """Discard only the controlled staging file for one active transaction."""

        transaction = self._write_transactions.pop(transaction_id, None)
        if transaction is None:
            raise BoundedACIError("aci_write_transaction_not_found")
        temporary = transaction["temporary"]
        if isinstance(temporary, Path) and not temporary.is_symlink():
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                self._write_transactions[transaction_id] = transaction
                raise BoundedACIError("aci_write_abort_failed") from exc
        return self._envelope(
            {"transaction_id": transaction_id, "aborted": True}
        )

    @staticmethod
    def _verify_argv(worktree: Path, argv: Sequence[str]) -> list[str]:
        if not 1 <= len(argv) <= 64 or any(not isinstance(item, str) for item in argv):
            raise BoundedACIError("aci_verify_argv_invalid")
        values = list(argv)
        executable = Path(values[0]).name
        if values[0] != executable or executable not in _ALLOWED_VERIFY_EXECUTABLES:
            raise BoundedACIError("aci_verify_command_forbidden")
        forbidden = {"-c", "-p", "-o", "--config", "--config-file", "--basetemp", "--rootdir"}
        lowered = [item.casefold() for item in values[1:]]
        if any(item in forbidden or any(item.startswith(f"{prefix}=") for prefix in forbidden) for item in lowered):
            raise BoundedACIError("aci_verify_command_forbidden")
        if executable == "ruff" and ("--fix" in lowered or "format" in lowered):
            raise BoundedACIError("aci_verify_command_forbidden")
        skip_next = False
        for index, item in enumerate(values[1:], start=1):
            if len(item) > 1_000 or "\x00" in item:
                raise BoundedACIError("aci_verify_argv_invalid")
            if skip_next:
                skip_next = False
                continue
            if item.casefold() in {"-k", "-m"} and executable == "pytest":
                skip_next = True
                continue
            if item.startswith("-"):
                continue
            candidate = item.split("::", 1)[0]
            candidate_path = Path(candidate)
            if candidate_path.is_absolute() or ".." in candidate_path.parts:
                raise BoundedACIError("aci_verify_path_forbidden")
            if candidate != "." and not (
                "/" in candidate
                or candidate_path.suffix in {".py", ".pyi"}
                or (worktree / candidate).exists()
            ):
                raise BoundedACIError("aci_verify_target_invalid")
            try:
                validate_workspace_path(worktree, candidate)
            except RepositoryAuthorizationError as exc:
                raise BoundedACIError(f"aci_{exc.category}", exc.evidence) from exc
        if skip_next:
            raise BoundedACIError("aci_verify_argv_invalid")
        return values

    def verify(self, *, argv: Sequence[str], timeout_seconds: float = 60.0) -> JsonObject:
        values = self._verify_argv(self.worktree, argv)
        if self.required_verification_argv is not None and tuple(values) != self.required_verification_argv:
            raise BoundedACIError("aci_verify_not_required_command")
        timeout = min(600.0, max(0.1, float(timeout_seconds)))
        before = self.git_state()
        executable = shutil.which(values[0])
        if executable is None:
            raise BoundedACIError("aci_verify_executable_missing")
        with tempfile.TemporaryFile(mode="w+b") as stdout, tempfile.TemporaryFile(mode="w+b") as stderr:
            try:
                process = subprocess.Popen(  # nosec B603 - executable and argv are allowlisted
                    [executable, *values[1:]],
                    cwd=self.worktree,
                    env=os.environ.copy(),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    start_new_session=os.name == "posix",
                )
            except OSError as exc:
                raise BoundedACIError("aci_verify_start_failed") from exc
            aborted: str | None = None
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if self.is_cancelled():
                        aborted = "aci_verify_cancelled"
                        self._terminate(process)
                        break
                    if time.monotonic() >= deadline:
                        aborted = "aci_verify_timeout"
                        self._terminate(process)
                        break
                    time.sleep(0.05)
                if aborted is None:
                    process.wait(timeout=2)
            finally:
                stdout.seek(0, os.SEEK_END)
                stdout_size = stdout.tell()
                stdout.seek(max(0, stdout_size - 8_000))
                stdout_tail = stdout.read().decode("utf-8", errors="replace")
                stderr.seek(0, os.SEEK_END)
                stderr_size = stderr.tell()
                stderr.seek(max(0, stderr_size - 8_000))
                stderr_tail = stderr.read().decode("utf-8", errors="replace")
        after = self.git_state()
        mutated = before.get("head") != after.get("head") or before.get("status_sha256") != after.get("status_sha256")
        return self._envelope(
            {
                "status": "PASS" if aborted is None and process.returncode == 0 and not mutated else "FAIL",
                "returncode": process.returncode,
                "aborted": aborted,
                "worktree_mutated": mutated,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_bytes": stdout_size,
                "stderr_bytes": stderr_size,
            }
        )
