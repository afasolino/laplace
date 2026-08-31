"""Bounded local task-outcome evidence for later adaptive policy decisions."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SCHEMA_VERSION: Final = 1
_MAX_TEXT: Final = 512
_MAX_FAILURES: Final = 16
_MAX_LIST_ITEMS: Final = 16
_MAX_RECORD_BYTES: Final = 32 * 1024


class TaskEvidenceError(ValueError):
    """A task-evidence record is unavailable or failed closed validation."""


@dataclass(frozen=True, slots=True)
class TaskOutcomeEvidence:
    task_id: str
    owner_id: str
    project_id: str
    outcome: str
    roles: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    runtime_profiles: tuple[str, ...] = ()
    manager_decision: str = "unknown"
    specialist_decision: str = "unknown"
    verifier_digest: str = ""
    failure_classes: tuple[str, ...] = ()
    context_refs: tuple[str, ...] = ()
    turns: int | None = None
    tool_calls: int | None = None
    reported_input_tokens: int | None = None
    reported_output_tokens: int | None = None
    wall_time_ms: int | None = None
    publication_sequence: int = 0
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for identifier, error in (
            (self.task_id, "task_id_invalid"),
            (self.owner_id, "owner_id_invalid"),
            (self.project_id, "project_id_invalid"),
        ):
            if not isinstance(identifier, str) or _ID.fullmatch(identifier) is None:
                raise ValueError(error)
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("task_evidence_schema_unsupported")
        if not isinstance(self.outcome, str) or not self.outcome or len(self.outcome) > _MAX_TEXT:
            raise ValueError("outcome_invalid")
        for sequence in (
            self.roles,
            self.model_ids,
            self.runtime_profiles,
            self.failure_classes,
            self.context_refs,
        ):
            if not isinstance(sequence, tuple) or len(sequence) > _MAX_LIST_ITEMS:
                raise ValueError("evidence_list_invalid")
        if len(self.failure_classes) > _MAX_FAILURES:
            raise ValueError("too_many_failure_classes")
        for seq in (
            self.roles,
            self.model_ids,
            self.runtime_profiles,
            self.failure_classes,
            self.context_refs,
        ):
            if any(not isinstance(value, str) or not value or len(value) > _MAX_TEXT for value in seq):
                raise ValueError("evidence_text_invalid")
        for decision in (self.manager_decision, self.specialist_decision):
            if not isinstance(decision, str) or not decision or len(decision) > _MAX_TEXT:
                raise ValueError("evidence_decision_invalid")
        if self.verifier_digest and (
            not isinstance(self.verifier_digest, str) or len(self.verifier_digest) > 128
        ):
            raise ValueError("verifier_digest_invalid")
        for count in (
            self.turns,
            self.tool_calls,
            self.reported_input_tokens,
            self.reported_output_tokens,
            self.wall_time_ms,
        ):
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError("evidence_count_negative")
        if (
            isinstance(self.publication_sequence, bool)
            or not isinstance(self.publication_sequence, int)
            or self.publication_sequence < 0
        ):
            raise ValueError("publication_sequence_invalid")


class TaskEvidenceStore:
    """Owner/project-scoped immutable JSON records using atomic rename."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, evidence: TaskOutcomeEvidence) -> Path:
        directory = self.root / evidence.owner_id / evidence.project_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{evidence.task_id}.json"
        lock = self._acquire_publication_lock(directory)
        try:
            if path.exists():
                raise FileExistsError("task_evidence_already_exists")
            published = replace(
                evidence,
                publication_sequence=self._next_publication_sequence(directory),
            )
            payload = json.dumps(
                asdict(published), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if len(payload) > _MAX_RECORD_BYTES:
                raise ValueError("task_evidence_record_too_large")
            temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Hard-linking the fully written temporary file gives an atomic
                # no-overwrite publication on filesystems that support links.
                # A concurrent writer therefore cannot replace an immutable record.
                os.link(temp, path)
                temp.unlink()
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        finally:
            lock.unlink(missing_ok=True)
        return path

    @staticmethod
    def _acquire_publication_lock(directory: Path) -> Path:
        lock = directory / ".publication.lock"
        for _ in range(100):
            try:
                descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                time.sleep(0.01)
                continue
            os.close(descriptor)
            return lock
        raise TaskEvidenceError("task_evidence_publication_busy")

    def _next_publication_sequence(self, directory: Path) -> int:
        sequences = []
        for path in directory.glob("*.json"):
            if not path.is_file():
                continue
            evidence = self.load(
                owner_id=directory.parent.name,
                project_id=directory.name,
                task_id=path.stem,
            )
            if evidence.publication_sequence >= 1:
                sequences.append(evidence.publication_sequence)
        return max(sequences, default=0) + 1

    def load(self, *, owner_id: str, project_id: str, task_id: str) -> TaskOutcomeEvidence:
        for identifier in (owner_id, project_id, task_id):
            if _ID.fullmatch(identifier) is None:
                raise ValueError("evidence_scope_invalid")
        path = self.root / owner_id / project_id / f"{task_id}.json"
        try:
            if path.stat().st_size > _MAX_RECORD_BYTES:
                raise TaskEvidenceError("task_evidence_record_too_large")
            raw_value: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_value, dict):
                raise TaskEvidenceError("task_evidence_corrupt")
            raw = dict(raw_value)
            if raw.get("schema_version") != _SCHEMA_VERSION:
                raise TaskEvidenceError("task_evidence_schema_unsupported")
            for key in ("roles", "model_ids", "runtime_profiles", "failure_classes", "context_refs"):
                sequence = raw.get(key, [])
                if not isinstance(sequence, list):
                    raise TaskEvidenceError("task_evidence_corrupt")
                raw[key] = tuple(sequence)
            evidence = TaskOutcomeEvidence(**raw)
        except FileNotFoundError as exc:
            raise TaskEvidenceError("task_evidence_not_found") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if isinstance(exc, TaskEvidenceError):
                raise
            raise TaskEvidenceError("task_evidence_corrupt") from exc
        if evidence.owner_id != owner_id or evidence.project_id != project_id or evidence.task_id != task_id:
            raise TaskEvidenceError("task_evidence_scope_mismatch")
        return evidence

    def recent(self, *, owner_id: str, project_id: str, limit: int = 4) -> tuple[TaskOutcomeEvidence, ...]:
        """Read a small deterministic advisory sample without altering evidence."""

        if not 1 <= limit <= _MAX_LIST_ITEMS or any(
            _ID.fullmatch(value) is None for value in (owner_id, project_id)
        ):
            raise ValueError("evidence_scope_invalid")
        directory = self.root / owner_id / project_id
        try:
            records = [
                self.load(owner_id=owner_id, project_id=project_id, task_id=path.stem)
                for path in directory.glob("*.json")
                if path.is_file()
            ]
        except OSError as exc:
            raise TaskEvidenceError("task_evidence_corrupt") from exc
        sequenced = [record for record in records if record.publication_sequence >= 1]
        return tuple(
            sorted(sequenced, key=lambda record: record.publication_sequence)[-limit:]
        )


def recent_evidence_for_policy(
    store: TaskEvidenceStore, *, owner_id: str, project_id: str, limit: int = 4
) -> tuple[TaskOutcomeEvidence, ...]:
    """Return no hint when evidence is absent, corrupt, or insufficient."""

    try:
        return store.recent(owner_id=owner_id, project_id=project_id, limit=limit)
    except (TaskEvidenceError, ValueError):
        return ()
