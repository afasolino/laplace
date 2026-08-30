"""Authoritative append-only trajectories with derived checkpoints.

Trajectory events are the source of truth for task reconstruction.  A
checkpoint is only a validated acceleration snapshot; it is never required
for recovery and is ignored when the event log remains intact.

The format is deliberately local JSONL plus one atomic JSON checkpoint.  It
is inspectable, content-addressed, owner-bound and independent of any model
or remote service.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias, cast

from .execution_records import (
    canonical_json_bytes,
    canonical_sha256,
    lock_file,
    unlock_file,
)

JsonObject: TypeAlias = dict[str, object]
TrajectorySourceKind = Literal[
    "user",
    "system",
    "model",
    "tool",
    "retrieval",
    "memory",
    "verification",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 128 * 1024
_MAX_PROVENANCE_BYTES = 16 * 1024
_MAX_PARTIAL_RECOVERY_BYTES = 256 * 1024
_FORBIDDEN_KEYS = frozenset({"prompt", "response", "source_code", "secret", "token"})


class TrajectoryEventType(StrEnum):
    """The closed vocabulary of replayable trajectory events."""

    TASK_STARTED = "task_started"
    TASK_RESUMED = "task_resumed"
    TASK_CANCELLED = "task_cancelled"
    RETRIEVAL_ACCESSED = "retrieval_accessed"
    MEMORY_ACCESSED = "memory_accessed"
    MODEL_CALLED = "model_called"
    TOOL_ACTION = "tool_action"
    EDIT = "edit"
    VERIFICATION = "verification"
    COMPACTION = "compaction"
    CHECKPOINT = "checkpoint"
    FAILURE = "failure"
    COMPLETION = "completion"


class TrajectoryError(RuntimeError):
    """Base error for fail-closed trajectory operations."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class TrajectoryValidationError(TrajectoryError):
    """A caller supplied malformed trajectory data."""


class TrajectoryAuthorizationError(TrajectoryError):
    """The requested trajectory belongs to another owner or project."""


class TrajectoryCorruptionError(TrajectoryError):
    """Durable trajectory data is malformed, altered or inconsistent."""


class TrajectorySchemaError(TrajectoryError):
    """The durable trajectory schema is not supported by this version."""


class TrajectoryConflictError(TrajectoryError):
    """An idempotency key was reused with different content."""


class TrajectoryCrashInjected(TrajectoryError):
    """A deterministic test hook simulated a process crash."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise TrajectoryValidationError(f"invalid_{label}")
    return value


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrajectoryValidationError(f"invalid_{label}")
    return value


def _json_value(value: object, *, label: str, maximum: int) -> object:
    """Validate strict finite JSON without silently coercing Python values."""

    def visit(item: object, path: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise TrajectoryValidationError(f"invalid_{label}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or "\x00" in key:
                    raise TrajectoryValidationError(f"invalid_{label}")
                if key.lower() in _FORBIDDEN_KEYS:
                    raise TrajectoryValidationError(f"forbidden_{label}_field")
                visit(child, f"{path}.{key}")
            return
        raise TrajectoryValidationError(f"invalid_{label}")

    visit(value, label)
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryValidationError(f"invalid_{label}") from exc
    if len(encoded) > maximum:
        raise TrajectoryValidationError(f"{label}_too_large")
    return value


def _object(value: object, *, label: str, maximum: int = _MAX_JSON_BYTES) -> JsonObject:
    checked = _json_value(value, label=label, maximum=maximum)
    if not isinstance(checked, dict):
        raise TrajectoryValidationError(f"invalid_{label}")
    return cast(JsonObject, checked)


@dataclass(frozen=True)
class TrajectoryIdentity:
    """The authorization and reconstruction scope of one trajectory."""

    owner_user_id: str
    project_id: str
    session_id: str
    task_id: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.owner_user_id, label="owner_user_id")
        _identifier(self.project_id, label="project_id")
        _identifier(self.session_id, label="session_id")
        _identifier(self.task_id, label="task_id")
        if self.schema_version != _SCHEMA_VERSION:
            raise TrajectorySchemaError("trajectory_schema_unsupported")

    def to_json(self) -> JsonObject:
        return {
            "owner_user_id": self.owner_user_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class TrajectoryProvenance:
    """Hashes and bounded metadata needed to explain an event without secrets."""

    source_kind: TrajectorySourceKind
    source_id: str
    actor_id: str
    input_sha256: str | None = None
    output_sha256: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_kind not in {
            "user",
            "system",
            "model",
            "tool",
            "retrieval",
            "memory",
            "verification",
        }:
            raise TrajectoryValidationError("invalid_provenance_source_kind")
        _identifier(self.source_id, label="provenance_source_id")
        _identifier(self.actor_id, label="provenance_actor_id")
        if self.input_sha256 is not None:
            _sha(self.input_sha256, label="provenance_input_sha256")
        if self.output_sha256 is not None:
            _sha(self.output_sha256, label="provenance_output_sha256")
        _object(dict(self.metadata), label="provenance_metadata", maximum=_MAX_PROVENANCE_BYTES)

    def to_json(self) -> JsonObject:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "actor_id": self.actor_id,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, value: object) -> TrajectoryProvenance:
        raw = _object(value, label="provenance", maximum=_MAX_PROVENANCE_BYTES)
        expected = {
            "source_kind",
            "source_id",
            "actor_id",
            "input_sha256",
            "output_sha256",
            "metadata",
        }
        if set(raw) != expected:
            raise TrajectoryCorruptionError("trajectory_provenance_schema_invalid")
        return cls(
            source_kind=cast(TrajectorySourceKind, raw["source_kind"]),
            source_id=cast(str, raw["source_id"]),
            actor_id=cast(str, raw["actor_id"]),
            input_sha256=cast(str | None, raw["input_sha256"]),
            output_sha256=cast(str | None, raw["output_sha256"]),
            metadata=cast(Mapping[str, object], raw["metadata"]),
        )


@dataclass(frozen=True)
class TrajectoryEvent:
    """One immutable, hash-chained event."""

    event_id: str
    event_sequence: int
    identity: TrajectoryIdentity
    event_type: TrajectoryEventType
    idempotency_key: str
    payload: JsonObject
    state_before_sha256: str
    state_after: JsonObject
    provenance: TrajectoryProvenance
    previous_event_sha256: str | None
    event_sha256: str
    created_at_utc: str
    schema_version: int = _SCHEMA_VERSION

    def to_json(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "identity": self.identity.to_json(),
            "event_type": self.event_type.value,
            "idempotency_key": self.idempotency_key,
            "payload": self.payload,
            "payload_sha256": canonical_sha256(self.payload),
            "state_before_sha256": self.state_before_sha256,
            "state_after": self.state_after,
            "state_after_sha256": canonical_sha256(self.state_after),
            "provenance": self.provenance.to_json(),
            "previous_event_sha256": self.previous_event_sha256,
            "event_sha256": self.event_sha256,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class TrajectoryReplay:
    """Exact state reconstructed from authoritative events."""

    identity: TrajectoryIdentity
    state: JsonObject
    events: tuple[TrajectoryEvent, ...]
    checkpoint_used: bool
    checkpoint_recovered: bool
    partial_event_recovered: bool

    def to_json(self) -> JsonObject:
        return {
            "identity": self.identity.to_json(),
            "state": self.state,
            "event_count": len(self.events),
            "checkpoint_used": self.checkpoint_used,
            "checkpoint_recovered": self.checkpoint_recovered,
            "partial_event_recovered": self.partial_event_recovered,
        }


class TrajectoryService:
    """Local append-only event store and derived-checkpoint manager."""

    def __init__(
        self,
        root: Path,
        *,
        failure_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.events_path = self.root / "events.jsonl"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.lock_path = self.root / "trajectory.lock"
        self.failure_hook = failure_hook

    def _fault(self, stage: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(stage)

    def _locked(self) -> BinaryIO:
        self.root.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        lock_file(handle)
        return handle

    @staticmethod
    def _identity_matches(left: TrajectoryIdentity, right: TrajectoryIdentity) -> bool:
        return left.to_json() == right.to_json()

    @staticmethod
    def _event_id(
        identity: TrajectoryIdentity,
        event_type: TrajectoryEventType,
        idempotency_key: str,
        payload_sha256: str,
        state_before_sha256: str,
        state_after_sha256: str,
        provenance: TrajectoryProvenance,
    ) -> str:
        return canonical_sha256(
            {
                "identity": identity.to_json(),
                "event_type": event_type.value,
                "idempotency_key": idempotency_key,
                "payload_sha256": payload_sha256,
                "state_before_sha256": state_before_sha256,
                "state_after_sha256": state_after_sha256,
                "provenance": provenance.to_json(),
            }
        )

    @staticmethod
    def _event_hash(raw: JsonObject) -> str:
        unsigned = dict(raw)
        unsigned.pop("event_sha256", None)
        return canonical_sha256(unsigned)

    def _recover_partial_final_line(self) -> bool:
        if not self.events_path.is_file():
            return False
        data = self.events_path.read_bytes()
        if not data or data.endswith(b"\n"):
            return False
        boundary = data.rfind(b"\n") + 1
        fragment = data[boundary:]
        if len(fragment) > _MAX_PARTIAL_RECOVERY_BYTES:
            raise TrajectoryCorruptionError("trajectory_partial_event_too_large")
        try:
            json.loads(fragment.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            digest = hashlib.sha256(fragment).hexdigest()
            recovery = self.root / f"events.partial.{digest[:16]}.bin"
            if not recovery.exists():
                recovery.write_bytes(fragment)
            with self.events_path.open("r+b") as handle:
                handle.truncate(boundary)
                handle.flush()
                os.fsync(handle.fileno())
            return True
        with self.events_path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True

    @staticmethod
    def _read_json_line(line: str, line_number: int) -> JsonObject:
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryCorruptionError(
                "trajectory_event_json_invalid", {"line": line_number}
            ) from exc
        return _object(raw, label="trajectory_event")

    def _parse_event(
        self,
        raw: JsonObject,
        requested: TrajectoryIdentity,
        expected_sequence: int,
        previous_hash: str | None,
    ) -> TrajectoryEvent:
        expected_keys = {
            "schema_version",
            "event_id",
            "event_sequence",
            "identity",
            "event_type",
            "idempotency_key",
            "payload",
            "payload_sha256",
            "state_before_sha256",
            "state_after",
            "state_after_sha256",
            "provenance",
            "previous_event_sha256",
            "event_sha256",
            "created_at_utc",
        }
        if set(raw) != expected_keys:
            raise TrajectoryCorruptionError("trajectory_event_schema_invalid")
        if raw["schema_version"] != _SCHEMA_VERSION:
            raise TrajectorySchemaError("trajectory_event_schema_unsupported")
        identity_raw = _object(raw["identity"], label="trajectory_identity")
        if set(identity_raw) != set(requested.to_json()):
            raise TrajectoryCorruptionError("trajectory_identity_schema_invalid")
        try:
            identity = TrajectoryIdentity(
                owner_user_id=cast(str, identity_raw["owner_user_id"]),
                project_id=cast(str, identity_raw["project_id"]),
                session_id=cast(str, identity_raw["session_id"]),
                task_id=cast(str, identity_raw["task_id"]),
                schema_version=cast(int, identity_raw["schema_version"]),
            )
        except TrajectoryAuthorizationError:
            raise
        if not self._identity_matches(identity, requested):
            raise TrajectoryAuthorizationError("trajectory_owner_or_project_denied")
        if raw["event_sequence"] != expected_sequence or not isinstance(
            raw["event_sequence"], int
        ):
            raise TrajectoryCorruptionError("trajectory_event_sequence_invalid")
        try:
            event_type = TrajectoryEventType(cast(str, raw["event_type"]))
        except ValueError as exc:
            raise TrajectoryCorruptionError("trajectory_event_type_invalid") from exc
        event_id = _sha(raw["event_id"], label="event_id")
        idempotency_key = _identifier(raw["idempotency_key"], label="idempotency_key")
        payload = _object(raw["payload"], label="trajectory_payload")
        state_before = _sha(raw["state_before_sha256"], label="state_before_sha256")
        state_after = _object(raw["state_after"], label="trajectory_state_after")
        state_after_hash = _sha(raw["state_after_sha256"], label="state_after_sha256")
        if canonical_sha256(state_after) != state_after_hash:
            raise TrajectoryCorruptionError("trajectory_state_hash_mismatch")
        payload_sha256 = _sha(raw["payload_sha256"], label="payload_sha256")
        if canonical_sha256(payload) != payload_sha256:
            raise TrajectoryCorruptionError("trajectory_payload_hash_mismatch")
        provenance = TrajectoryProvenance.from_json(raw["provenance"])
        previous = raw["previous_event_sha256"]
        if previous != previous_hash:
            raise TrajectoryCorruptionError("trajectory_hash_chain_mismatch")
        if previous is not None:
            _sha(previous, label="previous_event_sha256")
        event_hash = _sha(raw["event_sha256"], label="event_sha256")
        created_at = raw["created_at_utc"]
        if not isinstance(created_at, str) or not created_at or len(created_at) > 64:
            raise TrajectoryCorruptionError("trajectory_timestamp_invalid")
        expected_event_id = self._event_id(
            identity,
            event_type,
            idempotency_key,
            payload_sha256,
            state_before,
            state_after_hash,
            provenance,
        )
        if event_id != expected_event_id:
            raise TrajectoryCorruptionError("trajectory_event_id_mismatch")
        if event_hash != self._event_hash(raw):
            raise TrajectoryCorruptionError("trajectory_event_hash_mismatch")
        return TrajectoryEvent(
            event_id=event_id,
            event_sequence=expected_sequence,
            identity=identity,
            event_type=event_type,
            idempotency_key=idempotency_key,
            payload=payload,
            state_before_sha256=state_before,
            state_after=state_after,
            provenance=provenance,
            previous_event_sha256=previous,
            event_sha256=event_hash,
            created_at_utc=created_at,
        )

    def _read_events_unlocked(
        self, requested: TrajectoryIdentity
    ) -> tuple[list[TrajectoryEvent], bool]:
        partial_recovered = self._recover_partial_final_line()
        if not self.events_path.is_file():
            return [], partial_recovered
        events: list[TrajectoryEvent] = []
        previous_hash: str | None = None
        seen_ids: set[str] = set()
        seen_keys: set[str] = set()
        for line_number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue
            event = self._parse_event(
                self._read_json_line(line, line_number),
                requested,
                len(events) + 1,
                previous_hash,
            )
            if event.event_id in seen_ids or event.idempotency_key in seen_keys:
                raise TrajectoryCorruptionError("trajectory_duplicate_event")
            seen_ids.add(event.event_id)
            seen_keys.add(event.idempotency_key)
            events.append(event)
            previous_hash = event.event_sha256
        return events, partial_recovered

    @staticmethod
    def _replay_from_state(
        events: list[TrajectoryEvent],
        state: JsonObject,
        start_sequence: int,
    ) -> JsonObject:
        current = dict(state)
        for event in events[start_sequence:]:
            if event.state_before_sha256 != canonical_sha256(current):
                raise TrajectoryCorruptionError("trajectory_state_transition_mismatch")
            current = dict(event.state_after)
        return current

    def _read_checkpoint_unlocked(
        self,
        identity: TrajectoryIdentity,
        events: list[TrajectoryEvent],
    ) -> tuple[JsonObject, int, bool, bool]:
        if not self.checkpoint_path.is_file():
            return {}, 0, False, False
        try:
            raw: object = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            checkpoint = _object(raw, label="trajectory_checkpoint")
            expected_keys = {
                "schema_version",
                "identity",
                "through_sequence",
                "through_event_sha256",
                "state",
                "state_sha256",
                "created_at_utc",
            }
            if set(checkpoint) != expected_keys:
                raise ValueError("schema")
            if checkpoint["schema_version"] != _SCHEMA_VERSION:
                raise TrajectorySchemaError("trajectory_checkpoint_schema_unsupported")
            checkpoint_identity = _object(checkpoint["identity"], label="checkpoint_identity")
            if checkpoint_identity != identity.to_json():
                raise TrajectoryAuthorizationError("trajectory_owner_or_project_denied")
            through = checkpoint["through_sequence"]
            if not isinstance(through, int) or through < 0 or through > len(events):
                raise ValueError("sequence")
            state = _object(checkpoint["state"], label="checkpoint_state")
            state_hash = _sha(checkpoint["state_sha256"], label="checkpoint_state_sha256")
            if canonical_sha256(state) != state_hash:
                raise ValueError("state")
            event_hash = checkpoint["through_event_sha256"]
            expected_hash = events[through - 1].event_sha256 if through else None
            if event_hash != expected_hash:
                raise ValueError("event")
            if through and events[through - 1].state_after != state:
                raise ValueError("derived_state")
            return state, through, True, False
        except TrajectoryAuthorizationError:
            raise
        except TrajectorySchemaError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, TrajectoryError):
            return {}, 0, False, True

    def replay(self, identity: TrajectoryIdentity) -> TrajectoryReplay:
        """Validate and replay the trajectory, recovering only safe final fragments."""

        handle = self._locked()
        try:
            events, partial_recovered = self._read_events_unlocked(identity)
            checkpoint_state, through, used, checkpoint_recovered = self._read_checkpoint_unlocked(
                identity, events
            )
            if used:
                state = self._replay_from_state(events, checkpoint_state, through)
            else:
                state = self._replay_from_state(events, {}, 0)
            return TrajectoryReplay(
                identity=identity,
                state=state,
                events=tuple(events),
                checkpoint_used=used,
                checkpoint_recovered=checkpoint_recovered,
                partial_event_recovered=partial_recovered,
            )
        finally:
            unlock_file(handle)
            handle.close()

    def read_events(self, identity: TrajectoryIdentity) -> tuple[TrajectoryEvent, ...]:
        return self.replay(identity).events

    def append(
        self,
        identity: TrajectoryIdentity,
        *,
        event_type: TrajectoryEventType,
        idempotency_key: str,
        payload: JsonObject,
        state_before: JsonObject,
        state_after: JsonObject,
        provenance: TrajectoryProvenance,
    ) -> TrajectoryEvent:
        """Append one typed event, returning the existing event for duplicates."""

        if not isinstance(event_type, TrajectoryEventType):
            raise TrajectoryValidationError("invalid_event_type")
        key = _identifier(idempotency_key, label="idempotency_key")
        payload_checked = _object(payload, label="trajectory_payload")
        before = _object(state_before, label="trajectory_state_before")
        after = _object(state_after, label="trajectory_state_after")
        if not isinstance(provenance, TrajectoryProvenance):
            raise TrajectoryValidationError("invalid_provenance")
        before_hash = canonical_sha256(before)
        after_hash = canonical_sha256(after)
        payload_hash = canonical_sha256(payload_checked)
        event_id = self._event_id(
            identity,
            event_type,
            key,
            payload_hash,
            before_hash,
            after_hash,
            provenance,
        )
        handle = self._locked()
        try:
            self._fault("before_append")
            events, _ = self._read_events_unlocked(identity)
            for existing in events:
                if existing.idempotency_key == key:
                    if existing.event_id != event_id:
                        raise TrajectoryConflictError("trajectory_idempotency_conflict")
                    return existing
            current = events[-1].state_after if events else {}
            if before_hash != canonical_sha256(current):
                raise TrajectoryConflictError("trajectory_state_before_conflict")
            previous_hash = events[-1].event_sha256 if events else None
            raw: JsonObject = {
                "schema_version": _SCHEMA_VERSION,
                "event_id": event_id,
                "event_sequence": len(events) + 1,
                "identity": identity.to_json(),
                "event_type": event_type.value,
                "idempotency_key": key,
                "payload": payload_checked,
                "payload_sha256": payload_hash,
                "state_before_sha256": before_hash,
                "state_after": after,
                "state_after_sha256": after_hash,
                "provenance": provenance.to_json(),
                "previous_event_sha256": previous_hash,
                "created_at_utc": _now(),
            }
            raw["event_sha256"] = self._event_hash(raw)
            encoded = canonical_json_bytes(raw) + b"\n"
            self._fault("before_event_write")
            with self.events_path.open("ab") as writer:
                if self.failure_hook is not None:
                    writer.write(encoded[: max(1, len(encoded) // 2)])
                    writer.flush()
                    self._fault("after_partial_event_write")
                    writer.write(encoded[max(1, len(encoded) // 2) :])
                else:
                    writer.write(encoded)
                writer.flush()
                self._fault("after_event_write_before_fsync")
                os.fsync(writer.fileno())
            self._fault("after_event_fsync")
            return self._parse_event(raw, identity, len(events) + 1, previous_hash)
        finally:
            unlock_file(handle)
            handle.close()

    def checkpoint(self, identity: TrajectoryIdentity) -> Path:
        """Persist an atomic acceleration snapshot derived from replayable events."""

        handle = self._locked()
        try:
            events, _ = self._read_events_unlocked(identity)
            state = self._replay_from_state(events, {}, 0)
            self._fault("after_event_before_checkpoint")
            value: JsonObject = {
                "schema_version": _SCHEMA_VERSION,
                "identity": identity.to_json(),
                "through_sequence": len(events),
                "through_event_sha256": events[-1].event_sha256 if events else None,
                "state": state,
                "state_sha256": canonical_sha256(state),
                "created_at_utc": _now(),
            }
            temporary = self.checkpoint_path.with_name(
                f".{self.checkpoint_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
            with temporary.open("w", encoding="utf-8", newline="\n") as writer:
                writer.write(data)
                writer.flush()
                self._fault("after_checkpoint_write_before_fsync")
                os.fsync(writer.fileno())
            self._fault("before_checkpoint_replace")
            os.replace(temporary, self.checkpoint_path)
            if os.name != "nt":
                directory_fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return self.checkpoint_path
        finally:
            unlock_file(handle)
            handle.close()

    def migrate(self, *, target_schema_version: int = _SCHEMA_VERSION) -> JsonObject:
        """Validate the current format; unknown versions fail closed.

        Version 1 has no rewrite migration.  Future migrations must be added
        as explicit, tested transforms rather than silently coercing records.
        """

        if target_schema_version != _SCHEMA_VERSION:
            raise TrajectorySchemaError("trajectory_migration_target_unsupported")
        return {
            "status": "CURRENT",
            "schema_version": _SCHEMA_VERSION,
            "migration_policy": "explicit_transform_only",
        }

    def status(self, identity: TrajectoryIdentity) -> JsonObject:
        replay = self.replay(identity)
        return {
            "identity": identity.to_json(),
            "event_count": len(replay.events),
            "state_sha256": canonical_sha256(replay.state),
            "checkpoint_used": replay.checkpoint_used,
            "checkpoint_recovered": replay.checkpoint_recovered,
            "partial_event_recovered": replay.partial_event_recovered,
        }
