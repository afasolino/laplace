"""Bounded, shadow-only idle consolidation for Laplace.

The consolidator is deliberately a deterministic proposal engine.  It reads
bounded trajectory and memory snapshots, writes only an atomic proposal ledger,
and never changes authoritative memory, rules, routing, skills, source code, or
security policy.  Candidate harness changes use an explicit A/B evidence
record; a passing record still needs a human approval before any future
promotion path could consume it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
RecordLike: TypeAlias = Mapping[str, object] | object

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_DANGEROUS = re.compile(
    r"(?:recursive|unrestricted|self[- ]?modify|auto[- ]?promot|bypass|override|disable)"
    r"|\brm\s+-rf\b|\b(?:curl|wget)\b.*\|\s*(?:sh|bash)\b",
    re.IGNORECASE,
)
_FORBIDDEN_KEYS = frozenset(
    {"prompt", "response", "secret", "secrets", "token", "password", "source_code"}
)


class ProposalKind(str, Enum):
    EPISODIC_SUMMARY = "episodic_summary"
    DURABLE_FACT = "durable_fact"
    CONTRADICTION = "contradiction"
    OBSOLETE_MEMORY = "obsolete_memory"
    CANDIDATE_SKILL = "candidate_skill"
    RECURRING_FAILURE = "recurring_failure"
    PROCESS_IMPROVEMENT = "process_improvement"
    CODE_CHANGE = "code_change"


class ProposalState(str, Enum):
    PROPOSED = "proposed"
    HUMAN_APPROVED = "human-approved"
    REJECTED = "rejected"


class ConsolidationError(RuntimeError):
    """Base class for fail-closed consolidation errors."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class ConsolidationConflictError(ConsolidationError):
    """An idempotency, owner, scope, or evidence conflict was detected."""


class ConsolidationBudgetError(ConsolidationError):
    """A bounded maintenance resource was exhausted."""


class ConsolidationCorruptionError(ConsolidationError):
    """The persisted proposal ledger is malformed or unsupported."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConsolidationError("consolidation_json_invalid") from exc


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ConsolidationError(f"invalid_{label}")
    return value


def _text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ConsolidationError(f"invalid_{label}")
    return value


def _optional_text(value: object, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum)


def _validate_json(value: object, *, depth: int = 0) -> None:
    if depth > 8:
        raise ConsolidationError("consolidation_value_too_deep")
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and len(value) > 16_384:
            raise ConsolidationError("consolidation_value_too_large")
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ConsolidationError("consolidation_nonfinite_value")
        return
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise ConsolidationError("consolidation_mapping_too_large")
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ConsolidationError("consolidation_mapping_key_invalid")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ConsolidationError("consolidation_sensitive_key_rejected")
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 128:
            raise ConsolidationError("consolidation_sequence_too_large")
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    raise ConsolidationError("consolidation_value_type_invalid")


def _record_json(value: RecordLike, label: str) -> JsonObject:
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        to_json = getattr(value, "to_json", None)
        if not callable(to_json):
            raise ConsolidationError(f"{label}_record_invalid")
        converted = to_json()
        if not isinstance(converted, Mapping):
            raise ConsolidationError(f"{label}_record_invalid")
        raw = dict(converted)
    _validate_json(raw)
    return raw


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


@dataclass(frozen=True)
class MaintenanceBudget:
    """Hard limits for one local maintenance service."""

    max_trajectory_events: int = 256
    max_memory_records: int = 256
    max_input_bytes: int = 256_000
    max_proposals_per_cycle: int = 64
    max_cycles_stored: int = 128
    max_state_bytes: int = 4_000_000
    max_logical_tokens: int = 16_000
    min_interval_seconds: float = 60.0
    max_cycles_per_window: int = 1
    max_gpu_seconds: float = 0.0
    obsolete_after_days: int = 30

    def __post_init__(self) -> None:
        integer_fields = (
            "max_trajectory_events",
            "max_memory_records",
            "max_input_bytes",
            "max_proposals_per_cycle",
            "max_cycles_stored",
            "max_state_bytes",
            "max_logical_tokens",
            "max_cycles_per_window",
            "obsolete_after_days",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConsolidationError("maintenance_budget_invalid", {"field": name})
        if (
            isinstance(self.min_interval_seconds, bool)
            or not isinstance(self.min_interval_seconds, (int, float))
            or self.min_interval_seconds < 0
        ):
            raise ConsolidationError("maintenance_budget_interval_invalid")
        if (
            isinstance(self.max_gpu_seconds, bool)
            or not isinstance(self.max_gpu_seconds, (int, float))
            or self.max_gpu_seconds < 0
        ):
            raise ConsolidationError("maintenance_budget_gpu_invalid")

    def to_json(self) -> JsonObject:
        return {
            "max_trajectory_events": self.max_trajectory_events,
            "max_memory_records": self.max_memory_records,
            "max_input_bytes": self.max_input_bytes,
            "max_proposals_per_cycle": self.max_proposals_per_cycle,
            "max_cycles_stored": self.max_cycles_stored,
            "max_state_bytes": self.max_state_bytes,
            "max_logical_tokens": self.max_logical_tokens,
            "min_interval_seconds": float(self.min_interval_seconds),
            "max_cycles_per_window": self.max_cycles_per_window,
            "max_gpu_seconds": float(self.max_gpu_seconds),
            "obsolete_after_days": self.obsolete_after_days,
        }


@dataclass(frozen=True)
class ConsolidationProvenance:
    owner_id: str
    project_id: str
    session_id: str
    cycle_id: str
    source_event_ids: tuple[str, ...]
    source_memory_ids: tuple[str, ...]
    input_sha256: str
    analyzer: str = "deterministic-laplace-g10"
    policy_fingerprint: str = "shadow-only-no-production-mutation"
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.owner_id, "owner_id"),
            (self.project_id, "project_id"),
            (self.session_id, "session_id"),
            (self.cycle_id, "cycle_id"),
        ):
            _identifier(value, label)
        for values, label in (
            (self.source_event_ids, "source_event_ids"),
            (self.source_memory_ids, "source_memory_ids"),
        ):
            if len(values) > 256 or any(_IDENTIFIER.fullmatch(item) is None for item in values):
                raise ConsolidationError("consolidation_provenance_sources_invalid", {"field": label})
        if _SHA256.fullmatch(self.input_sha256) is None:
            raise ConsolidationError("consolidation_input_hash_invalid")
        _text(self.analyzer, "consolidation_analyzer", 128)
        _text(self.policy_fingerprint, "consolidation_policy_fingerprint", 256)
        _text(self.created_at_utc, "consolidation_created_at", 64)

    def to_json(self) -> JsonObject:
        return {
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "source_event_ids": list(self.source_event_ids),
            "source_memory_ids": list(self.source_memory_ids),
            "input_sha256": self.input_sha256,
            "analyzer": self.analyzer,
            "policy_fingerprint": self.policy_fingerprint,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class ConsolidationProposal:
    proposal_id: str
    kind: ProposalKind
    state: ProposalState
    owner_id: str
    project_id: str
    summary: str
    details: JsonObject
    provenance: ConsolidationProvenance
    requires_human_approval: bool = True
    active: bool = False

    def __post_init__(self) -> None:
        _identifier(self.proposal_id, "proposal_id")
        _identifier(self.owner_id, "owner_id")
        _identifier(self.project_id, "project_id")
        _text(self.summary, "proposal_summary", 2_048)
        _validate_json(self.details)
        if not isinstance(self.requires_human_approval, bool) or not isinstance(self.active, bool):
            raise ConsolidationError("proposal_flags_invalid")
        if self.active or not self.requires_human_approval:
            raise ConsolidationError("shadow_proposal_policy_violation")

    def to_json(self) -> JsonObject:
        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind.value,
            "state": self.state.value,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "summary": self.summary,
            "details": self.details,
            "provenance": self.provenance.to_json(),
            "requires_human_approval": self.requires_human_approval,
            "active": self.active,
        }


@dataclass(frozen=True)
class ABEvidence:
    """Frozen, hash-bound evidence for one candidate harness change."""

    baseline_result_sha256: str
    candidate_result_sha256: str
    frozen_task_ids: tuple[str, ...]
    development_task_ids: tuple[str, ...]
    held_out_task_ids: tuple[str, ...]
    baseline_correct: bool
    candidate_correct: bool
    security_regression: bool
    correctness_regression: bool = False
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.baseline_result_sha256, "baseline_result_sha256"),
            (self.candidate_result_sha256, "candidate_result_sha256"),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ConsolidationError("ab_result_hash_invalid", {"field": label})
        all_ids = self.frozen_task_ids + self.development_task_ids + self.held_out_task_ids
        if not all_ids or len(all_ids) > 256 or any(_IDENTIFIER.fullmatch(item) is None for item in all_ids):
            raise ConsolidationError("ab_task_ids_invalid")
        if set(self.frozen_task_ids) & set(self.development_task_ids):
            raise ConsolidationError("ab_development_overlap")
        if set(self.frozen_task_ids) & set(self.held_out_task_ids):
            raise ConsolidationError("ab_held_out_overlap")
        if set(self.development_task_ids) & set(self.held_out_task_ids):
            raise ConsolidationError("ab_held_out_overlap")
        if not all(isinstance(value, bool) for value in (self.baseline_correct, self.candidate_correct, self.security_regression, self.correctness_regression)):
            raise ConsolidationError("ab_flags_invalid")
        _text(self.observed_at_utc, "ab_observed_at", 64)

    @property
    def passes(self) -> bool:
        return (
            self.baseline_correct
            and self.candidate_correct
            and not self.security_regression
            and not self.correctness_regression
        )

    def to_json(self) -> JsonObject:
        return {
            "baseline_result_sha256": self.baseline_result_sha256,
            "candidate_result_sha256": self.candidate_result_sha256,
            "frozen_task_ids": list(self.frozen_task_ids),
            "development_task_ids": list(self.development_task_ids),
            "held_out_task_ids": list(self.held_out_task_ids),
            "baseline_correct": self.baseline_correct,
            "candidate_correct": self.candidate_correct,
            "security_regression": self.security_regression,
            "correctness_regression": self.correctness_regression,
            "observed_at_utc": self.observed_at_utc,
            "passes": self.passes,
        }


@dataclass(frozen=True)
class ConsolidationCycle:
    cycle_id: str
    window_id: str
    owner_id: str
    project_id: str
    session_id: str
    input_sha256: str
    proposal_ids: tuple[str, ...]
    duplicate_proposal_ids: tuple[str, ...]
    started_at_utc: str
    completed_at_utc: str
    status: str
    logical_tokens_used: int
    gpu_seconds_used: float

    def to_json(self) -> JsonObject:
        return {
            "cycle_id": self.cycle_id,
            "window_id": self.window_id,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "input_sha256": self.input_sha256,
            "proposal_ids": list(self.proposal_ids),
            "duplicate_proposal_ids": list(self.duplicate_proposal_ids),
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "status": self.status,
            "logical_tokens_used": self.logical_tokens_used,
            "gpu_seconds_used": self.gpu_seconds_used,
        }


@dataclass(frozen=True)
class ConsolidationReport:
    cycle: ConsolidationCycle
    proposals: tuple[ConsolidationProposal, ...]
    replayed: bool
    shadow_only: bool = True

    def __post_init__(self) -> None:
        if not self.shadow_only:
            raise ConsolidationError("consolidation_shadow_policy_violation")

    def to_json(self) -> JsonObject:
        return {
            "cycle": self.cycle.to_json(),
            "proposals": [proposal.to_json() for proposal in self.proposals],
            "replayed": self.replayed,
            "shadow_only": self.shadow_only,
        }


class IdleConsolidator:
    """Crash-safe deterministic proposal ledger for bounded idle windows."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: Path,
        *,
        budget: MaintenanceBudget | None = None,
        failure_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.state_path = self.root / "consolidation.json"
        self.budget = budget or MaintenanceBudget()
        self.failure_hook = failure_hook
        self._lock = threading.RLock()
        self._revision = 0
        self._cycles: dict[str, JsonObject] = {}
        self._proposals: dict[str, JsonObject] = {}
        self._windows: dict[str, JsonObject] = {}
        self._improvements: dict[str, JsonObject] = {}
        if self.state_path.is_symlink():
            raise ConsolidationCorruptionError("consolidation_state_symlink_rejected")
        self._load()

    def _fault(self, stage: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(stage)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw: object = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConsolidationCorruptionError("consolidation_state_unreadable") from exc
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version", "revision", "cycles", "proposals", "windows", "improvements"
        }:
            raise ConsolidationCorruptionError("consolidation_state_keys_invalid")
        if raw["schema_version"] != self.SCHEMA_VERSION or not isinstance(raw["revision"], int):
            raise ConsolidationCorruptionError("consolidation_state_schema_unsupported")
        for name in ("cycles", "proposals", "windows", "improvements"):
            if not isinstance(raw[name], dict) or not all(
                isinstance(key, str) and isinstance(value, dict) for key, value in cast(dict[object, object], raw[name]).items()
            ):
                raise ConsolidationCorruptionError(f"consolidation_state_{name}_invalid")
        self._revision = raw["revision"]
        self._cycles = cast(dict[str, JsonObject], raw["cycles"])
        self._proposals = cast(dict[str, JsonObject], raw["proposals"])
        self._windows = cast(dict[str, JsonObject], raw["windows"])
        self._improvements = cast(dict[str, JsonObject], raw["improvements"])
        if len(self._cycles) > self.budget.max_cycles_stored:
            raise ConsolidationCorruptionError("consolidation_state_cycle_limit_exceeded")
        for cycle in self._cycles.values():
            self._validate_cycle(cycle)
        for proposal in self._proposals.values():
            self._validate_proposal_json(proposal)
            self._proposal_from_json(proposal)
        for window in self._windows.values():
            self._validate_window(window)
        for improvement in self._improvements.values():
            self._validate_improvement(improvement)

    @staticmethod
    def _validate_cycle(value: Mapping[str, object]) -> None:
        expected = {
            "cycle_id", "window_id", "owner_id", "project_id", "session_id", "input_sha256",
            "proposal_ids", "duplicate_proposal_ids", "started_at_utc", "completed_at_utc",
            "status", "logical_tokens_used", "gpu_seconds_used",
        }
        if set(value) != expected:
            raise ConsolidationCorruptionError("consolidation_cycle_keys_invalid")
        for key in ("cycle_id", "window_id", "owner_id", "project_id", "session_id", "started_at_utc", "completed_at_utc", "status"):
            if not isinstance(value[key], str) or not value[key]:
                raise ConsolidationCorruptionError("consolidation_cycle_text_invalid")
        if _SHA256.fullmatch(cast(str, value["input_sha256"])) is None:
            raise ConsolidationCorruptionError("consolidation_cycle_hash_invalid")
        for key in ("proposal_ids", "duplicate_proposal_ids"):
            if not isinstance(value[key], list) or not all(isinstance(item, str) for item in cast(list[object], value[key])):
                raise ConsolidationCorruptionError("consolidation_cycle_proposals_invalid")
        if not isinstance(value["logical_tokens_used"], int) or value["logical_tokens_used"] < 0:
            raise ConsolidationCorruptionError("consolidation_cycle_tokens_invalid")
        if not isinstance(value["gpu_seconds_used"], (int, float)) or value["gpu_seconds_used"] != 0:
            raise ConsolidationCorruptionError("consolidation_cycle_gpu_invalid")

    @staticmethod
    def _validate_provenance_json(value: object) -> None:
        if not isinstance(value, dict) or set(value) != {
            "owner_id", "project_id", "session_id", "cycle_id", "source_event_ids", "source_memory_ids",
            "input_sha256", "analyzer", "policy_fingerprint", "created_at_utc",
        }:
            raise ConsolidationCorruptionError("consolidation_provenance_keys_invalid")
        for key in ("owner_id", "project_id", "session_id", "cycle_id", "analyzer", "policy_fingerprint", "created_at_utc"):
            if not isinstance(value[key], str) or not value[key]:
                raise ConsolidationCorruptionError("consolidation_provenance_text_invalid")
        for key in ("source_event_ids", "source_memory_ids"):
            if not isinstance(value[key], list) or not all(isinstance(item, str) for item in cast(list[object], value[key])):
                raise ConsolidationCorruptionError("consolidation_provenance_sources_invalid")
        if not isinstance(value["input_sha256"], str) or _SHA256.fullmatch(value["input_sha256"]) is None:
            raise ConsolidationCorruptionError("consolidation_provenance_hash_invalid")

    @classmethod
    def _validate_proposal_json(cls, value: Mapping[str, object]) -> None:
        expected = {
            "proposal_id", "kind", "state", "owner_id", "project_id", "summary", "details",
            "provenance", "requires_human_approval", "active",
        }
        if set(value) != expected:
            raise ConsolidationCorruptionError("consolidation_proposal_keys_invalid")
        if value["kind"] not in {item.value for item in ProposalKind} or value["state"] not in {item.value for item in ProposalState}:
            raise ConsolidationCorruptionError("consolidation_proposal_enum_invalid")
        for key in ("proposal_id", "owner_id", "project_id", "summary"):
            if not isinstance(value[key], str) or not value[key]:
                raise ConsolidationCorruptionError("consolidation_proposal_text_invalid")
        if not isinstance(value["details"], dict):
            raise ConsolidationCorruptionError("consolidation_proposal_details_invalid")
        cls._validate_provenance_json(value["provenance"])
        if value["requires_human_approval"] is not True or value["active"] is not False:
            raise ConsolidationCorruptionError("consolidation_proposal_policy_invalid")

    @staticmethod
    def _validate_window(value: Mapping[str, object]) -> None:
        if set(value) != {"owner_id", "project_id", "cycle_ids", "last_cycle_at"}:
            raise ConsolidationCorruptionError("consolidation_window_keys_invalid")
        if not all(isinstance(value[key], str) and value[key] for key in ("owner_id", "project_id", "last_cycle_at")):
            raise ConsolidationCorruptionError("consolidation_window_text_invalid")
        if not isinstance(value["cycle_ids"], list) or not all(isinstance(item, str) for item in cast(list[object], value["cycle_ids"])):
            raise ConsolidationCorruptionError("consolidation_window_cycles_invalid")

    @staticmethod
    def _validate_improvement(value: Mapping[str, object]) -> None:
        if set(value) != {"proposal_id", "ab_evidence", "decision", "human_approval"}:
            raise ConsolidationCorruptionError("consolidation_improvement_keys_invalid")
        if not isinstance(value["proposal_id"], str) or not isinstance(value["decision"], str):
            raise ConsolidationCorruptionError("consolidation_improvement_text_invalid")
        if value["ab_evidence"] is not None and not isinstance(value["ab_evidence"], dict):
            raise ConsolidationCorruptionError("consolidation_improvement_evidence_invalid")
        if value["human_approval"] is not None and not isinstance(value["human_approval"], dict):
            raise ConsolidationCorruptionError("consolidation_improvement_approval_invalid")

    def _payload(self) -> JsonObject:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "revision": self._revision + 1,
            "cycles": {key: self._cycles[key] for key in sorted(self._cycles)},
            "proposals": {key: self._proposals[key] for key in sorted(self._proposals)},
            "windows": {key: self._windows[key] for key in sorted(self._windows)},
            "improvements": {key: self._improvements[key] for key in sorted(self._improvements)},
        }

    def _persist_candidate(
        self,
        cycles: dict[str, JsonObject],
        proposals: dict[str, JsonObject],
        windows: dict[str, JsonObject],
        improvements: dict[str, JsonObject],
    ) -> None:
        candidate: JsonObject = {
            "schema_version": self.SCHEMA_VERSION,
            "revision": self._revision + 1,
            "cycles": {key: cycles[key] for key in sorted(cycles)},
            "proposals": {key: proposals[key] for key in sorted(proposals)},
            "windows": {key: windows[key] for key in sorted(windows)},
            "improvements": {key: improvements[key] for key in sorted(improvements)},
        }
        encoded = json.dumps(candidate, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        if len(encoded) > self.budget.max_state_bytes:
            raise ConsolidationBudgetError("consolidation_state_budget_exceeded", {"bytes": len(encoded)})
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._fault("before_state_replace")
            os.replace(temporary, self.state_path)
            if os.name != "nt":
                directory_fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError as exc:
            raise ConsolidationError("consolidation_state_persist_failed") from exc
        self._cycles = cycles
        self._proposals = proposals
        self._windows = windows
        self._improvements = improvements
        self._revision += 1

    @staticmethod
    def _proposal_from_json(raw: Mapping[str, object]) -> ConsolidationProposal:
        provenance_raw = cast(Mapping[str, object], raw["provenance"])
        source_events = tuple(cast(list[str], provenance_raw["source_event_ids"]))
        source_memories = tuple(cast(list[str], provenance_raw["source_memory_ids"]))
        provenance = ConsolidationProvenance(
            owner_id=cast(str, provenance_raw["owner_id"]),
            project_id=cast(str, provenance_raw["project_id"]),
            session_id=cast(str, provenance_raw["session_id"]),
            cycle_id=cast(str, provenance_raw["cycle_id"]),
            source_event_ids=source_events,
            source_memory_ids=source_memories,
            input_sha256=cast(str, provenance_raw["input_sha256"]),
            analyzer=cast(str, provenance_raw["analyzer"]),
            policy_fingerprint=cast(str, provenance_raw["policy_fingerprint"]),
            created_at_utc=cast(str, provenance_raw["created_at_utc"]),
        )
        return ConsolidationProposal(
            proposal_id=cast(str, raw["proposal_id"]),
            kind=ProposalKind(cast(str, raw["kind"])),
            state=ProposalState(cast(str, raw["state"])),
            owner_id=cast(str, raw["owner_id"]),
            project_id=cast(str, raw["project_id"]),
            summary=cast(str, raw["summary"]),
            details=cast(JsonObject, raw["details"]),
            provenance=provenance,
            requires_human_approval=cast(bool, raw["requires_human_approval"]),
            active=cast(bool, raw["active"]),
        )

    def _normalize_inputs(
        self,
        trajectory_events: Sequence[RecordLike],
        memories: Sequence[RecordLike],
        *,
        owner_id: str,
        project_id: str,
    ) -> tuple[list[JsonObject], list[JsonObject], str, int]:
        if len(trajectory_events) > self.budget.max_trajectory_events:
            raise ConsolidationBudgetError("trajectory_event_budget_exceeded")
        if len(memories) > self.budget.max_memory_records:
            raise ConsolidationBudgetError("memory_record_budget_exceeded")
        events = [_record_json(item, "trajectory") for item in trajectory_events]
        memory_records = [_record_json(item, "memory") for item in memories]
        for event in events:
            event_id = event.get("event_id")
            _identifier(event_id, "event_id")
            identity = event.get("identity")
            if isinstance(identity, Mapping):
                if identity.get("owner_user_id") != owner_id or identity.get("project_id") != project_id:
                    raise ConsolidationConflictError("trajectory_owner_or_project_denied")
        for memory in memory_records:
            _identifier(memory.get("memory_id"), "memory_id")
            if memory.get("owner_id") != owner_id or memory.get("project_id") != project_id:
                raise ConsolidationConflictError("memory_owner_or_project_denied")
        events.sort(key=lambda value: (value.get("event_sequence", 0), cast(str, value["event_id"])))
        memory_records.sort(key=lambda value: cast(str, value["memory_id"]))
        normalized = {"trajectory_events": events, "memories": memory_records}
        encoded = _canonical(normalized)
        if len(encoded) > self.budget.max_input_bytes:
            raise ConsolidationBudgetError("consolidation_input_bytes_exceeded", {"bytes": len(encoded)})
        logical_tokens = max(1, (len(encoded) + 3) // 4)
        if logical_tokens > self.budget.max_logical_tokens:
            raise ConsolidationBudgetError("consolidation_logical_token_budget_exceeded")
        return events, memory_records, _sha(normalized), logical_tokens

    @staticmethod
    def _source_ids(records: Sequence[Mapping[str, object]], key: str) -> tuple[str, ...]:
        return tuple(cast(str, record[key]) for record in records)

    def _make_proposal(
        self,
        *,
        kind: ProposalKind,
        owner_id: str,
        project_id: str,
        session_id: str,
        cycle_id: str,
        input_sha256: str,
        source_event_ids: Sequence[str],
        source_memory_ids: Sequence[str],
        summary: str,
        details: JsonObject,
        created_at_utc: str,
    ) -> ConsolidationProposal:
        if _DANGEROUS.search(summary) or _DANGEROUS.search(json.dumps(details, sort_keys=True)):
            raise ConsolidationError("unsafe_consolidation_proposal")
        provenance = ConsolidationProvenance(
            owner_id=owner_id,
            project_id=project_id,
            session_id=session_id,
            cycle_id=cycle_id,
            source_event_ids=tuple(sorted(set(source_event_ids))),
            source_memory_ids=tuple(sorted(set(source_memory_ids))),
            input_sha256=input_sha256,
            created_at_utc=created_at_utc,
        )
        identity = {
            "kind": kind.value,
            "owner_id": owner_id,
            "project_id": project_id,
            "summary": summary,
            "details": details,
            "source_event_ids": list(provenance.source_event_ids),
            "source_memory_ids": list(provenance.source_memory_ids),
        }
        proposal_id = f"prop_{_sha(identity)[:32]}"
        return ConsolidationProposal(
            proposal_id=proposal_id,
            kind=kind,
            state=ProposalState.PROPOSED,
            owner_id=owner_id,
            project_id=project_id,
            summary=summary,
            details=details,
            provenance=provenance,
        )

    def _analyze(
        self,
        events: Sequence[JsonObject],
        memories: Sequence[JsonObject],
        *,
        owner_id: str,
        project_id: str,
        session_id: str,
        cycle_id: str,
        input_sha256: str,
        created_at_utc: str,
    ) -> list[ConsolidationProposal]:
        proposals: list[ConsolidationProposal] = []
        event_ids = self._source_ids(events, "event_id")
        if events:
            last_type = str(events[-1].get("event_type", "unknown"))
            proposals.append(
                self._make_proposal(
                    kind=ProposalKind.EPISODIC_SUMMARY,
                    owner_id=owner_id,
                    project_id=project_id,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    input_sha256=input_sha256,
                    source_event_ids=event_ids,
                    source_memory_ids=(),
                    summary=f"Trajectory window contains {len(events)} recorded event(s); last type: {last_type}.",
                    details={"event_count": len(events), "last_event_type": last_type, "mode": "shadow"},
                    created_at_utc=created_at_utc,
                )
            )
        for event in events:
            event_type = str(event.get("event_type", ""))
            payload = event.get("payload")
            if event_type not in {"completion", "verification", "memory_accessed", "edit"} or not isinstance(payload, Mapping):
                continue
            fact = payload.get("fact")
            if isinstance(fact, str) and fact and len(fact) <= 2_048:
                proposals.append(
                    self._make_proposal(
                        kind=ProposalKind.DURABLE_FACT,
                        owner_id=owner_id,
                        project_id=project_id,
                        session_id=session_id,
                        cycle_id=cycle_id,
                        input_sha256=input_sha256,
                        source_event_ids=(cast(str, event["event_id"]),),
                        source_memory_ids=(),
                        summary="Candidate durable fact copied from an explicitly labelled event fact.",
                        details={"fact": fact, "write_mode": "requires_human_approval", "mode": "shadow"},
                        created_at_utc=created_at_utc,
                    )
                )
        failures: dict[str, list[JsonObject]] = {}
        for event in events:
            if str(event.get("event_type", "")) != "failure":
                continue
            payload = event.get("payload")
            category = "failure"
            if isinstance(payload, Mapping):
                candidate = payload.get("category", payload.get("reason", "failure"))
                if isinstance(candidate, str) and candidate:
                    category = _normalize(candidate)[:256]
            failures.setdefault(category, []).append(event)
        for category, grouped in sorted(failures.items()):
            if len(grouped) < 2:
                continue
            grouped_ids = self._source_ids(grouped, "event_id")
            common = {"category": category, "occurrences": len(grouped), "mode": "shadow"}
            proposals.append(
                self._make_proposal(
                    kind=ProposalKind.RECURRING_FAILURE,
                    owner_id=owner_id,
                    project_id=project_id,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    input_sha256=input_sha256,
                    source_event_ids=grouped_ids,
                    source_memory_ids=(),
                    summary=f"Recurring deterministic failure category observed: {category}.",
                    details=common,
                    created_at_utc=created_at_utc,
                )
            )
            proposals.append(
                self._make_proposal(
                    kind=ProposalKind.CANDIDATE_SKILL,
                    owner_id=owner_id,
                    project_id=project_id,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    input_sha256=input_sha256,
                    source_event_ids=grouped_ids,
                    source_memory_ids=(),
                    summary="Candidate skill proposed for human review of a recurring failure.",
                    details={
                        "skill_id": f"candidate-{_sha(category)[:16]}",
                        "trigger": [category],
                        "procedure": "Human review: inspect the cited failures and add a deterministic regression test.",
                        "required_tools": ["inspect_diff", "verify"],
                        "required_verifiers": ["pytest"],
                        "lifecycle": "candidate",
                        "mode": "shadow",
                    },
                    created_at_utc=created_at_utc,
                )
            )
            proposals.append(
                self._make_proposal(
                    kind=ProposalKind.PROCESS_IMPROVEMENT,
                    owner_id=owner_id,
                    project_id=project_id,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    input_sha256=input_sha256,
                    source_event_ids=grouped_ids,
                    source_memory_ids=(),
                    summary="Candidate process improvement: add a deterministic regression gate for the recurring failure.",
                    details={"change": "add_frozen_regression_gate", "category": category, "mode": "shadow"},
                    created_at_utc=created_at_utc,
                )
            )
            proposals.append(
                self._make_proposal(
                    kind=ProposalKind.CODE_CHANGE,
                    owner_id=owner_id,
                    project_id=project_id,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    input_sha256=input_sha256,
                    source_event_ids=grouped_ids,
                    source_memory_ids=(),
                    summary="Optional candidate code change recorded for human review only.",
                    details={"change": "no automatic source edit; create a narrowly scoped test first", "files": [], "mode": "shadow"},
                    created_at_utc=created_at_utc,
                )
            )
        active_memories = [memory for memory in memories if memory.get("state") == "ACTIVE"]
        by_contradiction: dict[str, list[JsonObject]] = {}
        for memory in active_memories:
            key = memory.get("contradiction_key")
            if isinstance(key, str) and key:
                by_contradiction.setdefault(key, []).append(memory)
        for key, grouped in sorted(by_contradiction.items()):
            contents = {_normalize(cast(str, memory["content"])) for memory in grouped if isinstance(memory.get("content"), str)}
            if len(grouped) > 1 and len(contents) > 1:
                memory_ids = self._source_ids(grouped, "memory_id")
                proposals.append(
                    self._make_proposal(
                        kind=ProposalKind.CONTRADICTION,
                        owner_id=owner_id,
                        project_id=project_id,
                        session_id=session_id,
                        cycle_id=cycle_id,
                        input_sha256=input_sha256,
                        source_event_ids=(),
                        source_memory_ids=memory_ids,
                        summary="Contradictory active memories require human supersession review.",
                        details={"contradiction_key": key, "memory_ids": list(memory_ids), "action": "review_supersession", "delete_allowed": False, "mode": "shadow"},
                        created_at_utc=created_at_utc,
                    )
                )
        by_content: dict[str, list[JsonObject]] = {}
        for memory in active_memories:
            content = memory.get("content")
            if isinstance(content, str):
                by_content.setdefault(_normalize(content), []).append(memory)
        for grouped in by_content.values():
            if len(grouped) < 2:
                continue
            memory_ids = self._source_ids(grouped, "memory_id")
            proposals.append(
                self._make_proposal(
                    kind=ProposalKind.OBSOLETE_MEMORY,
                    owner_id=owner_id,
                    project_id=project_id,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    input_sha256=input_sha256,
                    source_event_ids=(),
                    source_memory_ids=memory_ids,
                    summary="Duplicate active memories may be demoted after human review.",
                    details={"memory_ids": list(memory_ids), "action": "demote_duplicate", "delete_allowed": False, "mode": "shadow"},
                    created_at_utc=created_at_utc,
                )
            )
        now = _timestamp(created_at_utc)
        if now is not None:
            cutoff = now - timedelta(days=self.budget.obsolete_after_days)
            for memory in active_memories:
                updated = _timestamp(memory.get("updated_at_utc"))
                if updated is None or updated > cutoff or memory.get("kind") != "episodic":
                    continue
                memory_id = cast(str, memory["memory_id"])
                proposals.append(
                    self._make_proposal(
                        kind=ProposalKind.OBSOLETE_MEMORY,
                        owner_id=owner_id,
                        project_id=project_id,
                        session_id=session_id,
                        cycle_id=cycle_id,
                        input_sha256=input_sha256,
                        source_event_ids=(),
                        source_memory_ids=(memory_id,),
                        summary="Old episodic memory is a candidate for demotion after human review.",
                        details={"memory_id": memory_id, "action": "demote_obsolete", "delete_allowed": False, "mode": "shadow"},
                        created_at_utc=created_at_utc,
                    )
                )
        return proposals

    def _report_for_cycle(self, cycle: JsonObject, *, replayed: bool) -> ConsolidationReport:
        cycle_value = ConsolidationCycle(
            cycle_id=cast(str, cycle["cycle_id"]),
            window_id=cast(str, cycle["window_id"]),
            owner_id=cast(str, cycle["owner_id"]),
            project_id=cast(str, cycle["project_id"]),
            session_id=cast(str, cycle["session_id"]),
            input_sha256=cast(str, cycle["input_sha256"]),
            proposal_ids=tuple(cast(list[str], cycle["proposal_ids"])),
            duplicate_proposal_ids=tuple(cast(list[str], cycle["duplicate_proposal_ids"])),
            started_at_utc=cast(str, cycle["started_at_utc"]),
            completed_at_utc=cast(str, cycle["completed_at_utc"]),
            status=cast(str, cycle["status"]),
            logical_tokens_used=cast(int, cycle["logical_tokens_used"]),
            gpu_seconds_used=float(cast(float, cycle["gpu_seconds_used"])),
        )
        proposals = tuple(self._proposal_from_json(self._proposals[item]) for item in cycle_value.proposal_ids)
        return ConsolidationReport(cycle=cycle_value, proposals=proposals, replayed=replayed)

    def run_cycle(
        self,
        cycle_id: str,
        *,
        owner_id: str,
        project_id: str,
        session_id: str,
        trajectory_events: Sequence[RecordLike] = (),
        memories: Sequence[RecordLike] = (),
        window_id: str = "default",
        now_utc: str | None = None,
    ) -> ConsolidationReport:
        """Analyze one bounded idle window and atomically commit shadow proposals."""

        _identifier(cycle_id, "cycle_id")
        _identifier(owner_id, "owner_id")
        _identifier(project_id, "project_id")
        _identifier(session_id, "session_id")
        _identifier(window_id, "window_id")
        created_at = _text(now_utc or _now_utc(), "now_utc", 64)
        events, memory_records, input_sha256, logical_tokens = self._normalize_inputs(
            trajectory_events, memories, owner_id=owner_id, project_id=project_id
        )
        with self._lock:
            existing = self._cycles.get(cycle_id)
            if existing is not None:
                if any(
                    existing.get(key) != value
                    for key, value in (
                        ("owner_id", owner_id),
                        ("project_id", project_id),
                        ("session_id", session_id),
                        ("window_id", window_id),
                        ("input_sha256", input_sha256),
                    )
                ):
                    raise ConsolidationConflictError("consolidation_cycle_owner_conflict")
                return self._report_for_cycle(existing, replayed=True)
            window = self._windows.get(window_id)
            if window is not None:
                if window.get("owner_id") != owner_id or window.get("project_id") != project_id:
                    raise ConsolidationConflictError("consolidation_window_owner_conflict")
                cycle_ids = cast(list[str], window["cycle_ids"])
                if len(cycle_ids) >= self.budget.max_cycles_per_window:
                    raise ConsolidationBudgetError("maintenance_frequency_limit_exceeded")
                previous = _timestamp(window["last_cycle_at"])
                current = _timestamp(created_at)
                if previous is not None and current is not None and (current - previous).total_seconds() < self.budget.min_interval_seconds:
                    raise ConsolidationBudgetError("maintenance_minimum_interval_not_reached")
            if len(self._cycles) >= self.budget.max_cycles_stored:
                raise ConsolidationBudgetError("maintenance_cycle_storage_limit_exceeded")
            self._fault("after_input_validation")
            proposals = self._analyze(
                events,
                memory_records,
                owner_id=owner_id,
                project_id=project_id,
                session_id=session_id,
                cycle_id=cycle_id,
                input_sha256=input_sha256,
                created_at_utc=created_at,
            )
            self._fault("after_analysis")
            if len(proposals) > self.budget.max_proposals_per_cycle:
                raise ConsolidationBudgetError("maintenance_proposal_limit_exceeded")
            unique: dict[str, ConsolidationProposal] = {}
            duplicate_ids: list[str] = []
            for proposal in proposals:
                if proposal.proposal_id in self._proposals or proposal.proposal_id in unique:
                    duplicate_ids.append(proposal.proposal_id)
                else:
                    unique[proposal.proposal_id] = proposal
            proposal_ids = tuple(sorted(unique))
            cycle = ConsolidationCycle(
                cycle_id=cycle_id,
                window_id=window_id,
                owner_id=owner_id,
                project_id=project_id,
                session_id=session_id,
                input_sha256=input_sha256,
                proposal_ids=proposal_ids,
                duplicate_proposal_ids=tuple(sorted(set(duplicate_ids))),
                started_at_utc=created_at,
                completed_at_utc=created_at,
                status="SHADOW",
                logical_tokens_used=logical_tokens,
                gpu_seconds_used=0.0,
            )
            cycles = dict(self._cycles)
            proposals_state = dict(self._proposals)
            windows = dict(self._windows)
            improvements = dict(self._improvements)
            cycles[cycle_id] = cycle.to_json()
            proposals_state.update({proposal_id: unique[proposal_id].to_json() for proposal_id in proposal_ids})
            previous_ids = [] if window is None else list(cast(list[str], window["cycle_ids"]))
            windows[window_id] = {
                "owner_id": owner_id,
                "project_id": project_id,
                "cycle_ids": previous_ids + [cycle_id],
                "last_cycle_at": created_at,
            }
            self._fault("before_commit")
            self._persist_candidate(cycles, proposals_state, windows, improvements)
            self._fault("after_commit")
            return self._report_for_cycle(cycle.to_json(), replayed=False)

    run_idle_window = run_cycle

    def propose_harness_improvement(
        self,
        cycle_id: str,
        *,
        owner_id: str,
        project_id: str,
        session_id: str,
        description: str,
        source_event_ids: Sequence[str] = (),
        source_memory_ids: Sequence[str] = (),
        now_utc: str | None = None,
    ) -> ConsolidationProposal:
        """Create exactly one bounded harness candidate for a recorded cycle.

        This records the evaluation contract only.  It does not inspect or
        edit source files, run a model, choose tasks, or promote the result.
        """

        _identifier(cycle_id, "cycle_id")
        _identifier(owner_id, "owner_id")
        _identifier(project_id, "project_id")
        _identifier(session_id, "session_id")
        description = _text(description, "improvement_description", 2_048)
        for source_id in tuple(source_event_ids) + tuple(source_memory_ids):
            _identifier(source_id, "improvement_source_id")
        with self._lock:
            cycle = self._cycles.get(cycle_id)
            if cycle is None:
                raise ConsolidationConflictError("improvement_cycle_not_found")
            if cycle.get("owner_id") != owner_id or cycle.get("project_id") != project_id:
                raise ConsolidationConflictError("improvement_cycle_owner_conflict")
            cycle_proposal_ids = cast(list[str], cycle["proposal_ids"])
            for proposal_id in cycle_proposal_ids:
                existing = self._proposals.get(proposal_id)
                details = existing.get("details") if existing is not None else None
                if isinstance(details, Mapping) and details.get("harness_candidate") is True:
                    raise ConsolidationConflictError("one_harness_change_per_cycle")
            proposal = self._make_proposal(
                kind=ProposalKind.PROCESS_IMPROVEMENT,
                owner_id=owner_id,
                project_id=project_id,
                session_id=session_id,
                cycle_id=cycle_id,
                input_sha256=cast(str, cycle["input_sha256"]),
                source_event_ids=source_event_ids,
                source_memory_ids=source_memory_ids,
                summary="One bounded harness candidate requires frozen A/B and held-out evidence.",
                details={
                    "harness_candidate": True,
                    "description": description,
                    "evaluation_contract": {
                        "baseline_required": True,
                        "candidate_required": True,
                        "development_tasks_required": True,
                        "held_out_tasks_required": True,
                        "security_regression_rejects": True,
                        "correctness_regression_rejects": True,
                    },
                    "mode": "shadow",
                },
                created_at_utc=now_utc or _now_utc(),
            )
            if proposal.proposal_id in self._proposals:
                return self._proposal_from_json(self._proposals[proposal.proposal_id])
            updated_cycle = dict(cycle)
            updated_cycle["proposal_ids"] = sorted(cycle_proposal_ids + [proposal.proposal_id])
            cycles = dict(self._cycles)
            cycles[cycle_id] = updated_cycle
            proposals = dict(self._proposals)
            proposals[proposal.proposal_id] = proposal.to_json()
            self._persist_candidate(cycles, proposals, dict(self._windows), dict(self._improvements))
            return proposal

    propose_improvement = propose_harness_improvement

    def proposals(
        self, *, owner_id: str, project_id: str, state: ProposalState | None = None
    ) -> tuple[ConsolidationProposal, ...]:
        _identifier(owner_id, "owner_id")
        _identifier(project_id, "project_id")
        with self._lock:
            values = [self._proposal_from_json(value) for value in self._proposals.values() if value.get("owner_id") == owner_id and value.get("project_id") == project_id]
        if state is not None:
            values = [value for value in values if value.state is state]
        return tuple(sorted(values, key=lambda value: value.proposal_id))

    def record_ab_evidence(self, proposal_id: str, evidence: ABEvidence) -> JsonObject:
        """Record frozen A/B evidence; never activates the candidate."""

        _identifier(proposal_id, "proposal_id")
        if not isinstance(evidence, ABEvidence):
            raise ConsolidationError("ab_evidence_invalid")
        with self._lock:
            proposal_raw = self._proposals.get(proposal_id)
            if proposal_raw is None or proposal_raw.get("kind") not in {ProposalKind.PROCESS_IMPROVEMENT.value, ProposalKind.CODE_CHANGE.value, ProposalKind.CANDIDATE_SKILL.value}:
                raise ConsolidationConflictError("ab_candidate_not_found")
            if proposal_id in self._improvements:
                existing = self._improvements[proposal_id]
                if existing.get("ab_evidence") != evidence.to_json():
                    raise ConsolidationConflictError("ab_evidence_conflict")
                return dict(existing)
            decision = "awaiting_human_approval" if evidence.passes else "rejected_correctness_or_security_regression"
            updated: JsonObject = {
                "proposal_id": proposal_id,
                "ab_evidence": evidence.to_json(),
                "decision": decision,
                "human_approval": None,
            }
            improvements = dict(self._improvements)
            improvements[proposal_id] = updated
            self._persist_candidate(dict(self._cycles), dict(self._proposals), dict(self._windows), improvements)
            return dict(updated)

    def approve_improvement(self, proposal_id: str, *, approver_id: str) -> JsonObject:
        """Record explicit human approval without promoting or executing anything."""

        _identifier(proposal_id, "proposal_id")
        _identifier(approver_id, "approver_id")
        with self._lock:
            improvement = self._improvements.get(proposal_id)
            if improvement is None or improvement.get("decision") != "awaiting_human_approval":
                raise ConsolidationConflictError("ab_approval_not_ready")
            approved = dict(improvement)
            approved["decision"] = "human_approved_shadow_only"
            approved["human_approval"] = {"approver_id": approver_id, "approved_at_utc": _now_utc()}
            improvements = dict(self._improvements)
            improvements[proposal_id] = approved
            self._persist_candidate(dict(self._cycles), dict(self._proposals), dict(self._windows), improvements)
            return approved

    def maintenance_status(self) -> JsonObject:
        with self._lock:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "revision": self._revision,
                "mode": "SHADOW",
                "active_production_mutations": False,
                "auto_promotion": False,
                "cycles": len(self._cycles),
                "proposals": len(self._proposals),
                "improvements": len(self._improvements),
                "budget": self.budget.to_json(),
                "gpu_seconds_used": 0.0,
            }


ConsolidationService = IdleConsolidator
