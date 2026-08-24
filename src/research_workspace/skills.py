"""Human-governed procedural skills for the standalone Laplace core.

Skills are versioned, scoped procedure documents.  They are data supplied to a
caller, never executable plugins: this module does not import, invoke, or
grant authority to anything found in a skill document.  Lifecycle transitions
are deliberately explicit so a model-generated candidate cannot promote
itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
SkillScope = Literal["global", "user", "project"]

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_ACTOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DANGEROUS_PROCEDURE = re.compile(
    r"(?:ignore|override|bypass|disable|disregard)\s+(?:the\s+)?"
    r"(?:policy|policies|rules?|authorization|safety)|"
    r"(?:grant|acquire|assume)\s+(?:new\s+)?authority|"
    r"\brm\s+-rf\b|\b(?:curl|wget)\s+[^\n]*\|\s*(?:sh|bash)\b",
    re.IGNORECASE,
)
_ALLOWED_TOOLS = frozenset(
    {
        "repo_map",
        "find_symbol",
        "find_references",
        "search_text",
        "read_region",
        "inspect_diff",
        "edit_region",
        "create_text_file",
        "verify",
        "git_state",
        "retrieve",
        "evidence",
        "deterministic_python",
    }
)
_ALLOWED_VERIFIERS = frozenset(
    {
        "pytest",
        "ruff",
        "mypy",
        "bandit",
        "compileall",
        "iverilog",
        "verilator",
        "svlint",
        "git_diff_check",
        "deterministic_python",
    }
)


class SkillLifecycle(str, Enum):
    """The only states a skill can occupy."""

    CANDIDATE = "candidate"
    VALIDATED = "validated"
    A_B_TESTED = "A/B-tested"
    AB_TESTED = "A/B-tested"
    HUMAN_APPROVED = "human-approved"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"


class SkillRegistryError(RuntimeError):
    """A skill was malformed, unauthorized, ambiguous, or in an invalid state."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class SkillProvenance:
    """Traceable source information for a procedure document."""

    source_kind: Literal["human", "upstream", "model"] = "human"
    source_uri: str = "local"
    source_revision: str = "workspace"
    license_identifier: str = "internal"
    author: str = "human"
    generated_by_model: bool = False

    def to_json(self) -> JsonObject:
        return {
            "source_kind": self.source_kind,
            "source_uri": self.source_uri,
            "source_revision": self.source_revision,
            "license_identifier": self.license_identifier,
            "author": self.author,
            "generated_by_model": self.generated_by_model,
        }

    @classmethod
    def from_json(cls, raw: object) -> "SkillProvenance":
        if not isinstance(raw, dict) or set(raw) != {
            "source_kind",
            "source_uri",
            "source_revision",
            "license_identifier",
            "author",
            "generated_by_model",
        }:
            raise SkillRegistryError("invalid_skill_provenance")
        values = cast(Mapping[str, object], raw)
        kind = values["source_kind"]
        if kind not in {"human", "upstream", "model"}:
            raise SkillRegistryError("invalid_skill_provenance_kind")
        strings = ("source_uri", "source_revision", "license_identifier", "author")
        if not all(isinstance(values[name], str) and values[name] for name in strings):
            raise SkillRegistryError("invalid_skill_provenance_text")
        if not isinstance(values["generated_by_model"], bool):
            raise SkillRegistryError("invalid_skill_provenance_generation_flag")
        generated = values["generated_by_model"]
        if (kind == "model") != generated:
            raise SkillRegistryError("inconsistent_skill_provenance_generation")
        return cls(
            source_kind=kind,
            source_uri=cast(str, values["source_uri"]),
            source_revision=cast(str, values["source_revision"]),
            license_identifier=cast(str, values["license_identifier"]),
            author=cast(str, values["author"]),
            generated_by_model=generated,
        )


@dataclass(frozen=True)
class SkillSpec:
    """Immutable definition submitted to the registry as a new version."""

    skill_id: str
    version: str
    trigger: tuple[str, ...]
    scope: SkillScope
    procedure: str
    provenance: SkillProvenance = SkillProvenance()
    owner_id: str | None = None
    project_id: str | None = None
    required_tools: tuple[str, ...] = ()
    required_verifiers: tuple[str, ...] = ()
    description: str = ""
    content_sha256: str | None = None

    def to_json(self) -> JsonObject:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "trigger": list(self.trigger),
            "scope": self.scope,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "procedure": self.procedure,
            "required_tools": list(self.required_tools),
            "required_verifiers": list(self.required_verifiers),
            "description": self.description,
            "provenance": self.provenance.to_json(),
            "content_sha256": self.content_sha256 or _spec_digest(self),
        }


@dataclass(frozen=True)
class SkillStatistics:
    """Outcome and frozen A/B evidence kept with every version."""

    successes: int = 0
    failures: int = 0
    ab_tests: int = 0
    deterministic_correct: bool = False
    reliability_delta: float = 0.0
    efficiency_delta: float = 0.0
    benefit_justification: str = ""
    last_outcome: str | None = None

    def to_json(self) -> JsonObject:
        return {
            "successes": self.successes,
            "failures": self.failures,
            "ab_tests": self.ab_tests,
            "deterministic_correct": self.deterministic_correct,
            "reliability_delta": self.reliability_delta,
            "efficiency_delta": self.efficiency_delta,
            "benefit_justification": self.benefit_justification,
            "last_outcome": self.last_outcome,
        }

    @classmethod
    def from_json(cls, raw: object) -> "SkillStatistics":
        if not isinstance(raw, dict) or set(raw) != {
            "successes",
            "failures",
            "ab_tests",
            "deterministic_correct",
            "reliability_delta",
            "efficiency_delta",
            "benefit_justification",
            "last_outcome",
        }:
            raise SkillRegistryError("invalid_skill_statistics")
        value = cast(Mapping[str, object], raw)
        counts: list[int] = []
        for key in ("successes", "failures", "ab_tests"):
            count = value[key]
            if not isinstance(count, int) or count < 0:
                raise SkillRegistryError("invalid_skill_statistics_counts")
            counts.append(count)
        if not isinstance(value["deterministic_correct"], bool):
            raise SkillRegistryError("invalid_skill_statistics_correctness")
        if not all(isinstance(value[key], (int, float)) for key in (
            "reliability_delta",
            "efficiency_delta",
        )):
            raise SkillRegistryError("invalid_skill_statistics_deltas")
        if not isinstance(value["benefit_justification"], str):
            raise SkillRegistryError("invalid_skill_statistics_justification")
        last = value["last_outcome"]
        if last is not None and not isinstance(last, str):
            raise SkillRegistryError("invalid_skill_statistics_outcome")
        return cls(
            successes=counts[0],
            failures=counts[1],
            ab_tests=counts[2],
            deterministic_correct=value["deterministic_correct"],
            reliability_delta=float(cast(float, value["reliability_delta"])),
            efficiency_delta=float(cast(float, value["efficiency_delta"])),
            benefit_justification=value["benefit_justification"],
            last_outcome=last,
        )


@dataclass(frozen=True)
class SkillRecord:
    """Persisted skill version and its governance metadata."""

    spec: SkillSpec
    lifecycle: SkillLifecycle = SkillLifecycle.CANDIDATE
    statistics: SkillStatistics = SkillStatistics()
    created_at_utc: str = ""
    validated_by: str | None = None
    human_approved_by: str | None = None
    lifecycle_note: str = ""
    deactivation: JsonObject | None = None
    rollback: JsonObject | None = None

    @property
    def skill_id(self) -> str:
        return self.spec.skill_id

    @property
    def version(self) -> str:
        return self.spec.version

    def to_json(self) -> JsonObject:
        return {
            "spec": self.spec.to_json(),
            "lifecycle": self.lifecycle.value,
            "statistics": self.statistics.to_json(),
            "created_at_utc": self.created_at_utc,
            "validated_by": self.validated_by,
            "human_approved_by": self.human_approved_by,
            "lifecycle_note": self.lifecycle_note,
            "deactivation": self.deactivation,
            "rollback": self.rollback,
        }

    @classmethod
    def from_json(cls, raw: object) -> "SkillRecord":
        if not isinstance(raw, dict) or set(raw) != {
            "spec",
            "lifecycle",
            "statistics",
            "created_at_utc",
            "validated_by",
            "human_approved_by",
            "lifecycle_note",
            "deactivation",
            "rollback",
        }:
            raise SkillRegistryError("invalid_skill_record")
        value = cast(Mapping[str, object], raw)
        spec = _spec_from_json(value["spec"])
        state = value["lifecycle"]
        if not isinstance(state, str):
            raise SkillRegistryError("invalid_skill_lifecycle")
        try:
            lifecycle = SkillLifecycle(state)
        except ValueError as exc:
            raise SkillRegistryError("invalid_skill_lifecycle") from exc
        if not isinstance(value["created_at_utc"], str) or not value["created_at_utc"]:
            raise SkillRegistryError("invalid_skill_created_at")
        for key in ("validated_by", "human_approved_by"):
            if value[key] is not None and not isinstance(value[key], str):
                raise SkillRegistryError("invalid_skill_actor")
        if not isinstance(value["lifecycle_note"], str):
            raise SkillRegistryError("invalid_skill_lifecycle_note")
        for key in ("deactivation", "rollback"):
            if value[key] is not None and not isinstance(value[key], dict):
                raise SkillRegistryError("invalid_skill_recovery_metadata")
        return cls(
            spec=spec,
            lifecycle=lifecycle,
            statistics=SkillStatistics.from_json(value["statistics"]),
            created_at_utc=value["created_at_utc"],
            validated_by=cast(str | None, value["validated_by"]),
            human_approved_by=cast(str | None, value["human_approved_by"]),
            lifecycle_note=value["lifecycle_note"],
            deactivation=cast(JsonObject | None, value["deactivation"]),
            rollback=cast(JsonObject | None, value["rollback"]),
        )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _spec_digest(spec: SkillSpec) -> str:
    value: JsonObject = {
        "skill_id": spec.skill_id,
        "version": spec.version,
        "trigger": list(spec.trigger),
        "scope": spec.scope,
        "owner_id": spec.owner_id,
        "project_id": spec.project_id,
        "procedure": spec.procedure,
        "required_tools": list(spec.required_tools),
        "required_verifiers": list(spec.required_verifiers),
        "description": spec.description,
        "provenance": spec.provenance.to_json(),
        "content_sha256": None,
    }
    return hashlib.sha256(_canonical(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_text(value: object, category: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise SkillRegistryError(category)
    return value


def _spec_from_json(raw: object) -> SkillSpec:
    if not isinstance(raw, dict):
        raise SkillRegistryError("invalid_skill_spec")
    expected = {
        "skill_id",
        "version",
        "trigger",
        "scope",
        "owner_id",
        "project_id",
        "procedure",
        "required_tools",
        "required_verifiers",
        "description",
        "provenance",
        "content_sha256",
    }
    if set(raw) != expected:
        raise SkillRegistryError("invalid_skill_spec_keys")
    value = cast(Mapping[str, object], raw)
    trigger_raw = value["trigger"]
    tools_raw = value["required_tools"]
    verifiers_raw = value["required_verifiers"]
    if not all(isinstance(item, list) for item in (trigger_raw, tools_raw, verifiers_raw)):
        raise SkillRegistryError("invalid_skill_lists")
    trigger = cast(list[object], trigger_raw)
    tools = cast(list[object], tools_raw)
    verifiers = cast(list[object], verifiers_raw)
    if not all(isinstance(item, str) and item for item in trigger):
        raise SkillRegistryError("invalid_skill_trigger")
    if not all(isinstance(item, str) and item for item in tools + verifiers):
        raise SkillRegistryError("invalid_skill_requirements")
    scope = value["scope"]
    if scope not in {"global", "user", "project"}:
        raise SkillRegistryError("invalid_skill_scope")
    owner = value["owner_id"]
    project = value["project_id"]
    if owner is not None and (not isinstance(owner, str) or not owner):
        raise SkillRegistryError("invalid_skill_owner")
    if project is not None and (not isinstance(project, str) or not project):
        raise SkillRegistryError("invalid_skill_project")
    return SkillSpec(
        skill_id=_require_text(value["skill_id"], "invalid_skill_id", maximum=64),
        version=_require_text(value["version"], "invalid_skill_version", maximum=32),
        trigger=tuple(cast(str, item) for item in trigger),
        scope=scope,
        owner_id=owner,
        project_id=project,
        procedure=_require_text(value["procedure"], "invalid_skill_procedure", maximum=128_000),
        required_tools=tuple(cast(str, item) for item in tools),
        required_verifiers=tuple(cast(str, item) for item in verifiers),
        description=_require_text(value["description"], "invalid_skill_description", maximum=4096)
        if value["description"]
        else "",
        provenance=SkillProvenance.from_json(value["provenance"]),
        content_sha256=_require_text(value["content_sha256"], "invalid_skill_hash", maximum=64),
    )


class SkillRegistry:
    """Persistent, deterministic registry for human-governed skill versions.

    ``root`` is only a declared local skill source boundary.  ``state_path``
    is kept separately so importing a skill cannot overwrite its source.  The
    registry writes one atomically replaced JSON state file and never runs a
    script, shell command, verifier, or procedure from a skill.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, *, state_path: Path | None = None) -> None:
        self.root = root.resolve()
        self.state_path = (state_path or self.root / "registry.json").resolve()
        self._records: dict[tuple[str, str], SkillRecord] = {}
        self._revision = 0
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw: object = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillRegistryError("skill_registry_unreadable") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "revision", "skills"}:
            raise SkillRegistryError("invalid_skill_registry")
        if raw["schema_version"] != self.SCHEMA_VERSION or not isinstance(raw["revision"], int):
            raise SkillRegistryError("invalid_skill_registry_version")
        skills = raw["skills"]
        if not isinstance(skills, list):
            raise SkillRegistryError("invalid_skill_registry_skills")
        for item in skills:
            record = SkillRecord.from_json(item)
            self._validate_spec(record.spec)
            key = (record.skill_id, record.version)
            if key in self._records:
                raise SkillRegistryError("duplicate_skill_version")
            self._records[key] = record
        self._revision = raw["revision"]
        self._validate_active_uniqueness()

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload: JsonObject = {
            "schema_version": self.SCHEMA_VERSION,
            "revision": self._revision + 1,
            "skills": [
                record.to_json()
                for record in sorted(self._records.values(), key=lambda item: (item.skill_id, item.version))
            ],
        }
        temporary = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8"))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        except OSError as exc:
            raise SkillRegistryError("skill_registry_persist_failed") from exc
        self._revision += 1

    def _validate_active_uniqueness(self) -> None:
        active: dict[str, SkillRecord] = {}
        for record in self._records.values():
            if record.lifecycle is SkillLifecycle.ACTIVE:
                if record.skill_id in active:
                    raise SkillRegistryError("multiple_active_skill_versions")
                active[record.skill_id] = record

    @staticmethod
    def _validate_spec(spec: SkillSpec) -> None:
        if not _NAME.fullmatch(spec.skill_id) or not _VERSION.fullmatch(spec.version):
            raise SkillRegistryError("invalid_skill_identity")
        if not spec.trigger or len(spec.trigger) > 16:
            raise SkillRegistryError("invalid_skill_trigger")
        for trigger in spec.trigger:
            _require_text(trigger.strip().lower(), "invalid_skill_trigger", maximum=256)
        if spec.scope == "global" and (spec.owner_id is not None or spec.project_id is not None):
            raise SkillRegistryError("global_skill_has_owner_scope")
        if spec.scope == "user" and (not spec.owner_id or spec.project_id is not None):
            raise SkillRegistryError("user_skill_scope_mismatch")
        if spec.scope == "project" and (not spec.owner_id or not spec.project_id):
            raise SkillRegistryError("project_skill_scope_mismatch")
        if not spec.procedure.strip() or len(spec.procedure) > 128_000:
            raise SkillRegistryError("invalid_skill_procedure")
        if _DANGEROUS_PROCEDURE.search(spec.procedure):
            raise SkillRegistryError("skill_procedure_authority_violation")
        if any(tool not in _ALLOWED_TOOLS for tool in spec.required_tools):
            raise SkillRegistryError("skill_requires_unapproved_tool")
        if any(verifier not in _ALLOWED_VERIFIERS for verifier in spec.required_verifiers):
            raise SkillRegistryError("skill_requires_unapproved_verifier")
        if any(not _require_text(verifier, "invalid_skill_verifier", maximum=128) for verifier in spec.required_verifiers):
            raise SkillRegistryError("invalid_skill_verifier")
        if spec.content_sha256 is not None and not _SHA256.fullmatch(spec.content_sha256):
            raise SkillRegistryError("invalid_skill_hash")
        calculated = _spec_digest(spec)
        if spec.content_sha256 is not None and spec.content_sha256 != calculated:
            raise SkillRegistryError("skill_content_hash_mismatch")

    def records(self, *, include_deactivated: bool = True) -> tuple[SkillRecord, ...]:
        if not include_deactivated:
            return tuple(
                sorted(
                    (
                        record
                        for record in self._records.values()
                        if record.lifecycle is not SkillLifecycle.DEACTIVATED
                    ),
                    key=lambda item: (item.skill_id, item.version),
                )
            )
        return tuple(sorted(self._records.values(), key=lambda item: (item.skill_id, item.version)))

    def get(self, skill_id: str, version: str | None = None) -> SkillRecord:
        if not _NAME.fullmatch(skill_id):
            raise SkillRegistryError("invalid_skill_id")
        matches = [record for record in self._records.values() if record.skill_id == skill_id]
        if version is not None:
            record = self._records.get((skill_id, version))
            if record is None:
                raise SkillRegistryError("skill_not_found")
            return record
        if not matches:
            raise SkillRegistryError("skill_not_found")
        active = [record for record in matches if record.lifecycle is SkillLifecycle.ACTIVE]
        return sorted(active or matches, key=lambda item: item.version)[-1]

    def register(self, spec: SkillSpec, *, actor: str = "human") -> SkillRecord:
        """Register one immutable candidate version; never promote it."""

        self._validate_spec(spec)
        if not _ACTOR.fullmatch(actor):
            raise SkillRegistryError("invalid_skill_actor")
        key = (spec.skill_id, spec.version)
        if key in self._records:
            raise SkillRegistryError("skill_version_exists")
        record = SkillRecord(spec=spec, created_at_utc=_utc_now())
        self._records[key] = record
        self._persist()
        return record

    def register_directory(
        self,
        directory: Path,
        *,
        trigger: Sequence[str],
        scope: SkillScope,
        owner_id: str | None = None,
        project_id: str | None = None,
        provenance: SkillProvenance | None = None,
        actor: str = "human",
    ) -> SkillRecord:
        """Import a local SKILL.md/skill.json pair without executing support files."""

        candidate = directory.absolute()
        if candidate.is_symlink():
            raise SkillRegistryError("unsafe_skill_source")
        resolved = candidate.resolve()
        try:
            relative_root = resolved.relative_to(self.root)
        except ValueError as exc:
            raise SkillRegistryError("skill_source_outside_root") from exc
        if resolved.is_symlink() or not resolved.is_dir():
            raise SkillRegistryError("unsafe_skill_source")
        cursor = self.root
        for component in relative_root.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise SkillRegistryError("unsafe_skill_source")
        for path in (resolved / "SKILL.md", resolved / "skill.json"):
            if path.is_symlink() or not path.is_file():
                raise SkillRegistryError("incomplete_skill_source")
        try:
            metadata: object = json.loads((resolved / "skill.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillRegistryError("invalid_skill_metadata") from exc
        if not isinstance(metadata, dict):
            raise SkillRegistryError("invalid_skill_metadata")
        name = metadata.get("name")
        version = metadata.get("version")
        description = metadata.get("description", "")
        if not isinstance(name, str) or not isinstance(version, str) or not isinstance(description, str):
            raise SkillRegistryError("invalid_skill_metadata")
        procedure = (resolved / "SKILL.md").read_text(encoding="utf-8")
        return self.register(
            SkillSpec(
                skill_id=name,
                version=version,
                trigger=tuple(trigger),
                scope=scope,
                owner_id=owner_id,
                project_id=project_id,
                procedure=procedure,
                provenance=provenance or SkillProvenance(source_uri=str(resolved)),
                description=description,
                content_sha256=None,
            ),
            actor=actor,
        )

    def _replace(self, record: SkillRecord, **changes: object) -> SkillRecord:
        self._records[(record.skill_id, record.version)] = replace(record, **cast(Any, changes))
        self._persist()
        return self._records[(record.skill_id, record.version)]

    def validate(self, skill_id: str, version: str, *, validator: str, evidence: str) -> SkillRecord:
        record = self.get(skill_id, version)
        if record.lifecycle is not SkillLifecycle.CANDIDATE:
            raise SkillRegistryError("invalid_skill_transition")
        if not _ACTOR.fullmatch(validator) or not evidence.strip():
            raise SkillRegistryError("human_validation_required")
        return self._replace(
            record,
            lifecycle=SkillLifecycle.VALIDATED,
            validated_by=validator,
            lifecycle_note=evidence,
        )

    @staticmethod
    def _number(value: object, category: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SkillRegistryError(category)
        result = float(value)
        if result != result or result in {float("inf"), float("-inf")}:
            raise SkillRegistryError(category)
        return result

    def record_ab_test(
        self,
        skill_id: str,
        version: str,
        *,
        with_skill: Mapping[str, object],
        without_skill: Mapping[str, object],
        evaluator: str,
        justification: str,
    ) -> SkillRecord:
        """Record frozen-task A/B evidence and move validated -> A/B-tested."""

        record = self.get(skill_id, version)
        if record.lifecycle is not SkillLifecycle.VALIDATED:
            raise SkillRegistryError("invalid_skill_transition")
        if not _ACTOR.fullmatch(evaluator) or not justification.strip():
            raise SkillRegistryError("evidence_required")
        for arm in (with_skill, without_skill):
            if set(arm) != {"deterministic_correct", "reliability", "efficiency"}:
                raise SkillRegistryError("invalid_ab_test_evidence")
            if not isinstance(arm["deterministic_correct"], bool):
                raise SkillRegistryError("invalid_ab_test_correctness")
        correct = bool(with_skill["deterministic_correct"])
        reliability_delta = self._number(with_skill["reliability"], "invalid_ab_test_reliability") - self._number(
            without_skill["reliability"], "invalid_ab_test_reliability"
        )
        efficiency_delta = self._number(with_skill["efficiency"], "invalid_ab_test_efficiency") - self._number(
            without_skill["efficiency"], "invalid_ab_test_efficiency"
        )
        stats = replace(
            record.statistics,
            ab_tests=record.statistics.ab_tests + 1,
            deterministic_correct=correct,
            reliability_delta=reliability_delta,
            efficiency_delta=efficiency_delta,
            benefit_justification=justification,
        )
        return self._replace(
            record,
            lifecycle=SkillLifecycle.A_B_TESTED,
            statistics=stats,
            lifecycle_note=f"A/B evaluator {evaluator}: {justification}",
        )

    def human_approve(
        self, skill_id: str, version: str, *, approver: str, justification: str
    ) -> SkillRecord:
        """Require explicit human approval after deterministic, beneficial A/B evidence."""

        record = self.get(skill_id, version)
        if record.lifecycle is not SkillLifecycle.A_B_TESTED:
            raise SkillRegistryError("invalid_skill_transition")
        stats = record.statistics
        if not _ACTOR.fullmatch(approver) or approver.lower() in {"model", "llm", "agent"}:
            raise SkillRegistryError("human_approval_required")
        if not justification.strip() or not stats.deterministic_correct:
            raise SkillRegistryError("skill_benefit_gate_failed")
        if stats.reliability_delta <= 0 and stats.efficiency_delta <= 0:
            raise SkillRegistryError("skill_benefit_gate_failed")
        return self._replace(
            record,
            lifecycle=SkillLifecycle.HUMAN_APPROVED,
            human_approved_by=approver,
            lifecycle_note=justification,
        )

    def activate(self, skill_id: str, version: str, *, approver: str) -> SkillRecord:
        record = self.get(skill_id, version)
        if record.lifecycle is not SkillLifecycle.HUMAN_APPROVED:
            raise SkillRegistryError("human_approval_required")
        if record.human_approved_by != approver:
            raise SkillRegistryError("activation_approver_mismatch")
        for other in self._records.values():
            if other.skill_id == skill_id and other.lifecycle is SkillLifecycle.ACTIVE:
                raise SkillRegistryError("skill_already_active")
        return self._replace(record, lifecycle=SkillLifecycle.ACTIVE)

    def deactivate(self, skill_id: str, version: str, *, actor: str, reason: str) -> SkillRecord:
        record = self.get(skill_id, version)
        if record.lifecycle not in {SkillLifecycle.ACTIVE, SkillLifecycle.HUMAN_APPROVED}:
            raise SkillRegistryError("invalid_skill_transition")
        if not _ACTOR.fullmatch(actor) or not reason.strip():
            raise SkillRegistryError("deactivation_reason_required")
        return self._replace(
            record,
            lifecycle=SkillLifecycle.DEACTIVATED,
            deactivation={"actor": actor, "reason": reason, "at_utc": _utc_now()},
        )

    def rollback(
        self,
        skill_id: str,
        version: str,
        *,
        target_version: str,
        actor: str,
        reason: str,
    ) -> SkillRecord:
        """Deactivate one active version and activate an already approved target."""

        current = self.get(skill_id, version)
        target = self.get(skill_id, target_version)
        if current.lifecycle is not SkillLifecycle.ACTIVE:
            raise SkillRegistryError("rollback_source_not_active")
        if target.lifecycle is not SkillLifecycle.HUMAN_APPROVED:
            raise SkillRegistryError("rollback_target_not_approved")
        if not _ACTOR.fullmatch(actor) or not reason.strip():
            raise SkillRegistryError("rollback_reason_required")
        timestamp = _utc_now()
        current = self._replace(
            current,
            lifecycle=SkillLifecycle.DEACTIVATED,
            deactivation={"actor": actor, "reason": f"rollback: {reason}", "at_utc": timestamp},
            rollback={"target_version": target_version, "actor": actor, "reason": reason, "at_utc": timestamp},
        )
        del current
        return self._replace(
            target,
            lifecycle=SkillLifecycle.ACTIVE,
            rollback={"from_version": version, "actor": actor, "reason": reason, "at_utc": timestamp},
        )

    def record_outcome(self, skill_id: str, version: str, *, success: bool) -> SkillRecord:
        record = self.get(skill_id, version)
        stats = record.statistics
        stats = replace(
            stats,
            successes=stats.successes + int(success),
            failures=stats.failures + int(not success),
            last_outcome="success" if success else "failure",
        )
        return self._replace(record, statistics=stats)

    @staticmethod
    def _scope_matches(spec: SkillSpec, *, owner_id: str, project_id: str) -> bool:
        return (
            spec.scope == "global"
            or (spec.scope == "user" and spec.owner_id == owner_id)
            or (spec.scope == "project" and spec.owner_id == owner_id and spec.project_id == project_id)
        )

    @staticmethod
    def _trigger_score(trigger: str, query: str) -> int:
        normalized_trigger = " ".join(trigger.lower().split())
        normalized_query = " ".join(query.lower().split())
        if not normalized_trigger or not normalized_query:
            return 0
        if normalized_trigger == normalized_query:
            return 10_000 + len(normalized_trigger)
        if re.search(rf"(?<!\w){re.escape(normalized_trigger)}(?!\w)", normalized_query):
            return 1_000 + len(normalized_trigger)
        return 0

    def select(
        self,
        query: str,
        *,
        owner_id: str,
        project_id: str,
        available_tools: Sequence[str] = (),
        available_verifiers: Sequence[str] = (),
        enabled: bool = True,
    ) -> tuple[SkillRecord, ...]:
        """Select active skills deterministically; equal matches fail closed."""

        if not enabled:
            return ()
        if not query.strip() or not owner_id or not project_id:
            raise SkillRegistryError("invalid_skill_selection_context")
        tools = set(available_tools)
        verifiers = set(available_verifiers)
        scored: list[tuple[int, SkillRecord]] = []
        for record in self._records.values():
            spec = record.spec
            if record.lifecycle is not SkillLifecycle.ACTIVE or not self._scope_matches(
                spec, owner_id=owner_id, project_id=project_id
            ):
                continue
            if not set(spec.required_tools).issubset(tools) or not set(spec.required_verifiers).issubset(verifiers):
                continue
            score = max(self._trigger_score(trigger, query) for trigger in spec.trigger)
            if score:
                scored.append((score, record))
        if not scored:
            return ()
        maximum = max(score for score, _ in scored)
        winners = [record for score, record in scored if score == maximum]
        if len(winners) > 1:
            raise SkillRegistryError(
                "ambiguous_skill_trigger",
                {"skill_ids": sorted(record.skill_id for record in winners), "score": maximum},
            )
        return (winners[0],)

    def activation_packet(
        self,
        query: str,
        *,
        owner_id: str,
        project_id: str,
        available_tools: Sequence[str] = (),
        available_verifiers: Sequence[str] = (),
        enabled: bool = True,
    ) -> JsonObject:
        """Return advisory procedure text with explicit non-authority boundaries."""

        selected = self.select(
            query,
            owner_id=owner_id,
            project_id=project_id,
            available_tools=available_tools,
            available_verifiers=available_verifiers,
            enabled=enabled,
        )
        return {
            "schema_version": 1,
            "skills_enabled": enabled,
            "authority": {
                "advisory_only": True,
                "may_override_policy": False,
                "may_override_rules": False,
                "may_grant_authority": False,
                "may_execute_procedure": False,
            },
            "skills": [
                {
                    "skill_id": record.skill_id,
                    "version": record.version,
                    "content_sha256": record.spec.to_json()["content_sha256"],
                    "scope": record.spec.scope,
                    "required_tools": list(record.spec.required_tools),
                    "required_verifiers": list(record.spec.required_verifiers),
                    "procedure": record.spec.procedure,
                    "provenance": record.spec.provenance.to_json(),
                }
                for record in selected
            ],
        }

    def snapshot(self) -> JsonObject:
        payload: JsonObject = {
            "schema_version": self.SCHEMA_VERSION,
            "revision": self._revision,
            "skills": [record.to_json() for record in self.records()],
        }
        payload["snapshot_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        return payload
