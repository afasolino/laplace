"""Governed local software-engineering workflows for Laplace.

This module deliberately keeps orchestration deterministic.  It stores task
artifacts beside an existing Laplace project, treats curated material as
read-only evidence, and only invokes a small allowlist of verification tools.
It never turns model text into a shell command.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal

# LocalToolRunner constrains executable and argument allowlists.
import subprocess  # nosec B404
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

import yaml

from .documents import _chunks, _db
from .retrieval import embed


JsonValue: TypeAlias = object
JsonObject: TypeAlias = dict[str, object]
Domain = Literal["python", "c", "verilog", "systemverilog"]
Role = Literal["supervisor", "researcher", "implementer", "verifier", "reviewer"]
TaskState = Literal[
    "request",
    "requirements",
    "plan",
    "retrieval",
    "implementation",
    "verification",
    "review",
    "bounded_correction",
    "final_report",
    "blocked",
]

PYTHON_FOLDERS = (
    "00_policies",
    "10_language_stdlib",
    "20_architecture_patterns",
    "30_web_api",
    "40_testing",
    "50_typing_validation",
    "60_tooling_packaging",
    "70_agents_mcp",
    "80_security_performance",
    "90_manifests",
)
SYSTEMVERILOG_FOLDERS = (
    "00_policies",
    "10_rtl_patterns",
    "20_interfaces",
    "30_verification",
    "40_tooling",
    "50_vendor_flows",
    "90_manifests",
)
C_FOLDERS = (
    "00_policies",
    "10_language_library",
    "20_memory_integer_safety",
    "30_file_process_interfaces",
    "40_testing_sanitizers",
    "50_tooling_builds",
    "90_manifests",
)
VERILOG_FOLDERS = (
    "00_policies",
    "10_rtl_patterns",
    "20_handshake_storage",
    "30_verification",
    "40_tooling",
    "90_manifests",
)
_DOMAIN_FOLDERS: dict[Domain, tuple[str, ...]] = {
    "python": PYTHON_FOLDERS,
    "c": C_FOLDERS,
    "verilog": VERILOG_FOLDERS,
    "systemverilog": SYSTEMVERILOG_FOLDERS,
}
_DOMAIN_LIBRARY: dict[Domain, str] = {
    "python": "Python",
    "c": "C",
    "verilog": "Verilog",
    "systemverilog": "SystemVerilog",
}
_TASK_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    "request": {"requirements", "blocked"},
    "requirements": {"plan", "blocked"},
    "plan": {"retrieval", "blocked"},
    "retrieval": {"implementation", "blocked"},
    # A malformed or out-of-scope model patch can fail before verifier entry;
    # it must consume the same bounded repair budget rather than stranding the
    # persisted task in implementation.
    "implementation": {"verification", "bounded_correction", "blocked"},
    "verification": {"review", "blocked"},
    "review": {"bounded_correction", "final_report", "blocked"},
    "bounded_correction": {"implementation", "final_report", "blocked"},
    "final_report": set(),
    "blocked": set(),
}
_TASK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")


class EngineeringError(RuntimeError):
    """A safe, user-facing engineering-workflow error."""


class SchemaValidationError(EngineeringError):
    """A task did not satisfy the committed task schema."""


class ReferencePolicyError(EngineeringError):
    """A reference would violate its provenance or read-only policy."""


class ReferenceEvidenceError(EngineeringError):
    """A workflow requiring governed evidence could not retrieve any."""


class ToolExecutionError(EngineeringError):
    """An allowlisted local tool could not be executed safely."""


class InferenceBlockedError(EngineeringError):
    """A required CUDA inference capability is unavailable."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_shared_reference_root(explicit: Path | None = None) -> Path | None:
    """Resolve an explicitly configured shared FormalScience Library root."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.getenv("LAPLACE_SHARED_REFERENCE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    formal_root = os.getenv("FORMALSCIENCE_ROOT")
    if formal_root:
        return (Path(formal_root).expanduser() / "Library").resolve()
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineeringError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EngineeringError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: object, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    if readonly:
        path.chmod(0o444)


def _safe_relative(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or not _SAFE_RELATIVE.fullmatch(value):
        raise EngineeringError(f"{label} must be a safe, relative path")
    return candidate


def _inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise EngineeringError(f"Path escapes allowed root: {candidate}")
    return resolved


def _as_str_list(value: JsonValue | None, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EngineeringError(f"{label} must be a list of strings")
    return list(value)


def _schema_type_matches(value: JsonValue, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_schema(value: JsonValue, schema: JsonObject, path: str, errors: list[str]) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        expected = [expected_type]
    elif isinstance(expected_type, list) and all(isinstance(item, str) for item in expected_type):
        expected = list(expected_type)
    else:
        expected = []
    if expected and not any(_schema_type_matches(value, item) for item in expected):
        errors.append(f"{path}: expected {' or '.join(expected)}")
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    enumeration = schema.get("enum")
    if isinstance(enumeration, list) and value not in enumeration:
        errors.append(f"{path}: value is not an allowed enum member")
    minimum_length = schema.get("minLength")
    if isinstance(value, str) and isinstance(minimum_length, int) and len(value) < minimum_length:
        errors.append(f"{path}: string is shorter than {minimum_length}")
    pattern = schema.get("pattern")
    if isinstance(value, str) and isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
        errors.append(f"{path}: does not match required pattern")
    minimum_items = schema.get("minItems")
    if isinstance(value, list):
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            errors.append(f"{path}: array has fewer than {minimum_items} items")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_schema(item, items, f"{path}[{index}]", errors)
    exclusive_minimum = schema.get("exclusiveMinimum")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(exclusive_minimum, (int, float))
    ):
        if value <= exclusive_minimum:
            errors.append(f"{path}: must be greater than {exclusive_minimum}")
    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}: missing required property {key}")
        properties = schema.get("properties")
        property_schemas = properties if isinstance(properties, dict) else {}
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(property_schemas))
            for key in unexpected:
                errors.append(f"{path}: unexpected property {key}")
        for key, item in value.items():
            child_schema = property_schemas.get(key)
            if isinstance(child_schema, dict):
                _validate_schema(item, child_schema, f"{path}.{key}", errors)


def validate_task_spec(specification: JsonObject, schema_path: Path) -> None:
    schema = _load_json_object(schema_path)
    errors: list[str] = []
    _validate_schema(specification, schema, "$", errors)
    if errors:
        raise SchemaValidationError("Task schema validation failed: " + "; ".join(errors))


def domain_schema_path(repository_root: Path, domain: Domain) -> Path:
    filenames: dict[Domain, str] = {
        "python": "python_task_spec.schema.json",
        "c": "c_task_spec.schema.json",
        "verilog": "verilog_task_spec.schema.json",
        "systemverilog": "systemverilog_task_spec.schema.json",
    }
    filename = filenames[domain]
    return repository_root / "codex_a6000" / "templates" / filename


def normalize_task_spec(repository_root: Path, domain: Domain, raw: JsonObject) -> JsonObject:
    """Validate a task and add deterministic, evidence-required acceptance metadata."""
    validate_task_spec(raw, domain_schema_path(repository_root, domain))
    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
        raise SchemaValidationError("task_id must use only letters, digits, '.', '_' and '-'")
    normalized = dict(raw)
    normalized["normalized_domain"] = domain
    normalized["precedence"] = [
        "explicit_user_requirement",
        "target_repository_behavior_and_tests",
        "target_project_documentation_and_conventions",
        "private_curated_references",
        "governed_open_source_references",
        "model_prior_knowledge",
    ]
    requirements = raw.get("functional_requirements")
    required = (
        [item for item in requirements if isinstance(item, str)]
        if isinstance(requirements, list)
        else []
    )
    if domain == "python":
        gates = [
            "explicit_public_fixture_test",
            "adversarial_negative_path_test",
            "ruff_format_check",
            "ruff_lint",
            "strict_mypy",
            "pytest",
            "coverage_pytest",
            "bandit",
        ]
        focus = [
            "public interfaces and compatibility",
            "input and error behavior",
            "async lifecycle and transaction boundaries where applicable",
            "type constraints and negative-path behavior",
        ]
    elif domain == "c":
        gates = [
            "self_checking_public_unit_tests",
            "gcc_or_clang_warnings",
            "cmake_build",
            "ctest",
            "address_sanitizer",
            "undefined_behavior_sanitizer",
            "static_analysis_when_available",
        ]
        focus = [
            "public interfaces and ABI compatibility",
            "memory ownership, lifetime and cleanup",
            "integer conversions, overflow and undefined behavior",
            "error paths, partial I/O and deterministic resource release",
        ]
    else:
        gates = [
            "self_checking_public_simulation",
            "adversarial_protocol_simulation",
            "verilator_lint",
            "iverilog_compile",
            "vvp_simulation",
            "yosys_synthesis",
        ]
        focus = [
            f"explicit {domain} microarchitecture before RTL",
            "clock/reset and CDC assumptions",
            "ready/valid or bus stability under backpressure",
            "simultaneous events and boundary conditions",
        ]
    normalized["quality_contract"] = {
        "requirements": required,
        "required_gates": gates,
        "normalization_focus": focus,
        "review_requires_explicit_evidence_for_every_gate": True,
        "repair_budget": 2,
        "held_out_tests_available_to_implementation": False,
    }
    normalized["normalized_at"] = _now()
    return normalized


@dataclass(frozen=True)
class ReferenceFile:
    source_path: str
    selected_path: str
    sha256: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceManifest:
    reference_id: str
    domain: Domain
    repository: str
    commit: str
    licence_identifier: str
    licence_text_sha256: str
    permitted_use: str
    attribution: str
    source_kind: str
    registered_at: str
    files: tuple[ReferenceFile, ...]
    read_only: bool = True
    revision_kind: Literal["git_commit", "release"] = "git_commit"

    def to_json(self) -> JsonObject:
        value: JsonObject = {
            "reference_id": self.reference_id,
            "domain": self.domain,
            "repository": self.repository,
            "revision": self.commit,
            "revision_kind": self.revision_kind,
            "licence_identifier": self.licence_identifier,
            "licence_text_sha256": self.licence_text_sha256,
            "permitted_use": self.permitted_use,
            "attribution": self.attribution,
            "source_kind": self.source_kind,
            "registered_at": self.registered_at,
            "read_only": self.read_only,
            "files": [
                {
                    "source_path": item.source_path,
                    "selected_path": item.selected_path,
                    "sha256": item.sha256,
                    "topics": list(item.topics),
                }
                for item in self.files
            ],
        }
        if self.revision_kind == "git_commit":
            value["commit"] = self.commit
        return value


class ReferenceLibrary:
    """Hash-verified immutable references in project-local or shared layout."""

    def __init__(self, project_root: Path, domain: Domain, *, shared: bool = False) -> None:
        self.project_root = project_root.resolve()
        self.domain = domain
        self.shared = shared
        self.root = (
            self.project_root / _DOMAIN_LIBRARY[domain]
            if shared
            else self.project_root / "Data" / "References" / _DOMAIN_LIBRARY[domain]
        )

    @property
    def manifests(self) -> Path:
        return self.root / "90_manifests"

    @property
    def index_database(self) -> Path:
        """Dedicated derived index for this immutable reference library."""
        return self.root / "indexes" / "reference_index.db"

    def snapshot_hash(self) -> str | None:
        """Return a stable hash of immutable provenance and selected content."""
        manifests = self.manifests_list()
        if not manifests:
            return None
        payload = []
        for item in manifests:
            record = item.to_json()
            # Registration time is operational metadata, not corpus identity.
            # Rebuilding an identical overlay must produce the same fingerprint.
            record.pop("registered_at", None)
            payload.append(record)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def initialize(self, catalog_path: Path) -> JsonObject:
        catalog = catalog_path.resolve()
        if not catalog.is_file():
            raise ReferencePolicyError(f"Reference catalog is missing: {catalog}")
        try:
            raw_catalog: object = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ReferencePolicyError(f"Cannot read reference catalog: {exc}") from exc
        if not isinstance(raw_catalog, dict):
            raise ReferencePolicyError("Reference catalog must be an object")
        for folder in _DOMAIN_FOLDERS[self.domain]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for name in ("sources", "indexes", "selections"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        snapshot = self.manifests / "catalog_snapshot.json"
        _write_json_atomic(
            snapshot,
            {
                "domain": self.domain,
                "catalog_path": str(catalog),
                "catalog_sha256": _sha256(catalog),
                "initialized_at": _now(),
                "policy": raw_catalog.get("policy", {}),
                "source_ids": [
                    item.get("id")
                    for item in raw_catalog.get("sources", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ],
                "network_status": "NOT_CONTACTED_APPROVAL_REQUIRED",
            },
        )
        return self.status()

    def _manifest_path(self, reference_id: str) -> Path:
        if not _TASK_ID.fullmatch(reference_id):
            raise ReferencePolicyError("Reference id is unsafe")
        return self.manifests / f"{reference_id}.json"

    def _read_manifest(self, reference_id: str) -> ReferenceManifest:
        value = _load_json_object(self._manifest_path(reference_id))
        files_value = value.get("files")
        if not isinstance(files_value, list):
            raise ReferencePolicyError(f"Reference manifest {reference_id} has no file list")
        files: list[ReferenceFile] = []
        for item in files_value:
            if not isinstance(item, dict):
                raise ReferencePolicyError(
                    f"Reference manifest {reference_id} has malformed file data"
                )
            source_path = item.get("source_path")
            selected_path = item.get("selected_path")
            digest = item.get("sha256")
            topics = item.get("topics")
            if (
                not isinstance(source_path, str)
                or not isinstance(selected_path, str)
                or not isinstance(digest, str)
            ):
                raise ReferencePolicyError(
                    f"Reference manifest {reference_id} has malformed file data"
                )
            if not isinstance(topics, list) or not all(isinstance(topic, str) for topic in topics):
                raise ReferencePolicyError(
                    f"Reference manifest {reference_id} has malformed topics"
                )
            files.append(ReferenceFile(source_path, selected_path, digest, tuple(topics)))
        required_strings = (
            "repository",
            "licence_identifier",
            "licence_text_sha256",
            "permitted_use",
            "attribution",
            "source_kind",
            "registered_at",
        )
        values = {key: value.get(key) for key in required_strings}
        if not all(isinstance(item, str) for item in values.values()):
            raise ReferencePolicyError(f"Reference manifest {reference_id} has malformed metadata")
        revision = value.get("revision", value.get("commit"))
        revision_kind = value.get(
            "revision_kind", "git_commit" if value.get("commit") is not None else None
        )
        if not isinstance(revision, str) or revision_kind not in {"git_commit", "release"}:
            raise ReferencePolicyError(
                f"Reference manifest {reference_id} has malformed revision metadata"
            )
        return ReferenceManifest(
            reference_id=reference_id,
            domain=self.domain,
            repository=str(values["repository"]),
            commit=revision,
            licence_identifier=str(values["licence_identifier"]),
            licence_text_sha256=str(values["licence_text_sha256"]),
            permitted_use=str(values["permitted_use"]),
            attribution=str(values["attribution"]),
            source_kind=str(values["source_kind"]),
            registered_at=str(values["registered_at"]),
            files=tuple(files),
            read_only=bool(value.get("read_only", False)),
            revision_kind=revision_kind,
        )

    def register_local(
        self,
        *,
        reference_id: str,
        repository: str,
        commit: str,
        licence_identifier: str,
        licence_path: Path,
        selected_files: list[tuple[Path, str, tuple[str, ...]]],
        permitted_use: str,
        attribution: str,
    ) -> JsonObject:
        """Snapshot a focused local fixture or already-approved source tree.

        Network cloning is intentionally not implemented here: callers must
        obtain approval before material is made available locally.
        """
        if not _TASK_ID.fullmatch(reference_id) or not _COMMIT.fullmatch(commit):
            raise ReferencePolicyError("reference_id or commit is invalid")
        return self._register_local_revision(
            reference_id=reference_id,
            repository=repository,
            revision=commit,
            revision_kind="git_commit",
            licence_identifier=licence_identifier,
            licence_path=licence_path,
            selected_files=selected_files,
            permitted_use=permitted_use,
            attribution=attribution,
        )

    def register_local_release(
        self,
        *,
        reference_id: str,
        repository: str,
        release: str,
        licence_identifier: str,
        licence_path: Path,
        selected_files: list[tuple[Path, str, tuple[str, ...]]],
        permitted_use: str,
        attribution: str,
    ) -> JsonObject:
        """Register a content-hashed exact release without inventing a Git commit."""
        if not _TASK_ID.fullmatch(reference_id) or not _RELEASE.fullmatch(release):
            raise ReferencePolicyError("reference_id or release is invalid")
        return self._register_local_revision(
            reference_id=reference_id,
            repository=repository,
            revision=release,
            revision_kind="release",
            licence_identifier=licence_identifier,
            licence_path=licence_path,
            selected_files=selected_files,
            permitted_use=permitted_use,
            attribution=attribution,
        )

    def _register_local_revision(
        self,
        *,
        reference_id: str,
        repository: str,
        revision: str,
        revision_kind: Literal["git_commit", "release"],
        licence_identifier: str,
        licence_path: Path,
        selected_files: list[tuple[Path, str, tuple[str, ...]]],
        permitted_use: str,
        attribution: str,
    ) -> JsonObject:
        if not repository or not licence_identifier or not permitted_use or not attribution:
            raise ReferencePolicyError(
                "repository, licence, permitted use and attribution are required"
            )
        licence = licence_path.resolve()
        if not licence.is_file():
            raise ReferencePolicyError("Licence file is missing")
        if not selected_files:
            raise ReferencePolicyError("At least one focused reference file is required")
        source_root = self.root / "sources" / reference_id / revision
        records: list[ReferenceFile] = []
        for source, topic, topics in selected_files:
            selected_source = source.resolve()
            if not selected_source.is_file():
                raise ReferencePolicyError(f"Selected source is missing: {selected_source}")
            relative_topic = _safe_relative(topic, label="reference topic")
            if (
                not relative_topic.parts
                or relative_topic.parts[0] not in _DOMAIN_FOLDERS[self.domain]
            ):
                raise ReferencePolicyError(
                    "Reference topic must begin with a logical library folder"
                )
            source_name = _safe_relative(selected_source.name, label="reference filename")
            source_snapshot = source_root / source_name
            selected_target = self.root / relative_topic / source_name
            source_snapshot.parent.mkdir(parents=True, exist_ok=True)
            selected_target.parent.mkdir(parents=True, exist_ok=True)
            if source_snapshot.exists() and _sha256(source_snapshot) != _sha256(selected_source):
                raise ReferencePolicyError("Existing immutable source snapshot has drifted")
            if not source_snapshot.exists():
                shutil.copy2(selected_source, source_snapshot)
                source_snapshot.chmod(0o444)
            if selected_target.exists() and _sha256(selected_target) != _sha256(selected_source):
                raise ReferencePolicyError("Existing immutable selected reference has drifted")
            if not selected_target.exists():
                shutil.copy2(selected_source, selected_target)
                selected_target.chmod(0o444)
            records.append(
                ReferenceFile(
                    source_path=str(source_snapshot.relative_to(self.root)),
                    selected_path=str(selected_target.relative_to(self.root)),
                    sha256=_sha256(selected_source),
                    topics=topics,
                )
            )
        licence_snapshot = source_root / "LICENSE"
        if licence_snapshot.exists() and _sha256(licence_snapshot) != _sha256(licence):
            raise ReferencePolicyError("Existing immutable licence snapshot has drifted")
        if not licence_snapshot.exists():
            shutil.copy2(licence, licence_snapshot)
            licence_snapshot.chmod(0o444)
        manifest = ReferenceManifest(
            reference_id=reference_id,
            domain=self.domain,
            repository=repository,
            commit=revision,
            licence_identifier=licence_identifier,
            licence_text_sha256=_sha256(licence),
            permitted_use=permitted_use,
            attribution=attribution,
            source_kind="local_fixture_or_approved_local_snapshot",
            registered_at=_now(),
            files=tuple(records),
            revision_kind=revision_kind,
        )
        manifest_path = self._manifest_path(reference_id)
        if manifest_path.exists():
            previous = self._read_manifest(reference_id)
            candidate = manifest.to_json()
            candidate["registered_at"] = previous.registered_at
            if previous.to_json() != candidate:
                raise ReferencePolicyError(
                    "Reference manifest is immutable; use a new id for a new snapshot"
                )
        else:
            _write_json_atomic(manifest_path, manifest.to_json())
        return self.verify(reference_id)

    def manifests_list(self) -> list[ReferenceManifest]:
        if not self.manifests.is_dir():
            return []
        return [
            self._read_manifest(path.stem)
            for path in sorted(self.manifests.glob("*.json"))
            if path.name != "catalog_snapshot.json"
        ]

    def select(self, topics: list[str]) -> JsonObject:
        requested = sorted(set(topics))
        selected: list[JsonObject] = []
        for manifest in self.manifests_list():
            matching = [
                item
                for item in manifest.files
                if not requested or set(item.topics).intersection(requested)
            ]
            if matching:
                selected.append(
                    {
                        "reference_id": manifest.reference_id,
                        "files": [item.selected_path for item in matching],
                    }
                )
        target = (
            self.root
            / "selections"
            / f"selection_{hashlib.sha256(json.dumps(requested).encode()).hexdigest()[:16]}.json"
        )
        report: JsonObject = {
            "domain": self.domain,
            "topics": requested,
            "references": selected,
            "created_at": _now(),
        }
        _write_json_atomic(target, report)
        report["path"] = str(target)
        return report

    def synchronize(self) -> JsonObject:
        """Verify local immutable snapshots; never contact a network remote."""
        verification = self.verify()
        return {
            "domain": self.domain,
            "status": "SYNCHRONIZED_LOCAL"
            if verification["status"] == "VERIFIED"
            else "DRIFT_DETECTED",
            "network_status": "NOT_CONTACTED_APPROVAL_REQUIRED",
            "verification": verification,
            "next_step": "After explicit network and licence approval, register a new exact-commit snapshot rather than updating an immutable reference.",
        }

    def ingest(self, database: Path | None = None) -> JsonObject:
        """Index selected reference text in the existing project SQLite database."""
        target_database = (database or self.index_database).resolve()
        conn = _db(target_database)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reference_ingestions(reference_id TEXT NOT NULL, selected_path TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(reference_id, selected_path))"
        )
        counts: dict[str, int] = {"indexed": 0, "unchanged": 0, "skipped": 0}
        try:
            for manifest in self.manifests_list():
                for record in manifest.files:
                    selected = _inside(self.root, self.root / record.selected_path)
                    if not selected.is_file() or _sha256(selected) != record.sha256:
                        raise ReferencePolicyError(
                            f"Reference hash verification failed before ingestion: {record.selected_path}"
                        )
                    existing = conn.execute(
                        "SELECT 1 FROM reference_ingestions WHERE reference_id=? AND selected_path=? AND sha256=?",
                        (manifest.reference_id, record.selected_path, record.sha256),
                    ).fetchone()
                    if existing:
                        counts["unchanged"] += 1
                        continue
                    if selected.suffix.lower() not in {
                        ".md",
                        ".rst",
                        ".txt",
                        ".py",
                        ".c",
                        ".h",
                        ".sv",
                        ".v",
                        ".json",
                        ".yaml",
                        ".yml",
                    }:
                        counts["skipped"] += 1
                        continue
                    text = selected.read_text(encoding="utf-8", errors="replace")
                    document_id = hashlib.sha256(
                        (manifest.reference_id + record.sha256).encode()
                    ).hexdigest()[:20]
                    metadata = {
                        "absolute_source_path": str(selected),
                        "title": selected.name,
                        "source_class": "technical_documentation",
                        "source_kind": "governed_reference",
                        "reference_id": manifest.reference_id,
                        "commit": manifest.commit
                        if manifest.revision_kind == "git_commit"
                        else None,
                        "revision": manifest.commit,
                        "revision_kind": manifest.revision_kind,
                        "licence_identifier": manifest.licence_identifier,
                        "permitted_use": manifest.permitted_use,
                        "repository": manifest.repository,
                        "attribution": manifest.attribution,
                        "selected_path": record.selected_path,
                        "sha256": record.sha256,
                        "topics": list(record.topics),
                        "read_only": True,
                    }
                    conn.execute(
                        "INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)",
                        (
                            document_id,
                            record.sha256,
                            selected.name,
                            "technical_documentation",
                            json.dumps(metadata, sort_keys=True),
                        ),
                    )
                    for index, chunk in enumerate(_chunks(text)):
                        chunk_id = f"reference:{manifest.reference_id}:{record.sha256[:16]}:{index}"
                        conn.execute(
                            "INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?,?)",
                            (chunk_id, document_id, None, None, None, chunk),
                        )
                    conn.execute(
                        "INSERT INTO reference_ingestions VALUES(?,?,?)",
                        (manifest.reference_id, record.selected_path, record.sha256),
                    )
                    counts["indexed"] += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        report: JsonObject = {
            "domain": self.domain,
            "database": str(target_database),
            "counts": counts,
            "ingested_at": _now(),
            "snapshot_hash": self.snapshot_hash(),
        }
        target = self.root / "indexes" / "ingestion_report.json"
        _write_json_atomic(target, report)
        report["report_path"] = str(target)
        return report

    def search_chunks(
        self,
        query: str,
        *,
        limit: int = 6,
        token_budget: int = 3500,
        max_chunks_per_path: int = 2,
        max_chunks_per_reference: int = 3,
    ) -> list[JsonObject]:
        """Retrieve ranked, provenance-complete governed-reference content."""
        if limit <= 0 or token_budget <= 0:
            raise ReferencePolicyError("Reference search limits must be positive")
        if max_chunks_per_path <= 0 or max_chunks_per_reference <= 0:
            raise ReferencePolicyError("Reference diversity limits must be positive")
        manifests = self.manifests_list()
        if not manifests:
            return []
        verification = self.verify()
        if verification.get("status") != "VERIFIED":
            raise ReferencePolicyError("Governed reference library failed hash verification")
        self.ingest()
        conn = _db(self.index_database)
        try:
            rows = conn.execute(
                "SELECT c.id,c.text,d.filename,d.metadata "
                "FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.class=?",
                ("technical_documentation",),
            ).fetchall()
        finally:
            conn.close()
        query_tokens = set(re.findall(r"[\w.-]+", query.lower()))
        query_vector = embed(query)
        ranked: list[tuple[float, str, str, dict[str, object]]] = []
        for chunk_id, text, filename, metadata_text in rows:
            if not isinstance(chunk_id, str) or not isinstance(text, str):
                continue
            try:
                metadata_raw: object = json.loads(metadata_text or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata_raw = {}
            if not isinstance(metadata_raw, dict):
                metadata_raw = {}
            if metadata_raw.get("source_kind") != "governed_reference":
                continue
            metadata = {str(key): value for key, value in metadata_raw.items()}
            topics_raw = metadata.get("topics", [])
            topics = (
                [item for item in topics_raw if isinstance(item, str)]
                if isinstance(topics_raw, list)
                else []
            )
            selected_path = str(metadata.get("selected_path", filename or ""))
            searchable = " ".join([text, str(filename or ""), selected_path, *topics])
            tokens = set(re.findall(r"[\w.-]+", searchable.lower()))
            lexical = len(query_tokens.intersection(tokens)) / max(1, len(query_tokens))
            semantic = sum(a * b for a, b in zip(query_vector, embed(searchable)))
            topic_tokens = set(re.findall(r"[\w.-]+", " ".join(topics).lower()))
            topic_overlap = len(query_tokens.intersection(topic_tokens)) / max(1, len(query_tokens))
            score = 0.50 * semantic + 0.35 * lexical + 0.15 * topic_overlap
            if score <= 0:
                continue
            ranked.append((score, chunk_id, text, metadata))
        ordered = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)
        remaining_chars = token_budget * 4
        selected: list[JsonObject] = []
        seen_content: set[str] = set()
        path_counts: dict[str, int] = {}
        reference_counts: dict[str, int] = {}

        def consider(
            candidate: tuple[float, str, str, dict[str, object]], *, diversity_pass: bool
        ) -> None:
            nonlocal remaining_chars
            if len(selected) >= limit or remaining_chars <= 0:
                return
            score, chunk_id, text, metadata = candidate
            path = str(metadata.get("selected_path", ""))
            reference_id = str(metadata.get("reference_id", ""))
            if diversity_pass and path_counts.get(path, 0) > 0:
                return
            if path_counts.get(path, 0) >= max_chunks_per_path:
                return
            if reference_counts.get(reference_id, 0) >= max_chunks_per_reference:
                return
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen_content:
                return
            content = text[:remaining_chars]
            if not content.strip():
                return
            seen_content.add(fingerprint)
            path_counts[path] = path_counts.get(path, 0) + 1
            reference_counts[reference_id] = reference_counts.get(reference_id, 0) + 1
            topics_raw = metadata.get("topics", [])
            topics = (
                [item for item in topics_raw if isinstance(item, str)]
                if isinstance(topics_raw, list)
                else []
            )
            searchable_tokens = set(re.findall(r"[\w.-]+", " ".join([text, path, *topics]).lower()))
            matched_terms = sorted(query_tokens.intersection(searchable_tokens))
            selected.append(
                {
                    "kind": "governed_reference",
                    "reference_id": metadata.get("reference_id"),
                    "repository": metadata.get("repository"),
                    "commit": metadata.get("commit"),
                    "revision": metadata.get("revision", metadata.get("commit")),
                    "revision_kind": metadata.get("revision_kind", "git_commit"),
                    "licence_identifier": metadata.get("licence_identifier"),
                    "permitted_use": metadata.get("permitted_use"),
                    "attribution": metadata.get("attribution"),
                    "path": metadata.get("selected_path"),
                    "sha256": metadata.get("sha256"),
                    "topics": topics,
                    "chunk_id": chunk_id,
                    "score": round(score, 6),
                    "estimated_tokens": max(1, (len(content) + 3) // 4),
                    "matched_query_terms": matched_terms[:24],
                    "selection_pass": "diversity" if diversity_pass else "score_fill",
                    "content": content,
                }
            )
            remaining_chars -= len(content)

        for diversity_pass in (True, False):
            for candidate in ordered:
                consider(candidate, diversity_pass=diversity_pass)
                if len(selected) >= limit or remaining_chars <= 0:
                    break
        return selected

    def verify(self, reference_id: str | None = None) -> JsonObject:
        manifests = [self._read_manifest(reference_id)] if reference_id else self.manifests_list()
        errors: list[str] = []
        verified = 0
        for manifest in manifests:
            if manifest.revision_kind == "git_commit":
                if not _COMMIT.fullmatch(manifest.commit):
                    errors.append(f"{manifest.reference_id}: invalid commit")
            elif not _RELEASE.fullmatch(manifest.commit):
                errors.append(f"{manifest.reference_id}: invalid release")
            licence = self.root / "sources" / manifest.reference_id / manifest.commit / "LICENSE"
            if not licence.is_file() or _sha256(licence) != manifest.licence_text_sha256:
                errors.append(f"{manifest.reference_id}: licence hash mismatch")
            for record in manifest.files:
                selected = self.root / record.selected_path
                source = self.root / record.source_path
                if not selected.is_file() or not source.is_file():
                    errors.append(f"{manifest.reference_id}: missing selected source")
                    continue
                if _sha256(selected) != record.sha256 or _sha256(source) != record.sha256:
                    errors.append(f"{manifest.reference_id}: selected file hash mismatch")
                    continue
                if selected.stat().st_mode & 0o222:
                    errors.append(f"{manifest.reference_id}: selected file is writable")
                    continue
                verified += 1
        return {
            "domain": self.domain,
            "status": "VERIFIED" if not errors else "FAILED",
            "references": len(manifests),
            "files_verified": verified,
            "errors": errors,
        }

    def status(self) -> JsonObject:
        snapshot = self.manifests / "catalog_snapshot.json"
        return {
            "domain": self.domain,
            "root": str(self.root),
            "shared": self.shared,
            "initialized": snapshot.is_file(),
            "network_status": "NOT_CONTACTED_APPROVAL_REQUIRED",
            "references": [item.reference_id for item in self.manifests_list()],
            "verification": self.verify(),
        }


@dataclass(frozen=True)
class ToolResult:
    tool: str
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    status: str
    stdout: str
    stderr: str
    log_path: str

    def to_json(self) -> JsonObject:
        return {
            "tool": self.tool,
            "command": list(self.command),
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "log_path": self.log_path,
        }


def verilator_simulation_available() -> bool:
    """Return whether the installed Verilator supports timed ``--binary`` testbenches."""
    executable = shutil.which("verilator")
    if executable is None:
        return False
    try:
        completed = subprocess.run(  # nosec B603
            [executable, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    match = re.search(r"\bVerilator\s+(\d+)(?:\.|\b)", completed.stdout)
    return completed.returncode == 0 and match is not None and int(match.group(1)) >= 5


class LocalToolRunner:
    """Small, timeout-bound command allowlist with immutable captured logs."""

    def __init__(self, repository_root: Path, log_root: Path | None = None) -> None:
        self.repository_root = repository_root.resolve()
        self.log_root = (
            log_root or self.repository_root / "outputs" / "a6000_agent_team" / "tool_logs"
        ).resolve()

    def _target_paths(self, paths: list[str]) -> list[str]:
        values = paths or ["src", "tests"]
        output: list[str] = []
        for value in values:
            relative = _safe_relative(value, label="tool target")
            resolved = _inside(self.repository_root, self.repository_root / relative)
            if not resolved.exists():
                raise ToolExecutionError(f"Tool target does not exist: {value}")
            output.append(str(relative))
        return output

    def run(self, tool: str, command: list[str], *, timeout_seconds: int = 300) -> ToolResult:
        if tool not in {
            "ruff_format",
            "ruff",
            "mypy",
            "pytest",
            "coverage",
            "bandit",
            "verilator",
            "verilator_simulation",
            "iverilog",
            "vvp",
            "yosys",
            "cmake",
            "ctest",
            "gcc",
            "clang",
            "clang_tidy",
            "c_public_test",
            "sanitizer_probe",
            "sanitizer_compile",
            "sanitizer_test",
            "cuda_probe",
        }:
            raise ToolExecutionError(f"Tool is not allowlisted: {tool}")
        if timeout_seconds < 1 or timeout_seconds > 1800:
            raise ToolExecutionError("Tool timeout must be between 1 and 1800 seconds")
        started = time.monotonic()
        environment = os.environ.copy()
        for key in tuple(environment):
            if key.startswith("LAPLACE_ABLATION_") or key == "LAPLACE_SERVER_OWNER_TOKEN":
                environment.pop(key, None)
        try:
            # The command is built only by this module's allowlisted methods.
            process = subprocess.Popen(  # nosec B603
                command,
                cwd=self.repository_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )
        except OSError as exc:
            raise ToolExecutionError(f"Cannot start {tool}: {exc}") from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            returncode = process.returncode
            status = "PASS" if returncode == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                stdout, stderr = process.communicate()
            returncode = 124
            stderr = (stderr or "") + "\nTimed out and terminated process tree."
            status = "TIMEOUT"
        elapsed = time.monotonic() - started
        bounded_stdout = stdout[-100_000:]
        bounded_stderr = stderr[-100_000:]
        log = self.log_root / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{tool}.json"
        result = ToolResult(
            tool,
            tuple(command),
            returncode,
            elapsed,
            status,
            bounded_stdout,
            bounded_stderr,
            str(log),
        )
        _write_json_atomic(log, result.to_json(), readonly=True)
        return result

    def run_python_quality_gates(
        self,
        paths: list[str] | None = None,
        *,
        required_test_paths: list[str] | None = None,
        timeout_seconds: int = 300,
    ) -> JsonObject:
        """Run only task-scoped Python gates and explicitly declared public tests."""
        targets = self._target_paths(paths or [])
        python_targets = [target for target in targets if Path(target).suffix == ".py"]
        bandit_targets = python_targets or ["src"]
        required_tests = self._target_paths(required_test_paths or [])
        if not required_tests:
            raise ToolExecutionError("At least one explicit public Python test is required")
        if not all(Path(test).suffix == ".py" for test in required_tests):
            raise ToolExecutionError("Required Python tests must be .py files")
        public_test_commands: list[tuple[str, list[str]]] = []
        for test in required_tests:
            public_test_commands.append(
                (
                    "pytest",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        test,
                    ],
                )
            )
        commands: list[tuple[str, list[str]]] = [
            ("ruff_format", [sys.executable, "-m", "ruff", "format", "--check", *targets]),
            ("ruff", [sys.executable, "-m", "ruff", "check", *targets]),
            ("mypy", [sys.executable, "-m", "mypy", *targets]),
            *public_test_commands,
            (
                "coverage",
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "-m",
                    "pytest",
                    "-q",
                    *required_tests,
                ],
            ),
        ]
        if shutil.which("bandit"):
            commands.append(
                ("bandit", [sys.executable, "-m", "bandit", "-q", "-r", *bandit_targets])
            )
        results = [
            self.run(tool, command, timeout_seconds=timeout_seconds).to_json()
            for tool, command in commands
        ]
        passed = all(item["status"] == "PASS" for item in results)
        report: JsonObject = {
            "operation": "run_python_quality_gates",
            "repository_root": str(self.repository_root),
            "required_test_paths": required_tests,
            "scope_policy": "task_declared_sources_and_public_tests_only",
            "repository_wide_tests_executed": False,
            "passed": passed,
            "results": results,
            "created_at": _now(),
        }
        report_path = (
            self.log_root / f"quality_report_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        _write_json_atomic(report_path, report, readonly=True)
        report["report_path"] = str(report_path)
        return report

    def run_eda_flow(
        self,
        source_files: list[str],
        *,
        top_module: str | None = None,
        testbench: str | None = None,
        language: Literal["verilog", "systemverilog"] = "systemverilog",
        require_verilator_simulation: bool = False,
        required_tools: tuple[str, ...] = (),
        timeout_seconds: int = 300,
    ) -> JsonObject:
        if not source_files:
            raise ToolExecutionError("At least one SystemVerilog source is required")
        sources = self._target_paths(source_files)
        if not all(Path(value).suffix.lower() in {".sv", ".v"} for value in sources):
            raise ToolExecutionError("EDA sources must be .sv or .v files")
        ordered_sources = sorted(
            sources,
            key=lambda value: (
                0 if language == "systemverilog" and Path(value).stem.endswith("_pkg") else 1,
                sources.index(value),
            ),
        )
        simulation_sources = list(ordered_sources)
        if testbench is not None:
            testbench_path = self._target_paths([testbench])[0]
            if Path(testbench_path).suffix.lower() not in {".sv", ".v"}:
                raise ToolExecutionError("EDA testbench must be a .sv or .v file")
            simulation_sources.append(testbench_path)
        if top_module is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", top_module):
            raise ToolExecutionError("Top module name is unsafe")
        results: list[JsonObject] = []
        missing_tools: list[str] = []
        for tool in required_tools:
            executable = "verilator" if tool == "verilator_simulation" else tool
            if tool == "verilator_simulation":
                available = verilator_simulation_available()
            else:
                available = shutil.which(executable) is not None
            if not available:
                missing_tools.append(tool)
        if require_verilator_simulation and language == "systemverilog" and testbench is not None:
            if not verilator_simulation_available():
                missing_tools.append("verilator_simulation")
        missing_tools = sorted(set(missing_tools))
        if shutil.which("verilator"):
            language_flag = "1364-2001" if language == "verilog" else "1800-2017"
            results.append(
                self.run(
                    "verilator",
                    [
                        "verilator",
                        "--lint-only",
                        "--Wall",
                        "--language",
                        language_flag,
                        *sources,
                    ],
                    timeout_seconds=timeout_seconds,
                ).to_json()
            )
            if (
                language == "systemverilog"
                and testbench is not None
                and require_verilator_simulation
                and "verilator_simulation" not in missing_tools
            ):
                verilator_top = Path(testbench).stem
                verilator_root = self.log_root / f"verilator_sim_{uuid.uuid4().hex}"
                results.append(
                    self.run(
                        "verilator",
                        [
                            "verilator",
                            "--binary",
                            "--timing",
                            "--Wall",
                            "-Wno-fatal",
                            "--language",
                            language_flag,
                            "--top-module",
                            verilator_top,
                            "--Mdir",
                            str(verilator_root),
                            "-o",
                            "simv",
                            *simulation_sources,
                        ],
                        timeout_seconds=timeout_seconds,
                    ).to_json()
                )
                verilator_executable = verilator_root / "simv"
                if results[-1]["status"] == "PASS" and verilator_executable.is_file():
                    results.append(
                        self.run(
                            "verilator_simulation",
                            [str(verilator_executable)],
                            timeout_seconds=timeout_seconds,
                        ).to_json()
                    )
        output = self.log_root / f"eda_{uuid.uuid4().hex}.vvp"
        if shutil.which("iverilog"):
            generation = "-g2001" if language == "verilog" else "-g2012"
            compile_command = ["iverilog", generation, "-o", str(output)]
            simulation_top = Path(testbench).stem if testbench is not None else top_module
            if simulation_top:
                compile_command.extend(["-s", simulation_top])
            compile_command.extend(simulation_sources)
            results.append(
                self.run("iverilog", compile_command, timeout_seconds=timeout_seconds).to_json()
            )
            if testbench and results[-1]["status"] == "PASS" and shutil.which("vvp"):
                results.append(
                    self.run("vvp", ["vvp", str(output)], timeout_seconds=timeout_seconds).to_json()
                )
        if shutil.which("yosys") and top_module:
            script = self.log_root / f"synth_{uuid.uuid4().hex}.ys"
            _write_json_atomic(
                script.with_suffix(".json"), {"sources": sources, "top_module": top_module}
            )
            script.write_text(
                f"read_verilog {'-sv ' if language == 'systemverilog' else ''}{' '.join(ordered_sources)}\n"
                f"hierarchy -check -top {top_module}\nsynth -top {top_module}\n",
                encoding="utf-8",
            )
            results.append(
                self.run(
                    "yosys", ["yosys", "-s", str(script)], timeout_seconds=timeout_seconds
                ).to_json()
            )
        return {
            "operation": "run_eda_flow",
            "language": language,
            "require_verilator_simulation": require_verilator_simulation,
            "required_tools": list(required_tools),
            "sources": sources,
            "top_module": top_module,
            "results": results,
            "missing_tools": missing_tools,
            "passed": not missing_tools
            and bool(results)
            and all(item["status"] == "PASS" for item in results),
            "created_at": _now(),
        }

    def run_c_quality_gates(
        self,
        fixture: str,
        *,
        timeout_seconds: int = 300,
        sanitizers: bool = True,
        required_tools: tuple[str, ...] = (),
    ) -> JsonObject:
        """Run GCC minimum gates plus declared or available strengthening tools."""
        source_directory = self._target_paths([fixture])[0]
        source_root = self.repository_root / source_directory
        self.log_root.mkdir(parents=True, exist_ok=True)
        sources = sorted(source_root.glob("*.c"))
        if not sources:
            raise ToolExecutionError("C fixture must contain at least one .c source")
        compiler = shutil.which("gcc")
        missing_tools: list[str] = []
        if compiler is None:
            missing_tools.append("gcc")
        for tool in required_tools:
            if tool not in {"asan", "ubsan"} and shutil.which(tool) is None:
                missing_tools.append(tool)
        results: list[JsonObject] = []
        strengthening_results: list[JsonObject] = []
        relative_sources = [str(path.relative_to(self.repository_root)) for path in sources]
        if compiler is not None:
            direct_executable = self.log_root / f"c_direct_{uuid.uuid4().hex}"
            results.append(
                self.run(
                    "gcc",
                    [
                        compiler,
                        "-std=c11",
                        "-Wall",
                        "-Wextra",
                        "-Wpedantic",
                        "-Werror",
                        *relative_sources,
                        "-o",
                        str(direct_executable),
                    ],
                    timeout_seconds=timeout_seconds,
                ).to_json()
            )
            if results[-1]["status"] == "PASS":
                results.append(
                    self.run(
                        "c_public_test", [str(direct_executable)], timeout_seconds=timeout_seconds
                    ).to_json()
                )
        cmake_available = shutil.which("cmake") is not None and shutil.which("ctest") is not None
        if cmake_available and (source_root / "CMakeLists.txt").is_file():
            build_root = self.log_root / f"c_build_{uuid.uuid4().hex}"
            configure = [
                "cmake",
                "-S",
                source_directory,
                "-B",
                str(build_root),
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DCMAKE_C_STANDARD=11",
                "-DCMAKE_C_EXTENSIONS=OFF",
                "-DCMAKE_C_FLAGS=-Wall -Wextra -Wpedantic -Werror",
            ]
            results.append(self.run("cmake", configure, timeout_seconds=timeout_seconds).to_json())
            if results[-1]["status"] == "PASS":
                results.append(
                    self.run(
                        "cmake",
                        ["cmake", "--build", str(build_root), "--parallel", "2"],
                        timeout_seconds=timeout_seconds,
                    ).to_json()
                )
            if results[-1]["status"] == "PASS":
                results.append(
                    self.run(
                        "ctest",
                        ["ctest", "--test-dir", str(build_root), "--output-on-failure"],
                        timeout_seconds=timeout_seconds,
                    ).to_json()
                )
        if sanitizers:
            required_sanitizers = [item for item in ("asan", "ubsan") if item in required_tools]
            sanitizer_names = required_sanitizers or ["asan", "ubsan"]
            sanitizer_map = {"asan": "address", "ubsan": "undefined"}
            sanitizer_flag = "-fsanitize=" + ",".join(
                sanitizer_map[item] for item in sanitizer_names
            )
            probe_source = self.log_root / f"c_sanitizer_probe_{uuid.uuid4().hex}.c"
            probe_source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            probe_executable = probe_source.with_suffix("")
            sanitizer_compiler: str | None = None
            for candidate_name in ("gcc", "clang"):
                candidate = shutil.which(candidate_name)
                if candidate is None:
                    continue
                probe = self.run(
                    "sanitizer_probe",
                    [candidate, str(probe_source), sanitizer_flag, "-o", str(probe_executable)],
                    timeout_seconds=timeout_seconds,
                ).to_json()
                probe["compiler"] = candidate
                strengthening_results.append(probe)
                if probe["status"] == "PASS":
                    sanitizer_compiler = candidate
                    break
            if sanitizer_compiler is not None:
                sanitized_executable = self.log_root / f"c_sanitized_{uuid.uuid4().hex}"
                sanitized = self.run(
                    "sanitizer_compile",
                    [
                        sanitizer_compiler,
                        "-std=c11",
                        "-Wall",
                        "-Wextra",
                        "-Wpedantic",
                        "-Werror",
                        "-fno-omit-frame-pointer",
                        sanitizer_flag,
                        *relative_sources,
                        "-o",
                        str(sanitized_executable),
                    ],
                    timeout_seconds=timeout_seconds,
                ).to_json()
                results.append(sanitized)
                if sanitized["status"] == "PASS":
                    results.append(
                        self.run(
                            "sanitizer_test",
                            [str(sanitized_executable)],
                            timeout_seconds=timeout_seconds,
                        ).to_json()
                    )
            elif required_sanitizers:
                missing_tools.extend(required_sanitizers)
        clang = shutil.which("clang")
        if clang is not None:
            results.append(
                self.run(
                    "clang",
                    [
                        clang,
                        "-std=c11",
                        "-Wall",
                        "-Wextra",
                        "-Wpedantic",
                        "-Werror",
                        "-fsyntax-only",
                        *relative_sources,
                    ],
                    timeout_seconds=timeout_seconds,
                ).to_json()
            )
        missing_tools = sorted(set(missing_tools))
        report: JsonObject = {
            "operation": "run_c_quality_gates",
            "fixture": source_directory,
            "sanitizers": sanitizers,
            "required_tools": list(required_tools),
            "missing_tools": missing_tools,
            "results": results,
            "strengthening_results": strengthening_results,
            "passed": not missing_tools
            and bool(results)
            and all(item["status"] == "PASS" for item in results),
            "created_at": _now(),
        }
        report_path = self.log_root / f"c_quality_{uuid.uuid4().hex}.json"
        _write_json_atomic(report_path, report, readonly=True)
        report["report_path"] = str(report_path)
        return report


@dataclass(frozen=True)
class StateEvent:
    from_state: TaskState
    to_state: TaskState
    role: Role
    timestamp: str
    note: str


@dataclass
class AgentTask:
    task_id: str
    domain: Domain
    state: TaskState
    specification: JsonObject
    artifacts: dict[str, str] = field(default_factory=dict)
    transitions: list[StateEvent] = field(default_factory=list)
    correction_loops: int = 0
    blocked_reason: str | None = None

    def to_json(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "state": self.state,
            "specification": self.specification,
            "artifacts": self.artifacts,
            "transitions": [asdict(event) for event in self.transitions],
            "correction_loops": self.correction_loops,
            "blocked_reason": self.blocked_reason,
        }


class AgentTaskStore:
    """Persisted, resumable state machine using existing project-local data."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.root = self.project_root / "Data" / "AgentTeam" / "tasks"

    def _path(self, task_id: str) -> Path:
        if not _TASK_ID.fullmatch(task_id):
            raise EngineeringError("Unsafe task id")
        return self.root / task_id / "task.json"

    def create(self, domain: Domain, specification: JsonObject) -> AgentTask:
        task_id_value = specification.get("task_id")
        if not isinstance(task_id_value, str):
            raise EngineeringError("Task specification has no task_id")
        path = self._path(task_id_value)
        if path.exists():
            raise EngineeringError(f"Task already exists: {task_id_value}")
        task = AgentTask(task_id_value, domain, "request", specification)
        self.save(task)
        return task

    def load(self, task_id: str) -> AgentTask:
        value = _load_json_object(self._path(task_id))
        domain = value.get("domain")
        state = value.get("state")
        specification = value.get("specification")
        artifacts = value.get("artifacts")
        transitions = value.get("transitions")
        if (
            domain not in {"python", "c", "verilog", "systemverilog"}
            or state not in _TASK_TRANSITIONS
            or not isinstance(specification, dict)
        ):
            raise EngineeringError("Persisted task is malformed")
        if not isinstance(artifacts, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in artifacts.items()
        ):
            raise EngineeringError("Persisted task artifacts are malformed")
        events: list[StateEvent] = []
        if not isinstance(transitions, list):
            raise EngineeringError("Persisted task transitions are malformed")
        for item in transitions:
            if not isinstance(item, dict):
                raise EngineeringError("Persisted task transition is malformed")
            prior, target, role, timestamp, note = (
                item.get(key) for key in ("from_state", "to_state", "role", "timestamp", "note")
            )
            if (
                prior not in _TASK_TRANSITIONS
                or target not in _TASK_TRANSITIONS
                or role not in {"supervisor", "researcher", "implementer", "verifier", "reviewer"}
                or not isinstance(timestamp, str)
                or not isinstance(note, str)
            ):
                raise EngineeringError("Persisted task transition is malformed")
            events.append(StateEvent(prior, target, role, timestamp, note))
        loops = value.get("correction_loops", 0)
        if not isinstance(loops, int) or loops < 0 or loops > 2:
            raise EngineeringError("Persisted correction-loop count is malformed")
        blocked = value.get("blocked_reason")
        if blocked is not None and not isinstance(blocked, str):
            raise EngineeringError("Persisted blocked reason is malformed")
        return AgentTask(
            task_id, domain, state, specification, dict(artifacts), events, loops, blocked
        )

    def save(self, task: AgentTask) -> None:
        _write_json_atomic(self._path(task.task_id), task.to_json())

    def transition(self, task_id: str, target: TaskState, *, role: Role, note: str) -> AgentTask:
        if role != "supervisor":
            raise EngineeringError("Only the supervisor may transition task state")
        task = self.load(task_id)
        if target not in _TASK_TRANSITIONS[task.state]:
            raise EngineeringError(f"Invalid task transition {task.state} -> {target}")
        if target == "bounded_correction":
            if task.correction_loops >= 2:
                raise EngineeringError("Maximum of two correction loops reached")
            task.correction_loops += 1
        task.transitions.append(StateEvent(task.state, target, role, _now(), note))
        task.state = target
        if target == "blocked":
            task.blocked_reason = note
        self.save(task)
        return task

    def write_artifact(self, task_id: str, *, role: Role, name: str, payload: JsonObject) -> Path:
        if not _TASK_ID.fullmatch(name):
            raise EngineeringError("Artifact name is unsafe")
        allowed: dict[Role, set[str]] = {
            "supervisor": {"requirements", "plan", "final_report", "escalation"},
            "researcher": {"evidence_packet"},
            "implementer": {"implementation_report", "patch_manifest", "test_strategy"},
            "verifier": {"verification_report", "defect_report"},
            "reviewer": {"review_report"},
        }
        if name not in allowed[role]:
            raise EngineeringError(f"Role {role} cannot write artifact {name}")
        task = self.load(task_id)
        target = self._path(task_id).parent / "artifacts" / f"{name}.json"
        _write_json_atomic(target, payload)
        task.artifacts[name] = str(target)
        self.save(task)
        return target


def expand_engineering_query(task: AgentTask, query: str) -> tuple[str, list[str]]:
    """Expand narrow task text with deterministic invariant vocabulary."""
    base_terms = [query]
    objective = task.specification.get("objective")
    if isinstance(objective, str) and objective.strip() and objective.strip() != query.strip():
        base_terms.append(objective)
    requirements = task.specification.get("functional_requirements")
    if isinstance(requirements, list):
        base_terms.extend(item for item in requirements if isinstance(item, str))
    searchable = " ".join(base_terms).lower()
    expansions: list[str] = []
    if task.domain == "python":
        if any(token in searchable for token in ("pydantic", "strict", "coerc", "undeclared")):
            expansions.extend(
                [
                    "pydantic v2 ConfigDict strict true extra forbid",
                    "strict integer validation no coercion",
                    "model_config field validation",
                ]
            )
        if any(token in searchable for token in ("sqlite", "transaction", "conflict", "rollback")):
            expansions.extend(
                [
                    "sqlite transaction rollback conflict idempotent",
                    "exception ordering preserve ValueError",
                    "commit only after successful state transition",
                ]
            )
        if any(token in searchable for token in ("async", "deadline", "cancel")):
            expansions.extend(
                [
                    "asyncio cancellation cleanup await task",
                    "deadline timeout preserve successful result",
                ]
            )
    elif task.domain == "c":
        if any(token in searchable for token in ("buffer", "length", "parse", "string")):
            expansions.extend(
                [
                    "C bounded buffer size_t length termination partial parse",
                    "checked allocation ownership cleanup single exit path",
                ]
            )
        if any(token in searchable for token in ("integer", "overflow", "conversion", "shift")):
            expansions.extend(
                [
                    "C integer conversion overflow undefined behavior checked arithmetic",
                    "unsigned bounds before narrowing conversion",
                ]
            )
        if any(token in searchable for token in ("file", "stream", "process", "errno")):
            expansions.extend(
                [
                    "C stdio partial read write errno cleanup",
                    "portable process and file error handling",
                ]
            )
    else:
        if any(
            token in searchable
            for token in ("ready/valid", "ready valid", "buffer", "slot", "enqueue", "dequeue")
        ):
            expansions.extend(
                [
                    "ready valid skid buffer simultaneous enqueue dequeue",
                    "consume and replace payload stability under stall",
                    "in_ready equals empty or downstream ready",
                    "backpressure full empty transition no combinational handshake loop",
                ]
            )
        if any(token in searchable for token in ("axi", "wstrb", "write channel")):
            expansions.extend(
                [
                    "AXI4-Lite independent AW and W channel capture",
                    "write response after address and data handshakes",
                    "WSTRB byte enable register write",
                ]
            )
        if any(token in searchable for token in ("w1c", "write-one-to-clear", "irq", "pending")):
            expansions.extend(
                [
                    "write one to clear pending enable interrupt",
                    "byte strobe W1C priority event set clear",
                    "IRQ equals enable and pending",
                ]
            )
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in expansions:
        normalized = " ".join(item.split())
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        deduplicated.append(normalized)
    expanded = "\n".join([*base_terms, *deduplicated])
    return expanded, deduplicated


def _project_knowledge_cards(repository_root: Path, domain: Domain, query: str) -> list[JsonObject]:
    """Retrieve small project-authored invariant cards with exact hashes."""
    card_root = repository_root / "codex_a6000" / "knowledge_cards"
    candidates_by_domain: dict[Domain, list[Path]] = {
        "python": [card_root / "python_strict_validation_transactions.md"],
        "c": [card_root / "c_safety_portability.md"],
        "verilog": [card_root / "verilog2001_synthesis.md"],
        "systemverilog": [card_root / "systemverilog_handshake_axi_w1c.md"],
    }
    candidates = candidates_by_domain[domain]
    query_tokens = set(re.findall(r"[\w.-]+", query.lower()))
    anchors_by_domain: dict[Domain, set[str]] = {
        "python": {
            "pydantic",
            "strict",
            "coercion",
            "sqlite",
            "transaction",
            "rollback",
            "conflict",
        },
        "c": {
            "buffer",
            "ownership",
            "lifetime",
            "overflow",
            "conversion",
            "errno",
            "sanitizer",
        },
        "verilog": {
            "verilog",
            "counter",
            "fifo",
            "ready",
            "valid",
            "arbiter",
            "reset",
            "synthesis",
        },
        "systemverilog": {
            "ready",
            "valid",
            "buffer",
            "slot",
            "axi",
            "wstrb",
            "w1c",
            "irq",
            "pending",
        },
    }
    anchors = anchors_by_domain[domain]
    if not query_tokens.intersection(anchors):
        return []
    records: list[JsonObject] = []
    for path in candidates:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        card_tokens = set(re.findall(r"[\w.-]+", content.lower()))
        matched = sorted(query_tokens.intersection(card_tokens))
        if not matched:
            continue
        records.append(
            {
                "kind": "project_knowledge_card",
                "path": str(path.relative_to(repository_root)),
                "sha256": _sha256(path),
                "matched_query_terms": matched[:24],
                "content": content[:8000],
            }
        )
    return records


def retrieve_engineering_evidence(
    repository_root: Path,
    project_root: Path,
    task: AgentTask,
    *,
    query: str,
    shared_reference_root: Path | None = None,
    governed_chunk_limit: int = 6,
    governed_token_budget: int = 3500,
) -> JsonObject:
    """Return precedence-ordered project and governed evidence with provenance."""
    root = repository_root.resolve()
    project = project_root.resolve()
    task_paths_key = (
        "allowed_paths" if task.domain in {"python", "c"} else "files_allowed_to_change"
    )
    raw_paths = task.specification.get(task_paths_key, [])
    paths = _as_str_list(raw_paths if isinstance(raw_paths, list) else [], label=task_paths_key)
    expanded_query, expansion_terms = expand_engineering_query(task, query)
    target_project: list[JsonObject] = []
    terms = [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", expanded_query)]
    for item in paths:
        try:
            source = _inside(root, root / _safe_relative(item, label="task allowed path"))
        except EngineeringError:
            continue
        if source.is_file():
            content = source.read_text(encoding="utf-8", errors="replace")
            if not terms or any(term in content.lower() for term in terms):
                target_project.append(
                    {
                        "kind": "target_project",
                        "path": str(source.relative_to(root)),
                        "sha256": _sha256(source),
                        "excerpt": content[:1200],
                    }
                )
    library = (
        ReferenceLibrary(shared_reference_root, task.domain, shared=True)
        if shared_reference_root is not None
        else ReferenceLibrary(project, task.domain)
    )
    project_knowledge = _project_knowledge_cards(root, task.domain, expanded_query)
    governed = library.search_chunks(
        expanded_query,
        limit=governed_chunk_limit,
        token_budget=governed_token_budget,
        max_chunks_per_path=2,
        max_chunks_per_reference=3,
    )
    snapshot_hash = library.snapshot_hash()
    return {
        "query": query,
        "expanded_query": expanded_query,
        "query_expansion_terms": expansion_terms,
        "precedence": [
            "target_project",
            "project_knowledge_cards",
            "private_curated_references",
            "governed_open_source_references",
            "model_prior_knowledge",
        ],
        "target_project": target_project,
        "project_knowledge_cards": project_knowledge,
        "governed_references": governed,
        "governed_reference_library": {
            "root": str(library.root),
            "shared": shared_reference_root is not None,
            "snapshot_hash": snapshot_hash,
            "manifest_count": len(library.manifests_list()),
            "retrieved_chunk_count": len(governed),
            "token_budget": governed_token_budget,
        },
        "model_prior_knowledge_allowed_only_after_evidence": True,
    }


def collect_cuda_evidence(runner: LocalToolRunner) -> JsonObject:
    """Probe CUDA without claiming that a CPU path is an inference result."""
    probe = (
        "import json; import torch; "
        "value={'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,"
        "'available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}; "
        "value.update({'device':torch.cuda.get_device_name(0),'vram_gib':torch.cuda.get_device_properties(0).total_memory/1024**3} if value['available'] else {}); print(json.dumps(value))"
    )
    result = runner.run("cuda_probe", [sys.executable, "-c", probe], timeout_seconds=30)
    try:
        payload: object = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    details = payload if isinstance(payload, dict) else {}
    available = details.get("available") is True
    device = details.get("device")
    vram = details.get("vram_gib")
    a6000 = (
        available
        and isinstance(device, str)
        and "A6000" in device
        and isinstance(vram, (int, float))
        and float(vram) >= 45.0
    )
    return {
        "status": "CUDA_A6000_VERIFIED" if a6000 else "BLOCKED_GPU",
        "cuda": details,
        "probe_log": result.log_path,
        "probe_returncode": result.returncode,
        "reason": None
        if a6000
        else "Real A6000 CUDA inference is unavailable; CPU inference is prohibited for this workflow.",
    }
