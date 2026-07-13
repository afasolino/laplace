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


JsonValue: TypeAlias = object
JsonObject: TypeAlias = dict[str, object]
Domain = Literal["python", "systemverilog"]
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
_DOMAIN_FOLDERS: dict[Domain, tuple[str, ...]] = {
    "python": PYTHON_FOLDERS,
    "systemverilog": SYSTEMVERILOG_FOLDERS,
}
_DOMAIN_LIBRARY: dict[Domain, str] = {"python": "Python", "systemverilog": "SystemVerilog"}
_TASK_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    "request": {"requirements", "blocked"},
    "requirements": {"plan", "blocked"},
    "plan": {"retrieval", "blocked"},
    "retrieval": {"implementation", "blocked"},
    "implementation": {"verification", "blocked"},
    "verification": {"review", "blocked"},
    "review": {"bounded_correction", "final_report", "blocked"},
    "bounded_correction": {"implementation", "final_report", "blocked"},
    "final_report": set(),
    "blocked": set(),
}
_TASK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")


class EngineeringError(RuntimeError):
    """A safe, user-facing engineering-workflow error."""


class SchemaValidationError(EngineeringError):
    """A task did not satisfy the committed task schema."""


class ReferencePolicyError(EngineeringError):
    """A reference would violate its provenance or read-only policy."""


class ToolExecutionError(EngineeringError):
    """An allowlisted local tool could not be executed safely."""


class InferenceBlockedError(EngineeringError):
    """A required CUDA inference capability is unavailable."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
    filename = (
        "python_task_spec.schema.json"
        if domain == "python"
        else "systemverilog_task_spec.schema.json"
    )
    return repository_root / "codex_a6000" / "templates" / filename


def normalize_task_spec(repository_root: Path, domain: Domain, raw: JsonObject) -> JsonObject:
    """Validate the committed schema and add only deterministic workflow metadata."""
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

    def to_json(self) -> JsonObject:
        return {
            "reference_id": self.reference_id,
            "domain": self.domain,
            "repository": self.repository,
            "commit": self.commit,
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


class ReferenceLibrary:
    """Project-scoped, hash-verified and immutable curated references."""

    def __init__(self, project_root: Path, domain: Domain) -> None:
        self.project_root = project_root.resolve()
        self.domain = domain
        self.root = self.project_root / "Data" / "References" / _DOMAIN_LIBRARY[domain]

    @property
    def manifests(self) -> Path:
        return self.root / "90_manifests"

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
            "commit",
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
        return ReferenceManifest(
            reference_id=reference_id,
            domain=self.domain,
            repository=str(values["repository"]),
            commit=str(values["commit"]),
            licence_identifier=str(values["licence_identifier"]),
            licence_text_sha256=str(values["licence_text_sha256"]),
            permitted_use=str(values["permitted_use"]),
            attribution=str(values["attribution"]),
            source_kind=str(values["source_kind"]),
            registered_at=str(values["registered_at"]),
            files=tuple(files),
            read_only=bool(value.get("read_only", False)),
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
        if not repository or not licence_identifier or not permitted_use or not attribution:
            raise ReferencePolicyError(
                "repository, licence, permitted use and attribution are required"
            )
        licence = licence_path.resolve()
        if not licence.is_file():
            raise ReferencePolicyError("Licence file is missing")
        if not selected_files:
            raise ReferencePolicyError("At least one focused reference file is required")
        source_root = self.root / "sources" / reference_id / commit
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
            commit=commit,
            licence_identifier=licence_identifier,
            licence_text_sha256=_sha256(licence),
            permitted_use=permitted_use,
            attribution=attribution,
            source_kind="local_fixture_or_approved_local_snapshot",
            registered_at=_now(),
            files=tuple(records),
        )
        manifest_path = self._manifest_path(reference_id)
        if manifest_path.exists():
            previous = self._read_manifest(reference_id)
            if previous.to_json() != manifest.to_json():
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

    def ingest(self, database: Path) -> JsonObject:
        """Index selected reference text in the existing project SQLite database."""
        conn = _db(database)
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
                        "commit": manifest.commit,
                        "licence_identifier": manifest.licence_identifier,
                        "permitted_use": manifest.permitted_use,
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
            "database": str(database),
            "counts": counts,
            "ingested_at": _now(),
        }
        target = self.root / "indexes" / "ingestion_report.json"
        _write_json_atomic(target, report)
        report["report_path"] = str(target)
        return report

    def verify(self, reference_id: str | None = None) -> JsonObject:
        manifests = [self._read_manifest(reference_id)] if reference_id else self.manifests_list()
        errors: list[str] = []
        verified = 0
        for manifest in manifests:
            if not _COMMIT.fullmatch(manifest.commit):
                errors.append(f"{manifest.reference_id}: invalid commit")
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
            "iverilog",
            "vvp",
            "yosys",
            "cuda_probe",
        }:
            raise ToolExecutionError(f"Tool is not allowlisted: {tool}")
        if timeout_seconds < 1 or timeout_seconds > 1800:
            raise ToolExecutionError("Tool timeout must be between 1 and 1800 seconds")
        started = time.monotonic()
        try:
            # The command is built only by this module's allowlisted methods.
            process = subprocess.Popen(  # nosec B603
                command,
                cwd=self.repository_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
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
        self, paths: list[str] | None = None, *, timeout_seconds: int = 300
    ) -> JsonObject:
        targets = self._target_paths(paths or [])
        python_targets = [target for target in targets if Path(target).suffix == ".py"]
        bandit_targets = python_targets or ["src"]
        commands: list[tuple[str, list[str]]] = [
            ("ruff_format", [sys.executable, "-m", "ruff", "format", "--check", *targets]),
            ("ruff", [sys.executable, "-m", "ruff", "check", *targets]),
            ("mypy", [sys.executable, "-m", "mypy", "src"]),
            ("pytest", [sys.executable, "-m", "pytest"]),
            ("coverage", [sys.executable, "-m", "coverage", "run", "-m", "pytest"]),
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
        timeout_seconds: int = 300,
    ) -> JsonObject:
        if not source_files:
            raise ToolExecutionError("At least one SystemVerilog source is required")
        sources = self._target_paths(source_files)
        if not all(Path(value).suffix.lower() in {".sv", ".v"} for value in sources):
            raise ToolExecutionError("EDA sources must be .sv or .v files")
        simulation_sources = list(sources)
        if testbench is not None:
            testbench_path = self._target_paths([testbench])[0]
            if Path(testbench_path).suffix.lower() not in {".sv", ".v"}:
                raise ToolExecutionError("EDA testbench must be a .sv or .v file")
            simulation_sources.append(testbench_path)
        if top_module is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", top_module):
            raise ToolExecutionError("Top module name is unsafe")
        results: list[JsonObject] = []
        if shutil.which("verilator"):
            results.append(
                self.run(
                    "verilator",
                    ["verilator", "--lint-only", "--Wall", "--sv", *sources],
                    timeout_seconds=timeout_seconds,
                ).to_json()
            )
        output = self.log_root / f"eda_{uuid.uuid4().hex}.vvp"
        if shutil.which("iverilog"):
            compile_command = ["iverilog", "-g2012", "-o", str(output)]
            if top_module:
                compile_command.extend(["-s", top_module])
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
                f"read_verilog -sv {' '.join(sources)}\nhierarchy -top {top_module}\nsynth -top {top_module}\n",
                encoding="utf-8",
            )
            results.append(
                self.run(
                    "yosys", ["yosys", "-s", str(script)], timeout_seconds=timeout_seconds
                ).to_json()
            )
        return {
            "operation": "run_eda_flow",
            "sources": sources,
            "top_module": top_module,
            "results": results,
            "passed": bool(results) and all(item["status"] == "PASS" for item in results),
            "created_at": _now(),
        }


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
            domain not in {"python", "systemverilog"}
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
            "implementer": {"implementation_report", "patch_manifest"},
            "verifier": {"verification_report"},
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


def retrieve_engineering_evidence(
    repository_root: Path,
    project_root: Path,
    task: AgentTask,
    *,
    query: str,
) -> JsonObject:
    """Return precedence-ordered evidence without allowing source copying."""
    root = repository_root.resolve()
    project = project_root.resolve()
    task_paths_key = "allowed_paths" if task.domain == "python" else "files_allowed_to_change"
    raw_paths = task.specification.get(task_paths_key, [])
    paths = _as_str_list(raw_paths if isinstance(raw_paths, list) else [], label=task_paths_key)
    target_project: list[JsonObject] = []
    terms = [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query)]
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
    library = ReferenceLibrary(project, task.domain)
    governed: list[JsonObject] = []
    for manifest in library.manifests_list():
        for record in manifest.files:
            governed.append(
                {
                    "kind": "governed_reference",
                    "reference_id": manifest.reference_id,
                    "commit": manifest.commit,
                    "licence_identifier": manifest.licence_identifier,
                    "permitted_use": manifest.permitted_use,
                    "path": record.selected_path,
                    "sha256": record.sha256,
                }
            )
    return {
        "query": query,
        "precedence": [
            "target_project",
            "private_curated_references",
            "governed_open_source_references",
            "model_prior_knowledge",
        ],
        "target_project": target_project,
        "governed_references": governed,
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
