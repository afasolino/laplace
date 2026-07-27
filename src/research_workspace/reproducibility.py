"""Frozen skills, deterministic context packets, and reproducibility locks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

import yaml

from .execution_records import canonical_sha256

JsonObject: TypeAlias = dict[str, object]
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SAFE_ROLE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_HELD_OUT_PATH = re.compile(
    r"(?:^|[/_.-])(held[_-]?out|evaluator|tb_heldout|test_heldout)(?:$|[/_.-])",
    re.IGNORECASE,
)
_SECRET_PATH = re.compile(r"(?:^|/)(?:\.env(?:\..*)?|secrets?\.(?:json|ya?ml)|id_rsa)$")
_SECRET_CONTENT = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


class ReproducibilityError(RuntimeError):
    """Frozen run inputs are malformed, unsafe, or inconsistent."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normalized_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8", errors="strict")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


@dataclass(frozen=True)
class SkillRecord:
    name: str
    version: str
    roles: tuple[str, ...]
    files: tuple[JsonObject, ...]
    content_sha256: str

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "version": self.version,
            "roles": list(self.roles),
            "files": list(self.files),
            "content_sha256": self.content_sha256,
        }


class FrozenSkillRegistry:
    """Load strict Agent Skills folders and emit a deterministic lock."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @staticmethod
    def _frontmatter(path: Path) -> JsonObject:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ReproducibilityError(f"Skill frontmatter is missing: {path}")
        closing = text.find("\n---\n", 4)
        if closing < 0:
            raise ReproducibilityError(f"Skill frontmatter is unterminated: {path}")
        raw: object = yaml.safe_load(text[4:closing])
        if not isinstance(raw, dict) or set(raw) != {"name", "description"}:
            raise ReproducibilityError(
                f"Skill frontmatter must contain only name and description: {path}"
            )
        if not isinstance(raw.get("name"), str) or not isinstance(
            raw.get("description"), str
        ):
            raise ReproducibilityError(f"Skill frontmatter fields are invalid: {path}")
        return dict(raw)

    def records(self) -> tuple[SkillRecord, ...]:
        if not self.root.is_dir():
            raise ReproducibilityError(f"Skill root does not exist: {self.root}")
        records: list[SkillRecord] = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            name = directory.name
            if not _SAFE_NAME.fullmatch(name):
                raise ReproducibilityError(f"Unsafe skill directory: {name}")
            skill_path = directory / "SKILL.md"
            metadata_path = directory / "skill.json"
            if not skill_path.is_file() or not metadata_path.is_file():
                raise ReproducibilityError(f"Skill is incomplete: {name}")
            frontmatter = self._frontmatter(skill_path)
            try:
                metadata_raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ReproducibilityError(f"Invalid skill.json: {name}") from exc
            if not isinstance(metadata_raw, dict) or set(metadata_raw) != {
                "name",
                "roles",
                "schema_version",
                "version",
            }:
                raise ReproducibilityError(f"skill.json keys are invalid: {name}")
            if metadata_raw.get("name") != name or frontmatter.get("name") != name:
                raise ReproducibilityError(f"Skill names disagree: {name}")
            if metadata_raw.get("schema_version") != 1:
                raise ReproducibilityError(f"Unsupported skill schema: {name}")
            version = metadata_raw.get("version")
            roles = metadata_raw.get("roles")
            if not isinstance(version, str) or not re.fullmatch(
                r"\d+\.\d+\.\d+", version
            ):
                raise ReproducibilityError(f"Skill version is invalid: {name}")
            if not isinstance(roles, list) or not roles or not all(
                isinstance(item, str) and _SAFE_ROLE.fullmatch(item) for item in roles
            ):
                raise ReproducibilityError(f"Skill roles are invalid: {name}")
            files: list[JsonObject] = []
            aggregate = hashlib.sha256()
            for path in (skill_path, metadata_path):
                data = _normalized_bytes(path)
                relative = path.relative_to(self.root).as_posix()
                digest = hashlib.sha256(data).hexdigest()
                files.append(
                    {"path": relative, "sha256": digest, "bytes": len(data)}
                )
                aggregate.update(relative.encode("utf-8") + b"\0" + data)
            records.append(
                SkillRecord(
                    name=name,
                    version=version,
                    roles=tuple(str(item) for item in roles),
                    files=tuple(files),
                    content_sha256=aggregate.hexdigest(),
                )
            )
        if not records:
            raise ReproducibilityError("Skill root is empty")
        return tuple(records)

    def write_lock(self, path: Path) -> JsonObject:
        payload: JsonObject = {
            "schema_version": 1,
            "skills": [record.to_json() for record in self.records()],
        }
        payload["skills_lock_sha256"] = canonical_sha256(payload)
        _write_json(path, payload)
        return payload

    def skills_for_role(self, role: str) -> tuple[SkillRecord, ...]:
        if not _SAFE_ROLE.fullmatch(role):
            raise ReproducibilityError("Unsafe role")
        return tuple(record for record in self.records() if role in record.roles)


@dataclass(frozen=True)
class ContextPacket:
    role: str
    attempt: int
    context_path: str
    manifest_path: str
    sha256_path: str
    context_sha256: str


class ContextPacketBuilder:
    """Build role-specific context with deterministic ordering and exclusions."""

    def __init__(
        self,
        repository_root: Path,
        *,
        max_bytes: int = 256_000,
    ) -> None:
        self.repository_root = repository_root.resolve()
        if max_bytes < 4096 or max_bytes > 2_000_000:
            raise ReproducibilityError("Context packet limit is out of range")
        self.max_bytes = max_bytes

    def _source_record(self, path: Path) -> tuple[JsonObject | None, JsonObject | None]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.repository_root).as_posix()
        except ValueError as exc:
            raise ReproducibilityError(
                f"Context source is outside the declared repository: {path}"
            ) from exc
        if _HELD_OUT_PATH.search(relative):
            return None, {
                "path_sha256": canonical_sha256(relative),
                "reason": "held_out_path_excluded",
            }
        if _SECRET_PATH.search(relative):
            return None, {
                "path_sha256": canonical_sha256(relative),
                "reason": "secret_path_excluded",
            }
        data = _normalized_bytes(resolved)
        text = data.decode("utf-8")
        if _SECRET_CONTENT.search(text):
            return None, {
                "path_sha256": canonical_sha256(relative),
                "reason": "secret_content_excluded",
            }
        return (
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "content": text,
            },
            None,
        )

    def build(
        self,
        run_root: Path,
        *,
        role: str,
        attempt: int,
        sections: Mapping[str, str],
        source_paths: Sequence[Path],
        skills_lock_sha256: str,
        corpus_snapshot_sha256: str,
    ) -> ContextPacket:
        if not _SAFE_ROLE.fullmatch(role) or attempt < 0:
            raise ReproducibilityError("Context role or attempt is unsafe")
        for value in (skills_lock_sha256, corpus_snapshot_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ReproducibilityError("Context lock inputs must be SHA-256")
        normalized_sections: list[JsonObject] = []
        for name, content in sorted(sections.items()):
            if not _SAFE_ROLE.fullmatch(name):
                raise ReproducibilityError(f"Unsafe context section: {name}")
            normalized = content.replace("\r\n", "\n").replace("\r", "\n")
            if _SECRET_CONTENT.search(normalized):
                raise ReproducibilityError(
                    f"Context section contains forbidden material: {name}"
                )
            normalized_sections.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "content": normalized,
                }
            )
        sources: list[JsonObject] = []
        excluded: list[JsonObject] = []
        for source_path in sorted(source_paths, key=lambda item: item.as_posix()):
            record, exclusion = self._source_record(source_path)
            if record is not None:
                sources.append(record)
            if exclusion is not None:
                excluded.append(exclusion)
        lines = [
            "# Laplace frozen role context",
            "",
            f"- role: `{role}`",
            f"- attempt: `{attempt}`",
            f"- skills_lock_sha256: `{skills_lock_sha256}`",
            f"- corpus_snapshot_sha256: `{corpus_snapshot_sha256}`",
        ]
        for section in normalized_sections:
            lines.extend(
                [
                    "",
                    f"## {section['name']}",
                    "",
                    str(section["content"]).rstrip(),
                ]
            )
        for source in sources:
            lines.extend(
                [
                    "",
                    f"## Source `{source['path']}`",
                    "",
                    f"SHA-256: `{source['sha256']}`",
                    "",
                    "```text",
                    str(source["content"]).rstrip(),
                    "```",
                ]
            )
        context = "\n".join(lines).rstrip() + "\n"
        context_bytes = context.encode("utf-8")
        if len(context_bytes) > self.max_bytes:
            raise ReproducibilityError(
                f"Context packet exceeds {self.max_bytes} bytes"
            )
        context_sha256 = hashlib.sha256(context_bytes).hexdigest()
        target = run_root.resolve() / "context" / role / str(attempt)
        target.mkdir(parents=True, exist_ok=True)
        context_path = target / "context.md"
        manifest_path = target / "context_manifest.json"
        sha_path = target / "context.sha256"
        context_path.write_bytes(context_bytes)
        manifest: JsonObject = {
            "schema_version": 1,
            "role": role,
            "attempt": attempt,
            "skills_lock_sha256": skills_lock_sha256,
            "corpus_snapshot_sha256": corpus_snapshot_sha256,
            "max_bytes": self.max_bytes,
            "context_bytes": len(context_bytes),
            "context_sha256": context_sha256,
            "sections": [
                {"name": item["name"], "sha256": item["sha256"]}
                for item in normalized_sections
            ],
            "sources": [
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                }
                for item in sources
            ],
            "excluded": excluded,
        }
        _write_json(manifest_path, manifest)
        sha_path.write_text(context_sha256 + "\n", encoding="ascii", newline="\n")
        return ContextPacket(
            role,
            attempt,
            str(context_path),
            str(manifest_path),
            str(sha_path),
            context_sha256,
        )


def held_out_evaluator_identity(path: Path) -> JsonObject:
    """Hash evaluator identity metadata without reading held-out tests."""
    resolved = path.resolve()
    manifest = resolved / "manifest.json"
    manifest_sha256 = (
        hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
    )
    identity: JsonObject = {
        "directory_name": resolved.name,
        "manifest_filename": "manifest.json" if manifest.is_file() else None,
        "manifest_sha256": manifest_sha256,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def local_tool_versions(names: Sequence[str]) -> JsonObject:
    """Capture exact local versions without executing any document-derived command."""
    versions: JsonObject = {}
    commands = {
        "python": [os.fspath(Path(sys.executable)), "--version"],
        "verilator": ["verilator", "--version"],
        "iverilog": ["iverilog", "-V"],
        "vvp": ["vvp", "-V"],
        "yosys": ["yosys", "-V"],
    }
    for name in sorted(set(names)):
        command = commands.get(name)
        executable = shutil.which(command[0]) if command is not None else None
        if command is None or executable is None:
            versions[name] = {"available": False, "version": None}
            continue
        command[0] = executable
        completed = subprocess.run(  # nosec B603
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        output = (completed.stdout or completed.stderr).strip()[:2000]
        versions[name] = {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "executable": executable,
            "version": output,
        }
    return versions


def write_reproducibility_locks(
    run_root: Path,
    *,
    skills_lock: JsonObject,
    context_manifests: Sequence[Path],
    corpus_identity: JsonObject,
    models_identity: JsonObject,
    tools_identity: JsonObject,
    base_revision: str,
    experiment_configuration: object,
    request: object,
    held_out_identity: JsonObject,
) -> JsonObject:
    """Write subordinate locks and the hash-bound run lock."""
    run_root = run_root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", base_revision):
        raise ReproducibilityError("Base revision must be an exact Git SHA")
    skill_path = run_root / "skills.lock.json"
    if not skill_path.is_file():
        _write_json(skill_path, skills_lock)
    elif json.loads(skill_path.read_text(encoding="utf-8")) != skills_lock:
        raise ReproducibilityError("Existing skills lock is incompatible")
    context_records: list[JsonObject] = []
    for path in sorted(context_manifests, key=lambda item: item.as_posix()):
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ReproducibilityError(f"Context manifest is malformed: {path}")
        context_records.append(
            {
                "role": raw.get("role"),
                "attempt": raw.get("attempt"),
                "context_sha256": raw.get("context_sha256"),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    subordinate: dict[str, JsonObject] = {
        "context.lock.json": {
            "schema_version": 1,
            "contexts": context_records,
        },
        "corpus.lock.json": {
            "schema_version": 1,
            "store": "governed_corpus",
            "identity": corpus_identity,
        },
        "models.lock.json": {
            "schema_version": 1,
            "models": models_identity,
        },
        "tools.lock.json": {
            "schema_version": 1,
            "tools": tools_identity,
        },
    }
    subordinate_hashes: JsonObject = {
        "skills.lock.json": hashlib.sha256(skill_path.read_bytes()).hexdigest()
    }
    for filename, payload in subordinate.items():
        payload["lock_sha256"] = canonical_sha256(payload)
        path = run_root / filename
        _write_json(path, payload)
        subordinate_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    run_lock: JsonObject = {
        "schema_version": 1,
        "base_revision": base_revision,
        "experiment_configuration_sha256": canonical_sha256(
            experiment_configuration
        ),
        "request_sha256": canonical_sha256(request),
        "held_out_evaluator_identity_sha256": held_out_identity.get(
            "identity_sha256"
        ),
        "subordinate_locks": subordinate_hashes,
    }
    run_lock_sha256 = canonical_sha256(run_lock)
    run_lock["run_lock_sha256"] = run_lock_sha256
    _write_json(run_root / "run.lock.json", run_lock)
    return run_lock
