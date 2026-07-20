"""Strict structured model-output protocols for bounded local repairs and review."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .engineering import Domain, EngineeringError, JsonObject, _inside, _safe_relative


class StructuredOutputError(EngineeringError):
    """A model response did not satisfy the committed machine-readable protocol."""


ReplacementKind = Literal["source", "testbench"]
ReviewerDecision = Literal["approve", "request_changes", "block"]


@dataclass(frozen=True)
class FileReplacement:
    path: str
    language: Domain
    kind: ReplacementKind
    expected_sha256: str
    content: str


@dataclass(frozen=True)
class ReplacementPlan:
    schema_version: int
    replacements: tuple[FileReplacement, ...]


@dataclass(frozen=True)
class ReviewerVerdict:
    schema_version: int
    verdict: ReviewerDecision
    reason: str
    missing_evidence: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "reason": self.reason,
            "missing_evidence": list(self.missing_evidence),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _expected_kind(path: str) -> ReplacementKind:
    filename = Path(path).name.lower()
    if filename.startswith("tb_") or filename.startswith("test_"):
        return "testbench"
    return "source"


def source_state(root: Path, allowed_paths: list[str], domain: Domain) -> list[JsonObject]:
    """Return complete current source state with hashes for model-bound replacements."""
    records: list[JsonObject] = []
    for raw_path in allowed_paths:
        relative = _safe_relative(raw_path, label="allowed source path")
        path = _inside(root, root / relative)
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="strict")
        records.append(
            {
                "path": relative.as_posix(),
                "language": domain,
                "kind": _expected_kind(relative.as_posix()),
                "sha256": file_sha256(path),
                "content": content,
            }
        )
    return records


def replacement_plan_json_schema(*, allowed_paths: list[str], domain: Domain) -> JsonObject:
    """Return the strict request-time schema; worktree checks remain deterministic."""
    normalized_paths = sorted(
        {_safe_relative(path, label="allowed path").as_posix() for path in allowed_paths}
    )
    if not normalized_paths:
        raise StructuredOutputError("Replacement schema requires at least one allowed path")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "replacements"],
        "properties": {
            "schema_version": {"const": 1},
            "replacements": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(normalized_paths),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "path",
                        "language",
                        "kind",
                        "expected_sha256",
                        "content",
                    ],
                    "properties": {
                        "path": {"enum": normalized_paths},
                        "language": {"const": domain},
                        "kind": {"enum": ["source", "testbench"]},
                        "expected_sha256": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "content": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }


def _single_json_object(model_text: str, *, label: str) -> JsonObject:
    text = model_text.strip()
    if not text:
        raise StructuredOutputError(f"{label} response is empty")
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        if len(fences) != 1:
            raise StructuredOutputError(f"{label} response contains multiple fenced blocks")
        fenced_form = re.fullmatch(
            r"\s*```(?:json)?\s*.*?```\s*", text, flags=re.DOTALL | re.IGNORECASE
        )
        if fenced_form is None:
            raise StructuredOutputError(f"{label} response mixes JSON with prose or other content")
        text = fences[0].strip()
    elif "```" in text:
        raise StructuredOutputError(f"{label} response contains an incomplete or unsupported fence")
    try:
        value: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"{label} response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise StructuredOutputError(f"{label} response must be one JSON object")
    return value


def _require_exact_keys(value: JsonObject, expected: set[str], *, label: str) -> None:
    keys = set(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise StructuredOutputError(
            f"{label} keys are invalid; missing={missing}, unexpected={extra}"
        )


def parse_replacement_plan(
    model_text: str,
    *,
    root: Path,
    allowed_paths: list[str],
    domain: Domain,
    max_total_bytes: int = 1_000_000,
) -> ReplacementPlan:
    """Validate one exact structured replacement plan against current worktree hashes."""
    if "diff --git" in model_text or "--- a/" in model_text or "+++ b/" in model_text:
        raise StructuredOutputError("Raw model-generated patches are forbidden")
    value = _single_json_object(model_text, label="Replacement")
    _require_exact_keys(value, {"schema_version", "replacements"}, label="Replacement plan")
    if value.get("schema_version") != 1:
        raise StructuredOutputError("Replacement schema_version must equal 1")
    raw_replacements = value.get("replacements")
    if not isinstance(raw_replacements, list) or not raw_replacements:
        raise StructuredOutputError("Replacement plan must contain at least one replacement")
    allowed = {
        _safe_relative(path, label="allowed path").as_posix(): path for path in allowed_paths
    }
    seen: set[str] = set()
    total_bytes = 0
    replacements: list[FileReplacement] = []
    for index, raw in enumerate(raw_replacements):
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise StructuredOutputError(f"Replacement {index} must be a JSON object")
        item: JsonObject = dict(raw)
        _require_exact_keys(
            item,
            {"path", "language", "kind", "expected_sha256", "content"},
            label=f"Replacement {index}",
        )
        path_value = item.get("path")
        language_value = item.get("language")
        kind_value = item.get("kind")
        expected_hash = item.get("expected_sha256")
        content = item.get("content")
        if not isinstance(path_value, str):
            raise StructuredOutputError(f"Replacement {index} path must be a string")
        path = _safe_relative(path_value, label="replacement path").as_posix()
        if path not in allowed:
            raise StructuredOutputError(f"Replacement path is outside task scope: {path}")
        if path in seen:
            raise StructuredOutputError(f"Replacement path is duplicated: {path}")
        seen.add(path)
        if language_value != domain:
            raise StructuredOutputError(
                f"Replacement {path} language {language_value!r} does not match {domain!r}"
            )
        expected_kind = _expected_kind(path)
        if kind_value != expected_kind:
            raise StructuredOutputError(
                f"Replacement {path} kind must be {expected_kind!r}, not {kind_value!r}"
            )
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise StructuredOutputError(f"Replacement {path} has an invalid expected_sha256")
        if not isinstance(content, str) or not content.strip():
            raise StructuredOutputError(f"Replacement {path} content must be non-empty text")
        if "\x00" in content:
            raise StructuredOutputError(f"Replacement {path} contains a NUL byte")
        current = _inside(root, root / Path(path))
        if not current.is_file():
            raise StructuredOutputError(f"Replacement target does not exist: {path}")
        current_hash = file_sha256(current)
        if current_hash != expected_hash:
            raise StructuredOutputError(
                f"Replacement {path} is stale: expected {expected_hash}, current {current_hash}"
            )
        normalized_content = content.rstrip() + "\n"
        current_content = current.read_text(encoding="utf-8", errors="strict")
        if normalized_content == current_content:
            raise StructuredOutputError(f"Replacement {path} makes no source change")
        total_bytes += len(normalized_content.encode("utf-8"))
        if total_bytes > max_total_bytes:
            raise StructuredOutputError("Replacement plan exceeds the one MiB safety limit")
        replacements.append(
            FileReplacement(
                path=path,
                language=domain,
                kind=expected_kind,
                expected_sha256=expected_hash,
                content=normalized_content,
            )
        )
    return ReplacementPlan(schema_version=1, replacements=tuple(replacements))


def build_local_patch(plan: ReplacementPlan, *, root: Path) -> str:
    """Generate the only accepted unified diff locally from current worktree content."""
    chunks: list[str] = []
    for replacement in plan.replacements:
        relative = Path(replacement.path)
        source = _inside(root, root / relative)
        current = source.read_text(encoding="utf-8", errors="strict")
        diff = "".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                replacement.content.splitlines(keepends=True),
                fromfile=f"a/{replacement.path}",
                tofile=f"b/{replacement.path}",
            )
        )
        if not diff:
            raise StructuredOutputError(f"Replacement {replacement.path} makes no source change")
        chunks.append(f"diff --git a/{replacement.path} b/{replacement.path}\n{diff}")
    patch = "".join(chunks)
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def parse_reviewer_verdict(model_text: str) -> ReviewerVerdict:
    """Parse an operational reviewer verdict; invalid output is never approval."""
    value = _single_json_object(model_text, label="Reviewer")
    _require_exact_keys(
        value,
        {"schema_version", "verdict", "reason", "missing_evidence"},
        label="Reviewer verdict",
    )
    if value.get("schema_version") != 1:
        raise StructuredOutputError("Reviewer schema_version must equal 1")
    verdict = value.get("verdict")
    if verdict not in {"approve", "request_changes", "block"}:
        raise StructuredOutputError("Reviewer verdict is not an allowed value")
    reason = value.get("reason")
    missing = value.get("missing_evidence")
    if not isinstance(reason, str) or not reason.strip():
        raise StructuredOutputError("Reviewer reason must be non-empty text")
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise StructuredOutputError("Reviewer missing_evidence must be a list of strings")
    return ReviewerVerdict(
        schema_version=1,
        verdict=verdict,
        reason=reason.strip(),
        missing_evidence=tuple(item.strip() for item in missing if item.strip()),
    )
