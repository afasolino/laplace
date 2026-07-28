"""Provider-independent deterministic offline evaluation for v7 contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GPU_BLOCKED_STATUS = "BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE"
REQUIRED_CATEGORIES = (
    "retrieval_relevance",
    "citation_precision_recall",
    "unsupported_claim_detection",
    "personal_corpus_isolation",
    "shared_corpus_permissions",
    "conversation_persistence",
    "markdown_citation_rendering",
    "python_patch_applicability",
    "systemverilog_patch_applicability",
    "verification_gates",
    "worktree_isolation",
    "provider_routing",
    "cancellation_timeouts",
    "artifact_provenance",
)


class EvaluationError(RuntimeError):
    """Raised for an invalid or incomplete frozen evaluation suite."""


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    category: str
    inputs: dict[str, object]
    expected: dict[str, object]

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        if value not in REQUIRED_CATEGORIES:
            raise ValueError(f"unknown evaluation category: {value}")
        return value


class FrozenSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    suite_id: str
    fixture_only: Literal[True]
    cases: tuple[EvaluationCase, ...]


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: str
    status: Literal["PASS", "FAIL"]
    score: float = Field(ge=0, le=1)
    details: dict[str, object]


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    suite_id: str
    fixture_only: Literal[True] = True
    status: Literal["PASS", "FAIL"]
    infrastructure_correctness: dict[str, object]
    fixture_task_quality: dict[str, object]
    live_model_quality: dict[str, str]
    results: tuple[CaseResult, ...]


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"{name} must be a string list")
    return tuple(value)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationError(f"{name} must be an object")
    return dict(value)


def _retrieval(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    ranked = _strings(case.inputs.get("ranked_chunks"), "ranked_chunks")
    relevant = set(_strings(case.expected.get("relevant_chunks"), "relevant_chunks"))
    raw_top_k = case.expected.get("top_k", 1)
    if not isinstance(raw_top_k, int) or isinstance(raw_top_k, bool) or raw_top_k < 1:
        raise EvaluationError("top_k must be a positive integer")
    top_k = raw_top_k
    hits = len(set(ranked[:top_k]) & relevant)
    score = hits / max(1, len(relevant))
    return score, {"hits": hits, "relevant": len(relevant), "top_k": top_k}


def _citations(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    predicted = set(_strings(case.inputs.get("citations"), "citations"))
    required = set(_strings(case.expected.get("citations"), "citations"))
    true_positive = len(predicted & required)
    precision = true_positive / len(predicted) if predicted else float(not required)
    recall = true_positive / len(required) if required else 1.0
    return min(precision, recall), {"precision": precision, "recall": recall}


def _unsupported(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    predicted = _strings(case.inputs.get("unsupported_claim_ids"), "unsupported_claim_ids")
    expected = _strings(case.expected.get("unsupported_claim_ids"), "unsupported_claim_ids")
    score = float(predicted == expected)
    return score, {"predicted": list(predicted), "expected": list(expected)}


def _permission(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    owner = str(case.inputs.get("owner_id"))
    requester = str(case.inputs.get("requester_id"))
    scope = str(case.inputs.get("scope"))
    grants = _strings(case.inputs.get("grants", []), "grants")
    allowed = requester == owner if scope == "personal" else requester in grants
    expected = bool(case.expected.get("allowed"))
    return float(allowed == expected), {"allowed": allowed, "expected": expected}


def _conversation(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    serialized = json.dumps(case.inputs.get("messages"), sort_keys=True, separators=(",", ":"))
    reloaded = json.loads(serialized)
    expected = case.expected.get("messages")
    return float(reloaded == expected), {"message_count": len(reloaded)}


def _render(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    markdown = str(case.inputs.get("markdown", ""))
    citations = _strings(case.inputs.get("citations"), "citations")
    safe = "<script" not in markdown.lower() and "javascript:" not in markdown.lower()
    complete = all(
        all(label in citation for label in ("file=", "page=", "chunk="))
        for citation in citations
    )
    expected_safe = bool(case.expected.get("safe"))
    return float(safe == expected_safe and complete), {
        "safe": safe,
        "citation_contract_complete": complete,
    }


def _safe_patch_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.removeprefix("a/").removeprefix("b/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise EvaluationError("patch path is unsafe")
    return path


def _patch_applicable(case: EvaluationCase, fixture_root: Path) -> tuple[float, dict[str, object]]:
    repository = str(case.inputs.get("repository"))
    relative_file = _safe_patch_path(str(case.inputs.get("file")))
    old_text = str(case.inputs.get("old_text"))
    new_text = str(case.inputs.get("new_text"))
    target = (fixture_root / repository / relative_file).resolve()
    repository_root = (fixture_root / repository).resolve()
    safe = target.is_relative_to(repository_root) and target.is_file()
    content = target.read_text(encoding="utf-8") if safe else ""
    applicable = safe and old_text in content and old_text != new_text
    expected = bool(case.expected.get("applicable"))
    return float(applicable == expected), {
        "applicable": applicable,
        "expected": expected,
        "safe_path": safe,
    }


def _verification(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    required = set(_strings(case.inputs.get("required_gates"), "required_gates"))
    passed = set(_strings(case.inputs.get("passed_gates"), "passed_gates"))
    accepted = required <= passed
    expected = bool(case.expected.get("accepted"))
    return float(accepted == expected), {
        "accepted": accepted,
        "missing": sorted(required - passed),
    }


def _worktree(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    owner_namespace = str(case.inputs.get("owner_namespace"))
    logical_path = PurePosixPath(str(case.inputs.get("logical_path")))
    isolated = (
        not logical_path.is_absolute()
        and ".." not in logical_path.parts
        and logical_path.parts[:1] == (owner_namespace,)
    )
    expected = bool(case.expected.get("isolated"))
    return float(isolated == expected), {"isolated": isolated, "expected": expected}


def _routing(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    providers = case.inputs.get("providers")
    if not isinstance(providers, list):
        raise EvaluationError("providers must be a list")
    requires_tools = bool(case.inputs.get("requires_tools"))
    candidates = [
        str(provider["provider_id"])
        for provider in providers
        if isinstance(provider, dict)
        and (not requires_tools or provider.get("tools") is True)
        and provider.get("available") is True
    ]
    selected = candidates[0] if candidates else None
    expected = case.expected.get("provider_id")
    return float(selected == expected), {"selected_provider_id": selected}


def _cancellation(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    requested = str(case.inputs.get("terminal_event"))
    accepted = requested in {"CANCELLED", "TIMED_OUT"}
    expected = str(case.expected.get("terminal_state"))
    return float(accepted and requested == expected), {"terminal_state": requested}


def _provenance(case: EvaluationCase, _: Path) -> tuple[float, dict[str, object]]:
    record = _mapping(case.inputs.get("record"), "record")
    required = _strings(case.expected.get("required_fields"), "required_fields")
    missing = [field for field in required if record.get(field) in {None, ""}]
    return float(not missing), {"missing_fields": missing}


Evaluator = Callable[[EvaluationCase, Path], tuple[float, dict[str, object]]]
EVALUATORS: Mapping[str, Evaluator] = {
    "retrieval_relevance": _retrieval,
    "citation_precision_recall": _citations,
    "unsupported_claim_detection": _unsupported,
    "personal_corpus_isolation": _permission,
    "shared_corpus_permissions": _permission,
    "conversation_persistence": _conversation,
    "markdown_citation_rendering": _render,
    "python_patch_applicability": _patch_applicable,
    "systemverilog_patch_applicability": _patch_applicable,
    "verification_gates": _verification,
    "worktree_isolation": _worktree,
    "provider_routing": _routing,
    "cancellation_timeouts": _cancellation,
    "artifact_provenance": _provenance,
}


def load_suite(path: Path) -> FrozenSuite:
    suite = FrozenSuite.model_validate_json(path.read_text(encoding="utf-8"))
    categories = {case.category for case in suite.cases}
    missing = set(REQUIRED_CATEGORIES) - categories
    if missing:
        raise EvaluationError(f"evaluation categories missing: {sorted(missing)}")
    if len({case.case_id for case in suite.cases}) != len(suite.cases):
        raise EvaluationError("evaluation case IDs must be unique")
    return suite


def run_offline_evaluation(suite_path: Path, fixture_root: Path) -> EvaluationReport:
    suite = load_suite(suite_path)
    results: list[CaseResult] = []
    for case in suite.cases:
        score, details = EVALUATORS[case.category](case, fixture_root)
        results.append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                status="PASS" if score == 1.0 else "FAIL",
                score=score,
                details=details,
            )
        )
    passed = sum(result.status == "PASS" for result in results)
    categories = sorted({result.category for result in results})
    return EvaluationReport(
        suite_id=suite.suite_id,
        status="PASS" if passed == len(results) else "FAIL",
        infrastructure_correctness={
            "status": "PASS",
            "schema_valid": True,
            "fixture_only": True,
            "category_coverage": categories,
            "external_network_used": False,
            "provider_contacted": False,
        },
        fixture_task_quality={
            "status": "PASS" if passed == len(results) else "FAIL",
            "passed_cases": passed,
            "total_cases": len(results),
            "pass_rate": passed / len(results) if results else 0.0,
        },
        live_model_quality={
            "status": GPU_BLOCKED_STATUS,
            "reason": "GPU unavailable by user constraint; fixture scores are not live-model scores.",
        },
        results=tuple(results),
    )
