"""Deterministic role routing and auditable local dual-model calls.

The routing decision is derived entirely from persisted task metadata.  Model
output never selects its own role, endpoint, tools, or editable scope.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

from .engineering import EngineeringError, JsonObject, _write_json_atomic
from .inference import ServingCandidate, backend_for
from .llm import (
    GenerationResult,
    LocalGenerationBackend,
    ModelInvocationError,
    ModelRequired,
)


ModelRole = Literal[
    "planning_supervision",
    "retrieval_interpretation",
    "general_implementation",
    "rtl_contract_generation",
    "bounded_rtl_implementation",
    "bounded_rtl_repair",
    "review",
]
TaskKind = Literal["implementation", "repair", "integration"]
RtlScope = Literal[
    "bounded_module",
    "multi_file_subsystem",
    "protocol_integration",
    "software_rtl_codesign",
    "cdc_architecture",
    "uvm",
    "unresolved_architecture",
    "not_rtl",
]


@dataclass(frozen=True)
class RoutingTaskMetadata:
    task_id: str
    experiment_arm: str
    domain: Literal["python", "c", "verilog", "systemverilog"]
    task_kind: TaskKind
    rtl_scope: RtlScope
    worker_eligible: bool
    editable_sources: tuple[str, ...]
    module_count: int
    synthesizable: bool
    explicit_ports: bool
    cycle_behavior_specified: bool
    deterministic_verification: bool
    unresolved_architecture: bool = False

    @classmethod
    def single_model(
        cls,
        *,
        task_id: str,
        experiment_arm: str,
        domain: Literal["python", "c", "verilog", "systemverilog"],
    ) -> RoutingTaskMetadata:
        return cls(
            task_id=task_id,
            experiment_arm=experiment_arm,
            domain=domain,
            task_kind="implementation",
            rtl_scope="not_rtl" if domain in {"python", "c"} else "unresolved_architecture",
            worker_eligible=False,
            editable_sources=(),
            module_count=0,
            synthesizable=False,
            explicit_ports=False,
            cycle_behavior_specified=False,
            deterministic_verification=False,
        )


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reason: str
    failed_conditions: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "failed_conditions": list(self.failed_conditions),
        }


def assess_rtl_worker_eligibility(metadata: RoutingTaskMetadata) -> EligibilityDecision:
    """Apply the committed, non-model worker policy to one task."""
    failures: list[str] = []
    if metadata.domain not in {"verilog", "systemverilog"}:
        failures.append("domain_is_not_rtl")
    if metadata.rtl_scope != "bounded_module":
        failures.append(f"rtl_scope_is_{metadata.rtl_scope}")
    if metadata.module_count != 1:
        failures.append("module_count_is_not_one")
    if len(metadata.editable_sources) != 1:
        failures.append("editable_source_count_is_not_one")
    if not metadata.synthesizable:
        failures.append("task_is_not_synthesizable")
    if not metadata.explicit_ports:
        failures.append("ports_are_not_explicit")
    if not metadata.cycle_behavior_specified:
        failures.append("cycle_behavior_is_not_explicit")
    if not metadata.deterministic_verification:
        failures.append("deterministic_verification_is_missing")
    if metadata.unresolved_architecture:
        failures.append("architectural_decision_is_unresolved")
    mechanically_eligible = not failures
    if metadata.worker_eligible != mechanically_eligible:
        failures.append("declared_worker_eligibility_disagrees_with_policy")
        mechanically_eligible = False
    return EligibilityDecision(
        eligible=mechanically_eligible,
        reason=(
            "eligible_single_synthesizable_module_with_explicit_contract_and_tools"
            if mechanically_eligible
            else "main_model_required:" + ",".join(failures)
        ),
        failed_conditions=tuple(failures),
    )


@dataclass(frozen=True)
class DualModelConfiguration:
    main: ServingCandidate
    rtl_worker: ServingCandidate | None = None
    worker_contract_retries: int = 1
    worker_response_retries: int = 1
    fallback_to_main: bool = True

    def __post_init__(self) -> None:
        if self.worker_contract_retries < 0 or self.worker_contract_retries > 2:
            raise ValueError("worker_contract_retries must be between 0 and 2")
        if self.worker_response_retries < 0 or self.worker_response_retries > 2:
            raise ValueError("worker_response_retries must be between 0 and 2")
        if self.rtl_worker is not None and self.rtl_worker.endpoint == self.main.endpoint:
            if self.rtl_worker.model != self.main.model:
                raise ValueError("Different configured models cannot share one endpoint")

    @property
    def single_model(self) -> bool:
        return self.rtl_worker is None


@dataclass(frozen=True)
class RoutingDecision:
    role: ModelRole
    selected: Literal["main", "rtl_worker"]
    candidate: ServingCandidate
    eligibility: EligibilityDecision
    reason: str
    fallback_from: str | None = None

    def to_json(self) -> JsonObject:
        return {
            "role": self.role,
            "selected": self.selected,
            "model": self.candidate.model,
            "model_path": self.candidate.model_path,
            "endpoint": self.candidate.endpoint,
            "engine": self.candidate.engine,
            "revision": self.candidate.revision,
            "quantization": self.candidate.quantization,
            "context_tokens": self.candidate.context_tokens,
            "max_output_tokens": self.candidate.max_output_tokens,
            "temperature": self.candidate.temperature,
            "top_p": self.candidate.top_p,
            "seed": self.candidate.seed,
            "reason": self.reason,
            "fallback_from": self.fallback_from,
            "eligibility": self.eligibility.to_json(),
        }


class RoleRouter:
    """Select a configured endpoint for one explicit role."""

    def __init__(self, configuration: DualModelConfiguration) -> None:
        self.configuration = configuration

    def select(self, role: ModelRole, metadata: RoutingTaskMetadata) -> RoutingDecision:
        eligibility = assess_rtl_worker_eligibility(metadata)
        worker_role = role in {"bounded_rtl_implementation", "bounded_rtl_repair"}
        if worker_role and eligibility.eligible and self.configuration.rtl_worker is not None:
            return RoutingDecision(
                role,
                "rtl_worker",
                self.configuration.rtl_worker,
                eligibility,
                "bounded RTL role and deterministic eligibility policy selected the specialist",
            )
        reason = "role is reserved for the main model"
        if worker_role and self.configuration.rtl_worker is None:
            reason = "single-model operation: no RTL worker endpoint is configured"
        elif worker_role and not eligibility.eligible:
            reason = eligibility.reason
        return RoutingDecision(role, "main", self.configuration.main, eligibility, reason)

    def fallback(
        self, role: ModelRole, metadata: RoutingTaskMetadata, *, failed_reason: str
    ) -> RoutingDecision:
        if not self.configuration.fallback_to_main:
            raise ModelRequired("RTL worker fallback is disabled")
        eligibility = assess_rtl_worker_eligibility(metadata)
        return RoutingDecision(
            role,
            "main",
            self.configuration.main,
            eligibility,
            "bounded worker failure triggered configured main-model fallback",
            fallback_from=failed_reason,
        )


@dataclass(frozen=True)
class RoutedCall:
    result: GenerationResult
    decision: RoutingDecision
    response_valid: bool
    validation_error: str | None
    generation_seconds: float
    retry_index: int
    audit_path: str
    budget: JsonObject


ResponseValidator = Callable[[str], object]
PromptCompactor = Callable[[], str]


class ContextBudgetError(ModelRequired):
    """A prompt cannot fit a model's configured context without losing protected evidence."""

    def __init__(self, message: str, *, budget: JsonObject) -> None:
        super().__init__(message)
        self.category = "context_overflow"
        self.budget = budget


_ROLE_COMPLETION_CAPS: dict[ModelRole, int | None] = {
    "planning_supervision": 1536,
    "retrieval_interpretation": 1024,
    "general_implementation": None,
    "rtl_contract_generation": 2048,
    "bounded_rtl_implementation": None,
    "bounded_rtl_repair": None,
    "review": 768,
}


class AuditedModelCaller:
    """Reuse providers while writing one complete audit record per call."""

    def __init__(
        self,
        router: RoleRouter,
        audit_root: Path,
        *,
        backend_factory: Callable[[ServingCandidate], LocalGenerationBackend] = backend_for,
    ) -> None:
        self.router = router
        self.audit_root = audit_root.resolve()
        self.backend_factory = backend_factory
        self._backends: dict[ServingCandidate, LocalGenerationBackend] = {}

    def _backend(self, candidate: ServingCandidate) -> LocalGenerationBackend:
        backend = self._backends.get(candidate)
        if backend is None:
            backend = self.backend_factory(candidate)
            self._backends[candidate] = backend
        return backend

    def health(self, *, include_worker: bool) -> JsonObject:
        records: JsonObject = {"main": self._backend(self.router.configuration.main).health()}
        worker = self.router.configuration.rtl_worker
        if include_worker and worker is not None:
            records["rtl_worker"] = self._backend(worker).health()
        return records

    @staticmethod
    def _prompt_tokens(backend: LocalGenerationBackend, prompt: str) -> tuple[int, str, str | None]:
        counter = getattr(backend, "token_count", None)
        if callable(counter):
            try:
                count = counter(prompt)
                if isinstance(count, int) and count >= 0:
                    return count, "model_tokenizer_endpoint", None
            except ModelRequired as exc:
                tokenizer_error = str(exc)
            except (OSError, ValueError, TypeError) as exc:
                tokenizer_error = str(exc)
        else:
            tokenizer_error = "backend has no token_count method"
        # UTF-8 bytes / 3 is deliberately conservative for the configured code models.
        estimate = max(1, (len(prompt.encode("utf-8")) + 2) // 3)
        return estimate, "conservative_utf8_bytes_divided_by_3", tokenizer_error

    @staticmethod
    def _requested_completion_tokens(
        decision: RoutingDecision, role: ModelRole, override: int | None
    ) -> int:
        configured = decision.candidate.max_output_tokens
        role_cap = (
            decision.candidate.reviewer_max_output_tokens
            if role == "review"
            else _ROLE_COMPLETION_CAPS[role]
        )
        values = [configured]
        if role_cap is not None:
            values.append(role_cap)
        if override is not None:
            if override < 1:
                raise EngineeringError("Requested completion tokens must be positive")
            values.append(override)
        return min(values)

    def generate(
        self,
        prompt: str,
        *,
        role: ModelRole,
        metadata: RoutingTaskMetadata,
        retry_index: int = 0,
        fallback_reason: str | None = None,
        validator: ResponseValidator | None = None,
        requested_completion_tokens: int | None = None,
        compact_prompt: PromptCompactor | None = None,
    ) -> RoutedCall:
        if not prompt.strip():
            raise EngineeringError("Model prompt cannot be empty")
        decision = (
            self.router.fallback(role, metadata, failed_reason=fallback_reason)
            if fallback_reason is not None
            else self.router.select(role, metadata)
        )
        backend = self._backend(decision.candidate)
        call_id = uuid.uuid4().hex
        started = time.monotonic()
        created_at = datetime.now(UTC).isoformat()
        requested = self._requested_completion_tokens(decision, role, requested_completion_tokens)
        context_limit = decision.candidate.context_tokens
        safety_margin = decision.candidate.context_safety_margin_tokens
        minimum_completion = min(requested, decision.candidate.minimum_completion_tokens)
        effective_prompt = prompt
        prompt_tokens, token_count_method, tokenizer_error = self._prompt_tokens(
            backend, effective_prompt
        )
        capacity = max(0, context_limit - safety_margin - prompt_tokens)
        effective_completion = min(requested, capacity)
        compaction_occurred = False
        compaction_reason: str | None = None
        if effective_completion < minimum_completion and compact_prompt is not None:
            compacted = compact_prompt()
            if not compacted.strip():
                raise EngineeringError("Prompt compactor returned an empty prompt")
            if compacted != effective_prompt:
                effective_prompt = compacted
                compaction_occurred = True
                compaction_reason = "minimum_completion_budget_did_not_fit"
                prompt_tokens, token_count_method, tokenizer_error = self._prompt_tokens(
                    backend, effective_prompt
                )
                capacity = max(0, context_limit - safety_margin - prompt_tokens)
                effective_completion = min(requested, capacity)
        budget: JsonObject = {
            "prompt_tokens": prompt_tokens,
            "prompt_token_count_method": token_count_method,
            "tokenizer_error": tokenizer_error,
            "requested_completion_tokens": requested,
            "effective_completion_token_cap": effective_completion,
            "minimum_usable_completion_tokens": minimum_completion,
            "context_limit": context_limit,
            "safety_margin_tokens": safety_margin,
            "compaction_occurred": compaction_occurred,
            "compaction_reason": compaction_reason,
            "context_rejection_reason": None,
        }
        path = self.audit_root / f"{created_at.replace(':', '')}_{call_id}.json"

        def write_audit(payload: JsonObject) -> None:
            record: JsonObject = {
                "schema_version": 2,
                "call_id": call_id,
                "task_id": metadata.task_id,
                "experiment_arm": metadata.experiment_arm,
                "created_at": created_at,
                "routing": decision.to_json(),
                "prompt_sha256": hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
                "prompt_characters": len(effective_prompt),
                **budget,
                "retry_index": retry_index,
                "fallback_used": fallback_reason is not None,
                **payload,
            }
            _write_json_atomic(path, record, readonly=True)

        if effective_completion < minimum_completion:
            budget["context_rejection_reason"] = (
                f"prompt={prompt_tokens} + safety={safety_margin} leaves "
                f"{capacity} tokens; minimum completion is {minimum_completion}"
            )
            elapsed = time.monotonic() - started
            message = "Irreducible prompt context overflow after bounded compaction"
            write_audit(
                {
                    "completion_tokens": None,
                    "generation_seconds": elapsed,
                    "response_valid": False,
                    "status": "CONTEXT_REJECTED",
                    "failure_category": "context_overflow",
                    "error": message,
                }
            )
            raise ContextBudgetError(message, budget=budget)
        try:
            result = backend.generate(
                effective_prompt,
                context_tokens=context_limit,
                max_tokens=effective_completion,
            )
        except ModelRequired as exc:
            elapsed = time.monotonic() - started
            category = getattr(exc, "category", "endpoint_unavailable")
            write_audit(
                {
                    "completion_tokens": None,
                    "generation_seconds": elapsed,
                    "response_valid": False,
                    "status": "MODEL_REQUIRED",
                    "failure_category": category,
                    "http_status": exc.http_status
                    if isinstance(exc, ModelInvocationError)
                    else None,
                    "error": str(exc),
                }
            )
            raise
        elapsed = time.monotonic() - started
        valid = bool(result.text.strip())
        validation_error: str | None = None
        if validator is not None:
            try:
                validator(result.text)
            except (EngineeringError, ValueError) as exc:
                valid = False
                validation_error = str(exc)
        elif not valid:
            validation_error = "Model response is empty"
        write_audit(
            {
                "server_reported_prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "generation_seconds": elapsed,
                "ttft_seconds": result.ttft_seconds,
                "output_tokens_per_second": result.output_tokens_per_second,
                "response_valid": valid,
                "validation_error": validation_error,
                "status": result.status,
                "failure_category": None,
            }
        )
        return RoutedCall(
            result,
            decision,
            valid,
            validation_error,
            elapsed,
            retry_index,
            str(path),
            budget,
        )


def serving_candidate_from_json(value: object) -> ServingCandidate:
    """Load one exact serving profile while accepting legacy candidate files."""
    if not isinstance(value, dict):
        raise ValueError("Serving profile must be an object")
    required = {
        "engine",
        "endpoint",
        "model",
        "revision",
        "quantization",
        "kernel",
        "prefix_caching",
        "chunked_prefill",
        "cuda_graph_mode",
        "scheduler",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Serving profile is missing fields: {missing}")
    allowed = required | {
        "model_path",
        "context_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "request_timeout_seconds",
        "context_safety_margin_tokens",
        "minimum_completion_tokens",
        "reviewer_max_output_tokens",
    }
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"Serving profile has unexpected fields: {unexpected}")
    engine = value.get("engine")
    if engine not in {"vllm", "sglang"}:
        raise ValueError("Serving profile engine must be vllm or sglang")
    string_fields = required - {"engine", "prefix_caching", "chunked_prefill"}
    if any(not isinstance(value.get(key), str) or not value.get(key) for key in string_fields):
        raise ValueError("Serving profile string fields must be non-empty")
    if not isinstance(value.get("prefix_caching"), bool) or not isinstance(
        value.get("chunked_prefill"), bool
    ):
        raise ValueError("Serving profile cache settings must be booleans")
    integer_fields = {
        "context_tokens": 8192,
        "max_output_tokens": 2048,
        "request_timeout_seconds": 120,
        "context_safety_margin_tokens": 512,
        "minimum_completion_tokens": 256,
        "reviewer_max_output_tokens": 768,
    }
    integers: dict[str, int] = {}
    for key, default in integer_fields.items():
        raw = value.get(key, default)
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ValueError(f"Serving profile {key} must be an integer")
        integers[key] = raw
    temperature = value.get("temperature", 0.0)
    top_p = value.get("top_p", 1.0)
    seed = value.get("seed", 0)
    model_path = value.get("model_path")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("Serving profile temperature must be numeric")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValueError("Serving profile top_p must be numeric")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("Serving profile seed must be an integer or null")
    if model_path is not None and (not isinstance(model_path, str) or not model_path.strip()):
        raise ValueError("Serving profile model_path must be non-empty text or null")
    return ServingCandidate(
        engine=engine,
        endpoint=str(value["endpoint"]),
        model=str(value["model"]),
        revision=str(value["revision"]),
        quantization=str(value["quantization"]),
        kernel=str(value["kernel"]),
        prefix_caching=bool(value["prefix_caching"]),
        chunked_prefill=bool(value["chunked_prefill"]),
        cuda_graph_mode=str(value["cuda_graph_mode"]),
        scheduler=str(value["scheduler"]),
        model_path=model_path,
        context_tokens=integers["context_tokens"],
        max_output_tokens=integers["max_output_tokens"],
        temperature=float(temperature),
        top_p=float(top_p),
        seed=seed,
        request_timeout_seconds=integers["request_timeout_seconds"],
        context_safety_margin_tokens=integers["context_safety_margin_tokens"],
        minimum_completion_tokens=integers["minimum_completion_tokens"],
        reviewer_max_output_tokens=integers["reviewer_max_output_tokens"],
    )


def load_dual_model_configuration(path: Path) -> DualModelConfiguration:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load dual-model configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Dual-model configuration must be an object")
    expected = {
        "schema_version",
        "main",
        "rtl_worker",
        "worker_contract_retries",
        "worker_response_retries",
        "fallback_to_main",
    }
    if set(value) != expected or value.get("schema_version") != 1:
        raise ValueError("Dual-model configuration keys or schema_version are invalid")
    worker_raw = value.get("rtl_worker")
    worker = None if worker_raw is None else serving_candidate_from_json(worker_raw)
    contract_retries = value.get("worker_contract_retries")
    response_retries = value.get("worker_response_retries")
    fallback = value.get("fallback_to_main")
    if not isinstance(contract_retries, int) or isinstance(contract_retries, bool):
        raise ValueError("worker_contract_retries must be an integer")
    if not isinstance(response_retries, int) or isinstance(response_retries, bool):
        raise ValueError("worker_response_retries must be an integer")
    if not isinstance(fallback, bool):
        raise ValueError("fallback_to_main must be a boolean")
    return DualModelConfiguration(
        main=serving_candidate_from_json(value.get("main")),
        rtl_worker=worker,
        worker_contract_retries=contract_retries,
        worker_response_retries=response_retries,
        fallback_to_main=fallback,
    )
