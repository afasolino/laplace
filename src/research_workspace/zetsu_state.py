"""Checkpoint-compatible state objects for bounded Zetsu agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .agent_sandbox import AgentSessionBinding
from .service_tiers import ModelLane, ServiceTierError
from .verification_policy import VerificationPlan

JsonObject = dict[str, object]


def _usage_tokens(value: Mapping[str, object]) -> tuple[int | None, int | None, int | None]:
    response = value.get("response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return None, None, None

    def read(*names: str) -> int | None:
        for name in names:
            raw = usage.get(name)
            if isinstance(raw, int) and raw >= 0:
                return raw
        return None

    return (
        read("prompt_tokens", "input_tokens"),
        read("completion_tokens", "output_tokens"),
        read("cached_tokens", "cached_input_tokens"),
    )


@dataclass
class AgentTelemetry:
    qwen_input_tokens: int = 0
    qwen_output_tokens: int = 0
    qwen_cached_tokens: int = 0
    qwen_usage_reported_calls: int = 0
    qwen_calls: int = 0
    agent_steps: int = 0
    tool_calls: int = 0
    verification_calls: int = 0
    compactions: int = 0
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0
    approximate_active_context_tokens_before_last_compaction: int = 0
    approximate_active_context_tokens_after_last_compaction: int = 0
    last_model_reported_input_tokens: int | None = None

    def add_usage(self, value: Mapping[str, object], *, compaction: bool = False) -> None:
        prompt, completion, cached = _usage_tokens(value)
        self.qwen_calls += 1
        if prompt is not None or completion is not None or cached is not None:
            self.qwen_usage_reported_calls += 1
        if prompt is not None:
            self.qwen_input_tokens += prompt
            self.last_model_reported_input_tokens = prompt
        if completion is not None:
            self.qwen_output_tokens += completion
        if cached is not None:
            self.qwen_cached_tokens += cached
        if compaction:
            self.compaction_input_tokens += prompt or 0
            self.compaction_output_tokens += completion or 0

    @classmethod
    def from_mapping(cls, value: object) -> "AgentTelemetry":
        if not isinstance(value, Mapping):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        telemetry = cls()
        integer_fields = set(telemetry.__dataclass_fields__) - {"last_model_reported_input_tokens"}
        allowed = integer_fields | {
            "last_model_reported_input_tokens",
            "qwen_token_usage_source",
            "qwen_token_usage_complete",
            "approximate_context_token_method",
        }
        if set(value) - allowed:
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        for name in integer_fields:
            raw = value.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
            setattr(telemetry, name, raw)
        raw_last = value.get("last_model_reported_input_tokens")
        if raw_last is not None and (
            isinstance(raw_last, bool) or not isinstance(raw_last, int) or raw_last < 0
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        telemetry.last_model_reported_input_tokens = raw_last
        if value.get("qwen_token_usage_source") != "model_reported_per_request":
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        if value.get("approximate_context_token_method") != "utf8_json_bytes_div4":
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        complete = value.get("qwen_token_usage_complete")
        if not isinstance(complete, bool) or complete != (
            telemetry.qwen_usage_reported_calls == telemetry.qwen_calls
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        if (
            telemetry.qwen_usage_reported_calls > telemetry.qwen_calls
            or telemetry.verification_calls > telemetry.tool_calls
            or telemetry.compactions > telemetry.qwen_calls
            or telemetry.qwen_calls != telemetry.agent_steps + telemetry.compactions
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        return telemetry

    def as_dict(self) -> JsonObject:
        return {
            "qwen_input_tokens": self.qwen_input_tokens,
            "qwen_output_tokens": self.qwen_output_tokens,
            "qwen_cached_tokens": self.qwen_cached_tokens,
            "qwen_usage_reported_calls": self.qwen_usage_reported_calls,
            "qwen_calls": self.qwen_calls,
            "qwen_token_usage_source": "model_reported_per_request",
            "qwen_token_usage_complete": self.qwen_usage_reported_calls == self.qwen_calls,
            "approximate_context_token_method": "utf8_json_bytes_div4",
            "agent_steps": self.agent_steps,
            "tool_calls": self.tool_calls,
            "verification_calls": self.verification_calls,
            "compactions": self.compactions,
            "compaction_input_tokens": self.compaction_input_tokens,
            "compaction_output_tokens": self.compaction_output_tokens,
            "approximate_active_context_tokens_before_last_compaction": self.approximate_active_context_tokens_before_last_compaction,
            "approximate_active_context_tokens_after_last_compaction": self.approximate_active_context_tokens_after_last_compaction,
            "last_model_reported_input_tokens": self.last_model_reported_input_tokens,
        }


@dataclass
class AgentExecutionState:
    objective: str
    step: int = 0
    summary: str = ""
    recent_observations: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    worktree_head: str = ""
    worktree_status_sha256: str = ""
    lane: str = ""
    model_id: str = ""
    context_limit: int = 0
    required_verification_argv: list[str] | None = None
    required_verification_plan: list[JsonObject] | None = None
    validation_history: list[JsonObject] = field(default_factory=list)
    unresolved_failures: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    next_state: str = "choose_action"
    mutation_epoch: int = 0
    last_verified_epoch: int = -1
    candidate_fingerprint: str = ""
    verified_fingerprint: str = ""
    verifier_digest: str = ""
    assurance_state: str = "clean"
    command_count: int = 0
    consumed_wall_seconds: float = 0.0
    target_initial_head: str = ""
    target_initial_status_sha256: str = ""
    target_initial_snapshot_sha256: str = ""
    target_applied_status_sha256: str = ""
    target_applied_snapshot_sha256: str = ""
    applied_patch_sha256: str = ""
    output_cap_continuations: int = 0
    continuation_count: int = 0
    stagnation_count: int = 0
    exploration_only_quanta: int = 0
    recent_action_fingerprints: list[str] = field(default_factory=list)
    quantum_action_fingerprints: list[str] = field(default_factory=list)
    quantum_exploration_only: bool = True
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)


@dataclass(frozen=True)
class AgentRunContext:
    user_id: str
    session_id: str
    repo_id: str
    lane: ModelLane
    binding: AgentSessionBinding
    worktree: Path
    max_steps: int
    max_chars: int
    compaction_ratio: float
    model_id: str
    context_limit: int
    required_verification_argv: tuple[str, ...] | None
    run_started: float
    remaining_wall_seconds: float
    required_verification_plan: VerificationPlan | None = None
    task_label: str = "Repository Task"
    apply_to_repository: bool = False
    allow_mutation: bool = True
