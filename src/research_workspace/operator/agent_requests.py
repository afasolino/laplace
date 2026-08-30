"""Validated request models for bounded Operator agent turns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..manager_control import TaskComplexity


class AgentTaskComplexityRequest(BaseModel):
    """User-provided complexity evidence for advisory manager planning."""

    model_config = ConfigDict(extra="forbid", strict=True)

    file_count_hint: int = Field(default=0, ge=0, le=100_000)
    architecture_sensitive: bool = False
    security_sensitive: bool = False
    verification_recovery: bool = False
    ambiguous_requirements: bool = False
    rtl_involved: bool = False

    def as_task_complexity(self) -> TaskComplexity:
        return TaskComplexity(**self.model_dump())


class AgentRunRequest(BaseModel):
    """Bounded agent-turn parameters, independent of HTTP route implementation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    lane: Literal["quality", "standard", "economy"]
    instruction: str = Field(min_length=1, max_length=100_000)
    domain: str = Field(min_length=1, max_length=80)
    retrieval_selection: Literal["none", "personal", "shared", "both", "selected_personal"] = "none"
    personal_corpus_id: str | None = Field(default=None, pattern=r"^pc_[a-f0-9]{32}$")
    max_steps: int = Field(default=12, ge=1, le=32)
    max_chars: int = Field(default=8_000, ge=512, le=24_000)
    verification_argv: list[str] | None = Field(default=None, min_length=1, max_length=64)
    allow_mutation: bool = False
    wait_timeout_seconds: int = Field(default=1_800, ge=1, le=3_600)
    manager_complexity: AgentTaskComplexityRequest | None = None


class AgentAsyncRunRequest(AgentRunRequest):
    """Idempotent durable-turn submission for clients that cannot hold a long POST."""

    turn_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,159}$")
