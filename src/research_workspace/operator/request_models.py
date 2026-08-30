"""Validated non-agent request models for Operator transport routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..research_models import ResearchJobRequest


class RunPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    configuration: dict[str, object]
    run_id: str | None = Field(default=None, max_length=160)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str = Field(min_length=1, max_length=80)
    entity_id: str = Field(min_length=1, max_length=320)
    payload: dict[str, object] = Field(default_factory=dict)


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approve: bool


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approval_id: str | None = Field(default=None, max_length=160)


class ModelServerActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    approval_id: str | None = Field(default=None, max_length=160)


class ResearchCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    job: ResearchJobRequest
    research_job_id: str | None = Field(default=None, max_length=160)
    domain: str = Field(default="general", min_length=1, max_length=80)


class TierChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: str
    content: str = Field(min_length=1, max_length=100_000)


class TierChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lane: Literal["quality", "standard", "economy"]
    domain: str = Field(default="general", min_length=1, max_length=80)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    conversation_id: str | None = Field(default=None, pattern=r"^conv-[a-f0-9]{32}$")
    messages: list[TierChatMessage] = Field(min_length=1, max_length=200)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,159}$")
    retrieval_selection: Literal["none", "personal", "shared", "both", "selected_personal"] = "none"
    personal_corpus_id: str | None = Field(default=None, pattern=r"^pc_[a-f0-9]{32}$")


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1_024)


class ActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    email: str = Field(min_length=3, max_length=320)
    activation_code: str = Field(min_length=1, max_length=1_024)
    new_password: str = Field(min_length=12, max_length=1_024)


class PasswordChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    current_password: str = Field(min_length=1, max_length=1_024)
    new_password: str = Field(min_length=12, max_length=1_024)


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="New conversation", max_length=160)


class ConversationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str | None = Field(default=None, max_length=160)
    archived: bool | None = None
    draft: str | None = Field(default=None, max_length=100_000)


class AgentSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["read_file", "apply_patch", "run_tests"],
        min_length=1,
        max_length=20,
    )
    max_commands: int = Field(default=100, ge=1, le=1_000)
    max_wall_seconds: int = Field(default=1_800, ge=1, le=14_400)
    task_title: str = Field(default="New Agent task", min_length=1, max_length=200)
    idempotency_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$"
    )


class TierUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    tier: Literal["basic", "plus", "operator"]
    enabled: bool = True


class CapabilitySetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    capabilities: list[
        Literal[
            "chat", "agent", "research", "operator", "admin", "personal_corpus",
            "shared_corpus_ingest", "repository_admin", "model_admin",
        ]
    ] = Field(max_length=9)
    enabled: bool | None = None


class CorpusCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=160)


class CorpusUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str | None = Field(default=None, min_length=1, max_length=160)
    archived: bool | None = None


class UploadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    corpus_id: str = Field(pattern=r"^pc_[a-f0-9]{32}$")
    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$")


class IndexUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$")


class CorpusSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=4_000)
    corpus_id: str | None = Field(default=None, pattern=r"^pc_[a-f0-9]{32}$")
    limit: int = Field(default=8, ge=1, le=50)


class WorktreeDiscardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confirmation: str = Field(min_length=9, max_length=180)


class WorktreeExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    promotion: bool = False


class RepositoryRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    canonical_root: str = Field(min_length=1, max_length=4_096)


class RepositoryGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    base_revision: str = Field(default="HEAD", min_length=1, max_length=200)


class ServingProfileActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "stop"]
    profile_id: str | None = Field(default=None, pattern=r"^P[0-9]+(?:_[a-z0-9_]+)?$")


class ClientPairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=160)
    capabilities: dict[str, object]
    device_id: str | None = Field(default=None, pattern=r"^dev_[a-f0-9]{32}$")


class ClientHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    capabilities: dict[str, object]


class ClientOperationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    workspace_id: str = Field(pattern=r"^ws-[a-f0-9]{24}$")
    action: Literal["list", "read", "search", "write", "git", "run"]
    arguments: dict[str, object] = Field(default_factory=dict)


class ClientResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    result: dict[str, object]
    failed: bool = False
