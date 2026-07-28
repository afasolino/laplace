"""Provider-neutral, versioned internal contracts for both Laplace operating modes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictContract(BaseModel):
    """Base for persisted and inter-service records with fail-closed parsing."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CitationV1(StrictContract):
    schema_version: Literal[1] = 1
    source_id: str = Field(min_length=1, max_length=160)
    source_title: str = Field(min_length=1, max_length=500)
    chunk_id: str = Field(min_length=1, max_length=160)
    page: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=500)


class ConversationV1(StrictContract):
    schema_version: Literal[1] = 1
    conversation_id: str = Field(min_length=1, max_length=160)
    owner_id: str = Field(min_length=1, max_length=160)
    operating_mode: Literal["desktop", "server"]
    title: str = Field(min_length=1, max_length=500)
    created_at_utc: str
    updated_at_utc: str
    archived: bool = False


class MessageV1(StrictContract):
    schema_version: Literal[1] = 1
    message_id: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=1_000_000)
    created_at_utc: str
    citations: tuple[CitationV1, ...] = ()
    provider_id: str | None = Field(default=None, max_length=160)
    model_id: str | None = Field(default=None, max_length=320)


class AttachmentV1(StrictContract):
    schema_version: Literal[1] = 1
    attachment_id: str = Field(min_length=1, max_length=160)
    owner_id: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    logical_name: str = Field(min_length=1, max_length=500)
    media_type: str = Field(min_length=1, max_length=160)
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    retention_class: str = Field(min_length=1, max_length=80)


class CorpusV1(StrictContract):
    schema_version: Literal[1] = 1
    corpus_id: str = Field(min_length=1, max_length=160)
    owner_id: str | None = Field(default=None, max_length=160)
    corpus_class: Literal["project", "personal", "shared", "formal_science"]
    display_name: str = Field(min_length=1, max_length=500)
    snapshot_revision: str = Field(min_length=1, max_length=160)
    indexing_state: Literal["EMPTY", "STAGING", "INDEXING", "READY", "DEGRADED", "DELETED"]


class CorpusSourceV1(StrictContract):
    schema_version: Literal[1] = 1
    source_id: str = Field(min_length=1, max_length=160)
    corpus_id: str = Field(min_length=1, max_length=160)
    logical_name: str = Field(min_length=1, max_length=500)
    source_class: Literal[
        "user_work",
        "external_literature",
        "technical_documentation",
        "project_document",
        "experiment_record",
    ]
    original_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extracted_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_count: int = Field(ge=0)
    indexed_at_utc: str | None = None


class RetrievalSnapshotV1(StrictContract):
    schema_version: Literal[1] = 1
    snapshot_id: str = Field(min_length=1, max_length=160)
    owner_id: str = Field(min_length=1, max_length=160)
    corpus_revisions: tuple[str, ...]
    query_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    citations: tuple[CitationV1, ...]
    created_at_utc: str


class ArtifactV1(StrictContract):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(min_length=1, max_length=160)
    owner_id: str = Field(min_length=1, max_length=160)
    logical_name: str = Field(min_length=1, max_length=500)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: str = Field(min_length=1, max_length=160)
    size_bytes: int = Field(ge=0)
    retention_class: str = Field(min_length=1, max_length=80)
    created_at_utc: str


class ProvenanceV1(StrictContract):
    schema_version: Literal[1] = 1
    provenance_id: str = Field(min_length=1, max_length=160)
    subject_type: str = Field(min_length=1, max_length=80)
    subject_id: str = Field(min_length=1, max_length=160)
    action: str = Field(min_length=1, max_length=120)
    actor_id: str = Field(min_length=1, max_length=160)
    parent_ids: tuple[str, ...] = ()
    content_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    created_at_utc: str


class RepositoryGrantV1(StrictContract):
    schema_version: Literal[1] = 1
    grant_id: str = Field(min_length=1, max_length=160)
    user_id: str = Field(min_length=1, max_length=160)
    repository_id: str = Field(min_length=1, max_length=160)
    revision: int = Field(ge=1)
    active: bool
    created_at_utc: str
    expires_at_utc: str | None = None


class WorktreeV1(StrictContract):
    schema_version: Literal[1] = 1
    worktree_id: str = Field(min_length=1, max_length=160)
    owner_id: str = Field(min_length=1, max_length=160)
    repository_id: str = Field(min_length=1, max_length=160)
    base_commit: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    state: str = Field(min_length=1, max_length=80)
    changed_paths: tuple[str, ...] = ()
    created_at_utc: str
    updated_at_utc: str

    @field_validator("changed_paths")
    @classmethod
    def logical_paths_only(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item or item.startswith(("/", "\\")) or ".." in item.replace("\\", "/").split("/"):
                raise ValueError("worktree paths must be logical relative paths")
        return value


class JobV1(StrictContract):
    schema_version: Literal[1] = 1
    job_id: str = Field(min_length=1, max_length=160)
    owner_id: str = Field(min_length=1, max_length=160)
    job_type: str = Field(min_length=1, max_length=80)
    state: str = Field(min_length=1, max_length=80)
    request_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at_utc: str
    updated_at_utc: str


class ProviderCapabilitiesV1(StrictContract):
    schema_version: Literal[1] = 1
    streaming: bool
    tools: bool
    structured_output: bool
    embeddings: bool
    thinking_control: bool
    requires_gpu: bool
    supports_cpu: bool
    can_start: bool
    can_stop: bool


class ProviderV1(StrictContract):
    schema_version: Literal[1] = 1
    provider_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    display_name: str = Field(min_length=1, max_length=160)
    provider_type: Literal["fixture", "ollama", "vllm"]
    endpoint: str
    lifecycle: Literal["owned", "unowned", "fixture"]
    context_limit: int = Field(ge=256, le=10_000_000)
    output_limit: int = Field(ge=1, le=1_000_000)
    capabilities: ProviderCapabilitiesV1

    def public_summary(self) -> dict[str, object]:
        """Return frontend-safe capabilities without a transport endpoint."""

        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "provider_type": self.provider_type,
            "lifecycle": self.lifecycle,
            "context_limit": self.context_limit,
            "output_limit": self.output_limit,
            "capabilities": self.capabilities.model_dump(mode="json"),
            "endpoint_scope": "fixture" if self.lifecycle == "fixture" else "configured-local",
        }


class RouteV1(StrictContract):
    schema_version: Literal[1] = 1
    route_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    display_name: str = Field(min_length=1, max_length=160)
    provider_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    model_id: str = Field(min_length=1, max_length=320)
    lane: Literal["quality", "standard", "economy", "codev", "fixture"]
    enabled: bool


class CapabilityAssignmentV1(StrictContract):
    schema_version: Literal[1] = 1
    user_id: str = Field(min_length=1, max_length=160)
    capabilities: tuple[str, ...]
    revision: int = Field(ge=1)
    enabled: bool
    updated_at_utc: str


class AuditEventV1(StrictContract):
    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=160)
    occurred_at_utc: str
    actor_id: str = Field(min_length=1, max_length=160)
    action: str = Field(min_length=1, max_length=160)
    subject_type: str = Field(min_length=1, max_length=80)
    subject_id: str = Field(min_length=1, max_length=160)
    outcome: Literal["allowed", "denied", "completed", "failed"]
    trace_id: str = Field(min_length=1, max_length=160)


class ModelRequestV1(StrictContract):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=160)
    route_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    messages: tuple[MessageV1, ...] = Field(min_length=1, max_length=500)
    max_output_tokens: int = Field(ge=1, le=1_000_000)
    temperature: float = Field(ge=0.0, le=2.0)
    stream: bool = False
    structured_schema: dict[str, object] | None = None
    thinking: Literal["disabled", "enabled", "provider_default"] = "provider_default"


class ModelResponseV1(StrictContract):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=160)
    provider_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=320)
    text: str = Field(max_length=4_000_000)
    finish_reason: Literal["stop", "length", "cancelled", "fixture"]
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    fixture: bool = False


class EmbeddingRequestV1(StrictContract):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=160)
    model_id: str = Field(min_length=1, max_length=320)
    texts: tuple[str, ...] = Field(min_length=1, max_length=1_000)


class EmbeddingResponseV1(StrictContract):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=160)
    provider_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=320)
    vectors: tuple[tuple[float, ...], ...]
    fixture: bool = False
