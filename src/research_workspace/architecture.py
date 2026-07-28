"""Typed architecture boundaries shared by desktop and server modes.

Provider calls are asynchronous because they can wait on a local runtime. Persistent
stores remain synchronous transaction boundaries and are moved to worker threads by
HTTP adapters when needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import (
    ArtifactV1,
    AuditEventV1,
    CapabilityAssignmentV1,
    ConversationV1,
    CorpusSourceV1,
    CorpusV1,
    EmbeddingRequestV1,
    EmbeddingResponseV1,
    JobV1,
    MessageV1,
    ModelRequestV1,
    ModelResponseV1,
    ProvenanceV1,
    ProviderV1,
    RepositoryGrantV1,
    RetrievalSnapshotV1,
    RouteV1,
    WorktreeV1,
)


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def descriptor(self) -> ProviderV1: ...

    async def health(self) -> Mapping[str, object]: ...

    async def readiness(self) -> Mapping[str, object]: ...

    async def available_models(self) -> tuple[str, ...]: ...

    async def generate(self, request: ModelRequestV1) -> ModelResponseV1: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def descriptor(self) -> ProviderV1: ...

    async def embed(self, request: EmbeddingRequestV1) -> EmbeddingResponseV1: ...


@runtime_checkable
class ConversationStore(Protocol):
    def create(self, conversation: ConversationV1) -> ConversationV1: ...

    def append(self, message: MessageV1) -> MessageV1: ...

    def get(self, owner_id: str, conversation_id: str) -> ConversationV1: ...

    def messages(self, owner_id: str, conversation_id: str) -> tuple[MessageV1, ...]: ...


@runtime_checkable
class CorpusStore(Protocol):
    def put_corpus(self, corpus: CorpusV1) -> CorpusV1: ...

    def put_source(self, owner_id: str, source: CorpusSourceV1) -> CorpusSourceV1: ...

    def get_corpus(self, owner_id: str, corpus_id: str) -> CorpusV1: ...

    def sources(self, owner_id: str, corpus_id: str) -> tuple[CorpusSourceV1, ...]: ...


@runtime_checkable
class RetrievalService(Protocol):
    def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        corpus_ids: Sequence[str],
        limit: int,
    ) -> RetrievalSnapshotV1: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, artifact: ArtifactV1, content: bytes) -> ArtifactV1: ...

    def read(self, owner_id: str, artifact_id: str) -> bytes: ...


@runtime_checkable
class ProvenanceStore(Protocol):
    def append(self, record: ProvenanceV1) -> ProvenanceV1: ...

    def lineage(self, subject_id: str) -> tuple[ProvenanceV1, ...]: ...


@runtime_checkable
class RepositoryService(Protocol):
    def grants(self, user_id: str) -> tuple[RepositoryGrantV1, ...]: ...

    def require_active(self, user_id: str, repository_id: str) -> RepositoryGrantV1: ...


@runtime_checkable
class WorktreeService(Protocol):
    def create(self, worktree: WorktreeV1) -> WorktreeV1: ...

    def get(self, owner_id: str, worktree_id: str) -> WorktreeV1: ...

    def list_for_owner(self, owner_id: str) -> tuple[WorktreeV1, ...]: ...


@runtime_checkable
class IdentityService(Protocol):
    def enabled(self, user_id: str) -> bool: ...

    def revision(self, user_id: str) -> int: ...


@runtime_checkable
class CapabilityService(Protocol):
    def assignment(self, user_id: str) -> CapabilityAssignmentV1: ...

    def require(self, user_id: str, capability: str) -> CapabilityAssignmentV1: ...


@runtime_checkable
class JobService(Protocol):
    def create(self, job: JobV1) -> JobV1: ...

    def get(self, owner_id: str, job_id: str) -> JobV1: ...


@runtime_checkable
class AuditService(Protocol):
    def append(self, event: AuditEventV1) -> AuditEventV1: ...

    def for_subject(self, subject_id: str) -> tuple[AuditEventV1, ...]: ...


@runtime_checkable
class ConfigurationService(Protocol):
    def effective(self) -> Mapping[str, object]: ...

    def sanitized(self) -> Mapping[str, object]: ...

    def provenance(self) -> Mapping[str, str]: ...


class ExistingGenerationAdapter:
    """Compatibility adapter from a neutral response to legacy generation callers."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def generate_text(self, request: ModelRequestV1) -> str:
        return (await self.provider.generate(request)).text


class ExternalStatePath:
    """Operator-only wrapper preventing canonical paths from entering user records."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()

    def operator_path(self) -> Path:
        return self._path

    def __str__(self) -> str:
        return "<external-state-path>"


def public_provider_catalog(
    providers: Sequence[ModelProvider],
    routes: Sequence[RouteV1],
) -> dict[str, object]:
    """Produce the only provider catalog shape intended for ordinary frontends."""

    known = {provider.descriptor.provider_id for provider in providers}
    safe_routes = [
        route.model_dump(mode="json")
        for route in routes
        if route.enabled and route.provider_id in known
    ]
    return {
        "schema_version": 1,
        "providers": [provider.descriptor.public_summary() for provider in providers],
        "routes": safe_routes,
    }
