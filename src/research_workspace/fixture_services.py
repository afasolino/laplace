"""Deterministic in-memory implementations of every v7 architecture protocol."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from .contracts import (
    ArtifactV1,
    AuditEventV1,
    CapabilityAssignmentV1,
    CitationV1,
    ConversationV1,
    CorpusSourceV1,
    CorpusV1,
    JobV1,
    MessageV1,
    ProvenanceV1,
    RepositoryGrantV1,
    RetrievalSnapshotV1,
    WorktreeV1,
)


class FixtureAccessError(RuntimeError):
    """A fixture operation failed the same owner/capability boundary as production."""


class FixtureConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationV1] = {}
        self._messages: dict[str, list[MessageV1]] = {}

    def create(self, conversation: ConversationV1) -> ConversationV1:
        if conversation.conversation_id in self._conversations:
            raise FixtureAccessError("conversation_exists")
        self._conversations[conversation.conversation_id] = conversation
        self._messages[conversation.conversation_id] = []
        return conversation

    def append(self, message: MessageV1) -> MessageV1:
        conversation = self._conversations.get(message.conversation_id)
        if conversation is None:
            raise FixtureAccessError("conversation_not_found")
        if any(
            existing.message_id == message.message_id
            for existing in self._messages[message.conversation_id]
        ):
            raise FixtureAccessError("message_exists")
        self._messages[message.conversation_id].append(message)
        return message

    def get(self, owner_id: str, conversation_id: str) -> ConversationV1:
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.owner_id != owner_id:
            raise FixtureAccessError("conversation_not_found")
        return conversation

    def messages(self, owner_id: str, conversation_id: str) -> tuple[MessageV1, ...]:
        self.get(owner_id, conversation_id)
        return tuple(self._messages[conversation_id])


class FixtureCorpusStore:
    def __init__(self) -> None:
        self._corpora: dict[str, CorpusV1] = {}
        self._sources: dict[str, list[CorpusSourceV1]] = {}

    def put_corpus(self, corpus: CorpusV1) -> CorpusV1:
        existing = self._corpora.get(corpus.corpus_id)
        if existing is not None and existing.owner_id != corpus.owner_id:
            raise FixtureAccessError("corpus_owner_conflict")
        self._corpora[corpus.corpus_id] = corpus
        self._sources.setdefault(corpus.corpus_id, [])
        return corpus

    def put_source(self, owner_id: str, source: CorpusSourceV1) -> CorpusSourceV1:
        self.get_corpus(owner_id, source.corpus_id)
        values = self._sources[source.corpus_id]
        if any(item.source_id == source.source_id for item in values):
            raise FixtureAccessError("source_exists")
        values.append(source)
        return source

    def get_corpus(self, owner_id: str, corpus_id: str) -> CorpusV1:
        corpus = self._corpora.get(corpus_id)
        if corpus is None or corpus.owner_id not in {None, owner_id}:
            raise FixtureAccessError("corpus_not_found")
        return corpus

    def sources(self, owner_id: str, corpus_id: str) -> tuple[CorpusSourceV1, ...]:
        self.get_corpus(owner_id, corpus_id)
        return tuple(self._sources[corpus_id])


class FixtureRetrievalService:
    def __init__(
        self,
        corpora: FixtureCorpusStore,
        *,
        source_text: Mapping[str, str],
        clock: str,
    ) -> None:
        self._corpora = corpora
        self._source_text = dict(source_text)
        self._clock = clock

    def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        corpus_ids: Sequence[str],
        limit: int,
    ) -> RetrievalSnapshotV1:
        if limit < 1 or limit > 100:
            raise ValueError("fixture retrieval limit must be 1..100")
        terms = tuple(sorted(set(query.casefold().split())))
        candidates: list[tuple[int, CorpusSourceV1]] = []
        revisions: list[str] = []
        for corpus_id in corpus_ids:
            corpus = self._corpora.get_corpus(owner_id, corpus_id)
            revisions.append(f"{corpus.corpus_id}:{corpus.snapshot_revision}")
            for source in self._corpora.sources(owner_id, corpus_id):
                haystack = self._source_text.get(source.source_id, "").casefold()
                score = sum(1 for term in terms if term in haystack)
                if score:
                    candidates.append((score, source))
        candidates.sort(key=lambda item: (-item[0], item[1].source_id))
        citations = tuple(
            CitationV1(
                source_id=source.source_id,
                source_title=source.logical_name,
                chunk_id=f"{source.source_id}:chunk:0",
                page=None,
                section="fixture",
            )
            for _, source in candidates[:limit]
        )
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        identity = "\n".join([owner_id, query_hash, *sorted(revisions)])
        return RetrievalSnapshotV1(
            snapshot_id="fixture-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            owner_id=owner_id,
            corpus_revisions=tuple(sorted(revisions)),
            query_sha256=query_hash,
            citations=citations,
            created_at_utc=self._clock,
        )


class FixtureArtifactStore:
    def __init__(self) -> None:
        self._records: dict[str, ArtifactV1] = {}
        self._content: dict[str, bytes] = {}

    def put(self, artifact: ArtifactV1, content: bytes) -> ArtifactV1:
        if hashlib.sha256(content).hexdigest() != artifact.content_sha256:
            raise FixtureAccessError("artifact_hash_mismatch")
        existing = self._records.get(artifact.artifact_id)
        if existing is not None and existing != artifact:
            raise FixtureAccessError("artifact_identity_conflict")
        self._records[artifact.artifact_id] = artifact
        self._content[artifact.artifact_id] = bytes(content)
        return artifact

    def read(self, owner_id: str, artifact_id: str) -> bytes:
        record = self._records.get(artifact_id)
        if record is None or record.owner_id != owner_id:
            raise FixtureAccessError("artifact_not_found")
        return self._content[artifact_id]


class FixtureProvenanceStore:
    def __init__(self) -> None:
        self._records: list[ProvenanceV1] = []

    def append(self, record: ProvenanceV1) -> ProvenanceV1:
        if any(item.provenance_id == record.provenance_id for item in self._records):
            raise FixtureAccessError("provenance_exists")
        self._records.append(record)
        return record

    def lineage(self, subject_id: str) -> tuple[ProvenanceV1, ...]:
        return tuple(
            sorted(
                (item for item in self._records if item.subject_id == subject_id),
                key=lambda item: (item.created_at_utc, item.provenance_id),
            )
        )


class FixtureRepositoryService:
    def __init__(self, grants: Sequence[RepositoryGrantV1] = ()) -> None:
        self._grants = list(grants)

    def grants(self, user_id: str) -> tuple[RepositoryGrantV1, ...]:
        return tuple(
            sorted(
                (item for item in self._grants if item.user_id == user_id),
                key=lambda item: (item.repository_id, item.revision),
            )
        )

    def require_active(self, user_id: str, repository_id: str) -> RepositoryGrantV1:
        matches = [
            item
            for item in self._grants
            if item.user_id == user_id and item.repository_id == repository_id and item.active
        ]
        if not matches:
            raise FixtureAccessError("repository_not_authorized")
        return max(matches, key=lambda item: item.revision)


class FixtureWorktreeService:
    def __init__(self) -> None:
        self._worktrees: dict[str, WorktreeV1] = {}

    def create(self, worktree: WorktreeV1) -> WorktreeV1:
        existing = self._worktrees.get(worktree.worktree_id)
        if existing is not None and existing != worktree:
            raise FixtureAccessError("worktree_identity_conflict")
        self._worktrees[worktree.worktree_id] = worktree
        return worktree

    def get(self, owner_id: str, worktree_id: str) -> WorktreeV1:
        worktree = self._worktrees.get(worktree_id)
        if worktree is None or worktree.owner_id != owner_id:
            raise FixtureAccessError("worktree_not_found")
        return worktree

    def list_for_owner(self, owner_id: str) -> tuple[WorktreeV1, ...]:
        return tuple(
            sorted(
                (item for item in self._worktrees.values() if item.owner_id == owner_id),
                key=lambda item: (item.created_at_utc, item.worktree_id),
            )
        )


class FixtureIdentityService:
    def __init__(self, revisions: Mapping[str, int]) -> None:
        self._revisions = dict(revisions)

    def enabled(self, user_id: str) -> bool:
        return user_id in self._revisions

    def revision(self, user_id: str) -> int:
        if user_id not in self._revisions:
            raise FixtureAccessError("identity_disabled")
        return self._revisions[user_id]


class FixtureCapabilityService:
    def __init__(self, assignments: Sequence[CapabilityAssignmentV1]) -> None:
        self._assignments = {item.user_id: item for item in assignments}

    def assignment(self, user_id: str) -> CapabilityAssignmentV1:
        assignment = self._assignments.get(user_id)
        if assignment is None or not assignment.enabled:
            raise FixtureAccessError("identity_disabled")
        return assignment

    def require(self, user_id: str, capability: str) -> CapabilityAssignmentV1:
        assignment = self.assignment(user_id)
        if capability not in assignment.capabilities:
            raise FixtureAccessError("capability_required")
        return assignment


class FixtureJobService:
    def __init__(self) -> None:
        self._jobs: dict[str, JobV1] = {}

    def create(self, job: JobV1) -> JobV1:
        existing = self._jobs.get(job.job_id)
        if existing is not None and existing != job:
            raise FixtureAccessError("job_identity_conflict")
        self._jobs[job.job_id] = job
        return job

    def get(self, owner_id: str, job_id: str) -> JobV1:
        job = self._jobs.get(job_id)
        if job is None or job.owner_id != owner_id:
            raise FixtureAccessError("job_not_found")
        return job


class FixtureAuditService:
    def __init__(self) -> None:
        self._events: list[AuditEventV1] = []

    def append(self, event: AuditEventV1) -> AuditEventV1:
        if any(item.event_id == event.event_id for item in self._events):
            raise FixtureAccessError("audit_event_exists")
        self._events.append(event)
        return event

    def for_subject(self, subject_id: str) -> tuple[AuditEventV1, ...]:
        return tuple(
            sorted(
                (item for item in self._events if item.subject_id == subject_id),
                key=lambda item: (item.occurred_at_utc, item.event_id),
            )
        )


class FixtureConfigurationService:
    def __init__(
        self,
        value: Mapping[str, object],
        *,
        sanitized: Mapping[str, object],
        provenance: Mapping[str, str],
    ) -> None:
        self._value = dict(value)
        self._sanitized = dict(sanitized)
        self._provenance = dict(provenance)

    def effective(self) -> Mapping[str, object]:
        return dict(self._value)

    def sanitized(self) -> Mapping[str, object]:
        return dict(self._sanitized)

    def provenance(self) -> Mapping[str, str]:
        return dict(self._provenance)
