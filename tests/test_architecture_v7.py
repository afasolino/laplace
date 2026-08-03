from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest

from research_workspace.architecture import (
    AuditService,
    CapabilityService,
    ConfigurationService,
    ConversationStore,
    CorpusStore,
    ModelProvider,
    public_provider_catalog,
)
from research_workspace.configuration import (
    ConfigurationV7Error,
    load_configuration,
    write_diagnostic_export,
)
from research_workspace.contracts import (
    ArtifactV1,
    AuditEventV1,
    CapabilityAssignmentV1,
    ConversationV1,
    CorpusSourceV1,
    CorpusV1,
    MessageV1,
    ModelRequestV1,
    ProvenanceV1,
    RepositoryGrantV1,
    RouteV1,
    WorktreeV1,
)
from research_workspace.fixture_services import (
    FixtureAccessError,
    FixtureArtifactStore,
    FixtureAuditService,
    FixtureCapabilityService,
    FixtureConfigurationService,
    FixtureConversationStore,
    FixtureCorpusStore,
    FixtureProvenanceStore,
    FixtureRepositoryService,
    FixtureRetrievalService,
    FixtureWorktreeService,
)
from research_workspace.presentation import (
    NavigationItemV1,
    ProgressPresentationV1,
    capability_navigation,
)
from research_workspace.providers import FixtureModelProvider

NOW = "2026-07-28T00:00:00+00:00"


def _message(message_id: str, conversation_id: str, content: str) -> MessageV1:
    return MessageV1(
        message_id=message_id,
        conversation_id=conversation_id,
        role="user",
        content=content,
        created_at_utc=NOW,
    )


def test_strict_contracts_and_fixture_conversation_owner_boundary() -> None:
    store = FixtureConversationStore()
    assert isinstance(store, ConversationStore)
    conversation = store.create(
        ConversationV1(
            conversation_id="conv-1",
            owner_id="owner-a",
            operating_mode="server",
            title="Fixture",
            created_at_utc=NOW,
            updated_at_utc=NOW,
        )
    )
    store.append(_message("msg-1", conversation.conversation_id, "hello"))
    assert store.messages("owner-a", "conv-1")[0].content == "hello"
    with pytest.raises(FixtureAccessError, match="conversation_not_found"):
        store.messages("owner-b", "conv-1")
    with pytest.raises(ValueError):
        ConversationV1.model_validate(
            {
                **conversation.model_dump(mode="json"),
                "unknown": True,
            }
        )


def test_complete_fixture_services_preserve_owners_and_provenance() -> None:
    corpus_store = FixtureCorpusStore()
    assert isinstance(corpus_store, CorpusStore)
    corpus_store.put_corpus(
        CorpusV1(
            corpus_id="pc-a",
            owner_id="owner-a",
            corpus_class="personal",
            display_name="Notes",
            snapshot_revision="rev-1",
            indexing_state="READY",
        )
    )
    content = "Laplace fixture retrieval evidence"
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    source = corpus_store.put_source(
        "owner-a",
        CorpusSourceV1(
            source_id="source-a",
            corpus_id="pc-a",
            logical_name="notes.md",
            source_class="user_work",
            original_sha256=content_hash,
            extracted_sha256=content_hash,
            chunk_count=1,
            indexed_at_utc=NOW,
        ),
    )
    retrieval = FixtureRetrievalService(
        corpus_store,
        source_text={source.source_id: content},
        clock=NOW,
    )
    snapshot = retrieval.retrieve(
        owner_id="owner-a",
        query="fixture evidence",
        corpus_ids=["pc-a"],
        limit=3,
    )
    assert snapshot.citations[0].source_id == "source-a"
    with pytest.raises(FixtureAccessError, match="corpus_not_found"):
        retrieval.retrieve(
            owner_id="owner-b",
            query="fixture",
            corpus_ids=["pc-a"],
            limit=1,
        )

    artifact_content = b"result"
    artifact = ArtifactV1(
        artifact_id="artifact-a",
        owner_id="owner-a",
        logical_name="result.txt",
        content_sha256=hashlib.sha256(artifact_content).hexdigest(),
        media_type="text/plain",
        size_bytes=len(artifact_content),
        retention_class="fixture",
        created_at_utc=NOW,
    )
    artifacts = FixtureArtifactStore()
    artifacts.put(artifact, artifact_content)
    assert artifacts.read("owner-a", artifact.artifact_id) == artifact_content
    with pytest.raises(FixtureAccessError, match="artifact_not_found"):
        artifacts.read("owner-b", artifact.artifact_id)

    provenance = FixtureProvenanceStore()
    record = provenance.append(
        ProvenanceV1(
            provenance_id="prov-1",
            subject_type="artifact",
            subject_id=artifact.artifact_id,
            action="created",
            actor_id="owner-a",
            content_sha256=artifact.content_sha256,
            created_at_utc=NOW,
        )
    )
    assert provenance.lineage(artifact.artifact_id) == (record,)


def test_fixture_authorization_worktree_and_audit_protocols() -> None:
    assignment = CapabilityAssignmentV1(
        user_id="owner-a",
        capabilities=("chat", "agent"),
        revision=2,
        enabled=True,
        updated_at_utc=NOW,
    )
    capabilities = FixtureCapabilityService([assignment])
    assert isinstance(capabilities, CapabilityService)
    assert capabilities.require("owner-a", "agent") == assignment
    with pytest.raises(FixtureAccessError, match="capability_required"):
        capabilities.require("owner-a", "admin")

    grant = RepositoryGrantV1(
        grant_id="grant-1",
        user_id="owner-a",
        repository_id="repo-a",
        revision=1,
        active=True,
        created_at_utc=NOW,
    )
    repositories = FixtureRepositoryService([grant])
    assert repositories.require_active("owner-a", "repo-a") == grant
    worktrees = FixtureWorktreeService()
    record = worktrees.create(
        WorktreeV1(
            worktree_id="wt-1",
            owner_id="owner-a",
            repository_id="repo-a",
            base_commit="a" * 40,
            state="ACTIVE",
            changed_paths=("src/module.py",),
            created_at_utc=NOW,
            updated_at_utc=NOW,
        )
    )
    assert worktrees.get("owner-a", "wt-1") == record
    with pytest.raises(FixtureAccessError, match="worktree_not_found"):
        worktrees.get("owner-b", "wt-1")
    with pytest.raises(ValueError):
        WorktreeV1(
            worktree_id="wt-bad",
            owner_id="owner-a",
            repository_id="repo-a",
            base_commit="b" * 40,
            state="ACTIVE",
            changed_paths=("../escape",),
            created_at_utc=NOW,
            updated_at_utc=NOW,
        )

    audit = FixtureAuditService()
    assert isinstance(audit, AuditService)
    event = audit.append(
        AuditEventV1(
            event_id="event-1",
            occurred_at_utc=NOW,
            actor_id="owner-a",
            action="worktree.create",
            subject_type="worktree",
            subject_id="wt-1",
            outcome="completed",
            trace_id="trace-1",
        )
    )
    assert audit.for_subject("wt-1") == (event,)


def test_fixture_provider_is_deterministic_and_catalog_hides_endpoint() -> None:
    provider = FixtureModelProvider({"request-1": "known response"})
    assert isinstance(provider, ModelProvider)
    conversation = ConversationV1(
        conversation_id="conv-1",
        owner_id="owner-a",
        operating_mode="desktop",
        title="Fixture",
        created_at_utc=NOW,
        updated_at_utc=NOW,
    )
    request = ModelRequestV1(
        request_id="request-1",
        route_id="fixture",
        messages=(_message("msg-1", conversation.conversation_id, "hello"),),
        max_output_tokens=128,
        temperature=0.0,
    )
    first = asyncio.run(provider.generate(request))
    second = asyncio.run(provider.generate(request))
    assert first == second
    assert first.fixture is True
    catalog = public_provider_catalog(
        [provider],
        [
            RouteV1(
                route_id="fixture",
                display_name="Fixture",
                provider_id="fixture",
                model_id="fixture-model-v1",
                lane="fixture",
                enabled=True,
            )
        ],
    )
    assert "endpoint" not in catalog["providers"][0]  # type: ignore[index]
    assert catalog["providers"][0]["endpoint_scope"] == "fixture"  # type: ignore[index]


def test_configuration_precedence_unknown_keys_and_redaction(tmp_path: Path) -> None:
    repository = tmp_path / "repository.yaml"
    repository.write_text(
        "schema_version: 1\nlogging:\n  level: WARNING\n",
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment.yaml"
    deployment.write_text(
        "schema_version: 1\noperating_mode: server\nstorage:\n  state_root: state/server\n",
        encoding="utf-8",
    )
    deployment.chmod(0o600)
    configuration = load_configuration(
        repository_configuration=repository,
        deployment_configuration=deployment,
        environment={"LAPLACE_CONFIG_LOG_LEVEL": "ERROR"},
        cli_overrides={"logging": {"level": "DEBUG"}},
    )
    assert isinstance(configuration, ConfigurationService)
    assert configuration.configuration.operating_mode == "server"
    assert configuration.configuration.logging.level == "DEBUG"
    assert configuration.provenance()["logging.level"] == "explicit_cli_argument"
    diagnostic = configuration.diagnostic()
    serialized = str(diagnostic)
    assert "state/server" not in serialized
    assert "LAPLACE_SESSION_KEY" not in serialized
    target = write_diagnostic_export(tmp_path / "diagnostic.json", configuration)
    if os.name != "nt":
        assert target.stat().st_mode & 0o077 == 0

    with pytest.raises(ConfigurationV7Error, match="unknown_environment_override"):
        load_configuration(environment={"LAPLACE_CONFIG_UNSAFE": "yes"})
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ConfigurationV7Error, match="configuration_schema_invalid"):
        load_configuration(repository_configuration=invalid, environment={})


def test_fixture_configuration_and_shared_presentation_contracts() -> None:
    fixture = FixtureConfigurationService(
        {"mode": "fixture"},
        sanitized={"mode": "fixture"},
        provenance={"mode": "fixture"},
    )
    assert isinstance(fixture, ConfigurationService)
    assert fixture.sanitized()["mode"] == "fixture"
    definitions = (
        NavigationItemV1(item_id="chat", label="Chat", required_capability="chat"),
        NavigationItemV1(
            item_id="users",
            label="Users",
            required_capability="admin",
            mode="server",
        ),
    )
    assert [item.item_id for item in capability_navigation(
        definitions, capabilities=("chat",), mode="server"
    )] == ["chat"]
    progress = ProgressPresentationV1(
        state="GENERATING",
        elapsed_seconds=1.5,
        route_id="fixture",
        model_display_name="Fixture",
        trace_id="trace-1",
    )
    assert progress.percent_complete is None
    assert progress.private_reasoning is None
