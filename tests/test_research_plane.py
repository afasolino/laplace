from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from research_workspace.research_models import (
    ClaimAssertion,
    DiscoveredSource,
    FetchedSource,
    FixtureResearchAdapter,
    ResearchJobRequest,
)
from research_workspace.research_plane import (
    DeepResearchService,
    GovernedCorpusPromoter,
    RESEARCH_STAGES,
    ResearchPlaneError,
)


def _fixture_adapter() -> FixtureResearchAdapter:
    supporting = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://example.org/official",
            title="Official feature specification",
            backend="fixture",
            query="",
            source_type="official_documentation",
            license="CC-BY-4.0",
            authors=("Specification Group",),
            publication="Official Standards",
            publication_date="2026-01-01",
            revision="revision-1",
        ),
        content=b"The elastic buffer feature provides backpressure.",
        assertions=(
            ClaimAssertion(
                normalized_claim="The elastic buffer supports backpressure.",
                claim_key="elastic-buffer-backpressure",
                stance="supports",
                confidence=0.95,
            ),
        ),
        retrieved_at="2026-07-27T00:00:00+00:00",
    )
    duplicate = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://mirror.example.org/official-copy",
            title="Mirror of official specification",
            backend="fixture",
            query="",
            source_type="secondary_commentary",
            license="CC-BY-4.0",
        ),
        content=supporting.content,
        retrieved_at="2026-07-27T00:00:00+00:00",
    )
    contradicting = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://github.com/example/elastic-buffer",
            title="Elastic buffer implementation repository",
            backend="fixture",
            query="",
            source_type="repository",
            license="MIT",
            authors=("Example Authors",),
            revision="abc123",
        ),
        content=b"Repository notes describe a configuration without backpressure.",
        assertions=(
            ClaimAssertion(
                normalized_claim="The elastic buffer supports backpressure.",
                claim_key="elastic-buffer-backpressure",
                stance="contradicts",
                confidence=0.75,
            ),
        ),
        retrieved_at="2026-07-27T00:00:00+00:00",
    )
    return FixtureResearchAdapter([supporting, duplicate, contradicting])


def _request() -> ResearchJobRequest:
    return ResearchJobRequest(
        question="Does the elastic buffer support backpressure?",
        scope="deterministic fixture",
        research_mode="standard",
        search_backends=["fixture"],
        source_policy="primary_preferred",
        model_route="deterministic",
    )


def test_deterministic_research_job_builds_resolvable_evidence_ledger(
    tmp_path: Path,
) -> None:
    service = DeepResearchService(
        tmp_path,
        {"fixture": _fixture_adapter()},
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )
    service.create(_request(), job_id="research-fixture")
    result = service.run("research-fixture")
    root = tmp_path / "research/research-fixture"

    assert result["status"] == "COMPLETE"
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events[:-1]] == list(RESEARCH_STAGES)
    assert events[-1]["event_type"] == "research_complete"
    deduplicated = json.loads(
        (root / "deduplicated_sources.json").read_text(encoding="utf-8")
    )
    assert deduplicated["discovered_count"] > deduplicated["retained_count"]
    assert deduplicated["retained_count"] == 2

    ledger = json.loads(
        (root / "evidence_ledger.json").read_text(encoding="utf-8")
    )
    assert len(ledger["sources"]) == 2
    assert ledger["claims"][0]["status"] == "contested"
    assert ledger["claims"][0]["supporting_source_ids"]
    assert ledger["claims"][0]["contradicting_source_ids"]
    validation = json.loads(
        (root / "citation_validation.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["unresolved_citations"] == []
    assert (root / "report.md").is_file()
    assert (root / "report.html").is_file()
    assert (root / "research.lock.json").is_file()
    with zipfile.ZipFile(root / "research-fixture_compact.zip") as archive:
        assert "evidence_ledger.json" in archive.namelist()
        assert not any(name.startswith("sources/") for name in archive.namelist())

    first_lock = hashlib.sha256((root / "research.lock.json").read_bytes()).hexdigest()
    resumed = service.run("research-fixture")
    second_lock = hashlib.sha256((root / "research.lock.json").read_bytes()).hexdigest()
    assert resumed["status"] == "IDEMPOTENT_TERMINAL"
    assert first_lock == second_lock


def test_research_job_identity_conflict_is_structured(tmp_path: Path) -> None:
    service = DeepResearchService(tmp_path, {"fixture": _fixture_adapter()})
    service.create(_request(), job_id="research-fixture")
    changed = _request().model_copy(update={"question": "A different question?"})
    with pytest.raises(ResearchPlaneError) as caught:
        service.create(changed, job_id="research-fixture")
    assert caught.value.category == "research_job_identity_conflict"


def test_promotion_requires_approval_and_preserves_old_snapshot(
    tmp_path: Path,
) -> None:
    measured_lock = tmp_path / "measured-run/run.lock.json"
    measured_lock.parent.mkdir(parents=True)
    measured_lock.write_text('{"immutable":true}\n', encoding="utf-8")
    measured_lock_sha256 = hashlib.sha256(measured_lock.read_bytes()).hexdigest()
    service = DeepResearchService(
        tmp_path / "state",
        {"fixture": _fixture_adapter()},
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )
    service.create(_request(), job_id="research-fixture")
    service.run("research-fixture")
    ledger = json.loads(
        (
            tmp_path
            / "state/research/research-fixture/evidence_ledger.json"
        ).read_text(encoding="utf-8")
    )
    source_ids = [source["source_id"] for source in ledger["sources"]]
    approvals: set[str] = set()
    promoter = GovernedCorpusPromoter(
        tmp_path / "state/research",
        tmp_path / "governed",
        lambda approval_id, action, entity: (
            f"{approval_id}:{action}:{entity}" in approvals
        ),
    )
    kwargs = {
        "job_id": "research-fixture",
        "source_id": source_ids[0],
        "target_domain": "SystemVerilog",
        "approval_id": "approval-one",
        "permitted_use": "Local research and citation under the recorded licence",
        "topic_metadata": {"topic": "elastic buffer"},
        "curated_summary": "The elastic buffer evidence discusses backpressure.",
        "summary_origin": "human_curated",
        "relevance_tests": [
            {"query": "backpressure", "expected_terms": ["elastic", "backpressure"]}
        ],
        "approved_by": "fixture-approver",
    }
    with pytest.raises(ResearchPlaneError) as caught:
        promoter.promote(**kwargs)
    assert caught.value.category == "approval_required"

    approvals.add(
        f"approval-one:PROMOTE_RESEARCH_SOURCE:research-fixture:{source_ids[0]}"
    )
    first = promoter.promote(**kwargs)
    assert first["status"] == "PROMOTED"
    old_path = Path(str(first["snapshot_path"]))
    assert old_path.is_dir()

    second_kwargs = dict(kwargs)
    second_kwargs.update(
        {
            "source_id": source_ids[1],
            "approval_id": "approval-two",
            "curated_summary": (
                "The repository evidence discusses elastic buffer backpressure."
            ),
        }
    )
    approvals.add(
        f"approval-two:PROMOTE_RESEARCH_SOURCE:research-fixture:{source_ids[1]}"
    )
    second = promoter.promote(**second_kwargs)
    assert second["new_snapshot_sha256"] != first["new_snapshot_sha256"]
    assert second["previous_snapshot_sha256"] == first["new_snapshot_sha256"]
    assert second["old_snapshot_remains_readable"] is True
    assert old_path.is_dir()
    assert hashlib.sha256(measured_lock.read_bytes()).hexdigest() == measured_lock_sha256


def test_promotion_rejects_missing_license(tmp_path: Path) -> None:
    source = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://example.org/unknown",
            title="Unknown licence",
            backend="fixture",
            query="",
            source_type="official_documentation",
            license="UNKNOWN",
        ),
        content=b"Relevant content.",
        retrieved_at="2026-07-27T00:00:00+00:00",
    )
    service = DeepResearchService(
        tmp_path / "state",
        {"fixture": FixtureResearchAdapter([source])},
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )
    service.create(_request(), job_id="unknown-license")
    service.run("unknown-license")
    ledger = json.loads(
        (
            tmp_path / "state/research/unknown-license/evidence_ledger.json"
        ).read_text(encoding="utf-8")
    )
    source_id = ledger["sources"][0]["source_id"]
    promoter = GovernedCorpusPromoter(
        tmp_path / "state/research",
        tmp_path / "governed",
        lambda *_args: True,
    )
    with pytest.raises(ResearchPlaneError, match="promotion_validation_failure"):
        promoter.promote(
            job_id="unknown-license",
            source_id=source_id,
            target_domain="Test",
            approval_id="approved",
            permitted_use="Internal",
            topic_metadata={"topic": "test"},
            curated_summary="Relevant content.",
            summary_origin="human_curated",
            relevance_tests=[
                {"query": "relevant", "expected_terms": ["relevant"]}
            ],
            approved_by="approver",
        )
