#!/usr/bin/env python3
"""Preserve fixture research, contradiction, and controlled-promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from research_workspace.operator_service import OperatorService
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
    ResearchPlaneError,
)


def _source(
    url: str,
    content: bytes,
    *,
    stance: Literal["supports", "contradicts"],
    title: str,
) -> FetchedSource:
    return FetchedSource(
        discovered=DiscoveredSource(
            canonical_url=url,
            title=title,
            backend="fixture",
            query="",
            source_type="official_documentation",
            license="Apache-2.0",
        ),
        content=content,
        assertions=(
            ClaimAssertion(
                normalized_claim=(
                    "A two-entry elastic buffer preserves ordered payloads "
                    "under backpressure."
                ),
                claim_key="elastic-buffer-ordering",
                stance=stance,
                confidence=0.9,
            ),
        ),
        retrieved_at="2026-07-27T00:00:00+00:00",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--measured-run-lock", type=Path, required=True)
    arguments = parser.parse_args()
    measured_lock = arguments.measured_run_lock.resolve()
    before_lock_sha256 = hashlib.sha256(measured_lock.read_bytes()).hexdigest()
    supporting_content = (
        b"Elastic buffer evidence: ordered payloads survive randomized backpressure."
    )
    sources = [
        _source(
            "https://fixture.example/elastic-buffer",
            supporting_content,
            stance="supports",
            title="Elastic buffer fixture evidence",
        ),
        _source(
            "https://fixture.example/elastic-buffer-duplicate",
            supporting_content,
            stance="supports",
            title="Duplicate elastic buffer evidence",
        ),
        _source(
            "https://fixture.example/elastic-buffer-limit",
            b"A deliberately contradictory fixture claims ordering is not preserved.",
            stance="contradicts",
            title="Elastic buffer contradiction fixture",
        ),
    ]
    service = DeepResearchService(
        arguments.state_root,
        {"fixture": FixtureResearchAdapter(sources)},
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )
    request = ResearchJobRequest(
        question="Does the two-entry elastic buffer preserve ordering under backpressure?",
        scope="deterministic fixture and controlled promotion",
        research_mode="standard",
        search_backends=["fixture"],
        source_policy="fixture_with_explicit_contradiction",
        model_route="deterministic",
    )
    service.create(request, job_id=arguments.job_id)
    result = service.run(arguments.job_id)
    root = service.layout.research_jobs / arguments.job_id
    ledger = json.loads((root / "evidence_ledger.json").read_text(encoding="utf-8"))
    source_ids = [str(item["source_id"]) for item in ledger["sources"]]
    operator = OperatorService(arguments.repository_root, arguments.state_root)
    promoter = GovernedCorpusPromoter(
        service.layout.research_jobs,
        service.layout.governed_corpus,
        lambda approval_id, action, entity: operator.approval_is_valid(
            approval_id, action, entity, actor_role="read"
        ),
    )

    def promote(source_id: str, approval_id: str) -> dict[str, object]:
        return promoter.promote(
            job_id=arguments.job_id,
            source_id=source_id,
            target_domain="SystemVerilogFixture",
            approval_id=approval_id,
            permitted_use="Local research and citation under Apache-2.0",
            topic_metadata={"topic": "elastic buffer backpressure"},
            curated_summary=(
                "The fixture discusses two-entry elastic-buffer ordering "
                "under backpressure."
            ),
            summary_origin="human_curated",
            relevance_tests=[
                {
                    "query": "elastic buffer backpressure",
                    "expected_terms": ["elastic", "backpressure"],
                }
            ],
            approved_by="local-certification-admin",
        )

    approval_required = False
    try:
        promote(source_ids[0], "missing-approval")
    except ResearchPlaneError as exc:
        approval_required = exc.category == "approval_required"
    seed_request = operator.request_approval(
        "PROMOTE_RESEARCH_SOURCE",
        f"{arguments.job_id}:{source_ids[0]}",
        {"source_id": source_ids[0]},
        actor_role="operate",
    )
    seed_approval = str(seed_request["approval_id"])
    operator.decide_approval(seed_approval, approve=True, actor_role="admin")
    first = promote(source_ids[0], seed_approval)
    controlled_request = operator.request_approval(
        "PROMOTE_RESEARCH_SOURCE",
        f"{arguments.job_id}:{source_ids[1]}",
        {"source_id": source_ids[1]},
        actor_role="operate",
    )
    controlled_approval = str(controlled_request["approval_id"])
    operator.decide_approval(controlled_approval, approve=True, actor_role="admin")
    second = promote(source_ids[1], controlled_approval)
    after_lock_sha256 = hashlib.sha256(measured_lock.read_bytes()).hexdigest()
    deduplicated = json.loads(
        (root / "deduplicated_sources.json").read_text(encoding="utf-8")
    )
    acceptance = {
        "complete": result.get("status") in {"COMPLETE", "IDEMPOTENT_TERMINAL"},
        "question_decomposition_recorded": (root / "subquestions.json").is_file(),
        "source_discovery_recorded": (root / "discovered_sources.json").is_file(),
        "duplicate_sources_removed": (
            deduplicated["discovered_count"] > deduplicated["retained_count"]
        ),
        "claims_linked_to_sources": all(
            item["supporting_source_ids"] or item["contradicting_source_ids"]
            for item in ledger["claims"]
        ),
        "contradiction_represented": any(
            item["contradicting_source_ids"] for item in ledger["claims"]
        ),
        "citations_resolve": json.loads(
            (root / "citation_validation.json").read_text(encoding="utf-8")
        )["status"]
        == "PASS",
        "reports_present": (
            (root / "report.md").is_file() and (root / "report.html").is_file()
        ),
        "research_lock_present": (root / "research.lock.json").is_file(),
        "approval_required": approval_required,
        "license_metadata_required_and_present": all(
            item["license"] == "Apache-2.0" for item in ledger["sources"]
        ),
        "new_corpus_snapshot_generated": (
            first["new_snapshot_sha256"] != second["new_snapshot_sha256"]
        ),
        "old_snapshot_remains_readable": second["old_snapshot_remains_readable"] is True,
        "measured_run_lock_unchanged": before_lock_sha256 == after_lock_sha256,
    }
    evidence = {
        "schema_version": 1,
        "status": "PASS" if all(acceptance.values()) else "FAILED",
        "research_job_id": arguments.job_id,
        "acceptance": acceptance,
        "research_result": result,
        "first_promotion": first,
        "controlled_promotion": second,
        "measured_run_lock_sha256_before": before_lock_sha256,
        "measured_run_lock_sha256_after": after_lock_sha256,
    }
    (root / "fixture_research_smoke.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
