#!/usr/bin/env python3
"""Run the bounded web-enabled Research Plane certification smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research_workspace.research_models import (
    DirectUrlResearchAdapter,
    DiscoveredSource,
    ResearchJobRequest,
)
from research_workspace.research_plane import DeepResearchService
from research_workspace.research_web_adapters import (
    BoundedWebClient,
    GitHubRepositoryInspectionAdapter,
    GitHubRepositorySpec,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    arguments = parser.parse_args()
    official = DiscoveredSource(
        canonical_url="https://opentelemetry.io/docs/specs/otel/overview/",
        title="OpenTelemetry specification overview",
        backend="direct_url",
        query="",
        source_type="official_documentation",
        license="Apache-2.0",
        publication="OpenTelemetry",
    )
    adapters = {
        "direct_url": DirectUrlResearchAdapter(
            [official],
            timeout_seconds=20,
            maximum_bytes=4_000_000,
        ),
        "github_repository_inspection": GitHubRepositoryInspectionAdapter(
            [
                GitHubRepositorySpec(
                    repository_url="https://github.com/open-telemetry/opentelemetry-python",
                    revision="4ea521a73c0f7ad67b7c04dff1e1dc6d00324a69",
                    title="OpenTelemetry Python repository",
                    license="Apache-2.0",
                )
            ],
            client=BoundedWebClient(
                timeout_seconds=20,
                maximum_bytes=4_000_000,
                minimum_interval_seconds=1.0,
            ),
        ),
    }
    service = DeepResearchService(arguments.state_root, adapters)
    request = ResearchJobRequest(
        question=(
            "What official and repository evidence supports local append-only "
            "OpenTelemetry-compatible execution traces?"
        ),
        scope="OpenTelemetry trace structure and local evidence recording only",
        research_mode="quick",
        search_backends=["direct_url", "github_repository_inspection"],
        source_policy="official_and_exact_revision_repository_required",
        model_route="deterministic",
    )
    service.create(request, job_id=arguments.job_id)
    result = service.run(arguments.job_id)
    job = service.get(arguments.job_id)
    root = service.layout.research_jobs / arguments.job_id
    official_present = any(
        source.source_type in {"primary_source", "official_documentation"}
        for source in job.source_records
    )
    repository_or_paper_present = any(
        source.source_type
        in {"repository", "peer_reviewed_literature", "preprint"}
        for source in job.source_records
    )
    snapshots_valid = all(
        hashlib.sha256((root / source.local_snapshot_path).read_bytes()).hexdigest()
        == source.content_sha256
        for source in job.source_records
    )
    quality_labeled = all(
        source.quality_score > 0 and source.relevance_score >= 0
        for source in job.source_records
    )
    compact = root / f"{arguments.job_id}_compact.zip"
    acceptance = {
        "complete": result.get("status") in {"COMPLETE", "IDEMPOTENT_TERMINAL"},
        "official_or_primary_source": official_present,
        "repository_or_paper_source": repository_or_paper_present,
        "source_snapshots_hash_valid": snapshots_valid,
        "quality_labels_present": quality_labeled,
        "evidence_ledger_present": (root / "evidence_ledger.json").is_file(),
        "citation_validation_present": (root / "citation_validation.json").is_file(),
        "compact_export_present": compact.is_file(),
    }
    evidence = {
        "schema_version": 1,
        "status": "PASS" if all(acceptance.values()) else "FAILED",
        "research_job_id": arguments.job_id,
        "acceptance": acceptance,
        "result": result,
        "source_records": [
            source.model_dump(mode="json") for source in job.source_records
        ],
    }
    (root / "live_research_smoke.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
