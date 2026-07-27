"""CLI for resumable exploratory research and controlled corpus promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, cast

from pydantic import ValidationError

from .operator_service import OperatorService
from .research_models import (
    DirectUrlResearchAdapter,
    DiscoveredSource,
    LocalDirectoryResearchAdapter,
    ResearchAdapter,
    ResearchJobRequest,
    SourceType,
)
from .research_plane import (
    DeepResearchService,
    GovernedCorpusPromoter,
    ResearchPlaneError,
)


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchPlaneError(
            "invalid_research_input",
            {"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not isinstance(value, dict):
        raise ResearchPlaneError(
            "invalid_research_input", {"path": str(path), "reason": "expected object"}
        )
    return dict(value)


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _adapters(config_path: Path) -> dict[str, ResearchAdapter]:
    raw = _object(config_path)
    adapters: dict[str, ResearchAdapter] = {}
    source_types = {
        "primary_source",
        "official_documentation",
        "peer_reviewed_literature",
        "preprint",
        "repository",
        "secondary_commentary",
        "local_document",
    }
    for name, config in raw.items():
        if not isinstance(config, dict):
            raise ResearchPlaneError(
                "invalid_research_input", {"reason": f"adapter {name} must be an object"}
            )
        if name in {"local_governed_corpus", "local_uploaded_documents"}:
            root = config.get("root")
            if not isinstance(root, str) or not Path(root).is_absolute():
                raise ResearchPlaneError(
                    "invalid_research_input",
                    {"reason": f"adapter {name} requires an absolute root"},
                )
            adapters[name] = LocalDirectoryResearchAdapter(
                Path(root),
                name=name,
                source_type=(
                    "local_document"
                    if name == "local_uploaded_documents"
                    else "official_documentation"
                ),
            )
            continue
        if name != "direct_url" or not isinstance(config.get("sources"), list):
            raise ResearchPlaneError(
                "invalid_research_input", {"reason": f"unsupported adapter {name}"}
            )
        sources: list[DiscoveredSource] = []
        for item in config["sources"]:
            if not isinstance(item, dict) or item.get("source_type") not in source_types:
                raise ResearchPlaneError(
                    "invalid_research_input",
                    {"reason": "direct source has an invalid source type"},
                )
            authors = item.get("authors")
            sources.append(
                DiscoveredSource(
                    canonical_url=str(item["canonical_url"]),
                    title=str(item["title"]),
                    backend=name,
                    query="",
                    source_type=cast(SourceType, item["source_type"]),
                    license=str(item.get("license", "UNKNOWN")),
                    authors=(
                        tuple(str(author) for author in authors)
                        if isinstance(authors, list)
                        else ()
                    ),
                    publication=_optional_text(item.get("publication")),
                    publication_date=_optional_text(item.get("publication_date")),
                    revision=_optional_text(item.get("revision")),
                )
            )
        adapters[name] = DirectUrlResearchAdapter(sources)
    return adapters


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="research")
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=repository_root / "outputs/operator_plane",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--request", type=Path, required=True)
    create.add_argument("--adapters", type=Path, required=True)
    create.add_argument("--job")
    run = commands.add_parser("run")
    run.add_argument("--job", required=True)
    run.add_argument("--adapters", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--job", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--job", required=True)
    promote.add_argument("--source", required=True)
    promote.add_argument("--target-domain", required=True)
    promote.add_argument("--approval-id", required=True)
    promote.add_argument("--permitted-use", required=True)
    promote.add_argument("--topic-metadata", type=Path, required=True)
    promote.add_argument("--curated-summary", type=Path, required=True)
    promote.add_argument(
        "--summary-origin",
        choices=("human_curated", "independently_authored"),
        required=True,
    )
    promote.add_argument("--relevance-tests", type=Path, required=True)
    promote.add_argument("--approved-by", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command in {"create", "run"}:
            service = DeepResearchService(
                arguments.state_root,
                _adapters(arguments.adapters),
            )
            if arguments.command == "create":
                request = ResearchJobRequest.model_validate(
                    _object(arguments.request)
                )
                result: object = service.create(request, job_id=arguments.job)
            else:
                result = service.run(arguments.job)
        elif arguments.command == "status":
            service = DeepResearchService(arguments.state_root, {})
            result = service.get(arguments.job).model_dump(mode="json")
        else:
            operator = OperatorService(
                arguments.repository_root, arguments.state_root
            )
            topic_metadata = _object(arguments.topic_metadata)
            relevance_raw = _object(arguments.relevance_tests)
            tests = relevance_raw.get("tests")
            if not isinstance(tests, list):
                raise ResearchPlaneError(
                    "invalid_research_input",
                    {"reason": "relevance tests must contain a tests list"},
                )
            promoter = GovernedCorpusPromoter(
                arguments.state_root / "research",
                arguments.state_root / "stores/governed_corpus",
                lambda approval_id, action, entity_id: operator.approval_is_valid(
                    approval_id,
                    action,
                    entity_id,
                    actor_role="read",
                ),
            )
            result = promoter.promote(
                job_id=arguments.job,
                source_id=arguments.source,
                target_domain=arguments.target_domain,
                approval_id=arguments.approval_id,
                permitted_use=arguments.permitted_use,
                topic_metadata=topic_metadata,
                curated_summary=arguments.curated_summary.read_text(encoding="utf-8"),
                summary_origin=arguments.summary_origin,
                relevance_tests=[
                    dict(item) for item in tests if isinstance(item, dict)
                ],
                approved_by=arguments.approved_by,
            )
    except (ResearchPlaneError, ValidationError) as exc:
        if isinstance(exc, ResearchPlaneError):
            category = exc.category
            evidence: object = exc.evidence
        else:
            category = "invalid_research_input"
            evidence = {"validation_errors": exc.errors(include_url=False)}
        _emit(
            {
                "status": "ERROR",
                "failure_category": category,
                "evidence": evidence,
            }
        )
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

