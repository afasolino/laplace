"""Resumable, evidence-ledger-first exploratory research service."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import uuid
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeAlias

from .execution_records import (
    AppendOnlyEventLog,
    LocalTraceRecorder,
    canonical_json_bytes,
    canonical_sha256,
)
from .research_models import (
    ClaimAssertion,
    ClaimRecord,
    DiscoveredSource,
    ResearchAdapter,
    ResearchJobRecord,
    ResearchJobRequest,
    SourceRecord,
    assertions_from_json,
    assertions_to_json,
    canonical_source_key,
    canonicalize_url,
    discovered_from_json,
    discovered_to_json,
    source_content_sha256,
)

JsonObject: TypeAlias = dict[str, object]

RESEARCH_STAGES = (
    "question_normalization",
    "subquestion_decomposition",
    "search_plan",
    "source_discovery",
    "source_fetch",
    "source_deduplication",
    "source_quality_assessment",
    "claim_extraction",
    "contradiction_detection",
    "gap_analysis",
    "synthesis",
    "citation_validation",
    "report_generation",
)
_JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
_SOURCE_ID = re.compile(r"^src-[0-9a-f]{12}$")
_QUALITY = {
    "primary_source": 0.96,
    "official_documentation": 0.94,
    "peer_reviewed_literature": 0.91,
    "repository": 0.82,
    "preprint": 0.76,
    "local_document": 0.72,
    "secondary_commentary": 0.55,
}


class ResearchPlaneError(RuntimeError):
    """Research workflow evidence is missing, unsafe, or inconsistent."""

    def __init__(self, category: str, evidence: JsonObject) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )


def _jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(
            canonical_json_bytes(dict(value)).decode("utf-8") + "\n"
            for value in values
        ),
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchPlaneError(
            "research_artifact_invalid",
            {"path": str(path), "error_type": type(exc).__name__},
        ) from exc


def _relative_hashes(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _term_set(text: str) -> set[str]:
    return {
        value.lower()
        for value in re.findall(r"[A-Za-z0-9_]{3,}", text)
        if value.lower()
        not in {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "what",
            "which",
        }
    }


def _relevance(question: str, title: str, content: bytes) -> float:
    terms = _term_set(question)
    if not terms:
        return 0.0
    text = f"{title}\n{content[:100_000].decode('utf-8', errors='replace')}".lower()
    matched = sum(term in text for term in terms)
    return round(min(1.0, matched / len(terms)), 4)


def _markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


class ResearchStoreLayout:
    """Three physically separate stores; only exploratory is written by jobs."""

    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root.resolve()
        self.governed_corpus = self.state_root / "stores/governed_corpus"
        self.exploratory_research_store = (
            self.state_root / "stores/exploratory_research_store"
        )
        self.personal_workspace_store = (
            self.state_root / "stores/personal_workspace_store"
        )
        self.research_jobs = self.state_root / "research"

    def initialize(self) -> None:
        for path in (
            self.governed_corpus,
            self.exploratory_research_store,
            self.personal_workspace_store,
            self.research_jobs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            self.state_root / "stores/store_policy.json",
            {
                "schema_version": 1,
                "measured_run_readable_stores": ["governed_corpus"],
                "exploratory_research_readable_stores": [
                    "governed_corpus",
                    "exploratory_research_store",
                ],
                "personal_workspace_injection_into_measured_runs": False,
                "automatic_research_promotion": False,
            },
        )


class DeepResearchService:
    """Run the explicit research state machine and preserve all evidence."""

    def __init__(
        self,
        state_root: Path,
        adapters: Mapping[str, ResearchAdapter],
        *,
        clock: Callable[[], str] = _now,
    ) -> None:
        self.layout = ResearchStoreLayout(state_root)
        self.layout.initialize()
        self.adapters = dict(adapters)
        self.clock = clock

    def _job_root(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise ResearchPlaneError(
                "invalid_research_job_id", {"research_job_id": job_id}
            )
        return self.layout.research_jobs / job_id

    def create(
        self,
        request: ResearchJobRequest,
        *,
        job_id: str | None = None,
    ) -> JsonObject:
        missing = sorted(set(request.search_backends).difference(self.adapters))
        if missing:
            raise ResearchPlaneError(
                "research_backend_unavailable", {"missing_backends": missing}
            )
        selected_job_id = job_id or (
            f"research-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:10]}"
        )
        root = self._job_root(selected_job_id)
        request_object = request.model_dump(mode="json")
        request_sha256 = canonical_sha256(request_object)
        job_path = root / "job.json"
        if job_path.is_file():
            existing = ResearchJobRecord.model_validate(_read_json(job_path))
            if existing.request_sha256 != request_sha256:
                raise ResearchPlaneError(
                    "research_job_identity_conflict",
                    {
                        "research_job_id": selected_job_id,
                        "existing_request_sha256": existing.request_sha256,
                        "requested_request_sha256": request_sha256,
                    },
                )
            return {
                "status": "IDEMPOTENT_EXISTING_JOB",
                "research_job_id": selected_job_id,
                "job": existing.model_dump(mode="json"),
            }
        root.mkdir(parents=True, exist_ok=False)
        (root / "sources").mkdir()
        trace_id = uuid.uuid4().hex
        job = ResearchJobRecord(
            research_job_id=selected_job_id,
            question=request.question,
            scope=request.scope,
            research_mode=request.research_mode,
            search_backends=request.search_backends,
            source_policy=request.source_policy,
            model_route=request.model_route,
            created_at=self.clock(),
            status="CREATED",
            current_stage=None,
            completed_stages=[],
            subquestions=[],
            source_records=[],
            claims=[],
            contradictions=[],
            report_path=None,
            evidence_ledger_path=None,
            trace_id=trace_id,
            request_sha256=request_sha256,
        )
        _atomic_json(job_path, job.model_dump(mode="json"))
        _atomic_json(root / "request.json", request_object)
        return {
            "status": "CREATED",
            "research_job_id": selected_job_id,
            "job": job.model_dump(mode="json"),
        }

    def get(self, job_id: str) -> ResearchJobRecord:
        return ResearchJobRecord.model_validate(
            _read_json(self._job_root(job_id) / "job.json")
        )

    def _save_job(self, root: Path, job: ResearchJobRecord) -> None:
        _atomic_json(root / "job.json", job.model_dump(mode="json"))

    def _complete_stage(
        self,
        *,
        root: Path,
        job: ResearchJobRecord,
        stage: str,
        output_paths: Sequence[Path],
        event_log: AppendOnlyEventLog,
    ) -> ResearchJobRecord:
        # A stage may update compatibility projections in job.json. Reload
        # before recording its completion so those updates are not overwritten.
        job = ResearchJobRecord.model_validate(_read_json(root / "job.json"))
        artifact_hashes = _relative_hashes(root, output_paths)
        event_log.append(
            attempt=0,
            event_type=stage,
            from_state=job.current_stage,
            to_state=stage,
            source_state_fingerprint=None,
            payload={
                "stage": stage,
                "artifact_hashes": artifact_hashes,
                "artifact_count": len(artifact_hashes),
            },
        )
        completed = list(job.completed_stages)
        if stage not in completed:
            completed.append(stage)
        updated = job.model_copy(
            update={
                "status": "RUNNING",
                "current_stage": stage,
                "completed_stages": completed,
            }
        )
        self._save_job(root, updated)
        return updated

    def run(self, job_id: str) -> JsonObject:
        root = self._job_root(job_id)
        job = self.get(job_id)
        if job.status == "COMPLETE":
            return {
                "status": "IDEMPOTENT_TERMINAL",
                "research_job_id": job_id,
                "research_lock_path": str(root / "research.lock.json"),
                "report_path": job.report_path,
                "evidence_ledger_path": job.evidence_ledger_path,
            }
        event_log = AppendOnlyEventLog(
            root / "events.jsonl",
            run_id=job_id,
            task_id="research",
            arm_id=job.research_mode,
        )
        trace = LocalTraceRecorder(
            root / "traces.jsonl", trace_id=job.trace_id
        )
        for stage in RESEARCH_STAGES:
            if stage in job.completed_stages:
                continue
            try:
                with trace.span(
                    f"research_{stage}",
                    attributes={
                        "job_id_hash": hashlib.sha256(job_id.encode()).hexdigest(),
                        "stage_index": RESEARCH_STAGES.index(stage),
                    },
                ):
                    paths = self._execute_stage(stage, root, job)
                job = self._complete_stage(
                    root=root,
                    job=job,
                    stage=stage,
                    output_paths=paths,
                    event_log=event_log,
                )
            except Exception as exc:
                event_log.append(
                    attempt=0,
                    event_type=f"{stage}_failure",
                    from_state=job.current_stage,
                    to_state="FAILED",
                    source_state_fingerprint=None,
                    payload={
                        "stage": stage,
                        "failure_category": "research_stage_failure",
                        "error_type": type(exc).__name__,
                    },
                )
                failed = job.model_copy(
                    update={"status": "FAILED", "current_stage": stage}
                )
                self._save_job(root, failed)
                raise
        final = self.get(job_id).model_copy(
            update={
                "status": "COMPLETE",
                "current_stage": "report_generation",
                "report_path": str(root / "report.md"),
                "evidence_ledger_path": str(root / "evidence_ledger.json"),
            }
        )
        self._save_job(root, final)
        event_log.append(
            attempt=0,
            event_type="research_complete",
            from_state="report_generation",
            to_state="COMPLETE",
            source_state_fingerprint=None,
            payload={
                "research_lock_sha256": hashlib.sha256(
                    (root / "research.lock.json").read_bytes()
                ).hexdigest()
            },
        )
        return {
            "status": "COMPLETE",
            "research_job_id": job_id,
            "report_path": final.report_path,
            "report_html_path": str(root / "report.html"),
            "evidence_ledger_path": final.evidence_ledger_path,
            "research_lock_path": str(root / "research.lock.json"),
            "compact_export_path": str(root / f"{job_id}_compact.zip"),
            "trace_id": final.trace_id,
        }

    def _execute_stage(
        self, stage: str, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        handler = getattr(self, f"_stage_{stage}", None)
        if handler is None:
            raise ResearchPlaneError(
                "research_stage_missing", {"stage": stage}
            )
        result: object = handler(root, job)
        if not isinstance(result, list) or not all(
            isinstance(path, Path) for path in result
        ):
            raise ResearchPlaneError(
                "research_stage_invalid_result", {"stage": stage}
            )
        return result

    def _stage_question_normalization(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        normalized = re.sub(r"\s+", " ", job.question).strip()
        path = root / "question_normalized.json"
        _atomic_json(path, {"question": normalized})
        return [path]

    def _stage_subquestion_decomposition(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        raw = _read_json(root / "question_normalized.json")
        if not isinstance(raw, dict) or not isinstance(raw.get("question"), str):
            raise ResearchPlaneError("research_artifact_invalid", {"stage": "subquestions"})
        question = raw["question"]
        maximum = {
            "quick": 1,
            "standard": 3,
            "deep": 5,
            "systematic_literature": 5,
            "repository_audit": 4,
            "hardware_state_of_the_art": 5,
        }[job.research_mode]
        candidates = [question]
        if maximum >= 2:
            candidates.append(f"What primary or official evidence directly addresses: {question}")
        if maximum >= 3:
            candidates.append(f"What credible evidence contradicts or limits: {question}")
        if maximum >= 4:
            candidates.append(f"What implementation evidence is relevant to: {question}")
        if maximum >= 5:
            candidates.append(f"What remains unknown or weakly evidenced about: {question}")
        subquestions = candidates[:maximum]
        path = root / "subquestions.json"
        _atomic_json(path, {"subquestions": subquestions})
        updated = job.model_copy(update={"subquestions": subquestions})
        self._save_job(root, updated)
        return [path]

    def _stage_search_plan(self, root: Path, job: ResearchJobRecord) -> list[Path]:
        raw = _read_json(root / "subquestions.json")
        subquestions = raw.get("subquestions") if isinstance(raw, dict) else None
        if not isinstance(subquestions, list):
            raise ResearchPlaneError("research_artifact_invalid", {"stage": "search_plan"})
        plan = [
            {
                "query_id": f"query-{index:04d}",
                "query": question,
                "backend": backend,
                "limit": 8 if job.research_mode in {"deep", "systematic_literature"} else 5,
            }
            for index, (question, backend) in enumerate(
                (
                    (str(question), backend)
                    for question in subquestions
                    for backend in job.search_backends
                ),
                start=1,
            )
        ]
        path = root / "search_plan.json"
        _atomic_json(path, {"queries": plan})
        return [path]

    def _stage_source_discovery(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        raw = _read_json(root / "search_plan.json")
        plan = raw.get("queries") if isinstance(raw, dict) else None
        if not isinstance(plan, list):
            raise ResearchPlaneError(
                "research_artifact_invalid", {"stage": "source_discovery"}
            )
        discovered: list[dict[str, object]] = []
        queries: list[dict[str, object]] = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            backend = str(item["backend"])
            query = str(item["query"])
            limit = int(item["limit"])
            results = self.adapters[backend].discover(query, limit=limit)
            query_results = []
            for source in results:
                normalized = DiscoveredSource(
                    canonical_url=canonicalize_url(source.canonical_url),
                    title=source.title[:1_000],
                    backend=backend,
                    query=query,
                    source_type=source.source_type,
                    license=source.license[:300],
                    authors=source.authors[:100],
                    publication=source.publication,
                    publication_date=source.publication_date,
                    revision=source.revision,
                )
                record = discovered_to_json(normalized)
                record["discovery_key"] = canonical_source_key(normalized)
                discovered.append(record)
                query_results.append(
                    {
                        "returned_url": normalized.canonical_url,
                        "title": normalized.title,
                    }
                )
            queries.append(
                {
                    "query_id": item["query_id"],
                    "query": query,
                    "backend": backend,
                    "timestamp": self.clock(),
                    "results": query_results,
                }
            )
        discovered_path = root / "discovered_sources.json"
        queries_path = root / "search_queries.jsonl"
        _atomic_json(discovered_path, {"sources": discovered})
        _jsonl(queries_path, queries)
        return [discovered_path, queries_path]

    def _stage_source_fetch(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        del job
        raw = _read_json(root / "discovered_sources.json")
        discovered = raw.get("sources") if isinstance(raw, dict) else None
        if not isinstance(discovered, list):
            raise ResearchPlaneError(
                "research_artifact_invalid", {"stage": "source_fetch"}
            )
        fetched_records: list[dict[str, object]] = []
        output_paths: list[Path] = []
        for index, raw_source in enumerate(discovered, start=1):
            if not isinstance(raw_source, dict):
                continue
            source = discovered_from_json(raw_source)
            fetched = self.adapters[source.backend].fetch(source)
            digest = source_content_sha256(fetched)
            suffix = ".json" if "json" in fetched.content_type else ".txt"
            snapshot = root / "sources" / f"fetched-{index:04d}-{digest[:12]}{suffix}"
            if snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise ResearchPlaneError(
                    "research_snapshot_conflict", {"path": str(snapshot)}
                )
            if not snapshot.exists():
                _atomic_bytes(snapshot, fetched.content)
            fetched_records.append(
                {
                    "discovered": discovered_to_json(source),
                    "content_sha256": digest,
                    "content_type": fetched.content_type,
                    "local_snapshot_path": str(snapshot.relative_to(root)),
                    "retrieved_at": fetched.retrieved_at or self.clock(),
                    "assertions": assertions_to_json(fetched.assertions),
                }
            )
            output_paths.append(snapshot)
        fetched_path = root / "fetched_sources.json"
        _atomic_json(fetched_path, {"sources": fetched_records})
        output_paths.append(fetched_path)
        return output_paths

    def _stage_source_deduplication(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        del job
        raw = _read_json(root / "fetched_sources.json")
        fetched = raw.get("sources") if isinstance(raw, dict) else None
        if not isinstance(fetched, list):
            raise ResearchPlaneError(
                "research_artifact_invalid", {"stage": "source_deduplication"}
            )
        retained: dict[str, dict[str, object]] = {}
        content_seen: dict[str, str] = {}
        duplicates: list[dict[str, object]] = []
        for item in fetched:
            if not isinstance(item, dict) or not isinstance(item.get("discovered"), dict):
                continue
            source = discovered_from_json(item["discovered"])
            canonical = canonicalize_url(source.canonical_url)
            digest = str(item["content_sha256"])
            key = canonical
            duplicate_of = retained.get(key)
            if duplicate_of is None and digest in content_seen:
                duplicate_of = retained[content_seen[digest]]
            if duplicate_of is not None:
                queries = duplicate_of["discovery_queries"]
                if isinstance(queries, list) and source.query not in queries:
                    queries.append(source.query)
                duplicates.append(
                    {
                        "canonical_url": canonical,
                        "content_sha256": digest,
                        "duplicate_of_url": duplicate_of["canonical_url"],
                    }
                )
                continue
            source_id = "src-" + hashlib.sha256(
                f"{canonical}\0{digest}".encode("utf-8")
            ).hexdigest()[:12]
            record: dict[str, object] = {
                "source_id": source_id,
                "canonical_url": canonical,
                "title": source.title,
                "authors": list(source.authors),
                "publication": source.publication,
                "publication_date": source.publication_date,
                "retrieved_at": str(item["retrieved_at"]),
                "source_type": source.source_type,
                "license": source.license,
                "content_sha256": digest,
                "local_snapshot_path": str(item["local_snapshot_path"]),
                "quality_score": 0.0,
                "relevance_score": 0.0,
                "used_claim_ids": [],
                "backend": source.backend,
                "discovery_queries": [source.query],
                "revision": source.revision,
                "assertions": item.get("assertions", []),
            }
            retained[canonical] = record
            content_seen[digest] = canonical
        path = root / "deduplicated_sources.json"
        _atomic_json(
            path,
            {
                "sources": list(retained.values()),
                "duplicates_removed": duplicates,
                "discovered_count": len(fetched),
                "retained_count": len(retained),
            },
        )
        return [path]

    def _stage_source_quality_assessment(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        raw = _read_json(root / "deduplicated_sources.json")
        records = raw.get("sources") if isinstance(raw, dict) else None
        if not isinstance(records, list):
            raise ResearchPlaneError(
                "research_artifact_invalid", {"stage": "quality"}
            )
        assessed: list[dict[str, object]] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            snapshot = root / str(item["local_snapshot_path"])
            updated = dict(item)
            updated["quality_score"] = _QUALITY[str(item["source_type"])]
            updated["relevance_score"] = _relevance(
                job.question, str(item["title"]), snapshot.read_bytes()
            )
            assessed.append(updated)
        path = root / "assessed_sources.json"
        _atomic_json(path, {"sources": assessed})
        return [path]

    def _stage_claim_extraction(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        raw = _read_json(root / "assessed_sources.json")
        records = raw.get("sources") if isinstance(raw, dict) else None
        if not isinstance(records, list):
            raise ResearchPlaneError(
                "research_artifact_invalid", {"stage": "claim_extraction"}
            )
        aggregates: dict[str, dict[str, object]] = {}
        source_records: list[dict[str, object]] = []
        source_claims: dict[str, list[str]] = defaultdict(list)
        for source in records:
            if not isinstance(source, dict):
                continue
            source_id = str(source["source_id"])
            assertions = assertions_from_json(source.get("assertions"))
            if not assertions:
                assertions = (
                    ClaimAssertion(
                        normalized_claim=(
                            f'The source "{source["title"]}" was retrieved and '
                            "preserved in the evidence ledger."
                        ),
                        claim_key=f"source-preserved:{source_id}",
                        confidence=1.0,
                    ),
                )
            for assertion in assertions:
                key = assertion.claim_key or re.sub(
                    r"\s+", " ", assertion.normalized_claim
                ).strip().lower()
                aggregate = aggregates.setdefault(
                    key,
                    {
                        "normalized_claim": assertion.normalized_claim,
                        "supporting_source_ids": [],
                        "contradicting_source_ids": [],
                        "confidences": [],
                    },
                )
                target = (
                    "supporting_source_ids"
                    if assertion.stance == "supports"
                    else "contradicting_source_ids"
                )
                identifiers = aggregate[target]
                if isinstance(identifiers, list) and source_id not in identifiers:
                    identifiers.append(source_id)
                confidences = aggregate["confidences"]
                if isinstance(confidences, list):
                    confidences.append(assertion.confidence)
        claims: list[ClaimRecord] = []
        for key, aggregate in sorted(aggregates.items()):
            supporting_raw = aggregate["supporting_source_ids"]
            contradicting_raw = aggregate["contradicting_source_ids"]
            confidences_raw = aggregate["confidences"]
            if (
                not isinstance(supporting_raw, list)
                or not isinstance(contradicting_raw, list)
                or not isinstance(confidences_raw, list)
            ):
                raise ResearchPlaneError(
                    "research_artifact_invalid",
                    {"reason": "claim aggregate is malformed"},
                )
            supporting = sorted(str(item) for item in supporting_raw)
            contradicting = sorted(
                str(item) for item in contradicting_raw
            )
            confidence_values = [float(item) for item in confidences_raw]
            claim = ClaimRecord(
                claim_id="claim-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
                normalized_claim=str(aggregate["normalized_claim"]),
                supporting_source_ids=supporting,
                contradicting_source_ids=contradicting,
                confidence=round(
                    sum(confidence_values) / len(confidence_values), 4
                ),
                status=(
                    "contested"
                    if supporting and contradicting
                    else "supported"
                    if supporting
                    else "unsupported"
                ),
            )
            claims.append(claim)
            for source_id in supporting + contradicting:
                source_claims[source_id].append(claim.claim_id)
        for source in records:
            if not isinstance(source, dict):
                continue
            clean = dict(source)
            clean.pop("assertions", None)
            clean["used_claim_ids"] = sorted(source_claims[str(source["source_id"])])
            source_records.append(clean)
        validated_sources = [
            SourceRecord.model_validate(item).model_dump(mode="json")
            for item in source_records
        ]
        manifest_path = root / "source_manifest.json"
        claims_path = root / "claims.jsonl"
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "sources": validated_sources,
                "source_count": len(validated_sources),
            },
        )
        _jsonl(
            claims_path,
            [claim.model_dump(mode="json") for claim in claims],
        )
        self._save_job(
            root,
            job.model_copy(
                update={
                    "source_records": [
                        SourceRecord.model_validate(item) for item in validated_sources
                    ],
                    "claims": claims,
                }
            ),
        )
        return [manifest_path, claims_path]

    def _load_claims(self, root: Path) -> list[ClaimRecord]:
        claims: list[ClaimRecord] = []
        try:
            lines = (root / "claims.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ResearchPlaneError(
                "research_artifact_invalid", {"path": str(root / "claims.jsonl")}
            ) from exc
        for line in lines:
            if line:
                claims.append(ClaimRecord.model_validate(json.loads(line)))
        return claims

    def _load_sources(self, root: Path) -> list[SourceRecord]:
        raw = _read_json(root / "source_manifest.json")
        sources = raw.get("sources") if isinstance(raw, dict) else None
        if not isinstance(sources, list):
            raise ResearchPlaneError(
                "research_artifact_invalid",
                {"path": str(root / "source_manifest.json")},
            )
        return [SourceRecord.model_validate(item) for item in sources]

    def _stage_contradiction_detection(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        contradictions = [
            {
                "claim_id": claim.claim_id,
                "normalized_claim": claim.normalized_claim,
                "supporting_source_ids": claim.supporting_source_ids,
                "contradicting_source_ids": claim.contradicting_source_ids,
                "status": "SOURCE_DISAGREEMENT",
            }
            for claim in self._load_claims(root)
            if claim.status == "contested"
        ]
        path = root / "contradictions.json"
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "contradictions": contradictions,
                "count": len(contradictions),
            },
        )
        self._save_job(
            root, job.model_copy(update={"contradictions": contradictions})
        )
        return [path]

    def _stage_gap_analysis(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        claims = self._load_claims(root)
        sources = self._load_sources(root)
        gaps: list[str] = []
        if not sources:
            gaps.append("No sources were retained.")
        if not any(source.source_type in {"primary_source", "official_documentation"} for source in sources):
            gaps.append("No primary or official source was retained.")
        if any(claim.status == "contested" for claim in claims):
            gaps.append("At least one claim remains contested.")
        if not gaps:
            gaps.append(f"Further evidence may still refine the bounded answer to: {job.question}")
        path = root / "gap_analysis.json"
        _atomic_json(path, {"open_questions": gaps})
        return [path]

    def _stage_synthesis(self, root: Path, job: ResearchJobRecord) -> list[Path]:
        sources = self._load_sources(root)
        claims = self._load_claims(root)
        contradictions_raw = _read_json(root / "contradictions.json")
        contradictions = (
            contradictions_raw.get("contradictions")
            if isinstance(contradictions_raw, dict)
            else []
        )
        gaps_raw = _read_json(root / "gap_analysis.json")
        gaps = gaps_raw.get("open_questions") if isinstance(gaps_raw, dict) else []
        source_by_id = {source.source_id: source for source in sources}
        lines = [
            f"# Research report: {_markdown_text(job.question)}",
            "",
            "## Grounded facts",
            "",
        ]
        supported = [claim for claim in claims if claim.status == "supported"]
        if not supported:
            lines.append("- No uncontested substantive claim was extracted.")
        for claim in supported:
            citations = " ".join(f"[{item}]" for item in claim.supporting_source_ids)
            lines.append(
                f"- {_markdown_text(claim.normalized_claim)} {citations}".rstrip()
            )
        lines.extend(["", "## Source disagreement", ""])
        contested = [claim for claim in claims if claim.status == "contested"]
        if not contested:
            lines.append("- No explicit source disagreement was detected.")
        for claim in contested:
            supporting_citations = " ".join(
                f"[{item}]" for item in claim.supporting_source_ids
            )
            contradicting_citations = " ".join(
                f"[{item}]" for item in claim.contradicting_source_ids
            )
            lines.append(
                f"- {_markdown_text(claim.normalized_claim)} "
                f"(supports: {supporting_citations}; contradicts: "
                f"{contradicting_citations})"
            )
        lines.extend(
            [
                "",
                "## Model inference",
                "",
                "- No uncited model inference is presented as a fact.",
                "",
                "## Open questions",
                "",
            ]
        )
        for gap in gaps if isinstance(gaps, list) else []:
            lines.append(f"- {_markdown_text(str(gap))}")
        lines.extend(
            [
                "",
                "## Implementation recommendation",
                "",
                "- Validate decisions against the evidence ledger and resolve contested claims first.",
                "",
                "## Evidence table",
                "",
                "| ID | Type | Quality | Relevance | Title |",
                "|---|---|---:|---:|---|",
            ]
        )
        for source in sources:
            lines.append(
                f"| {source.source_id} | {source.source_type} | "
                f"{source.quality_score:.2f} | {source.relevance_score:.2f} | "
                f"{_markdown_text(source.title)} |"
            )
        lines.extend(["", "## References", ""])
        for source in sources:
            lines.append(
                f"- <a id=\"{source.source_id}\"></a>[{source.source_id}] "
                f"{_markdown_text(source.title)} — {source.canonical_url}"
            )
        report_path = root / "report.md"
        _atomic_text(report_path, "\n".join(lines) + "\n")
        ledger = {
            "schema_version": 1,
            "research_job_id": job.research_job_id,
            "question": job.question,
            "sources": [source.model_dump(mode="json") for source in sources],
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "contradictions": contradictions,
            "citation_map": {
                source_id: {
                    "source_id": source.source_id,
                    "canonical_url": source.canonical_url,
                    "content_sha256": source.content_sha256,
                    "local_snapshot_path": source.local_snapshot_path,
                }
                for source_id, source in source_by_id.items()
            },
        }
        ledger_path = root / "evidence_ledger.json"
        _atomic_json(ledger_path, ledger)
        graph = {
            "nodes": [
                *[
                    {"id": source.source_id, "kind": "source", "label": source.title}
                    for source in sources
                ],
                *[
                    {
                        "id": claim.claim_id,
                        "kind": "claim",
                        "label": claim.normalized_claim,
                        "status": claim.status,
                    }
                    for claim in claims
                ],
            ],
            "edges": [
                *[
                    {
                        "source": source_id,
                        "target": claim.claim_id,
                        "relation": "supports",
                    }
                    for claim in claims
                    for source_id in claim.supporting_source_ids
                ],
                *[
                    {
                        "source": source_id,
                        "target": claim.claim_id,
                        "relation": "contradicts",
                    }
                    for claim in claims
                    for source_id in claim.contradicting_source_ids
                ],
            ],
        }
        graph_path = root / "claim_source_graph.json"
        _atomic_json(graph_path, graph)
        self._save_job(
            root,
            job.model_copy(
                update={
                    "report_path": str(report_path),
                    "evidence_ledger_path": str(ledger_path),
                }
            ),
        )
        return [report_path, ledger_path, graph_path]

    def _stage_citation_validation(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        del job
        ledger_raw = _read_json(root / "evidence_ledger.json")
        citation_map = (
            ledger_raw.get("citation_map") if isinstance(ledger_raw, dict) else None
        )
        if not isinstance(citation_map, dict):
            raise ResearchPlaneError(
                "citation_validation_failure", {"reason": "citation map absent"}
            )
        report = (root / "report.md").read_text(encoding="utf-8")
        citations = sorted(set(re.findall(r"\[(src-[0-9a-f]{12})\]", report)))
        unresolved = sorted(set(citations).difference(citation_map))
        invalid_ledger = sorted(
            source_id
            for source_id, record in citation_map.items()
            if not _SOURCE_ID.fullmatch(source_id)
            or not isinstance(record, dict)
            or not isinstance(record.get("content_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["content_sha256"])
        )
        result: JsonObject = {
            "status": "PASS" if not unresolved and not invalid_ledger else "FAIL",
            "citations": citations,
            "unresolved_citations": unresolved,
            "invalid_ledger_entries": invalid_ledger,
        }
        path = root / "citation_validation.json"
        _atomic_json(path, result)
        if result["status"] != "PASS":
            raise ResearchPlaneError("citation_validation_failure", result)
        return [path]

    def _stage_report_generation(
        self, root: Path, job: ResearchJobRecord
    ) -> list[Path]:
        report = (root / "report.md").read_text(encoding="utf-8")
        ledger = _read_json(root / "evidence_ledger.json")
        sources = self._load_sources(root)
        claims = self._load_claims(root)
        html_path = root / "report.html"
        escaped_report = html.escape(report)
        graph_json = (root / "claim_source_graph.json").read_text(encoding="utf-8")
        html_document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laplace research report</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f4f1e8;color:#17231f}}
main{{max-width:1080px;margin:auto;padding:clamp(1rem,4vw,3rem)}}
.panel{{background:white;border:1px solid #cbc7ba;border-radius:16px;padding:1.25rem}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}} details{{margin:.75rem 0}}
@media(max-width:600px){{main{{padding:.75rem}}.panel{{border-radius:10px}}}}
</style>
</head>
<body><main><h1>Evidence-led research</h1>
<section class="panel" aria-label="Research report"><pre>{escaped_report}</pre></section>
<details><summary>Evidence ledger</summary><pre>{html.escape(json.dumps(ledger, indent=2))}</pre></details>
<script type="application/json" id="claim-source-graph">{html.escape(graph_json)}</script>
</main></body></html>
"""
        _atomic_text(html_path, html_document)

        bibtex_lines: list[str] = []
        csl: list[dict[str, object]] = []
        for source in sources:
            cite_key = source.source_id.replace("-", "_")
            bibtex_lines.extend(
                [
                    f"@misc{{{cite_key},",
                    f"  title = {{{source.title.replace('{', '').replace('}', '')}}},",
                    f"  url = {{{source.canonical_url}}},",
                    f"  note = {{Retrieved {source.retrieved_at}}}",
                    "}",
                    "",
                ]
            )
            csl.append(
                {
                    "id": source.source_id,
                    "type": (
                        "article-journal"
                        if source.source_type == "peer_reviewed_literature"
                        else "webpage"
                    ),
                    "title": source.title,
                    "author": [
                        {"literal": author} for author in source.authors
                    ],
                    "URL": source.canonical_url,
                    "note": f"SHA-256 {source.content_sha256}",
                }
            )
        bibtex_path = root / "references.bib"
        csl_path = root / "references.csl.json"
        _atomic_text(bibtex_path, "\n".join(bibtex_lines))
        _atomic_json(csl_path, csl)

        lock_inputs = [
            root / "request.json",
            root / "source_manifest.json",
            root / "claims.jsonl",
            root / "contradictions.json",
            root / "evidence_ledger.json",
            root / "citation_validation.json",
            root / "report.md",
            html_path,
            root / "claim_source_graph.json",
            bibtex_path,
            csl_path,
        ]
        lock = {
            "schema_version": 1,
            "research_job_id": job.research_job_id,
            "request_sha256": job.request_sha256,
            "adapter_names": sorted(job.search_backends),
            "model_route": job.model_route,
            "artifact_sha256": _relative_hashes(root, lock_inputs),
            "source_snapshot_sha256": {
                source.local_snapshot_path: source.content_sha256 for source in sources
            },
            "claim_count": len(claims),
            "citation_validation_status": "PASS",
            "automatic_governed_corpus_promotion": False,
        }
        lock["research_lock_sha256"] = canonical_sha256(lock)
        lock_path = root / "research.lock.json"
        _atomic_json(lock_path, lock)
        archive_path = root / f"{job.research_job_id}_compact.zip"
        archive_inputs = [
            root / "report.md",
            html_path,
            root / "evidence_ledger.json",
            root / "source_manifest.json",
            root / "claims.jsonl",
            root / "contradictions.json",
            root / "claim_source_graph.json",
            bibtex_path,
            csl_path,
            lock_path,
        ]
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(archive_inputs):
                info = zipfile.ZipInfo(str(path.relative_to(root)))
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.external_attr = 0o644 << 16
                archive.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
        return [html_path, bibtex_path, csl_path, lock_path, archive_path]


class GovernedCorpusPromoter:
    """Create immutable governed snapshots from approved research evidence."""

    def __init__(
        self,
        research_root: Path,
        governed_root: Path,
        approval_validator: Callable[[str, str, str], bool],
    ) -> None:
        self.research_root = research_root.resolve()
        self.governed_root = governed_root.resolve()
        self.approval_validator = approval_validator

    def promote(
        self,
        *,
        job_id: str,
        source_id: str,
        target_domain: str,
        approval_id: str,
        permitted_use: str,
        topic_metadata: Mapping[str, object],
        curated_summary: str,
        summary_origin: str,
        relevance_tests: Sequence[Mapping[str, object]],
        approved_by: str,
    ) -> JsonObject:
        if not _JOB_ID.fullmatch(job_id) or not _SOURCE_ID.fullmatch(source_id):
            raise ResearchPlaneError(
                "promotion_validation_failure", {"reason": "unsafe identity"}
            )
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", target_domain):
            raise ResearchPlaneError(
                "promotion_validation_failure", {"reason": "unsafe target domain"}
            )
        if not self.approval_validator(
            approval_id, "PROMOTE_RESEARCH_SOURCE", f"{job_id}:{source_id}"
        ):
            raise ResearchPlaneError(
                "approval_required",
                {
                    "approval_id": approval_id,
                    "required_action": "PROMOTE_RESEARCH_SOURCE",
                },
            )
        if not permitted_use.strip():
            raise ResearchPlaneError(
                "promotion_validation_failure", {"reason": "permitted use is required"}
            )
        if summary_origin not in {"human_curated", "independently_authored"}:
            raise ResearchPlaneError(
                "promotion_validation_failure",
                {"reason": "model-generated synthesis cannot be promoted"},
            )
        if not curated_summary.strip() or not topic_metadata or not relevance_tests:
            raise ResearchPlaneError(
                "promotion_validation_failure",
                {"reason": "summary, topic metadata, and relevance tests are required"},
            )
        job_root = self.research_root / job_id
        manifest_raw = _read_json(job_root / "source_manifest.json")
        sources = (
            manifest_raw.get("sources")
            if isinstance(manifest_raw, dict)
            else None
        )
        if not isinstance(sources, list):
            raise ResearchPlaneError(
                "promotion_validation_failure", {"reason": "source manifest absent"}
            )
        source = next(
            (
                SourceRecord.model_validate(item)
                for item in sources
                if isinstance(item, dict) and item.get("source_id") == source_id
            ),
            None,
        )
        if source is None:
            raise ResearchPlaneError(
                "promotion_validation_failure", {"reason": "source not found"}
            )
        if source.license.strip().upper() in {"", "UNKNOWN", "UNSPECIFIED"}:
            raise ResearchPlaneError(
                "promotion_validation_failure",
                {"reason": "license metadata is required"},
            )
        snapshot = job_root / source.local_snapshot_path
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source.content_sha256:
            raise ResearchPlaneError(
                "promotion_validation_failure",
                {"reason": "research snapshot hash mismatch"},
            )
        searchable = (
            curated_summary + "\n" + snapshot.read_text(encoding="utf-8", errors="replace")
        ).lower()
        test_results: list[JsonObject] = []
        for index, test in enumerate(relevance_tests, start=1):
            query = test.get("query")
            expected_terms = test.get("expected_terms")
            if (
                not isinstance(query, str)
                or not isinstance(expected_terms, list)
                or not expected_terms
                or not all(isinstance(term, str) for term in expected_terms)
            ):
                raise ResearchPlaneError(
                    "promotion_validation_failure",
                    {"reason": f"invalid relevance test {index}"},
                )
            matched = [
                term for term in expected_terms if term.lower() in searchable
            ]
            passed = len(matched) == len(expected_terms)
            test_results.append(
                {
                    "query": query,
                    "expected_terms": expected_terms,
                    "matched_terms": matched,
                    "passed": passed,
                }
            )
        if not all(bool(result["passed"]) for result in test_results):
            raise ResearchPlaneError(
                "promotion_relevance_failure", {"tests": test_results}
            )
        record = {
            "schema_version": 1,
            "source": source.model_dump(mode="json"),
            "permitted_use": permitted_use,
            "topic_metadata": dict(topic_metadata),
            "curated_summary": curated_summary,
            "summary_origin": summary_origin,
            "relevance_tests": test_results,
            "approval": {
                "approval_id": approval_id,
                "approved_by": approved_by,
                "approved_at": _now(),
            },
        }
        snapshot_sha256 = canonical_sha256(record)
        domain_root = self.governed_root / target_domain
        snapshot_root = domain_root / "snapshots" / snapshot_sha256
        if snapshot_root.exists():
            existing = _read_json(snapshot_root / "promotion.json")
            if existing != record:
                raise ResearchPlaneError(
                    "governed_snapshot_conflict",
                    {"snapshot_sha256": snapshot_sha256},
                )
        else:
            snapshot_root.mkdir(parents=True)
            _atomic_json(snapshot_root / "promotion.json", record)
            _atomic_text(snapshot_root / "curated_summary.md", curated_summary + "\n")
            shutil.copyfile(snapshot, snapshot_root / f"{source_id}.snapshot")
        prior_current = (
            _read_json(domain_root / "current.json")
            if (domain_root / "current.json").is_file()
            else None
        )
        prior_snapshot_sha256 = (
            prior_current.get("snapshot_sha256")
            if isinstance(prior_current, dict)
            else None
        )
        current = {
            "schema_version": 1,
            "domain": target_domain,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_path": str(snapshot_root),
            "previous_snapshot_sha256": prior_snapshot_sha256,
        }
        _atomic_json(domain_root / "current.json", current)
        return {
            "status": "PROMOTED",
            "job_id": job_id,
            "source_id": source_id,
            "target_domain": target_domain,
            "new_snapshot_sha256": snapshot_sha256,
            "snapshot_path": str(snapshot_root),
            "previous_snapshot_sha256": current["previous_snapshot_sha256"],
            "old_snapshot_remains_readable": (
                prior_snapshot_sha256 is None
                or (
                    domain_root
                    / "snapshots"
                    / str(prior_snapshot_sha256)
                ).is_dir()
            ),
        }
