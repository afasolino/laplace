"""Bounded, deterministic read-only views used by v3.1 routing."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

JsonObject: TypeAlias = dict[str, object]


class _CorpusClient(Protocol):
    def personal_corpora(self, *, include_archived: bool = False) -> JsonObject: ...

    def personal_corpus(self, corpus_id: str) -> JsonObject: ...


def runtime_metrics_view(capabilities: Mapping[str, object]) -> JsonObject:
    """Return only telemetry the Operator currently exposes truthfully.

    v3.1 intentionally does not infer decode throughput.  Exact serving
    instrumentation belongs to the later adaptive-runtime ladder step.
    """
    lanes = capabilities.get("model_lanes")
    rendered_lanes = [str(item) for item in lanes] if isinstance(lanes, Sequence) and not isinstance(lanes, (str, bytes)) else []
    return {
        "status": "OK",
        "evidence_kind": "runtime_metrics",
        "model_lanes": rendered_lanes,
        "completion_tokens_per_second": None,
        "streaming_decode_tokens_per_second": None,
        "gpu_memory_bytes": None,
        "gpu_utilization": None,
        "mtp_depth": None,
        "throughput_status": "UNAVAILABLE",
        "throughput_reason": "decode_throughput_not_exposed_by_operator_v31",
        "measurement_policy": "measured_or_explicitly_unavailable_never_inferred",
    }


def corpus_overview(client: _CorpusClient, *, max_corpora: int = 20, max_sources: int = 80) -> JsonObject:
    """Summarize owner-authorized corpus metadata without semantic invention."""
    list_corpora = client.personal_corpora
    get_corpus = client.personal_corpus
    listing = list_corpora(include_archived=False)
    raw_corpora = listing.get("corpora") if isinstance(listing, Mapping) else None
    corpora = [item for item in raw_corpora if isinstance(item, Mapping)] if isinstance(raw_corpora, list) else []
    corpora = corpora[:max_corpora]
    media_types: Counter[str] = Counter()
    source_names: list[str] = []
    corpus_rows: list[JsonObject] = []
    source_count = 0
    for corpus in corpora:
        corpus_id = corpus.get("corpus_id")
        if not isinstance(corpus_id, str) or not corpus_id:
            continue
        detail = get_corpus(corpus_id)
        raw_sources = detail.get("sources") if isinstance(detail, Mapping) else None
        sources = [item for item in raw_sources if isinstance(item, Mapping)] if isinstance(raw_sources, list) else []
        source_count += len(sources)
        for source in sources:
            media = source.get("media_type") or source.get("type") or "unknown"
            media_types[str(media)] += 1
            name = source.get("logical_path") or source.get("name") or source.get("file")
            if isinstance(name, str) and name and len(source_names) < max_sources:
                source_names.append(name[:500])
        corpus_rows.append(
            {
                "corpus_id": corpus_id,
                "name": corpus.get("name"),
                "revision": corpus.get("revision"),
                "state": corpus.get("state"),
                "source_count": len(sources),
            }
        )
    return {
        "status": "OK",
        "evidence_kind": "corpus_overview",
        "corpus_count": len(corpora),
        "source_count": source_count,
        "corpora": corpus_rows,
        "media_type_counts": dict(sorted(media_types.items())),
        "sample_source_names": source_names,
        "bounded": True,
        "inference_policy": "bounded_owner_metadata_only_no_freeform_topic_inference",
    }
