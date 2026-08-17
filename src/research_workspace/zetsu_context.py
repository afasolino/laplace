"""Compact, progressive evidence responses for Zetsu."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Protocol, cast

DEFAULT_MAX_RESULTS = 6
DEFAULT_MAX_CHARS = 6000
MIN_MAX_CHARS = 512
MAX_MAX_CHARS = 24000


class EvidenceLike(Protocol):
    @property
    def filename(self) -> str: ...

    @property
    def page(self) -> int | None: ...

    @property
    def section(self) -> str | None: ...

    @property
    def chunk_id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def score(self) -> float: ...


@dataclass(frozen=True)
class CompactEvidence:
    evidence_id: str
    filename: str
    page: int | None
    section: str | None
    chunk_id: str
    score: float
    excerpt: str
    source_class: str = "governed_document"


@dataclass(frozen=True)
class CompactPacket:
    query: str
    grounded: bool
    evidence: tuple[CompactEvidence, ...]
    truncated: bool
    max_chars: int

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "grounded": self.grounded,
            "evidence": [asdict(item) for item in self.evidence],
            "truncated": self.truncated,
            "max_chars": self.max_chars,
        }


def normalize_budget(max_chars: int) -> int:
    if max_chars < MIN_MAX_CHARS or max_chars > MAX_MAX_CHARS:
        raise ValueError(f"max_chars must be between {MIN_MAX_CHARS} and {MAX_MAX_CHARS}")
    return max_chars


def compact_evidence(
    query: str,
    items: Iterable[EvidenceLike],
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> CompactPacket:
    """Return ranked evidence excerpts within a hard character budget."""

    budget = normalize_budget(max_chars)
    if max_results < 1 or max_results > 20:
        raise ValueError("max_results must be between 1 and 20")
    result: list[CompactEvidence] = []
    used = len(query)
    truncated = False
    for index, item in enumerate(items, start=1):
        if len(result) >= max_results:
            truncated = True
            break
        fixed = 120 + len(item.filename) + len(item.section or "") + len(item.chunk_id)
        available = budget - used - fixed
        if available <= 32:
            truncated = True
            break
        excerpt = " ".join(item.text.split())
        if len(excerpt) > available:
            excerpt = excerpt[: max(0, available - 1)].rstrip() + "…"
            truncated = True
        result.append(
            CompactEvidence(
                evidence_id=f"E{index}",
                filename=item.filename,
                page=item.page,
                section=item.section,
                chunk_id=item.chunk_id,
                score=round(float(item.score), 6),
                excerpt=excerpt,
            )
        )
        used += fixed + len(excerpt)
        if truncated:
            break
    return CompactPacket(
        query=query,
        grounded=bool(result),
        evidence=tuple(result),
        truncated=truncated,
        max_chars=budget,
    )


@dataclass(frozen=True)
class _RecordEvidence:
    filename: str
    page: int | None
    section: str | None
    chunk_id: str
    text: str
    score: float


def compact_personal_results(
    query: str,
    raw: Mapping[str, object],
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, object]:
    """Compact owner-authorized corpus search output without losing citations."""

    values = raw.get("results")
    rows = values if isinstance(values, list) else []
    evidence: list[_RecordEvidence] = []
    snapshots: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, dict):
            continue
        evidence.append(
            _RecordEvidence(
                filename=str(value.get("file", "")),
                page=cast(int | None, value.get("page")),
                section=cast(str | None, value.get("section")),
                chunk_id=str(value.get("chunk_id", "")),
                text=str(value.get("text", "")),
                score=float(value.get("score", 0.0)),
            )
        )
        snapshot = {
            "corpus_id": value.get("corpus_id"),
            "snapshot_revision": value.get("snapshot_revision"),
        }
        if snapshot not in snapshots:
            snapshots.append(snapshot)
    packet = compact_evidence(
        query,
        evidence,
        max_results=max_results,
        max_chars=max_chars,
    ).as_dict()
    packet["source_scope"] = "authenticated_user_personal_corpus"
    packet["snapshots"] = snapshots
    return packet


def compact_expanded_results(
    raw: Mapping[str, object],
    *,
    max_chars: int = 12_000,
) -> dict[str, object]:
    """Bound selective evidence expansion by the same hard response budget."""

    budget = normalize_budget(max_chars)
    values = raw.get("results")
    rows = values if isinstance(values, list) else []
    output: list[dict[str, object]] = []
    used = 0
    truncated = False
    for value in rows:
        if not isinstance(value, dict):
            continue
        text = " ".join(str(value.get("text", "")).split())
        fixed = 160 + len(str(value.get("file", ""))) + len(str(value.get("chunk_id", "")))
        available = budget - used - fixed
        if available <= 32:
            truncated = True
            break
        if len(text) > available:
            text = text[: max(0, available - 1)].rstrip() + "…"
            truncated = True
        output.append(
            {
                key: value.get(key)
                for key in (
                    "corpus_id",
                    "source_id",
                    "file",
                    "page",
                    "section",
                    "chunk_id",
                    "chunk_ordinal",
                    "snapshot_revision",
                    "content_sha256",
                )
            }
            | {"text": text, "source_class": "governed_document"}
        )
        used += fixed + len(text)
        if truncated:
            break
    return {
        "evidence": output,
        "truncated": truncated,
        "max_chars": budget,
        "source_scope": "authenticated_user_personal_corpus",
    }
