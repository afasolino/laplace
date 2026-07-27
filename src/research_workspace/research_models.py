"""Strict schemas and adapters for the isolated exploratory Research Plane."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


ResearchMode = Literal[
    "quick",
    "standard",
    "deep",
    "systematic_literature",
    "repository_audit",
    "hardware_state_of_the_art",
]
SourceType = Literal[
    "primary_source",
    "official_documentation",
    "peer_reviewed_literature",
    "preprint",
    "repository",
    "secondary_commentary",
    "local_document",
]


class ResearchJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=3, max_length=12_000)
    scope: str = Field(default="bounded", min_length=1, max_length=2_000)
    research_mode: ResearchMode = "standard"
    search_backends: list[str] = Field(min_length=1, max_length=8)
    source_policy: str = Field(default="primary_preferred", max_length=500)
    model_route: str = Field(default="deterministic", max_length=160)

    @field_validator("search_backends")
    @classmethod
    def validate_backends(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("search backends must be unique")
        if not all(re.fullmatch(r"[a-z0-9_-]{1,64}", item) for item in value):
            raise ValueError("invalid search backend")
        return value


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str
    canonical_url: str
    title: str
    authors: list[str]
    publication: str | None
    publication_date: str | None
    retrieved_at: str
    source_type: SourceType
    license: str
    content_sha256: str
    local_snapshot_path: str
    quality_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    used_claim_ids: list[str]
    backend: str
    discovery_queries: list[str]
    revision: str | None = None


class ClaimRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: str
    normalized_claim: str
    supporting_source_ids: list[str]
    contradicting_source_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    status: Literal["supported", "contested", "unsupported"]


class ResearchJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = 1
    research_job_id: str
    question: str
    scope: str
    research_mode: ResearchMode
    search_backends: list[str]
    source_policy: str
    model_route: str
    created_at: str
    status: str
    current_stage: str | None
    completed_stages: list[str]
    subquestions: list[str]
    source_records: list[SourceRecord]
    claims: list[ClaimRecord]
    contradictions: list[dict[str, object]]
    report_path: str | None
    evidence_ledger_path: str | None
    trace_id: str
    request_sha256: str


@dataclass(frozen=True)
class ClaimAssertion:
    """A source-scoped assertion supplied by an extractor or fixture."""

    normalized_claim: str
    stance: Literal["supports", "contradicts"] = "supports"
    claim_key: str | None = None
    confidence: float = 0.8


@dataclass(frozen=True)
class DiscoveredSource:
    canonical_url: str
    title: str
    backend: str
    query: str
    source_type: SourceType
    license: str
    authors: tuple[str, ...] = ()
    publication: str | None = None
    publication_date: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class FetchedSource:
    discovered: DiscoveredSource
    content: bytes
    content_type: str = "text/plain"
    assertions: tuple[ClaimAssertion, ...] = ()
    retrieved_at: str | None = None


class ResearchAdapter(Protocol):
    """Pluggable source discovery/fetch contract."""

    name: str

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        """Return bounded source candidates."""

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        """Fetch one candidate with bounded content."""


@dataclass
class FixtureResearchAdapter:
    """Deterministic adapter used by CPU certification and tests."""

    sources: Sequence[FetchedSource]
    name: str = "fixture"

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        return [
            DiscoveredSource(
                canonical_url=item.discovered.canonical_url,
                title=item.discovered.title,
                backend=self.name,
                query=query,
                source_type=item.discovered.source_type,
                license=item.discovered.license,
                authors=item.discovered.authors,
                publication=item.discovered.publication,
                publication_date=item.discovered.publication_date,
                revision=item.discovered.revision,
            )
            for item in self.sources[:limit]
        ]

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        for item in self.sources:
            if canonicalize_url(item.discovered.canonical_url) == canonicalize_url(
                source.canonical_url
            ):
                return FetchedSource(
                    discovered=source,
                    content=item.content,
                    content_type=item.content_type,
                    assertions=item.assertions,
                    retrieved_at=item.retrieved_at,
                )
        raise KeyError(source.canonical_url)


@dataclass
class LocalDirectoryResearchAdapter:
    """Search authorized local text documents without mutating their source tree."""

    root: Path
    name: str
    source_type: SourceType = "local_document"
    license_by_path: Mapping[str, str] = field(default_factory=dict)
    maximum_files: int = 2_000
    maximum_bytes: int = 4_000_000

    def _documents(self) -> list[Path]:
        root = self.root.resolve()
        if not root.is_dir():
            return []
        allowed = {".md", ".txt", ".json", ".rst"}
        return [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in allowed
        ][: self.maximum_files]

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9_]{3,}", query)
        }
        scored: list[tuple[int, Path]] = []
        root = self.root.resolve()
        for path in self._documents():
            try:
                if path.stat().st_size > self.maximum_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            haystack = f"{path.name}\n{text[:100_000]}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        return [
            DiscoveredSource(
                canonical_url=path.as_uri(),
                title=path.stem,
                backend=self.name,
                query=query,
                source_type=self.source_type,
                license=self.license_by_path.get(
                    str(path.relative_to(root)), "UNKNOWN"
                ),
            )
            for _, path in scored[:limit]
        ]

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        parsed = urllib.parse.urlsplit(source.canonical_url)
        path = Path(urllib.request.url2pathname(parsed.path)).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("local source escapes adapter root")
        data = path.read_bytes()
        if len(data) > self.maximum_bytes:
            raise ValueError("local research source exceeds content limit")
        return FetchedSource(discovered=source, content=data)


def canonicalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme == "file":
        return urllib.parse.urlunsplit(("file", "", parsed.path, "", ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("research URL must use HTTP(S) or file")
    host = parsed.hostname.lower()
    port = parsed.port
    authority = host if port is None else f"{host}:{port}"
    path = parsed.path or "/"
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [
        (key, value)
        for key, value in query
        if not key.lower().startswith(("utm_", "fbclid", "gclid"))
    ]
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            authority,
            path,
            urllib.parse.urlencode(sorted(query)),
            "",
        )
    )


def _assert_public_hostname(hostname: str) -> None:
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise ValueError("research hostname cannot be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("direct URL resolves to a non-public address")


@dataclass
class DirectUrlResearchAdapter:
    """Fetch an explicit, bounded list of public URLs with robots checks."""

    urls: Sequence[DiscoveredSource]
    name: str = "direct_url"
    timeout_seconds: float = 10
    maximum_bytes: int = 4_000_000
    user_agent: str = "LaplaceLocalResearch/1.0"

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        return [
            DiscoveredSource(
                **{
                    **item.__dict__,
                    "backend": self.name,
                    "query": query,
                    "canonical_url": canonicalize_url(item.canonical_url),
                }
            )
            for item in self.urls[:limit]
        ]

    def _robots_allowed(self, url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        robots_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, "/robots.txt", "", "")
        )
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        request = urllib.request.Request(  # nosec B310 - public host checked.
            robots_url, headers={"User-Agent": self.user_agent}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                body = response.read(512_000).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code == 404
        except (OSError, urllib.error.URLError, TimeoutError):
            # Fail closed when robots policy cannot be established.
            return False
        parser.parse(body.splitlines())
        return parser.can_fetch(self.user_agent, url)

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        url = canonicalize_url(source.canonical_url)
        parsed = urllib.parse.urlsplit(url)
        assert parsed.hostname is not None
        _assert_public_hostname(parsed.hostname)
        if not self._robots_allowed(url):
            raise ValueError("robots policy does not permit this fetch")
        request = urllib.request.Request(  # nosec B310 - public host checked.
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                content_length = response.headers.get("Content-Length")
                if (
                    content_length is not None
                    and int(content_length) > self.maximum_bytes
                ):
                    raise ValueError("research response exceeds content limit")
                content = response.read(self.maximum_bytes + 1)
                content_type = response.headers.get_content_type()
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise ValueError("research source fetch failed") from exc
        if len(content) > self.maximum_bytes:
            raise ValueError("research response exceeds content limit")
        return FetchedSource(
            discovered=source,
            content=content,
            content_type=content_type,
        )


def source_content_sha256(source: FetchedSource) -> str:
    return hashlib.sha256(source.content).hexdigest()


def assertions_to_json(assertions: Sequence[ClaimAssertion]) -> list[dict[str, object]]:
    return [
        {
            "normalized_claim": item.normalized_claim,
            "stance": item.stance,
            "claim_key": item.claim_key,
            "confidence": item.confidence,
        }
        for item in assertions
    ]


def assertions_from_json(value: object) -> tuple[ClaimAssertion, ...]:
    if not isinstance(value, list):
        return ()
    result: list[ClaimAssertion] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        stance = raw.get("stance")
        if stance not in {"supports", "contradicts"}:
            continue
        normalized_claim = raw.get("normalized_claim")
        confidence = raw.get("confidence")
        if not isinstance(normalized_claim, str) or not isinstance(
            confidence, int | float
        ):
            continue
        claim_key = raw.get("claim_key")
        result.append(
            ClaimAssertion(
                normalized_claim=normalized_claim,
                stance=stance,
                claim_key=claim_key if isinstance(claim_key, str) else None,
                confidence=float(confidence),
            )
        )
    return tuple(result)


def discovered_to_json(source: DiscoveredSource) -> dict[str, object]:
    return {
        "canonical_url": source.canonical_url,
        "title": source.title,
        "backend": source.backend,
        "query": source.query,
        "source_type": source.source_type,
        "license": source.license,
        "authors": list(source.authors),
        "publication": source.publication,
        "publication_date": source.publication_date,
        "revision": source.revision,
    }


def discovered_from_json(value: Mapping[str, object]) -> DiscoveredSource:
    source_type = value.get("source_type")
    if source_type not in {
        "primary_source",
        "official_documentation",
        "peer_reviewed_literature",
        "preprint",
        "repository",
        "secondary_commentary",
        "local_document",
    }:
        raise ValueError("invalid persisted source type")
    authors = value.get("authors")
    return DiscoveredSource(
        canonical_url=str(value["canonical_url"]),
        title=str(value["title"]),
        backend=str(value["backend"]),
        query=str(value["query"]),
        source_type=source_type,
        license=str(value["license"]),
        authors=tuple(str(item) for item in authors) if isinstance(authors, list) else (),
        publication=(
            str(value["publication"]) if value.get("publication") is not None else None
        ),
        publication_date=(
            str(value["publication_date"])
            if value.get("publication_date") is not None
            else None
        ),
        revision=str(value["revision"]) if value.get("revision") is not None else None,
    )


def canonical_source_key(source: DiscoveredSource) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "url": canonicalize_url(source.canonical_url),
                "title": source.title,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
