"""Bounded optional web discovery adapters for the exploratory Research Plane."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Callable, Sequence

from defusedxml import ElementTree as ET

from .research_models import (
    DirectUrlResearchAdapter,
    DiscoveredSource,
    FetchedSource,
    ResearchAdapter,
    assert_public_hostname,
    canonicalize_url,
)


class WebAdapterError(RuntimeError):
    """An optional research web backend failed safely."""


@dataclass
class BoundedWebClient:
    """Rate-limited public HTTP client with robots and response-size bounds."""

    timeout_seconds: float = 10
    maximum_bytes: int = 4_000_000
    minimum_interval_seconds: float = 1.0
    user_agent: str = "LaplaceLocalResearch/1.0"
    sleeper: Callable[[float], None] = time.sleep
    _last_request: dict[str, float] = field(default_factory=dict)
    _robots: dict[str, urllib.robotparser.RobotFileParser | None] = field(
        default_factory=dict
    )

    def _check_public(self, url: str) -> urllib.parse.SplitResult:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname is None:
            raise WebAdapterError("web adapters require public HTTPS endpoints")
        assert_public_hostname(parsed.hostname)
        return parsed

    def _rate_limit(self, hostname: str) -> None:
        elapsed = time.monotonic() - self._last_request.get(hostname, 0.0)
        remaining = self.minimum_interval_seconds - elapsed
        if remaining > 0:
            self.sleeper(remaining)
        self._last_request[hostname] = time.monotonic()

    def _robots_allowed(self, parsed: urllib.parse.SplitResult, url: str) -> bool:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = origin + "/robots.txt"
            request = urllib.request.Request(  # nosec B310
                robots_url, headers={"User-Agent": self.user_agent}
            )
            try:
                self._rate_limit(str(parsed.hostname))
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                    body = response.read(512_000).decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    self._robots[origin] = None
                else:
                    raise WebAdapterError("robots policy request failed") from exc
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                raise WebAdapterError("robots policy is unavailable") from exc
            else:
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(body.splitlines())
                self._robots[origin] = parser
        cached_parser = self._robots[origin]
        return cached_parser is None or cached_parser.can_fetch(self.user_agent, url)

    def get(self, url: str, *, accept: str) -> bytes:
        parsed = self._check_public(url)
        if not self._robots_allowed(parsed, url):
            raise WebAdapterError("robots policy does not permit the request")
        self._rate_limit(str(parsed.hostname))
        request = urllib.request.Request(  # nosec B310
            url,
            headers={"Accept": accept, "User-Agent": self.user_agent},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > self.maximum_bytes:
                    raise WebAdapterError("web response exceeds content limit")
                body_raw: object = response.read(self.maximum_bytes + 1)
                if not isinstance(body_raw, bytes):
                    raise WebAdapterError("web backend returned non-byte content")
                body = body_raw
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise WebAdapterError("web research request failed") from exc
        if len(body) > self.maximum_bytes:
            raise WebAdapterError("web response exceeds content limit")
        return body

    def get_json(self, url: str) -> object:
        try:
            return json.loads(
                self.get(url, accept="application/json").decode(
                    "utf-8", errors="strict"
                )
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebAdapterError("web backend returned invalid JSON") from exc


@dataclass
class SearXNGResearchAdapter:
    """Optional self-hosted SearXNG discovery with bounded direct fetching."""

    endpoint: str
    name: str = "searxng"
    timeout_seconds: float = 10
    maximum_bytes: int = 4_000_000
    _discovered: dict[str, DiscoveredSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("SearXNG endpoint must be local HTTP")

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        url = self.endpoint.rstrip("/") + "/search?" + urllib.parse.urlencode(
            {"q": query, "format": "json"}
        )
        request = urllib.request.Request(  # nosec B310
            url,
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                body = response.read(self.maximum_bytes + 1)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise WebAdapterError("SearXNG is unavailable") from exc
        if len(body) > self.maximum_bytes:
            raise WebAdapterError("SearXNG response exceeds content limit")
        try:
            payload: object = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebAdapterError("SearXNG returned invalid JSON") from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        discovered: list[DiscoveredSource] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            source = DiscoveredSource(
                canonical_url=canonicalize_url(item["url"]),
                title=str(item.get("title") or item["url"])[:1_000],
                backend=self.name,
                query=query,
                source_type="secondary_commentary",
                license="UNKNOWN",
            )
            self._discovered[source.canonical_url] = source
            discovered.append(source)
        return discovered[:limit]

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        direct = DirectUrlResearchAdapter(
            [source],
            timeout_seconds=self.timeout_seconds,
            maximum_bytes=self.maximum_bytes,
        )
        return direct.fetch(source)


@dataclass
class _CachedMetadataAdapter:
    name: str
    client: BoundedWebClient = field(default_factory=BoundedWebClient)
    _content: dict[str, bytes] = field(default_factory=dict)

    def _remember(self, source: DiscoveredSource, value: object) -> DiscoveredSource:
        self._content[source.canonical_url] = json.dumps(
            value, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        return source

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        content = self._content.get(source.canonical_url)
        if content is None:
            raise WebAdapterError("source metadata is not present in the discovery cache")
        return FetchedSource(
            discovered=source,
            content=content,
            content_type="application/json",
        )


@dataclass
class ArxivResearchAdapter(_CachedMetadataAdapter):
    """arXiv Atom API adapter preserving exact entry metadata."""

    name: str = "arxiv"

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(limit, 20),
            }
        )
        body = self.client.get(url, accept="application/atom+xml")
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise WebAdapterError("arXiv returned invalid Atom XML") from exc
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        discovered: list[DiscoveredSource] = []
        for entry in root.findall("atom:entry", namespace)[:limit]:
            identifier = (entry.findtext("atom:id", "", namespace) or "").strip()
            title = re.sub(
                r"\s+", " ", entry.findtext("atom:title", "", namespace)
            ).strip()
            if not identifier or not title:
                continue
            authors = tuple(
                name.text.strip()
                for name in entry.findall("atom:author/atom:name", namespace)
                if name.text
            )
            source = DiscoveredSource(
                canonical_url=canonicalize_url(identifier),
                title=title,
                backend=self.name,
                query=query,
                source_type="preprint",
                license="arXiv item licence must be verified before promotion",
                authors=authors,
                publication="arXiv",
                publication_date=entry.findtext("atom:published", None, namespace),
                revision=identifier.rstrip("/").rsplit("/", 1)[-1],
            )
            metadata = {
                "id": identifier,
                "title": title,
                "authors": authors,
                "published": source.publication_date,
                "summary": re.sub(
                    r"\s+", " ", entry.findtext("atom:summary", "", namespace)
                ).strip(),
            }
            discovered.append(self._remember(source, metadata))
        return discovered


@dataclass
class CrossrefResearchAdapter(_CachedMetadataAdapter):
    """Crossref metadata adapter for DOI-backed literature discovery."""

    name: str = "crossref"

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query": query, "rows": min(limit, 20), "select": "DOI,title,author,published,type,container-title,license"}
        )
        payload = self.client.get_json(url)
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        discovered: list[DiscoveredSource] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("DOI"), str):
                continue
            title_raw = item.get("title")
            title = (
                str(title_raw[0])
                if isinstance(title_raw, list) and title_raw
                else item["DOI"]
            )
            author_raw = item.get("author")
            authors = tuple(
                " ".join(
                    part
                    for part in (str(author.get("given", "")), str(author.get("family", "")))
                    if part
                )
                for author in author_raw
                if isinstance(author, dict)
            ) if isinstance(author_raw, list) else ()
            container = item.get("container-title")
            source = DiscoveredSource(
                canonical_url=canonicalize_url(
                    f"https://doi.org/{item['DOI']}"
                ),
                title=title,
                backend=self.name,
                query=query,
                source_type="peer_reviewed_literature",
                license=(
                    str(item["license"][0].get("URL", "UNKNOWN"))
                    if isinstance(item.get("license"), list)
                    and item["license"]
                    and isinstance(item["license"][0], dict)
                    else "UNKNOWN"
                ),
                authors=authors,
                publication=(
                    str(container[0])
                    if isinstance(container, list) and container
                    else None
                ),
                revision=item["DOI"],
            )
            discovered.append(self._remember(source, item))
        return discovered


@dataclass
class SemanticScholarResearchAdapter(_CachedMetadataAdapter):
    """Public Semantic Scholar Graph API adapter without credential storage."""

    name: str = "semantic_scholar"

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
            {
                "query": query,
                "limit": min(limit, 20),
                "fields": "paperId,title,authors,year,venue,externalIds,openAccessPdf",
            }
        )
        payload = self.client.get_json(url)
        data = payload.get("data") if isinstance(payload, dict) else None
        discovered: list[DiscoveredSource] = []
        for item in data if isinstance(data, list) else []:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("paperId"), str)
                or not isinstance(item.get("title"), str)
            ):
                continue
            authors_raw = item.get("authors")
            authors = tuple(
                str(author["name"])
                for author in authors_raw
                if isinstance(author, dict) and isinstance(author.get("name"), str)
            ) if isinstance(authors_raw, list) else ()
            source = DiscoveredSource(
                canonical_url=canonicalize_url(
                    f"https://www.semanticscholar.org/paper/{item['paperId']}"
                ),
                title=item["title"],
                backend=self.name,
                query=query,
                source_type="peer_reviewed_literature",
                license="UNKNOWN",
                authors=authors,
                publication=(
                    str(item["venue"]) if item.get("venue") else None
                ),
                publication_date=(
                    str(item["year"]) if item.get("year") else None
                ),
                revision=item["paperId"],
            )
            discovered.append(self._remember(source, item))
        return discovered


@dataclass(frozen=True)
class GitHubRepositorySpec:
    repository_url: str
    revision: str
    title: str
    license: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", self.repository_url):
            raise ValueError("GitHub repository URL is invalid")
        if not re.fullmatch(r"[0-9a-f]{7,64}", self.revision):
            raise ValueError("GitHub repository inspection requires an exact revision")


@dataclass
class GitHubRepositoryInspectionAdapter:
    """Inspect selected repository README content at exact audited revisions."""

    repositories: Sequence[GitHubRepositorySpec]
    client: BoundedWebClient = field(default_factory=BoundedWebClient)
    name: str = "github_repository_inspection"
    _content: dict[str, bytes] = field(default_factory=dict)

    def discover(self, query: str, *, limit: int) -> Sequence[DiscoveredSource]:
        terms = _term_set(query)
        selected = [
            repository
            for repository in self.repositories
            if not terms
            or any(
                term in f"{repository.title} {repository.repository_url}".lower()
                for term in terms
            )
        ]
        return [
            DiscoveredSource(
                canonical_url=canonicalize_url(repository.repository_url),
                title=repository.title,
                backend=self.name,
                query=query,
                source_type="repository",
                license=repository.license,
                revision=repository.revision,
            )
            for repository in selected[:limit]
        ]

    def fetch(self, source: DiscoveredSource) -> FetchedSource:
        if source.revision is None:
            raise WebAdapterError("repository revision is missing")
        parsed = urllib.parse.urlsplit(source.canonical_url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 2:
            raise WebAdapterError("repository URL is malformed")
        api_url = (
            f"https://api.github.com/repos/{parts[0]}/{parts[1]}/readme?"
            + urllib.parse.urlencode({"ref": source.revision})
        )
        payload = self.client.get_json(api_url)
        if (
            not isinstance(payload, dict)
            or payload.get("encoding") != "base64"
            or not isinstance(payload.get("content"), str)
        ):
            raise WebAdapterError("GitHub README response is malformed")
        try:
            content = base64.b64decode(payload["content"], validate=False)
        except ValueError as exc:
            raise WebAdapterError("GitHub README content is invalid") from exc
        if len(content) > self.client.maximum_bytes:
            raise WebAdapterError("GitHub README exceeds content limit")
        return FetchedSource(
            discovered=source,
            content=content,
            content_type="text/markdown",
        )


def _term_set(text: str) -> set[str]:
    return {
        item.lower()
        for item in re.findall(r"[A-Za-z0-9_]{3,}", text)
        if item.lower() not in {"the", "and", "for", "with", "what"}
    }


def supported_web_adapter_names() -> tuple[str, ...]:
    """Stable names exposed in diagnostics and GUI source policy controls."""

    adapter_types: tuple[type[ResearchAdapter], ...] = (
        SearXNGResearchAdapter,
        ArxivResearchAdapter,
        CrossrefResearchAdapter,
        SemanticScholarResearchAdapter,
        GitHubRepositoryInspectionAdapter,
    )
    del adapter_types
    return (
        "searxng",
        "arxiv",
        "crossref",
        "semantic_scholar",
        "direct_url",
        "github_repository_inspection",
    )
