from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, Callable, Iterable

try:
    import certifi as _certifi
except ImportError:
    _certifi = None  # type: ignore[assignment]


class ProviderStatus:
    AVAILABLE = "AVAILABLE"
    API_KEY_REQUIRED = "API_KEY_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    ACCESS_DENIED = "ACCESS_DENIED"
    NETWORK_ERROR = "NETWORK_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True)
class SearchResult:
    provider: str
    provider_id: str | None
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    doi: str | None
    ieee_article_number: str | None
    canonical_url: str | None
    pdf_url: str | None
    open_access: bool | None
    citation_count: int | None
    retrieved_at: str
    raw_record_hash: str
    query: str
    rank: int
    access_level: str = "METADATA_ONLY"


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    status: str
    query: str
    results: list[SearchResult]
    error: str | None = None
    next_cursor: str | None = None


def _hash_record(record: Any) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def _redact(value: str) -> str:
    secrets = [
        os.getenv("IEEE_XPLORE_API_KEY"),
        os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        os.getenv("GENERAL_SEARCH_API_KEY"),
    ]
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", html.unescape(str(value))).strip()
    return text or None


def _year(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
    max_bytes: int = 10_000_000,
    retries: int = 2,
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FormalScience/0.1 (local research; contact configured user)",
            **(headers or {}),
        },
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            context = (
                ssl.create_default_context(cafile=_certifi.where())
                if _certifi is not None
                else ssl.create_default_context()
            )
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError("response exceeds configured maximum size")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(min(64 * 1024, max_bytes - total + 1))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("response exceeds configured maximum size")
                return b"".join(chunks), {
                    key.lower(): value for key, value in response.headers.items()
                }
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403, 429}:
                raise
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(str(last))


def _result(
    provider: str,
    record: dict[str, Any],
    query: str,
    rank: int,
    *,
    title: Any,
    authors: Iterable[Any],
    year: Any,
    venue: Any,
    abstract: Any,
    doi: Any,
    canonical: Any,
    pdf: Any,
    open_access: bool | None,
    citations: Any,
    provider_id: Any,
    ieee_number: Any = None,
) -> SearchResult:
    normalized_doi = _clean(doi)
    return SearchResult(
        provider,
        _clean(provider_id),
        _clean(title) or "[UNTITLED]",
        [author for author in (_clean(a) for a in authors) if author],
        _year(year),
        _clean(venue),
        _clean(abstract),
        normalized_doi,
        _clean(ieee_number),
        _clean(canonical),
        _clean(pdf),
        open_access,
        _year(citations),
        datetime.now(UTC).isoformat(),
        _hash_record(record),
        query,
        rank,
        "ABSTRACT_ONLY" if abstract else "METADATA_ONLY",
    )


def _query_url(base: str, params: dict[str, Any]) -> str:
    return (
        base
        + "?"
        + urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
    )


def search_crossref(
    query: str, *, limit: int = 10, offset: int = 0, timeout: int = 20
) -> ProviderResponse:
    limit = max(1, min(limit, 100))
    url = _query_url(
        "https://api.crossref.org/works",
        {"query.bibliographic": query, "rows": limit, "offset": offset},
    )
    try:
        payload, _ = _request_bytes(url, timeout=timeout)
        message = json.loads(payload).get("message", {})
        results = [
            _result(
                "crossref",
                item,
                query,
                index + 1 + offset,
                title=(item.get("title") or [None])[0],
                authors=[
                    a.get("given", "") + " " + a.get("family", "") for a in item.get("author", [])
                ],
                year=(item.get("published-print") or item.get("published-online") or {}).get(
                    "date-parts", [[None]]
                )[0][0],
                venue=(item.get("container-title") or [None])[0],
                abstract=item.get("abstract"),
                doi=item.get("DOI"),
                canonical=item.get("URL"),
                pdf=next(
                    (
                        link.get("URL")
                        for link in item.get("link", [])
                        if link.get("content-type") == "application/pdf"
                    ),
                    None,
                ),
                open_access=(item.get("license") is not None),
                citations=item.get("is-referenced-by-count"),
                provider_id=item.get("DOI"),
            )
            for index, item in enumerate(message.get("items", []))
        ]
        return ProviderResponse("crossref", ProviderStatus.AVAILABLE, query, results)
    except urllib.error.HTTPError as exc:
        return ProviderResponse(
            "crossref",
            ProviderStatus.RATE_LIMITED
            if exc.code == 429
            else ProviderStatus.ACCESS_DENIED
            if exc.code in {401, 403}
            else ProviderStatus.PROVIDER_ERROR,
            query,
            [],
            str(exc),
        )
    except Exception as exc:
        return ProviderResponse("crossref", ProviderStatus.NETWORK_ERROR, query, [], str(exc))


def search_openalex(
    query: str, *, limit: int = 10, cursor: str = "*", timeout: int = 20
) -> ProviderResponse:
    limit = max(1, min(limit, 100))
    url = _query_url(
        "https://api.openalex.org/works",
        {
            "search": query,
            "per-page": limit,
            "cursor": cursor,
            "mailto": os.getenv("OPENALEX_MAILTO"),
        },
    )
    try:
        payload, _ = _request_bytes(url, timeout=timeout)
        data = json.loads(payload)
        results = []
        for index, item in enumerate(data.get("results", [])):
            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}
            pdf_record = primary_location.get("pdf") or {}
            results.append(
                _result(
                    "openalex",
                    item,
                    query,
                    index + 1,
                    title=item.get("title"),
                    authors=[
                        (a.get("author") or {}).get("display_name")
                        for a in item.get("authorships", [])
                    ],
                    year=item.get("publication_year"),
                    venue=source.get("display_name"),
                    abstract=None,
                    doi=item.get("doi"),
                    canonical=item.get("id"),
                    pdf=pdf_record.get("url"),
                    open_access=((item.get("open_access") or {}).get("is_oa")),
                    citations=item.get("cited_by_count"),
                    provider_id=item.get("id"),
                )
            )
        return ProviderResponse(
            "openalex",
            ProviderStatus.AVAILABLE,
            query,
            results,
            next_cursor=(data.get("meta") or {}).get("next_cursor"),
        )
    except urllib.error.HTTPError as exc:
        return ProviderResponse(
            "openalex",
            ProviderStatus.RATE_LIMITED
            if exc.code == 429
            else ProviderStatus.ACCESS_DENIED
            if exc.code in {401, 403}
            else ProviderStatus.PROVIDER_ERROR,
            query,
            [],
            str(exc),
        )
    except Exception as exc:
        return ProviderResponse("openalex", ProviderStatus.NETWORK_ERROR, query, [], str(exc))


def search_arxiv(
    query: str, *, limit: int = 10, start: int = 0, timeout: int = 20
) -> ProviderResponse:
    url = _query_url(
        "https://export.arxiv.org/api/query",
        {"search_query": f"all:{query}", "start": start, "max_results": max(1, min(limit, 50))},
    )
    try:
        payload, _ = _request_bytes(url, timeout=timeout)
        root = ET.fromstring(payload)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        results = []
        for index, item in enumerate(root.findall("a:entry", ns), start + 1):
            links = item.findall("a:link", ns)
            pdf = next(
                (link.attrib.get("href") for link in links if link.attrib.get("title") == "pdf"),
                None,
            )
            results.append(
                _result(
                    "arxiv",
                    {"xml": ET.tostring(item, encoding="unicode")},
                    query,
                    index,
                    title=item.findtext("a:title", namespaces=ns),
                    authors=[
                        a.findtext("a:name", default="", namespaces=ns)
                        for a in item.findall("a:author", ns)
                    ],
                    year=(item.findtext("a:published", namespaces=ns) or "")[:4],
                    venue="arXiv",
                    abstract=item.findtext("a:summary", namespaces=ns),
                    doi=None,
                    canonical=item.findtext("a:id", namespaces=ns),
                    pdf=pdf,
                    open_access=True,
                    citations=None,
                    provider_id=item.findtext("a:id", namespaces=ns),
                )
            )
        return ProviderResponse("arxiv", ProviderStatus.AVAILABLE, query, results)
    except Exception as exc:
        return ProviderResponse("arxiv", ProviderStatus.NETWORK_ERROR, query, [], str(exc))


def search_ieee(
    query: str,
    *,
    limit: int = 10,
    start: int = 1,
    timeout: int = 20,
    title: str | None = None,
    author: str | None = None,
    doi: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> ProviderResponse:
    api_key = os.getenv("IEEE_XPLORE_API_KEY")
    if not api_key:
        return ProviderResponse(
            "ieee",
            ProviderStatus.API_KEY_REQUIRED,
            query,
            [],
            "Set IEEE_XPLORE_API_KEY in the environment; the key is never logged or stored.",
        )
    params: dict[str, Any] = {
        "apikey": api_key,
        "querytext": query,
        "max_records": max(1, min(limit, 100)),
        "start_record": max(1, start),
    }
    if title:
        params["article_title"] = title
    if author:
        params["author"] = author
    if doi:
        params["doi"] = doi
    if year_from:
        params["start_year"] = year_from
    if year_to:
        params["end_year"] = year_to
    try:
        payload, _ = _request_bytes(
            _query_url("https://ieeexploreapi.ieee.org/api/v1/search/articles", params),
            timeout=timeout,
        )
        data = json.loads(payload)
        records = data.get("articles", [])
        results = [
            _result(
                "ieee",
                item,
                query,
                index + start,
                title=item.get("title"),
                authors=[
                    a.get("full_name") or a.get("name")
                    for a in item.get("authors", {}).get("authors", [])
                ]
                if isinstance(item.get("authors"), dict)
                else [],
                year=item.get("publication_year"),
                venue=item.get("publication_title"),
                abstract=item.get("abstract"),
                doi=item.get("doi"),
                canonical=item.get("html_url") or item.get("pdf_url"),
                pdf=item.get("pdf_url"),
                open_access=item.get("is_open_access")
                if isinstance(item.get("is_open_access"), bool)
                else None,
                citations=item.get("citing_paper_count"),
                provider_id=item.get("article_number"),
                ieee_number=item.get("article_number"),
            )
            for index, item in enumerate(records)
        ]
        return ProviderResponse("ieee", ProviderStatus.AVAILABLE, query, results)
    except urllib.error.HTTPError as exc:
        return ProviderResponse(
            "ieee",
            ProviderStatus.RATE_LIMITED
            if exc.code == 429
            else ProviderStatus.ACCESS_DENIED
            if exc.code in {401, 403}
            else ProviderStatus.PROVIDER_ERROR,
            query,
            [],
            f"HTTP {exc.code}",
        )
    except Exception as exc:
        return ProviderResponse("ieee", ProviderStatus.NETWORK_ERROR, query, [], _redact(str(exc)))


def search_semantic_scholar(query: str, *, limit: int = 10, timeout: int = 20) -> ProviderResponse:
    url = _query_url(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": "title,authors,year,venue,abstract,externalIds,openAccessPdf,citationCount,url",
        },
    )
    semantic_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers: dict[str, str] = {"x-api-key": semantic_key} if semantic_key else {}
    try:
        payload, _ = _request_bytes(url, headers=headers, timeout=timeout)
        data = json.loads(payload)
        results = []
        for index, item in enumerate(data.get("data", []), 1):
            external = item.get("externalIds") or {}
            oa = item.get("openAccessPdf") or {}
            results.append(
                _result(
                    "semantic_scholar",
                    item,
                    query,
                    index,
                    title=item.get("title"),
                    authors=[author.get("name") for author in item.get("authors", [])],
                    year=item.get("year"),
                    venue=item.get("venue"),
                    abstract=item.get("abstract"),
                    doi=external.get("DOI"),
                    canonical=item.get("url"),
                    pdf=oa.get("url"),
                    open_access=bool(oa.get("url")),
                    citations=item.get("citationCount"),
                    provider_id=item.get("paperId"),
                )
            )
        return ProviderResponse("semantic_scholar", ProviderStatus.AVAILABLE, query, results)
    except urllib.error.HTTPError as exc:
        return ProviderResponse(
            "semantic_scholar",
            ProviderStatus.RATE_LIMITED
            if exc.code == 429
            else ProviderStatus.ACCESS_DENIED
            if exc.code in {401, 403}
            else ProviderStatus.PROVIDER_ERROR,
            query,
            [],
            f"HTTP {exc.code}",
        )
    except Exception as exc:
        return ProviderResponse(
            "semantic_scholar", ProviderStatus.NETWORK_ERROR, query, [], str(exc)
        )


def search_general_web(query: str, *, limit: int = 10, timeout: int = 20) -> ProviderResponse:
    base = os.getenv("GENERAL_SEARCH_BASE_URL")
    if not base:
        return ProviderResponse(
            "general_web",
            ProviderStatus.PROVIDER_ERROR,
            query,
            [],
            "GENERAL_SEARCH_BASE_URL is not configured; HTML search scraping is not used",
        )
    url = _query_url(
        base.rstrip("/"), {"q": query, "format": "json", "limit": max(1, min(limit, 50))}
    )
    try:
        general_key = os.getenv("GENERAL_SEARCH_API_KEY")
        headers: dict[str, str] = {"Authorization": f"Bearer {general_key}"} if general_key else {}
        payload, _ = _request_bytes(
            url,
            headers=headers,
            timeout=timeout,
        )
        data = json.loads(payload)
        results = [
            _result(
                "general_web",
                item,
                query,
                index,
                title=item.get("title"),
                authors=[],
                year=None,
                venue=None,
                abstract=item.get("content") or item.get("snippet"),
                doi=None,
                canonical=item.get("url"),
                pdf=None,
                open_access=None,
                citations=None,
                provider_id=item.get("url"),
            )
            for index, item in enumerate(data.get("results", []), 1)
        ]
        return ProviderResponse("general_web", ProviderStatus.AVAILABLE, query, results)
    except Exception as exc:
        return ProviderResponse("general_web", ProviderStatus.NETWORK_ERROR, query, [], str(exc))


def deduplicate(results: Iterable[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    output: list[SearchResult] = []
    for result in results:
        keys = [
            result.doi.lower() if result.doi else None,
            re.sub(r"\W+", "", result.title.lower()),
            result.raw_record_hash,
        ]
        key = next((value for value in keys if value), result.provider_id or result.title)
        if key not in seen:
            seen.add(key)
            output.append(result)
    return output


def search_scholarly(
    query: str, *, providers: list[str] | None = None, limit: int = 10, offline: bool = False
) -> dict[str, Any]:
    providers = providers or ["crossref", "openalex", "arxiv"]
    if offline:
        return {"query": query, "offline": True, "responses": []}
    functions: dict[str, Callable[..., ProviderResponse]] = {
        "crossref": search_crossref,
        "openalex": search_openalex,
        "arxiv": search_arxiv,
        "ieee": search_ieee,
        "semantic_scholar": search_semantic_scholar,
        "general_web": search_general_web,
    }
    responses = [functions[name](query, limit=limit) for name in providers if name in functions]
    all_results = deduplicate(result for response in responses for result in response.results)
    return {
        "query": query,
        "providers": providers,
        "responses": [
            {
                "provider": response.provider,
                "status": response.status,
                "error": response.error,
                "results": [asdict(result) for result in response.results],
                "next_cursor": response.next_cursor,
            }
            for response in responses
        ],
        "deduplicated_results": [asdict(result) for result in all_results],
    }


def _blocked_host(host: str) -> bool:
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return True
    addresses: set[Any] = set()
    try:
        if host:
            try:
                addresses.add(ipaddress.ip_address(host))
            except ValueError:
                pass
        addresses.update(
            ipaddress.ip_address(value[4][0]) for value in socket.getaddrinfo(host, None)
        )
        return any(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            for address in addresses
        )
    except (ValueError, socket.gaierror):
        return True


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise urllib.error.HTTPError(
            req.full_url, code, "redirect blocked by safe fetcher", headers, None
        )


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.text: list[str] = []
        self.canonical: str | None = None
        self._title = False
        self._blocked = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "title":
            self._title = True
        if tag in {"script", "style", "noscript"}:
            self._blocked = True
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonical = attributes.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._title = False
        if tag in {"script", "style", "noscript"}:
            self._blocked = False

    def handle_data(self, data: str) -> None:
        if self._blocked:
            return
        if self._title:
            self.title.append(data)
        self.text.append(data)


def fetch_public_webpage(
    url: str, *, timeout: int = 20, max_bytes: int = 5_000_000
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or _blocked_host(parsed.hostname)
    ):
        raise ValueError("Only public HTTP(S) hosts are allowed")
    context = (
        ssl.create_default_context(cafile=_certifi.where())
        if _certifi is not None
        else ssl.create_default_context()
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), _NoRedirect()
    )
    request = urllib.request.Request(url, headers={"User-Agent": "FormalScience/0.1"})
    try:
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {
                "text/html",
                "text/plain",
                "application/json",
                "application/pdf",
            }:
                raise ValueError(f"unsupported content type: {content_type}")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("response exceeds maximum size")
            result: dict[str, Any] = {
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "content_type": content_type,
                "bytes": len(body),
            }
            if content_type == "text/html":
                parser = _PageParser()
                parser.feed(body.decode("utf-8", errors="replace"))
                result.update(
                    {
                        "title": _clean(" ".join(parser.title)),
                        "text": _clean(" ".join(parser.text)),
                        "canonical_url": urllib.parse.urljoin(url, parser.canonical)
                        if parser.canonical
                        else url,
                    }
                )
            elif content_type == "application/json":
                result["json"] = json.loads(body)
            elif content_type == "text/plain":
                result["text"] = body.decode("utf-8", errors="replace")
            else:
                result["pdf_bytes"] = len(body)
                result["pdf_magic"] = body.startswith(b"%PDF-")
            return result
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
