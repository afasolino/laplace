from __future__ import annotations

import json

import pytest

from research_workspace.research_web_adapters import (
    ArxivResearchAdapter,
    CrossrefResearchAdapter,
    GitHubRepositorySpec,
    SearXNGResearchAdapter,
    SemanticScholarResearchAdapter,
    supported_web_adapter_names,
)


class _Client:
    maximum_bytes = 4_000_000

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def get_json(self, _url: str) -> object:
        return self.payload

    def get(self, _url: str, *, accept: str) -> bytes:
        assert accept == "application/atom+xml"
        assert isinstance(self.payload, bytes)
        return self.payload


def test_arxiv_adapter_preserves_exact_entry_metadata() -> None:
    atom = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2601.00001v2</id>
        <title>Bounded Local Research</title>
        <published>2026-01-02T00:00:00Z</published>
        <summary>A reproducible method.</summary>
        <author><name>A. Author</name></author>
      </entry>
    </feed>"""
    adapter = ArxivResearchAdapter(client=_Client(atom))  # type: ignore[arg-type]
    sources = adapter.discover("local research", limit=2)
    assert len(sources) == 1
    assert sources[0].source_type == "preprint"
    assert sources[0].revision == "2601.00001v2"
    fetched = adapter.fetch(sources[0])
    assert json.loads(fetched.content)["title"] == "Bounded Local Research"


def test_crossref_and_semantic_scholar_are_distinct_metadata_adapters() -> None:
    crossref = CrossrefResearchAdapter(
        client=_Client(  # type: ignore[arg-type]
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.1234/example",
                            "title": ["Paper"],
                            "author": [{"given": "Ada", "family": "Lovelace"}],
                            "container-title": ["Journal"],
                        }
                    ]
                }
            }
        )
    )
    source = crossref.discover("paper", limit=1)[0]
    assert source.canonical_url == "https://doi.org/10.1234/example"
    assert source.source_type == "peer_reviewed_literature"

    semantic = SemanticScholarResearchAdapter(
        client=_Client(  # type: ignore[arg-type]
            {
                "data": [
                    {
                        "paperId": "paper-id",
                        "title": "Paper",
                        "authors": [{"name": "Grace Hopper"}],
                        "year": 2025,
                    }
                ]
            }
        )
    )
    semantic_source = semantic.discover("paper", limit=1)[0]
    assert semantic_source.backend == "semantic_scholar"
    assert semantic_source.revision == "paper-id"


def test_searxng_must_be_self_hosted_and_github_requires_exact_revision() -> None:
    with pytest.raises(ValueError, match="local"):
        SearXNGResearchAdapter("https://search.example.org")
    SearXNGResearchAdapter("http://127.0.0.1:8080")
    with pytest.raises(ValueError, match="exact revision"):
        GitHubRepositorySpec(
            repository_url="https://github.com/example/project",
            revision="main",
            title="Project",
            license="MIT",
        )
    spec = GitHubRepositorySpec(
        repository_url="https://github.com/example/project",
        revision="a" * 40,
        title="Project",
        license="MIT",
    )
    assert spec.revision == "a" * 40
    assert {
        "searxng",
        "arxiv",
        "crossref",
        "semantic_scholar",
        "direct_url",
        "github_repository_inspection",
    } == set(supported_web_adapter_names())

