from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.analysis import compare_runs, parse_log
from research_workspace.acquisition import download_open_access, promote_download
from research_workspace.core import ConfigurationError, load_settings
from research_workspace.documents import ingest
from research_workspace.drafting import ground_text, preserve_numbers
from research_workspace.extraction import Provenance, extract_metrics
from research_workspace.ieee_browser import browser_init, browser_profile, queue_candidate
from research_workspace.library import COLLECTION_MAP
from research_workspace.llm import MockProvider, benchmark
from research_workspace.online import (
    ProviderStatus,
    SearchResult,
    deduplicate,
    fetch_public_webpage,
    search_crossref,
    search_ieee,
)
from research_workspace.api import create_app
from research_workspace.optional import npu_status
from research_workspace.probe import collect_probe
from research_workspace.projects import ProjectError, init_project, validate_project
from research_workspace.retrieval import evidence_packet, search, validate_citations


def test_config_and_loopback() -> None:
    settings = load_settings()
    assert settings.bind_host == "127.0.0.1"
    assert settings.context_tokens == 8192
    with pytest.raises(ConfigurationError):
        import os

        old = os.environ.get("RW_BIND_HOST")
        os.environ["RW_BIND_HOST"] = "0.0.0.0"
        try:
            load_settings()
        finally:
            if old is None:
                os.environ.pop("RW_BIND_HOST", None)
            else:
                os.environ["RW_BIND_HOST"] = old


def test_probe_missing_tool() -> None:
    result = collect_probe(lambda command: (_ for _ in ()).throw(FileNotFoundError()))
    assert result["nvidia"]["status"] in {"unavailable", "unsupported"}


def test_ingest_retrieval_and_citation(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text(
        "Our accelerator reports 12.5 mW latency and uses a router.", encoding="utf-8"
    )
    root = tmp_path
    database = root / "data" / "sqlite" / "workspace.db"
    result = ingest(source, root, database, "user_work")
    assert result["status"] == "ingested"
    evidence = search(database, "router accelerator", limit=3)
    packet = evidence_packet("router accelerator", evidence)
    assert packet["grounded"]
    assert validate_citations(
        packet,
        [
            {
                "filename": evidence[0].filename,
                "page": evidence[0].page,
                "chunk_id": evidence[0].chunk_id,
            }
        ],
    )
    duplicate = ingest(source, root, database, "user_work")
    assert duplicate["status"] == "duplicate"


def test_extraction_requires_provenance_and_unit() -> None:
    record = extract_metrics(
        "power=12.5 mW", Provenance(filename="x.txt", page=1, chunk_id="d:p1:c0")
    )
    assert record.metrics[0].unit == "mW"
    assert record.metrics[0].provenance.page == 1


def test_analysis_first_error_and_compare() -> None:
    failed = parse_log("compile: 10 ms\nfatal error: missing header\nlatency=3 ms")
    assert failed["first_actionable_error"]["line"] == 2
    assert compare_runs({"metrics": [{"name": "latency", "value": "2", "unit": "ms"}]}, failed)[
        "observed"
    ]


def test_drafting_and_mock_benchmark() -> None:
    assert "[SOURCE REQUIRED]" in ground_text("Claim", evidence_packet("x", []))
    assert preserve_numbers("latency 3 ms", "latency three ms") == ["3 ms"]
    assert benchmark(MockProvider(), ["hello"])["results"][0]["status"] == "mock"


def test_api_local_and_upload_validation(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(tmp_path, tmp_path / "db.sqlite"))
    assert client.get("/dashboard").status_code == 200
    assert client.get("/dashboard").headers["content-type"].startswith("text/html")
    response = client.post("/ingest", files={"file": ("bad.exe", b"x")})
    assert response.status_code == 415
    assert npu_status().status in {"NPU_OPTIONAL_NOT_BENEFICIAL", "AVAILABLE_UNVALIDATED"}


def test_malformed_pdf_is_quarantined(tmp_path: Path) -> None:
    source = tmp_path / "bad.pdf"
    source.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        ingest(source, tmp_path, tmp_path / "db.sqlite", "project_document")
    assert list((tmp_path / "data" / "quarantine").glob("*.json"))


def test_project_initialization_safety_and_nonoverwrite(tmp_path: Path) -> None:
    paths = init_project("Example_Project", root=tmp_path)
    assert validate_project("Example_Project", root=tmp_path)["valid"]
    with pytest.raises(ProjectError):
        init_project("../escape", root=tmp_path)
    with pytest.raises(ProjectError):
        init_project("Example_Project", root=tmp_path)
    assert paths.config.is_file()


def test_library_collection_mapping() -> None:
    assert COLLECTION_MAP["MyWorks"]["source_class"] == "user_work"
    assert COLLECTION_MAP["Documentations"]["source_class"] == "technical_documentation"


def test_provider_normalization_dedup_and_ieee_key(monkeypatch: pytest.MonkeyPatch) -> None:
    first = SearchResult(
        "crossref",
        "1",
        "Same Title",
        [],
        2024,
        None,
        None,
        "10.1/x",
        None,
        None,
        None,
        None,
        None,
        "now",
        "hash1",
        "q",
        1,
    )
    second = SearchResult(
        "openalex",
        "2",
        "Same Title",
        [],
        2024,
        None,
        None,
        "10.1/x",
        None,
        None,
        None,
        None,
        None,
        "now",
        "hash2",
        "q",
        1,
    )
    assert len(deduplicate([first, second])) == 1
    monkeypatch.delenv("IEEE_XPLORE_API_KEY", raising=False)
    assert search_ieee("sram", limit=1).status == ProviderStatus.API_KEY_REQUIRED


def test_ieee_secret_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "not-a-real-secret"
    monkeypatch.setenv("IEEE_XPLORE_API_KEY", secret)
    monkeypatch.setattr(
        "research_workspace.online._request_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    response = search_ieee("sram", limit=1)
    assert response.status == ProviderStatus.NETWORK_ERROR
    assert secret not in (response.error or "")


def test_crossref_mock_pagination_and_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1/test",
                    "title": ["Test"],
                    "author": [{"given": "A", "family": "B"}],
                    "published": {"date-parts": [[2024]]},
                }
            ]
        }
    }
    monkeypatch.setattr(
        "research_workspace.online._request_bytes",
        lambda *args, **kwargs: (json_bytes(payload), {}),
    )
    result = search_crossref("test", limit=1)
    assert result.status == ProviderStatus.AVAILABLE
    assert result.results[0].doi == "10.1/test"


def test_safe_fetch_and_open_access_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fetch_public_webpage("http://127.0.0.1:11434/api/tags")
    init_project("Acquisition", root=tmp_path)
    with pytest.raises(PermissionError):
        download_open_access(
            "Acquisition",
            {"title": "No proof", "pdf_url": "https://example.org/file.pdf", "open_access": False},
            root=tmp_path,
        )
    with pytest.raises(PermissionError):
        promote_download("Acquisition", "missing.pdf", "MyTopics", root=tmp_path)


def test_browser_profile_isolated_and_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "browser-profile"
    monkeypatch.setenv("FORMALSCIENCE_BROWSER_PROFILE", str(profile))
    assert browser_profile() == profile.resolve()
    assert browser_init()["profile"] == str(profile.resolve())
    init_project("BrowserProject", root=tmp_path)
    queued = queue_candidate(
        "BrowserProject", {"title": "Candidate", "provider": "crossref"}, root=tmp_path
    )
    assert queued["status"] == "QUEUED"


def json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value).encode()
