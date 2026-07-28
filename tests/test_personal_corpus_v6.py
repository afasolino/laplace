from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from research_workspace.personal_corpus import (
    CorpusError,
    PersonalCorpusPolicy,
    PersonalCorpusStore,
)


def _docx(*, macro: bool = False, external: bool = False) -> bytes:
    value = io.BytesIO()
    with zipfile.ZipFile(value, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="urn:test"><w:body><w:p><w:r>'
                "<w:t>Laplace DOCX evidence</w:t>"
                "</w:r></w:p></w:body></w:document>"
            ),
        )
        if macro:
            archive.writestr("word/vbaProject.bin", b"macro")
        if external:
            archive.writestr(
                "word/_rels/document.xml.rels",
                b'<Relationship TargetMode="External" Target="https://example.test"/>',
            )
    return value.getvalue()


def _pdf() -> bytes:
    value = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(value)
    return value.getvalue()


def test_personal_corpus_policy_manifest_index_search_delete_and_isolation(
    tmp_path: Path,
) -> None:
    policy = PersonalCorpusPolicy(min_free_disk_bytes=1)
    store = PersonalCorpusStore(tmp_path, policy=policy)
    corpus = store.create_corpus("owner-one", "Private references")
    corpus_id = str(corpus["corpus_id"])
    upload = store.create_upload(
        "owner-one", corpus_id, idempotency_key="upload:test-one"
    )
    upload_id = str(upload["upload_id"])

    markdown = store.stage_file(
        "owner-one",
        upload_id,
        logical_path="notes/evidence.md",
        content=b"# Evidence\nLaplace private retrieval marker.\n",
        client_mime="text/markdown",
    )
    assert markdown["state"] == "ACCEPTED"
    assert markdown["support_label"] == "retrieval_only"
    resumed = store.stage_file(
        "owner-one",
        upload_id,
        logical_path="notes/evidence.md",
        content=b"# Evidence\nLaplace private retrieval marker.\n",
        client_mime="text/markdown",
    )
    assert resumed["state"] == "ACCEPTED"
    assert resumed["idempotent"] is True
    python = store.stage_file(
        "owner-one",
        upload_id,
        logical_path="src/check.py",
        content=b'API_TOKEN = "secret-value-with-many-characters"\n',
        client_mime="text/x-python",
    )
    assert python["state"] == "ACCEPTED"
    assert python["support_label"] == "python_reference"
    assert "api_token" in python["warnings"]
    rejected = store.stage_file(
        "owner-one",
        upload_id,
        logical_path="../escape.py",
        content=b"print('no')\n",
        client_mime="text/x-python",
    )
    assert rejected["state"] == "REJECTED"
    assert rejected["reason"] == "upload_path_traversal"

    manifest = store.upload_manifest("owner-one", upload_id)
    assert manifest["accepted_count"] == 2
    assert manifest["rejected_count"] == 1
    indexed = store.index_upload(
        "owner-one", upload_id, idempotency_key="index:test-one"
    )
    assert indexed["status"] == "INDEXED"
    assert indexed["indexed_sources"] == 2
    assert store.index_upload(
        "owner-one", upload_id, idempotency_key="index:test-one"
    ) == indexed

    found = store.search(
        "owner-one", "private retrieval marker", corpus_id=corpus_id
    )
    assert found["retrieval_used"] is True
    first = found["results"][0]
    assert first["file"] == "notes/evidence.md"
    assert str(first["chunk_id"]).startswith("chk_")
    assert first["citation"] == {
        "file": "notes/evidence.md",
        "page": None,
        "section": None,
        "chunk_id": first["chunk_id"],
    }
    assert store.list_corpora("owner-two") == []
    with pytest.raises(CorpusError, match="corpus_not_found"):
        store.search("owner-two", "private", corpus_id=corpus_id)

    source_id = str(first["source_id"])
    deleted = store.delete_source("owner-one", corpus_id, source_id)
    assert deleted["status"] == "SOFT_DELETED"
    assert (
        store.search("owner-one", "private retrieval marker", corpus_id=corpus_id)[
            "retrieval_used"
        ]
        is False
    )
    owner_directories = [path.name for path in (store.root / "owners").iterdir()]
    assert owner_directories and "owner-one" not in " ".join(owner_directories)


def test_format_magic_json_docx_pdf_and_controlled_zip_validation(
    tmp_path: Path,
) -> None:
    store = PersonalCorpusStore(
        tmp_path, policy=PersonalCorpusPolicy(min_free_disk_bytes=1)
    )
    corpus_id = str(store.create_corpus("owner", "Mixed")["corpus_id"])
    upload_id = str(
        store.create_upload(
            "owner", corpus_id, idempotency_key="upload:mixed-one"
        )["upload_id"]
    )
    assert store.stage_file(
        "owner",
        upload_id,
        logical_path="records.jsonl",
        content=b'{"valid": true}\n{"second": 2}\n',
        client_mime="application/x-ndjson",
    )["state"] == "ACCEPTED"
    assert store.stage_file(
        "owner",
        upload_id,
        logical_path="paper.pdf",
        content=_pdf(),
        client_mime="application/pdf",
    )["state"] == "ACCEPTED"
    assert store.stage_file(
        "owner",
        upload_id,
        logical_path="report.docx",
        content=_docx(),
        client_mime=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )["state"] == "ACCEPTED"
    macro = store.stage_file(
        "owner",
        upload_id,
        logical_path="macro.docx",
        content=_docx(macro=True),
        client_mime="application/octet-stream",
    )
    assert macro["state"] == "REJECTED"
    assert macro["reason"] == "macro_or_embedded_executable_rejected"
    bad_json = store.stage_file(
        "owner",
        upload_id,
        logical_path="invalid.json",
        content=b"{not valid}",
        client_mime="application/json",
    )
    assert bad_json["state"] == "REJECTED"
    assert bad_json["reason"] == "invalid_json"
    assert store.index_upload(
        "owner", upload_id, idempotency_key="index:mixed-one"
    )["indexed_sources"] == 3

    zip_upload = str(
        store.create_upload(
            "owner", corpus_id, idempotency_key="upload:zip-safe"
        )["upload_id"]
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("folder/Makefile", "all:\n\t@echo safe\n")
        value.writestr("folder/design.sv", "module design; endmodule\n")
    result = store.stage_zip_fallback(
        "owner", zip_upload, content=archive.getvalue()
    )
    assert result["accepted"] == 2

    unsafe = io.BytesIO()
    with zipfile.ZipFile(unsafe, "w") as value:
        value.writestr("../escape.txt", "unsafe")
    with pytest.raises(CorpusError, match="upload_path_traversal"):
        store.stage_zip_fallback("owner", zip_upload, content=unsafe.getvalue())


def test_policy_rejects_old_office_executable_nested_archive_and_collisions(
    tmp_path: Path,
) -> None:
    store = PersonalCorpusStore(
        tmp_path, policy=PersonalCorpusPolicy(min_free_disk_bytes=1)
    )
    corpus_id = str(store.create_corpus("owner", "Reject")["corpus_id"])
    upload_id = str(
        store.create_upload(
            "owner", corpus_id, idempotency_key="upload:reject-one"
        )["upload_id"]
    )
    cases = [
        ("legacy.doc", b"\xd0\xcf\x11\xe0", "unsupported_extension"),
        ("binary.py", b"\x7fELFpayload", "executable_rejected"),
        ("nested.zip", b"PK\x03\x04", "zip_fallback_requires_controlled_endpoint"),
    ]
    for name, content, reason in cases:
        result = store.stage_file(
            "owner",
            upload_id,
            logical_path=name,
            content=content,
            client_mime="application/octet-stream",
        )
        assert result["state"] == "REJECTED"
        assert result["reason"] == reason
    store.stage_file(
        "owner",
        upload_id,
        logical_path="Résumé.txt",
        content=b"first",
        client_mime="text/plain",
    )
    with pytest.raises(CorpusError, match="duplicate_or_normalization_collision"):
        store.stage_file(
            "owner",
            upload_id,
            logical_path="Re\u0301sume\u0301.txt",
            content=b"second",
            client_mime="text/plain",
        )
    assert json.loads(store.audit_path.read_text().splitlines()[0])["owner_pseudonym"]
