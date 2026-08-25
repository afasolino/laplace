from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from research_workspace.bounded_aci import BoundedACIError, BoundedRepositoryACI


def _repo(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _aci(root: Path, *, cancelled: list[bool] | None = None) -> BoundedRepositoryACI:
    return BoundedRepositoryACI(
        root,
        owner_user_id="owner",
        session_id="c3-session",
        allow_mutation=True,
        is_cancelled=(lambda: bool(cancelled and cancelled[0])) if cancelled is not None else None,
    )


def test_large_utf8_read_pages_reconstruct_exact_source_and_detect_drift(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    content = "".join(f"αβγ line {index}\n" for index in range(1_501))
    source = root / "large.py"
    source.write_text(content, encoding="utf-8")
    aci = _aci(root)
    pages: list[str] = []
    cursor: str | None = None
    snapshot: str | None = None
    while True:
        page = aci.read_page(path="large.py", cursor=cursor, max_lines=100)
        pages.append(str(page["content"]))
        assert len(str(page["content"]).encode("utf-8")) <= 32_000
        snapshot = str(page["snapshot_sha256"])
        cursor_value = page["next_cursor"]
        if cursor_value is None:
            break
        cursor = str(cursor_value)
    assert "".join(pages) == content
    assert snapshot == hashlib.sha256(content.encode("utf-8")).hexdigest()

    source.write_text(content + "drift\n", encoding="utf-8")
    assert cursor is not None
    with pytest.raises(BoundedACIError, match="aci_read_cursor_stale"):
        aci.read_page(path="large.py", cursor=cursor, max_lines=100)


def test_read_cursor_and_path_validation_fail_closed(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    (root / "source.sv").write_text("module top;\nendmodule\n", encoding="utf-8")
    aci = _aci(root)
    with pytest.raises(BoundedACIError, match="aci_read_cursor_invalid"):
        aci.read_page(path="source.sv", cursor="not-a-cursor", max_lines=10)
    with pytest.raises(BoundedACIError, match="aci_path_escape"):
        aci.read_page(path="../outside.sv", max_lines=10)
    with pytest.raises(BoundedACIError, match="aci_git_metadata_forbidden"):
        aci.read_page(path=".git/config", max_lines=10)


def test_large_chunked_create_is_ordered_idempotent_and_atomic(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    aci = _aci(root)
    chunks = ["αβγ\n" * 4_000 for _ in range(12)]
    expected = "".join(chunks)
    transaction = aci.begin_file_write(
        path="large.sv",
        expected_base_sha256=None,
        expected_bytes=len(expected.encode("utf-8")),
    )
    transaction_id = str(transaction["transaction_id"])
    offset = 0
    for sequence, chunk in enumerate(chunks):
        raw = chunk.encode("utf-8")
        chunk_hash = hashlib.sha256(raw).hexdigest()
        if sequence == 0:
            first = aci.write_file_chunk(
                transaction_id=transaction_id,
                sequence=sequence,
                offset=offset,
                content=chunk,
                chunk_sha256=chunk_hash,
            )
            retry = aci.write_file_chunk(
                transaction_id=transaction_id,
                sequence=sequence,
                offset=offset,
                content=chunk,
                chunk_sha256=chunk_hash,
            )
            assert first["idempotent_retry"] is False
            assert retry["idempotent_retry"] is True
        else:
            aci.write_file_chunk(
                transaction_id=transaction_id,
                sequence=sequence,
                offset=offset,
                content=chunk,
                chunk_sha256=chunk_hash,
            )
        offset += len(raw)
    finalized = aci.finalize_file_write(
        transaction_id=transaction_id,
        content_sha256=hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    )
    assert finalized["finalized"] is True
    assert (root / "large.sv").read_text(encoding="utf-8") == expected
    assert not list(root.glob(".*.aci.tmp"))


def test_chunk_conflict_order_hash_mismatch_and_abort_are_safe(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    aci = _aci(root)
    transaction = aci.begin_file_write(path="target.py", expected_bytes=6)
    transaction_id = str(transaction["transaction_id"])
    first = "abc"
    first_hash = hashlib.sha256(first.encode()).hexdigest()
    aci.write_file_chunk(
        transaction_id=transaction_id,
        sequence=0,
        offset=0,
        content=first,
        chunk_sha256=first_hash,
    )
    with pytest.raises(BoundedACIError, match="aci_write_duplicate_chunk_conflict"):
        aci.write_file_chunk(
            transaction_id=transaction_id,
            sequence=0,
            offset=0,
            content="xyz",
            chunk_sha256=hashlib.sha256(b"xyz").hexdigest(),
        )
    with pytest.raises(BoundedACIError, match="aci_write_chunk_out_of_order"):
        aci.write_file_chunk(
            transaction_id=transaction_id,
            sequence=2,
            offset=3,
            content="def",
            chunk_sha256=hashlib.sha256(b"def").hexdigest(),
        )
    with pytest.raises(BoundedACIError, match="aci_write_chunk_hash_mismatch"):
        aci.write_file_chunk(
            transaction_id=transaction_id,
            sequence=1,
            offset=3,
            content="def",
            chunk_sha256="0" * 64,
        )
    aci.abort_file_write(transaction_id)
    assert not (root / "target.py").exists()
    assert not list(root.glob(".*.aci.tmp"))
    with pytest.raises(BoundedACIError, match="aci_write_transaction_not_found"):
        aci.finalize_file_write(transaction_id=transaction_id, content_sha256="0" * 64)


def test_finalize_hash_drift_and_cancellation_preserve_previous_state(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    target = root / "source.py"
    target.write_text("old\n", encoding="utf-8")
    aci = _aci(root)
    base_hash = hashlib.sha256(b"old\n").hexdigest()
    transaction = aci.begin_file_write(path="source.py", expected_base_sha256=base_hash)
    transaction_id = str(transaction["transaction_id"])
    value = "new\n"
    aci.write_file_chunk(
        transaction_id=transaction_id,
        sequence=0,
        offset=0,
        content=value,
        chunk_sha256=hashlib.sha256(value.encode()).hexdigest(),
    )
    target.write_text("external\n", encoding="utf-8")
    with pytest.raises(BoundedACIError, match="aci_write_target_drift"):
        aci.finalize_file_write(
            transaction_id=transaction_id,
            content_sha256=hashlib.sha256(value.encode()).hexdigest(),
        )
    aci.abort_file_write(transaction_id)
    assert target.read_text(encoding="utf-8") == "external\n"

    cancelled = [False]
    cancellable = _aci(root, cancelled=cancelled)
    new_transaction = cancellable.begin_file_write(path="cancelled.py", expected_bytes=4)
    cancelled[0] = True
    with pytest.raises(BoundedACIError, match="aci_write_cancelled"):
        cancellable.write_file_chunk(
            transaction_id=str(new_transaction["transaction_id"]),
            sequence=0,
            offset=0,
            content="data",
            chunk_sha256=hashlib.sha256(b"data").hexdigest(),
        )
    assert not (root / "cancelled.py").exists()
    assert not list(root.glob(".*.aci.tmp"))
