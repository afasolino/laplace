"""Prepare and validate one immutable, language-separated experiment corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from .engineering import (
    Domain,
    JsonObject,
    ReferenceLibrary,
    ReferencePolicyError,
    _inside,
)


_DOMAINS: tuple[Domain, ...] = ("python", "c", "verilog", "systemverilog")
_EXTERNAL_MANIFEST = Path("codex_a6000/governed_corpus/installed_external_references.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReferencePolicyError(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ReferencePolicyError(
            f"{label} keys are invalid; missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def load_bundled_corpus_manifest(repository_root: Path) -> JsonObject:
    path = repository_root.resolve() / "codex_a6000" / "governed_corpus" / "manifest.json"
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferencePolicyError(f"Cannot read bundled corpus manifest: {exc}") from exc
    value = _object(raw, label="Bundled corpus manifest")
    _exact_keys(
        value,
        {
            "schema_version",
            "release",
            "repository",
            "licence_identifier",
            "licence_path",
            "licence_sha256",
            "permitted_use",
            "attribution",
            "references",
        },
        label="Bundled corpus manifest",
    )
    if value.get("schema_version") != 1:
        raise ReferencePolicyError("Bundled corpus schema_version must equal 1")
    licence_raw = value.get("licence_path")
    licence_hash = value.get("licence_sha256")
    if not isinstance(licence_raw, str) or not isinstance(licence_hash, str):
        raise ReferencePolicyError("Bundled corpus licence metadata is malformed")
    licence = _inside(repository_root.resolve(), repository_root.resolve() / licence_raw)
    if not licence.is_file() or _sha256(licence) != licence_hash:
        raise ReferencePolicyError("Bundled corpus licence hash verification failed")
    references = value.get("references")
    if not isinstance(references, list) or len(references) != 2:
        raise ReferencePolicyError("Bundled corpus must contain C and Verilog references")
    seen_domains: set[str] = set()
    for index, raw_reference in enumerate(references):
        reference = _object(raw_reference, label=f"Bundled reference {index}")
        _exact_keys(
            reference,
            {"reference_id", "domain", "files"},
            label=f"Bundled reference {index}",
        )
        domain = reference.get("domain")
        if domain not in {"c", "verilog"} or domain in seen_domains:
            raise ReferencePolicyError("Bundled reference domains must be unique C and Verilog")
        seen_domains.add(str(domain))
        files = reference.get("files")
        if not isinstance(files, list) or not files:
            raise ReferencePolicyError(f"Bundled reference {index} has no selected files")
        for file_index, raw_file in enumerate(files):
            item = _object(raw_file, label=f"Bundled reference {index} file {file_index}")
            _exact_keys(
                item,
                {"path", "selected_topic", "sha256", "topics"},
                label=f"Bundled reference {index} file {file_index}",
            )
            source_path = item.get("path")
            digest = item.get("sha256")
            topics = item.get("topics")
            if (
                not isinstance(source_path, str)
                or not isinstance(digest, str)
                or not isinstance(topics, list)
                or not topics
                or not all(isinstance(topic, str) and topic for topic in topics)
            ):
                raise ReferencePolicyError("Bundled selected-file metadata is malformed")
            source = _inside(repository_root.resolve(), repository_root.resolve() / source_path)
            if not source.is_file() or _sha256(source) != digest:
                raise ReferencePolicyError(f"Bundled selected-file hash failed: {source_path}")
    return value


def load_installed_external_manifest(repository_root: Path) -> JsonObject:
    """Load only actually acquired, hash-complete C/Verilog external sources."""
    root = repository_root.resolve()
    path = root / _EXTERNAL_MANIFEST
    if not path.is_file():
        raise ReferencePolicyError(
            "Required external C/Verilog governed references are not installed; run "
            "scripts/acquire_multilanguage_governed_references.py with approved network access"
        )
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferencePolicyError(f"Cannot read installed external manifest: {exc}") from exc
    value = _object(raw, label="Installed external manifest")
    _exact_keys(
        value,
        {"schema_version", "release", "sources"},
        label="Installed external manifest",
    )
    if value.get("schema_version") != 1:
        raise ReferencePolicyError("Installed external manifest schema_version must equal 1")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ReferencePolicyError("Installed external manifest has no sources")
    domains: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = _object(raw_source, label=f"Installed external source {index}")
        _exact_keys(
            source,
            {
                "reference_id",
                "domain",
                "repository",
                "revision_kind",
                "revision",
                "resolved_commit",
                "licence_identifier",
                "licence_path",
                "licence_sha256",
                "permitted_use",
                "attribution",
                "files",
            },
            label=f"Installed external source {index}",
        )
        domain = source.get("domain")
        if domain not in {"c", "verilog"}:
            raise ReferencePolicyError("External governed source domain must be C or Verilog")
        domains.add(str(domain))
        revision_kind = source.get("revision_kind")
        revision = source.get("revision")
        resolved_commit = source.get("resolved_commit")
        if revision_kind not in {"commit", "release"} or not isinstance(revision, str):
            raise ReferencePolicyError("External governed revision metadata is malformed")
        if (
            not isinstance(resolved_commit, str)
            or len(resolved_commit) != 40
            or any(character not in "0123456789abcdef" for character in resolved_commit)
        ):
            raise ReferencePolicyError("External resolved commit must be an exact SHA-1")
        if revision_kind == "commit" and len(revision) != 40:
            raise ReferencePolicyError(
                "External commit revisions must be exact 40-character hashes"
            )
        if revision_kind == "commit" and revision != resolved_commit:
            raise ReferencePolicyError("External commit and resolved commit differ")
        licence_path = source.get("licence_path")
        licence_hash = source.get("licence_sha256")
        if not isinstance(licence_path, str) or not isinstance(licence_hash, str):
            raise ReferencePolicyError("External licence metadata is malformed")
        licence = _inside(root, root / licence_path)
        if not licence.is_file() or _sha256(licence) != licence_hash:
            raise ReferencePolicyError(f"External licence hash failed: {licence_path}")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise ReferencePolicyError("External governed source has no selected files")
        for raw_file in files:
            item = _object(raw_file, label="External selected file")
            _exact_keys(
                item,
                {"path", "selected_topic", "topics", "sha256"},
                label="External selected file",
            )
            file_path = item.get("path")
            digest = item.get("sha256")
            topics = item.get("topics")
            if (
                not isinstance(file_path, str)
                or not isinstance(digest, str)
                or not isinstance(topics, list)
                or not topics
                or not all(isinstance(topic, str) and topic for topic in topics)
            ):
                raise ReferencePolicyError("External selected-file metadata is malformed")
            selected = _inside(root, root / file_path)
            if not selected.is_file() or _sha256(selected) != digest:
                raise ReferencePolicyError(f"External selected-file hash failed: {file_path}")
    if domains != {"c", "verilog"}:
        raise ReferencePolicyError("Installed external corpus must cover both C and Verilog")
    return value


def _copy_registered_domain(base_root: Path, overlay_root: Path, domain: Domain) -> JsonObject:
    source_library = ReferenceLibrary(base_root, domain, shared=True)
    verification = source_library.verify()
    if verification.get("status") != "VERIFIED":
        raise ReferencePolicyError(f"Base {domain} corpus is not verified")
    target_library = ReferenceLibrary(overlay_root, domain, shared=True)
    catalog = (
        Path(__file__).resolve().parents[2]
        / "codex_a6000"
        / "reference_sources"
        / f"{domain}_sources.yaml"
    )
    target_library.initialize(catalog)
    for manifest in source_library.manifests_list():
        licence = (
            source_library.root / "sources" / manifest.reference_id / manifest.commit / "LICENSE"
        )
        selected = [
            (
                source_library.root / record.selected_path,
                str(Path(record.selected_path).parent),
                record.topics,
            )
            for record in manifest.files
        ]
        if manifest.revision_kind == "release":
            target_library.register_local_release(
                reference_id=manifest.reference_id,
                repository=manifest.repository,
                release=manifest.commit,
                licence_identifier=manifest.licence_identifier,
                licence_path=licence,
                selected_files=selected,
                permitted_use=manifest.permitted_use,
                attribution=manifest.attribution,
            )
        else:
            target_library.register_local(
                reference_id=manifest.reference_id,
                repository=manifest.repository,
                commit=manifest.commit,
                licence_identifier=manifest.licence_identifier,
                licence_path=licence,
                selected_files=selected,
                permitted_use=manifest.permitted_use,
                attribution=manifest.attribution,
            )
    return {
        "domain": domain,
        "verification": target_library.verify(),
        "ingestion": target_library.ingest(),
        "snapshot_hash": target_library.snapshot_hash(),
    }


def bootstrap_bundled_extensions(repository_root: Path, overlay_root: Path) -> JsonObject:
    root = repository_root.resolve()
    manifest = load_bundled_corpus_manifest(root)
    licence = _inside(root, root / str(manifest["licence_path"]))
    references = manifest["references"]
    if not isinstance(references, list):
        raise ReferencePolicyError("Bundled reference list is malformed")
    results: list[JsonObject] = []
    for raw_reference in references:
        reference = _object(raw_reference, label="Bundled reference")
        domain_raw = reference.get("domain")
        if domain_raw == "c":
            domain: Domain = "c"
        elif domain_raw == "verilog":
            domain = "verilog"
        else:
            raise ReferencePolicyError("Bundled extension domain is invalid")
        library = ReferenceLibrary(overlay_root, domain, shared=True)
        library.initialize(root / "codex_a6000" / "reference_sources" / f"{domain}_sources.yaml")
        files_raw = reference.get("files")
        if not isinstance(files_raw, list):
            raise ReferencePolicyError("Bundled reference files are malformed")
        selected: list[tuple[Path, str, tuple[str, ...]]] = []
        for raw_file in files_raw:
            item = _object(raw_file, label="Bundled selected file")
            topics = item.get("topics")
            if not isinstance(topics, list) or not all(isinstance(topic, str) for topic in topics):
                raise ReferencePolicyError("Bundled selected file topics are malformed")
            selected.append(
                (
                    _inside(root, root / str(item["path"])),
                    str(item["selected_topic"]),
                    tuple(topics),
                )
            )
        library.register_local_release(
            reference_id=str(reference["reference_id"]),
            repository=str(manifest["repository"]),
            release=str(manifest["release"]),
            licence_identifier=str(manifest["licence_identifier"]),
            licence_path=licence,
            selected_files=selected,
            permitted_use=str(manifest["permitted_use"]),
            attribution=str(manifest["attribution"]),
        )
        results.append(
            {
                "domain": domain,
                "verification": library.verify(),
                "ingestion": library.ingest(),
                "snapshot_hash": library.snapshot_hash(),
            }
        )
    return {"status": "EXTENSIONS_REGISTERED", "domains": results}


def bootstrap_external_extensions(repository_root: Path, overlay_root: Path) -> JsonObject:
    root = repository_root.resolve()
    manifest = load_installed_external_manifest(root)
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise ReferencePolicyError("Installed external source list is malformed")
    records: list[JsonObject] = []
    for raw_source in raw_sources:
        source = _object(raw_source, label="Installed external source")
        domain = cast(Domain, source["domain"])
        library = ReferenceLibrary(overlay_root, domain, shared=True)
        library.initialize(root / "codex_a6000" / "reference_sources" / f"{domain}_sources.yaml")
        selected: list[tuple[Path, str, tuple[str, ...]]] = []
        raw_files = source.get("files")
        if not isinstance(raw_files, list):
            raise ReferencePolicyError("External selected files are malformed")
        for raw_file in raw_files:
            item = _object(raw_file, label="External selected file")
            selected.append(
                (
                    _inside(root, root / str(item["path"])),
                    str(item["selected_topic"]),
                    tuple(cast(list[str], item["topics"])),
                )
            )
        if source["revision_kind"] == "release":
            library.register_local_release(
                reference_id=str(source["reference_id"]),
                repository=str(source["repository"]),
                release=str(source["revision"]),
                licence_identifier=str(source["licence_identifier"]),
                licence_path=_inside(root, root / str(source["licence_path"])),
                selected_files=selected,
                permitted_use=str(source["permitted_use"]),
                attribution=str(source["attribution"]),
            )
        else:
            library.register_local(
                reference_id=str(source["reference_id"]),
                repository=str(source["repository"]),
                commit=str(source["revision"]),
                licence_identifier=str(source["licence_identifier"]),
                licence_path=_inside(root, root / str(source["licence_path"])),
                selected_files=selected,
                permitted_use=str(source["permitted_use"]),
                attribution=str(source["attribution"]),
            )
        records.append(
            {
                "reference_id": source["reference_id"],
                "domain": domain,
                "verification": library.verify(str(source["reference_id"])),
                "ingestion": library.ingest(),
            }
        )
    return {"status": "EXTERNAL_EXTENSIONS_REGISTERED", "sources": records}


def prepare_corpus_overlay(
    repository_root: Path,
    base_root: Path,
    overlay_root: Path,
    *,
    require_external: bool = True,
) -> JsonObject:
    """Copy verified Python/SV evidence and add bundled C/Verilog releases."""
    base = base_root.expanduser().resolve()
    overlay = overlay_root.expanduser().resolve()
    if base == overlay:
        raise ReferencePolicyError("Corpus overlay must not modify the authoritative base root")
    copied = [
        _copy_registered_domain(base, overlay, "python"),
        _copy_registered_domain(base, overlay, "systemverilog"),
    ]
    extensions = bootstrap_bundled_extensions(repository_root, overlay)
    external = (
        bootstrap_external_extensions(repository_root, overlay)
        if require_external
        else {"status": "EXTERNAL_ACQUISITION_NOT_REQUIRED_FOR_TEST_FIXTURE"}
    )
    return {
        "status": "CORPUS_OVERLAY_READY",
        "base_root": str(base),
        "overlay_root": str(overlay),
        "copied": copied,
        "extensions": extensions,
        "external": external,
    }


def validate_corpus_retrieval(overlay_root: Path, *, require_external: bool = True) -> JsonObject:
    """Prove non-empty, domain-separated retrieval for all four languages."""
    queries: dict[Domain, str] = {
        "python": "strict Pydantic validation asyncio cancellation SQLite transaction",
        "c": "C11 ownership lifetime integer overflow sanitizer bounded buffer",
        "verilog": "Verilog-2001 ready valid parameterized synthesis reset Icarus Yosys",
        "systemverilog": "SystemVerilog AXI4-Lite WSTRB W1C ready valid assertions",
    }
    records: list[JsonObject] = []
    passed = True
    for domain in _DOMAINS:
        library = ReferenceLibrary(overlay_root, domain, shared=True)
        verification = library.verify()
        chunks = library.search_chunks(queries[domain], limit=3, token_budget=1200)
        ranked_chunks = [dict(chunk, rank=index) for index, chunk in enumerate(chunks, start=1)]
        manifests = library.manifests_list()
        has_external = any(not manifest.repository.startswith("local://") for manifest in manifests)
        external_required = require_external and domain in {"c", "verilog"}
        domain_passed = (
            verification.get("status") == "VERIFIED"
            and bool(chunks)
            and (has_external or not external_required)
        )
        passed = passed and domain_passed
        records.append(
            {
                "domain": domain,
                "passed": domain_passed,
                "verification": verification,
                "snapshot_hash": library.snapshot_hash(),
                "retrieved_chunk_count": len(chunks),
                "retrieved": ranked_chunks,
                "external_reference_required": external_required,
                "external_reference_present": has_external,
            }
        )
    return {
        "status": "VERIFIED_NON_EMPTY" if passed else "FAILED",
        "language_separated": True,
        "domains": records,
    }
