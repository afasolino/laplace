#!/usr/bin/env python3
"""Register an already-approved exact-commit reference snapshot in shared Laplace storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, cast

from research_workspace.engineering import JsonObject, ReferenceLibrary, ReferencePolicyError


Domain = Literal["python", "systemverilog"]


def _object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReferencePolicyError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferencePolicyError(f"{label} must be a non-empty string")
    return value.strip()


def _load_descriptor(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferencePolicyError(f"Cannot read registration descriptor: {exc}") from exc
    return _object(value, label="registration descriptor")


def _selected_files(
    descriptor: JsonObject, descriptor_path: Path
) -> list[tuple[Path, str, tuple[str, ...]]]:
    raw = descriptor.get("selected_files")
    if not isinstance(raw, list) or not raw:
        raise ReferencePolicyError("selected_files must be a non-empty list")
    selected: list[tuple[Path, str, tuple[str, ...]]] = []
    for index, item_raw in enumerate(raw):
        item = _object(item_raw, label=f"selected_files[{index}]")
        source_value = _text(item.get("path"), label=f"selected_files[{index}].path")
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = descriptor_path.parent / source
        topic = _text(item.get("topic"), label=f"selected_files[{index}].topic")
        topics_raw = item.get("topics")
        if (
            not isinstance(topics_raw, list)
            or not topics_raw
            or not all(
                isinstance(topic_value, str) and topic_value.strip() for topic_value in topics_raw
            )
        ):
            raise ReferencePolicyError(
                f"selected_files[{index}].topics must be a non-empty string list"
            )
        selected.append(
            (source.resolve(), topic, tuple(str(value).strip() for value in topics_raw))
        )
    return selected


def register_shared_reference(
    repository_root: Path,
    library_root: Path,
    domain: Domain,
    descriptor_path: Path,
) -> JsonObject:
    descriptor_file = descriptor_path.expanduser().resolve()
    descriptor = _load_descriptor(descriptor_file)
    licence_value = _text(descriptor.get("licence_path"), label="licence_path")
    licence = Path(licence_value).expanduser()
    if not licence.is_absolute():
        licence = descriptor_file.parent / licence
    library = ReferenceLibrary(library_root.expanduser().resolve(), domain, shared=True)
    catalog = (
        repository_root.resolve() / "codex_a6000" / "reference_sources" / f"{domain}_sources.yaml"
    )
    if not library.status().get("initialized"):
        library.initialize(catalog)
    verification = library.register_local(
        reference_id=_text(descriptor.get("reference_id"), label="reference_id"),
        repository=_text(descriptor.get("repository"), label="repository"),
        commit=_text(descriptor.get("commit"), label="commit"),
        licence_identifier=_text(descriptor.get("licence_identifier"), label="licence_identifier"),
        licence_path=licence.resolve(),
        selected_files=_selected_files(descriptor, descriptor_file),
        permitted_use=_text(descriptor.get("permitted_use"), label="permitted_use"),
        attribution=_text(descriptor.get("attribution"), label="attribution"),
    )
    ingestion = library.ingest()
    return {
        "status": "REGISTERED_AND_INDEXED",
        "domain": domain,
        "library_root": str(library.root),
        "snapshot_hash": library.snapshot_hash(),
        "verification": verification,
        "ingestion": ingestion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register an approved local reference descriptor into shared Laplace storage."
    )
    parser.add_argument("--library-root", required=True, type=Path)
    parser.add_argument("--domain", required=True, choices=("python", "systemverilog"))
    parser.add_argument("--descriptor", required=True, type=Path)
    arguments = parser.parse_args()
    result = register_shared_reference(
        Path(__file__).resolve().parents[1],
        arguments.library_root,
        cast(Domain, arguments.domain),
        arguments.descriptor,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
