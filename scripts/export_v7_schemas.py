#!/usr/bin/env python3
"""Export deterministic JSON schemas for v7 persisted/inter-service contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from research_workspace.contracts import (
    ArtifactV1,
    AttachmentV1,
    AuditEventV1,
    CapabilityAssignmentV1,
    ConversationV1,
    CorpusSourceV1,
    CorpusV1,
    JobV1,
    MessageV1,
    ProviderV1,
    ProvenanceV1,
    RepositoryGrantV1,
    RetrievalSnapshotV1,
    RouteV1,
    WorktreeV1,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {
    "artifact.v1": ArtifactV1,
    "attachment.v1": AttachmentV1,
    "audit_event.v1": AuditEventV1,
    "capability_assignment.v1": CapabilityAssignmentV1,
    "conversation.v1": ConversationV1,
    "corpus.v1": CorpusV1,
    "corpus_source.v1": CorpusSourceV1,
    "job.v1": JobV1,
    "message.v1": MessageV1,
    "provider.v1": ProviderV1,
    "provenance.v1": ProvenanceV1,
    "repository_grant.v1": RepositoryGrantV1,
    "retrieval_snapshot.v1": RetrievalSnapshotV1,
    "route.v1": RouteV1,
    "worktree.v1": WorktreeV1,
}


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("schema export accepts no arguments")
    target = ROOT / "schemas/v7"
    target.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for name, model in sorted(SCHEMAS.items()):
        path = target / f"{name}.schema.json"
        encoded = (
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
        path.write_bytes(encoded)
        files.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
            }
        )
    manifest = {
        "schema_version": 1,
        "generator": "scripts/export_v7_schemas.py",
        "count": len(files),
        "files": files,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "count": len(files), "target": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
