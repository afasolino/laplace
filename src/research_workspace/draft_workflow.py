from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .projects import load_project
from .real_benchmark import generate_measured
from .retrieval import evidence_packet, search, validate_citations


def grounded_related_work(
    project: str,
    query: str,
    *,
    root: Path | None = None,
    endpoint: str = "http://127.0.0.1:11434",
    model: str = "qwen3:4b",
) -> dict[str, Any]:
    paths, _ = load_project(project, root=root)
    evidence = search(
        paths.data / "Metadata" / "workspace.db",
        query,
        mode="hybrid",
        limit=6,
        document_class="user_work",
    )
    packet = evidence_packet(query, evidence)
    compact = [
        {
            "filename": item.filename,
            "page": item.page,
            "section": item.section,
            "chunk_id": item.chunk_id,
            "text": item.text,
        }
        for item in evidence
    ]
    prompt = (
        "Return ONLY JSON with keys draft and claims. Draft one concise formal IEEE-style related-work paragraph using only the supplied complete local PDF evidence. Each claim must cite an exact filename, page, and chunk_id from the evidence. Unsupported claims must be [SOURCE REQUIRED]. Do not use online metadata as full-text evidence. Evidence: "
        + json.dumps(compact, ensure_ascii=False)
    )
    generation = generate_measured(
        endpoint, model, prompt, context_tokens=8192, think=False, num_predict=280, json_mode=True
    )
    output = generation.get("output", "")
    citations: list[dict[str, object]] = [
        {"filename": item.filename, "page": item.page, "chunk_id": item.chunk_id}
        for item in evidence
        if item.filename in output and item.chunk_id in output
    ]
    citation_valid = validate_citations(packet, citations) if citations else False
    try:
        structured = json.loads(output)
        draft = str(structured.get("draft", output))
        claims = structured.get("claims", [])
    except json.JSONDecodeError:
        draft = output
        claims = []
    model_citation_valid = citation_valid
    unsupported = (
        []
        if citation_valid
        else [
            "The qwen3:4b draft was rejected because it did not reproduce valid local provenance citations."
        ]
    )
    fallback_used = False
    if not citation_valid and evidence:
        fallback_used = True
        fallback_lines = [
            "Extractive evidence-grounded related-work draft (model draft rejected; no unsupported inference added):"
        ]
        fallback_citations: list[dict[str, object]] = []
        for item in evidence[:2]:
            fallback_lines.append(
                f"- {item.text[:420].rstrip()} [{item.filename}, p. {item.page}, chunk {item.chunk_id}]"
            )
            fallback_citations.append(
                {"filename": item.filename, "page": item.page, "chunk_id": item.chunk_id}
            )
        draft = "\n".join(fallback_lines)
        citations = fallback_citations
        citation_valid = validate_citations(packet, citations)
    if unsupported and "[SOURCE REQUIRED]" not in draft and not fallback_used:
        draft += "\n\n[SOURCE REQUIRED]"
    evidence_path = paths.outputs / "EvidencePackets" / "related_work.json"
    draft_path = paths.outputs / "Drafts" / "related_work.md"
    unsupported_path = paths.outputs / "Reports" / "unsupported_claims.json"
    provenance_path = paths.outputs / "Reports" / "provenance_validation.json"
    for path in (evidence_path, draft_path, unsupported_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            {
                "packet": packet,
                "citations_checked": citations,
                "model": model,
                "availability": "COMPLETE_LOCAL_PDF",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    draft_path.write_text(draft + "\n", encoding="utf-8")
    unsupported_path.write_text(
        json.dumps({"claims": claims, "unsupported": unsupported}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    provenance_path.write_text(
        json.dumps(
            {
                "model_citation_valid": model_citation_valid,
                "citation_valid": citation_valid,
                "fallback_used": fallback_used,
                "citations_checked": citations,
                "evidence_chunks": len(evidence),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "status": "GROUNDED_EXTRACTIVE_FALLBACK"
        if fallback_used and citation_valid
        else "GROUNDED"
        if citation_valid
        else "REVIEW_REQUIRED",
        "draft_path": str(draft_path),
        "evidence_packet": str(evidence_path),
        "unsupported_claims": str(unsupported_path),
        "provenance_report": str(provenance_path),
        "retrieved_chunks": len(evidence),
        "citation_valid": citation_valid,
        "model_citation_valid": model_citation_valid,
        "fallback_used": fallback_used,
        "generation": generation,
    }
