#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_workspace.personal_corpus import PersonalCorpusStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite.get("cases") if isinstance(suite, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite_cases_missing")

    store = PersonalCorpusStore(Path(args.state_root))
    rows = []
    scored = recall_hits = no_evidence_total = no_evidence_passes = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case_invalid")
        case_id = str(case["id"])
        query = str(case["query"])
        limit = int(case.get("limit", 8))
        expected = case.get("expected_sources", [])
        if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
            raise ValueError(f"expected_sources_invalid:{case_id}")
        expected_set = set(expected)
        expect_none = bool(case.get("expect_no_evidence", False))

        result = store.search(args.owner, query, limit=limit)
        found = [
            str(item["source_title"])
            for item in result.get("results", [])
            if isinstance(item, dict) and isinstance(item.get("source_title"), str)
        ]

        rank = next(
            (i for i, source in enumerate(found, start=1) if source in expected_set),
            None,
        )
        recall = bool(expected_set.intersection(found)) if expected_set else False
        if expected_set:
            scored += 1
            recall_hits += int(recall)
            if rank is not None:
                reciprocal_rank_sum += 1.0 / rank

        no_evidence_ok = None
        if expect_none:
            no_evidence_total += 1
            no_evidence_ok = len(found) == 0
            no_evidence_passes += int(no_evidence_ok)

        rows.append(
            {
                "id": case_id,
                "query": query,
                "expected_sources": sorted(expected_set),
                "found_sources": found,
                "recall_at_k": recall,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "no_evidence_pass": no_evidence_ok,
            }
        )

    report = {
        "schema_version": 1,
        "case_count": len(cases),
        "retrieval_case_count": scored,
        "recall_at_k": None if scored == 0 else recall_hits / scored,
        "mrr": None if scored == 0 else reciprocal_rank_sum / scored,
        "no_evidence_accuracy": None if no_evidence_total == 0 else no_evidence_passes / no_evidence_total,
        "cases": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
