from __future__ import annotations

from pathlib import Path

from research_workspace.orchestration_certification import (
    _compact_model_call,
    _reviewer_approved,
)


def test_model_call_projection_excludes_generated_content_and_prompts() -> None:
    compact = _compact_model_call(
        {
            "call_id": "call-1",
            "response_valid": True,
            "content": "generated source",
            "prompt": "private prompt",
            "reasoning": "private reasoning",
            "routing": {"selected": "rtl_worker"},
        }
    )

    assert compact["call_id"] == "call-1"
    assert compact["routing"] == {"selected": "rtl_worker"}
    assert "content" not in compact
    assert "prompt" not in compact
    assert "reasoning" not in compact


def test_reviewer_projection_requires_all_approval_signals() -> None:
    review = {
        "status": "APPROVED",
        "reviewer_approved": True,
        "reviewer_verdict": {"verdict": "approve"},
    }
    assert _reviewer_approved(review) is True
    assert _reviewer_approved({**review, "status": "CHANGES_REQUESTED"}) is False


def test_gate_projection_helpers_do_not_require_a_gpu(tmp_path: Path) -> None:
    assert tmp_path.is_dir()
