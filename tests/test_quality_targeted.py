from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace import quality_improvement
from research_workspace.quality_improvement import TARGETED_TASK_IDS


def test_targeted_six_task_set_is_exact_and_deduplicated() -> None:
    assert TARGETED_TASK_IDS == (
        "py_fastapi_strict_endpoint",
        "py_unseen_sqlite_state",
        "sv_ready_valid_buffer",
        "sv_axi_lite_irq_regs",
        "sv_unseen_rv_slot",
        "sv_unseen_w1c_event",
    )
    assert len(set(TARGETED_TASK_IDS)) == 6


def test_analysis_only_does_not_require_a_serving_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        quality_improvement,
        "write_failure_analysis",
        lambda repository_root, output_root: observed.append((repository_root, output_root)) or {},
    )

    assert (
        quality_improvement.main(
            ["--analysis-only", "--output-root", str(tmp_path / "analysis")]
        )
        == 0
    )
    assert observed == [(Path.cwd().resolve(), (Path.cwd() / tmp_path / "analysis").resolve())]


def test_analysis_only_reports_missing_recorded_evidence_without_a_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        quality_improvement,
        "write_failure_analysis",
        lambda _repository_root, _output_root: (_ for _ in ()).throw(
            RuntimeError("original_quality_evidence_missing")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        quality_improvement.main(
            ["--analysis-only", "--output-root", str(tmp_path / "analysis")]
        )
    assert exc_info.value.code == 2
    assert "analysis-only requires the recorded quality evidence" in capsys.readouterr().err
