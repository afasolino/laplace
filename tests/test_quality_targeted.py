from __future__ import annotations

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
