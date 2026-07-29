from __future__ import annotations

from research_workspace.gpu_coordination import (
    ProcessEvidence,
    classify_compute_ownership,
)


def _reader(values: dict[int, ProcessEvidence]):
    def read(pid: int) -> ProcessEvidence:
        if pid not in values:
            raise RuntimeError("missing fixture process")
        return values[pid]

    return read


def _process(
    pid: int,
    parent: int,
    *,
    protected: tuple[str, ...] = (),
) -> ProcessEvidence:
    return ProcessEvidence(
        pid=pid,
        parent_pid=parent,
        executable_name="python",
        command_sha256=f"{pid:064x}"[-64:],
        cwd_classification="SPECDEC_PROTECTED" if protected else "OTHER",
        protected_markers=protected,
    )


def test_empty_gpu_is_clear() -> None:
    assert classify_compute_ownership(())["status"] == "GPU_CLEAR"


def test_specdec_process_is_protected() -> None:
    result = classify_compute_ownership(
        (120,),
        reader=_reader({120: _process(120, 1, protected=("specdec_ladder",))}),
    )
    assert result["status"] == "BLOCKED_BY_SPECDEC_ACTIVE"


def test_unrelated_or_unresolvable_process_fails_closed() -> None:
    unrelated = classify_compute_ownership(
        (120,),
        reader=_reader({120: _process(120, 1)}),
    )
    unresolved = classify_compute_ownership((999,), reader=_reader({}))
    assert unrelated["status"] == "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP"
    assert unresolved["status"] == "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP"


def test_laplace_owned_child_is_allowed() -> None:
    values = {
        200: _process(200, 100),
        100: _process(100, 1),
    }
    result = classify_compute_ownership(
        (200,),
        allowed_laplace_roots=(100,),
        reader=_reader(values),
    )
    assert result["status"] == "GPU_CLEAR_LAPLACE_OWNED_ONLY"
