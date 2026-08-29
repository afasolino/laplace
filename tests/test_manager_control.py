from pathlib import Path
from typing import cast

import pytest

from research_workspace.laplace_core import LaplaceCore, LaplaceCoreError
from research_workspace.manager_control import ManagerPlan, ManagerUnavailableError, TaskComplexity
from research_workspace.service_tiers import ModelLane


class _Worker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"status": "SUCCESS", "worker_instruction": kwargs["instruction"]}


class _Manager:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def plan(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def _core(worker: _Worker, manager: object | None = None) -> LaplaceCore:
    return LaplaceCore(
        Path.cwd(),
        cast(object, object()),
        cast(object, object()),
        repository_agent_service=cast(object, worker),
        manager_provider=cast(object, manager),
    )


def _run(core: LaplaceCore, *, complexity: TaskComplexity | None = None) -> dict[str, object]:
    return core.repository_agent(
        user_id="owner",
        repo_id="repo",
        instruction="architecture-sensitive change",
        lane=ModelLane.QUALITY,
        session_id="session-a",
        max_steps=4,
        max_chars=512,
        verification_argv=("pytest",),
        apply_to_repository=False,
        wait_timeout_seconds=10,
        task_complexity=complexity,
        allow_mutation=False,
    )


def test_default_core_admission_preserves_worker_result_without_manager() -> None:
    worker = _Worker()
    result = _run(_core(worker))

    assert result == {"status": "SUCCESS", "worker_instruction": "architecture-sensitive change"}
    assert len(worker.calls) == 1


def test_real_core_admission_calls_manager_before_one_worker_dispatch() -> None:
    worker = _Worker()
    manager = _Manager(ManagerPlan(objective="Plan", milestones=("Inspect", "Verify")))

    result = _run(_core(worker, manager), complexity=TaskComplexity(architecture_sensitive=True))

    assert result["manager_control"]["decision"] == "plan"
    assert len(manager.calls) == len(worker.calls) == 1
    assert "Advisory manager plan" in str(worker.calls[0]["instruction"])
    assert worker.calls[0]["allow_mutation"] is False


def test_invalid_manager_plan_stops_before_repository_worker() -> None:
    worker = _Worker()

    with pytest.raises(LaplaceCoreError, match="manager_plan_invalid"):
        _run(_core(worker, _Manager({"objective": "missing milestones"})), complexity=TaskComplexity(file_count_hint=3))

    assert worker.calls == []


def test_manager_outage_falls_back_without_authority_expansion() -> None:
    class _DownManager:
        def plan(self, **_kwargs: object) -> object:
            raise ManagerUnavailableError("offline")

    worker = _Worker()
    result = _run(_core(worker, _DownManager()), complexity=TaskComplexity(verification_recovery=True))

    assert result["manager_control"] == {"decision": "plan", "fallback": True, "plan": None}
    assert worker.calls[0]["allow_mutation"] is False
    assert worker.calls[0]["verification_argv"] == ["pytest"]
