from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from research_workspace.laplace_core import LaplaceCore, LaplaceCoreError
from research_workspace.repository_agent_service import RepositoryAgentService
from research_workspace.service_tiers import ModelLane
from research_workspace.user_capabilities import Capability
from research_workspace.verification_policy import (
    validate_verification_argv,
    verification_qualifies_for_promotion,
)
from research_workspace.zetsu_agent import ZetsuAgentCoordinator
from research_workspace.zetsu_mcp import ZetsuService


class _AgentService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"status": "SUCCESS", "content": "neutral-service"}

    def scheduler_status(self, *, user_id: str) -> dict[str, object]:
        return {"status": "READY", "user_id": user_id}

    def task_status(self, *, user_id: str, session_id: str) -> dict[str, object]:
        return {"status": "ACTIVE", "user_id": user_id, "session_id": session_id}

    def cancel_queued(self, *, user_id: str, session_id: str) -> dict[str, object]:
        return {"status": "CANCELLED", "user_id": user_id, "session_id": session_id}

    def handoff_evidence(self, session_id: str, *, max_chars: int) -> dict[str, object]:
        return {"session_id": session_id, "max_chars": max_chars}


class _Tiered:
    def effective_capabilities(self, _user_id: str) -> frozenset[Capability]:
        return frozenset({Capability.AGENT})


def _core(service: RepositoryAgentService) -> LaplaceCore:
    return LaplaceCore(Path.cwd(), cast(object, object()), cast(object, _Tiered()), repository_agent_service=service)


def test_core_constructs_without_loading_zetsu_or_mcp_modules() -> None:
    code = (
        "import sys; from pathlib import Path; "
        "from research_workspace.laplace_core import LaplaceCore; "
        "LaplaceCore(Path.cwd(), object(), object()); "
        "print('zetsu_agent' in sys.modules, 'research_workspace.zetsu_mcp' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


def test_standalone_core_repository_agent_uses_neutral_service() -> None:
    service = _AgentService()
    result = _core(service).repository_agent(
        user_id="owner",
        repo_id="repo",
        instruction="inspect",
        lane=ModelLane.QUALITY,
        session_id=None,
        max_steps=2,
        max_chars=512,
        verification_argv=None,
        apply_to_repository=False,
        wait_timeout_seconds=10,
    )
    assert result["status"] == "SUCCESS"
    assert service.calls[0]["repo_id"] == "repo"


def test_zetsu_adapter_uses_the_same_bound_core_service() -> None:
    service = _AgentService()
    core = _core(service)
    zetsu = ZetsuService(Path.cwd(), cast(object, object()), cast(object, _Tiered()), core=core)
    result = zetsu.call(
        "owner",
        "agent_task",
        {"repo_id": "repo", "instruction": "inspect", "max_chars": 512},
    )
    assert result["status"] == "SUCCESS"
    assert core.repository_agent_service is service
    assert service.calls[0]["repo_id"] == "repo"


def test_neutral_and_zetsu_verifier_adapters_are_identical(tmp_path: Path) -> None:
    valid = ["pytest", ".", "-q"]
    assert validate_verification_argv(tmp_path, valid) == ZetsuAgentCoordinator._verify_argv(
        tmp_path, valid
    )
    for invalid in (["pytest", "-c", "unsafe.ini"], ["/bin/pytest", "."]):
        with pytest.raises(Exception) as neutral:
            validate_verification_argv(tmp_path, invalid)
        with pytest.raises(Exception) as adapter:
            ZetsuAgentCoordinator._verify_argv(tmp_path, invalid)
        assert type(neutral.value) is type(adapter.value)
        assert str(neutral.value) == str(adapter.value)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("pytest", "tests/test_value.py", "-q"), True),
        (("pytest", "--collect-only"), False),
        (("ruff", "check", "src"), True),
        (("ruff", "check", "--show-files", "src"), False),
        (("mypy", "src/research_workspace"), True),
        (("mypy", "--help"), False),
        (("mypy", "--install-types", "src/research_workspace"), False),
    ],
)
def test_final_verifier_qualification_requires_a_real_read_only_check(
    argv: tuple[str, ...], expected: bool
) -> None:
    assert verification_qualifies_for_promotion(argv) is expected


def test_core_contains_no_private_zetsu_verifier_dependency() -> None:
    source = inspect.getsource(LaplaceCore)
    assert "ZetsuAgentCoordinator" not in source
    assert "_verify_argv" not in source
    with pytest.raises(LaplaceCoreError, match="repository_agent_unavailable"):
        LaplaceCore(Path.cwd(), cast(object, object()), cast(object, _Tiered())).repository_agent(
            user_id="owner",
            repo_id="repo",
            instruction="inspect",
            lane=ModelLane.QUALITY,
            session_id=None,
            max_steps=1,
            max_chars=512,
            verification_argv=None,
            apply_to_repository=False,
            wait_timeout_seconds=1,
        )
