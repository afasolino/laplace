from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from research_workspace.laplace_core import LaplaceCore, LaplaceCoreError
from research_workspace.service_tiers import ModelLane
from research_workspace.user_capabilities import Capability
from research_workspace.verification_gates import VerificationGateRegistry
from research_workspace.zetsu_mcp import ZetsuService


class _Corpus:
    def search(self, user_id: str, query: str, *, corpus_id: str | None, limit: int) -> dict[str, object]:
        if user_id != "owner-a":
            raise RuntimeError("corpus_not_found")
        return {
            "retrieval_used": True,
            "results": [
                {
                    "chunk_id": "chk_" + "a" * 32,
                    "file": "notes.md",
                    "page": 2,
                    "section": "Results",
                    "text": f"{query}:{corpus_id}:{limit}",
                }
            ],
        }

    def evidence(self, user_id: str, chunk_ids: tuple[str, ...]) -> dict[str, object]:
        if user_id != "owner-a":
            raise RuntimeError("corpus_not_found")
        return {"results": [{"chunk_id": chunk_ids[0], "file": "notes.md", "page": 2}]}


class _Tiered:
    class _Policy:
        def __init__(self) -> None:
            self.routes = {
                ModelLane.QUALITY: type("Route", (), {"model_id": "qwen-quality"})(),
                ModelLane.STANDARD: type("Route", (), {"model_id": "qwen-standard"})(),
                ModelLane.ECONOMY: type("Route", (), {"model_id": "codev-economy"})(),
            }

    def __init__(self) -> None:
        self.lane_policy = self._Policy()
        self.calls: list[dict[str, object]] = []

    def effective_capabilities(self, _user_id: str) -> frozenset[Capability]:
        return frozenset({Capability.CHAT, Capability.AGENT, Capability.PERSONAL_CORPUS})

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.calls.append({"operation": "chat", **kwargs})
        return {
            "status": "SUCCESS",
            "model_id": "qwen-standard",
            "effective_lane": "standard",
            "response": {"content": "fixture answer", "finish_reason": "stop"},
        }

    def agent(self, **kwargs: object) -> dict[str, object]:
        self.calls.append({"operation": "agent", **kwargs})
        return {
            "status": "SUCCESS",
            "model_id": "codev-economy",
            "effective_lane": "economy",
            "response": {"content": "fixture specialist result"},
        }

    def agent_session_status(self, **kwargs: object) -> dict[str, object]:
        self.calls.append({"operation": "status", **kwargs})
        return {"status": "ACTIVE", "session_id": kwargs["session_id"]}

    def cancel_agent_session(self, **kwargs: object) -> dict[str, object]:
        self.calls.append({"operation": "cancel", **kwargs})
        return {"status": "CANCELLED", "session_id": kwargs["session_id"]}


class _Coordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "status": "SUCCESS",
            "effective_lane": cast(ModelLane, kwargs["lane"]).value,
        }

    def scheduler_status(self, *, user_id: str) -> dict[str, object]:
        return {"status": "READY", "user_id": user_id}


def _core() -> tuple[LaplaceCore, _Tiered, _Coordinator]:
    tiered = _Tiered()
    coordinator = _Coordinator()
    return (
        LaplaceCore(
            Path.cwd(),
            cast(Any, _Corpus()),
            cast(Any, tiered),
            agent_coordinator=cast(Any, coordinator),
        ),
        tiered,
        coordinator,
    )


def test_core_retrieval_and_evidence_are_owner_scoped_and_grounded() -> None:
    core, _tiered, _coordinator = _core()
    result = core.retrieve("owner-a", "marker", corpus_id="pc_fixture", limit=3)
    results = cast(list[dict[str, object]], result["results"])
    assert results[0]["file"] == "notes.md"
    assert results[0]["page"] == 2
    assert results[0]["section"] == "Results"
    assert str(results[0]["chunk_id"]).startswith("chk_")
    evidence = cast(
        list[dict[str, object]],
        core.evidence("owner-a", (str(results[0]["chunk_id"]),))["results"],
    )
    assert evidence[0]["page"] == 2
    with pytest.raises(RuntimeError, match="corpus_not_found"):
        core.retrieve("owner-b", "marker")


def test_core_routes_chat_qwen_agent_and_rtl_without_mcp() -> None:
    core, tiered, coordinator = _core()
    chat = core.chat(
        user_id="owner-a",
        lane=ModelLane.STANDARD,
        messages=({"role": "user", "content": "hello"},),
    )
    assert chat["effective_lane"] == "standard"
    agent = core.repository_agent(
        user_id="owner-a",
        repo_id="laplace-v2",
        instruction="inspect",
        lane=ModelLane.QUALITY,
        session_id="core-session",
        max_steps=2,
        max_chars=512,
        verification_argv=("pytest", "tests/test_laplace_core_g1.py", "-q"),
        apply_to_repository=False,
        wait_timeout_seconds=10,
    )
    assert agent["status"] == "SUCCESS"
    assert coordinator.calls[-1]["lane"] is ModelLane.QUALITY
    rtl = core.rtl_task(
        user_id="owner-a",
        session_id="rtl-session",
        instruction="implement the bounded module",
        task_kind="implementation",
        editable_sources=("rtl/counter.sv",),
        module_count=1,
    )
    assert rtl["effective_lane"] == "economy"
    assert tiered.calls[-1]["domain"] == "systemverilog"
    assert tiered.calls[-1]["lane"] is ModelLane.ECONOMY


def test_core_verification_uses_registry_and_rejects_unsafe_commands() -> None:
    definitions = VerificationGateRegistry.definitions("python", scope="public")
    gate_results = {
        item.gate_id: {"status": "PASS", "executed": True, "tool": item.tool}
        for item in definitions
    }
    summary = LaplaceCore.deterministic_verification(
        "python",
        gate_results,
        scope="public",
        available_tools={item.tool: True for item in definitions},
    )
    assert summary["verification_status"] == "PASSED"
    assert summary["passed"] is True
    assert LaplaceCore.validate_verification_command(
        Path.cwd(), ["pytest", "tests/test_laplace_core_g1.py", "-q"]
    )[0] == "pytest"
    with pytest.raises(LaplaceCoreError, match="zetsu_agent_verify_command_forbidden"):
        LaplaceCore.validate_verification_command(Path.cwd(), ["pytest", "-c", "/tmp/x"])


def test_zetsu_is_an_adapter_over_the_same_injected_core() -> None:
    core, tiered, _coordinator = _core()
    service = ZetsuService(Path.cwd(), core.corpus, tiered, core=core)  # type: ignore[arg-type]
    result = service.call(
        "owner-a",
        "delegate",
        {"instruction": "hello", "lane": "standard", "max_chars": 512},
    )
    direct = core.chat(
        user_id="owner-a",
        lane=ModelLane.STANDARD,
        messages=(
            {"role": "system", "content": "Local work stays in Codex."},
            {"role": "user", "content": "hello"},
        ),
    )
    assert service.core is core
    assert result["model_id"] == direct["model_id"]
    assert result["effective_lane"] == direct["effective_lane"]
    assert [call["operation"] for call in tiered.calls] == ["chat", "chat"]
