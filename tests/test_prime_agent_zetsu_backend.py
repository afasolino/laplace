from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research_workspace.prime_agent_harness import PrimeAgentRunResult, PrimeToolExecution
from research_workspace.zetsu_agent import ZetsuAgentCoordinator
from research_workspace.zetsu_state import AgentExecutionState


def _prime_result() -> PrimeAgentRunResult:
    events = (
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "laplace-local-p8",
                "model": "laplace-quality-qwen38-mtp8",
                "usage": {"input": 20, "output": 4, "cacheRead": 2, "totalTokens": 24},
                "content": [{"type": "text", "text": "verified repair"}],
            },
        },
        {
            "type": "turn_end",
            "message": {"role": "assistant"},
            "toolResults": [],
        },
    )
    tools = (
        PrimeToolExecution(
            tool_call_id="tool-1",
            tool_name="ipython",
            args={"code": "Path('target.py').write_text('fixed')"},
            result={"text": "fixed"},
            is_error=False,
        ),
    )
    return PrimeAgentRunResult(
        returncode=0,
        final_text="verified repair",
        events=events,
        tool_executions=tools,
        stdout="\n".join(json.dumps(item) for item in events),
        stderr="",
        elapsed_seconds=1.0,
        bwrap_used=True,
    )


def test_prime_backend_reuses_laplace_authoritative_verification(
    tmp_path: Path, monkeypatch
) -> None:
    sandbox_root = tmp_path / "laplace-state" / "sandboxes"
    sandbox_root.mkdir(parents=True)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    canonical = tmp_path / "canonical"
    (canonical / ".runtime").mkdir(parents=True)

    coordinator = object.__new__(ZetsuAgentCoordinator)
    coordinator.tiered = SimpleNamespace(
        lane_policy=SimpleNamespace(routes={"quality": object()}),
        sandboxes=SimpleNamespace(sandbox_root=sandbox_root),
    )
    staged: list[str] = []
    coordinator.results = SimpleNamespace(
        stage_stream=lambda _session, name, _stream: staged.append(name)
    )
    state = AgentExecutionState(objective="repair")
    ctx = SimpleNamespace(
        lane="quality",
        session_id="prime-test",
        user_id="user",
        worktree=worktree,
        allow_mutation=True,
        apply_to_repository=False,
        required_verification_plan=((".", ("pytest", "tests/test_target.py", "-q")),),
        binding=SimpleNamespace(
            canonical_repository_root=str(canonical),
            tool_policy=SimpleNamespace(max_commands=100),
        ),
    )

    import research_workspace.zetsu_agent as module

    monkeypatch.setattr(module, "resolve_prime_agent_executable", lambda: Path("/prime-agent"))
    monkeypatch.setattr(
        module, "resolve_prime_kernel_python", lambda: Path("/kernel/bin/python")
    )
    monkeypatch.setattr(module, "profile_from_route", lambda _route: object())
    monkeypatch.setattr(module, "run_prime_agent", lambda **_kwargs: _prime_result())
    monkeypatch.setattr(
        module.AgentSandboxManager,
        "fixed_environment",
        staticmethod(lambda _binding: {"PATH": "/usr/bin"}),
    )
    monkeypatch.setattr(coordinator, "_remaining_wall", lambda _ctx, _state: 30.0)
    monkeypatch.setattr(coordinator, "_ensure_active", lambda _ctx, _state: None)
    monkeypatch.setattr(
        coordinator,
        "_worktree_state",
        lambda _root: ("head", "status", ["target.py"]),
    )

    def sync(_ctx, current: AgentExecutionState):
        current.mutation_epoch = 1
        current.last_verified_epoch = 0
        current.changed_paths = ["target.py"]
        return {}

    monkeypatch.setattr(coordinator, "_sync_authoritative_assurance", sync)

    def consume(_ctx, current: AgentExecutionState):
        current.command_count += 1
        current.telemetry.tool_calls += 1

    monkeypatch.setattr(coordinator, "_consume_tool_budget", consume)

    def verify(_ctx, current: AgentExecutionState, _action):
        current.last_verified_epoch = current.mutation_epoch
        current.telemetry.verification_calls += 1
        current.validation_history.append({"passed": True})
        return {"passed": True}

    monkeypatch.setattr(coordinator, "_verify", verify)
    monkeypatch.setattr(
        coordinator,
        "_finish_allowed",
        lambda current: current.last_verified_epoch == current.mutation_epoch,
    )

    final, evidence = coordinator._run_prime_harness(
        ctx, state, instruction="repair", max_steps=12
    )
    assert final == "verified repair"
    assert evidence["authoritative_verification"] == {"passed": True}
    assert state.last_verified_epoch == state.mutation_epoch == 1
    assert state.telemetry.verification_calls == 1
    assert state.telemetry.qwen_input_tokens == 20
    assert "prime-agent.events.jsonl" in staged
    assert "prime-agent.stderr.log" in staged
