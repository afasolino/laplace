from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from research_workspace.prime_agent_harness import (
    PRIME_AGENT_DIR_ENV,
    PRIME_AGENT_PROVIDER_ID,
    PRIME_AGENT_KERNEL_PYTHON_ENV,
    PrimeAgentHarnessError,
    PrimeAgentProfile,
    _bwrap_argv,
    _count_child_usage_attributions,
    _prime_environment,
    parse_prime_events,
    parse_prime_version,
    prepare_prime_agent_state,
    prime_models_payload,
    prime_settings_payload,
    repository_agent_backend,
    resolve_prime_agent_executable,
    resolve_prime_kernel_python,
    require_prime_agent_version,
    run_prime_agent,
)


def _profile() -> PrimeAgentProfile:
    return PrimeAgentProfile(
        provider_id=PRIME_AGENT_PROVIDER_ID,
        model_id="laplace-quality-qwen38-mtp8",
        endpoint="http://127.0.0.1:8207",
        base_url="http://127.0.0.1:8207/v1",
        context_window=131072,
        max_tokens=4096,
    )


def _selected(repository: Path) -> None:
    (repository / "configs").mkdir(parents=True)
    (repository / ".runtime").mkdir()
    (repository / "configs/selected_serving_profiles.json").write_text(
        json.dumps(
            {
                "default_profile_id": "P8_qwen38_w4a16_mtp",
                "routes": {
                    "quality": {
                        "model_id": "laplace-quality-qwen38-mtp8",
                        "endpoint": "http://127.0.0.1:8207",
                        "context_limit": 131072,
                        "output_limit": 4096,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_models_payload_uses_pinned_local_vllm_contract() -> None:
    payload = prime_models_payload(_profile())
    provider = payload["providers"][PRIME_AGENT_PROVIDER_ID]  # type: ignore[index]
    assert provider["baseUrl"] == "http://127.0.0.1:8207/v1"  # type: ignore[index]
    assert provider["api"] == "openai-completions"  # type: ignore[index]
    compat = provider["compat"]  # type: ignore[index]
    assert compat["supportsDeveloperRole"] is False  # type: ignore[index]
    assert compat["supportsReasoningEffort"] is False  # type: ignore[index]
    assert compat["thinkingFormat"] == "qwen-chat-template"  # type: ignore[index]


def test_settings_disable_refinement_and_never_export_recursive_agent_task() -> None:
    payload = prime_settings_payload(enable_zetsu_readonly=True, rlm_max_depth=1)
    assert payload["rlmMaxDepth"] == 1
    assert payload["autoRefine"] == {"enabled": False}
    assert payload["telemetry"] == {"enabled": False}
    server = payload["mcpServers"]["zetsu"]  # type: ignore[index]
    assert "agent_task" not in server["enabledTools"]  # type: ignore[index]
    assert "rtl_task" not in server["enabledTools"]  # type: ignore[index]
    production = prime_settings_payload(enable_zetsu_readonly=False)
    assert "mcpServers" not in production


def test_prepare_state_remains_repository_local_for_qualification(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _selected(repository)
    state = repository / ".runtime" / "prime-pilot"
    profile = prepare_prime_agent_state(repository, state)
    assert profile.model_id == "laplace-quality-qwen38-mtp8"
    assert (state / "agent/models.json").is_file()
    assert stat.S_IMODE((state / "agent/models.json").stat().st_mode) == 0o600
    with pytest.raises(
        PrimeAgentHarnessError, match="prime_state_must_be_under_repository_runtime"
    ):
        prepare_prime_agent_state(repository, tmp_path / "outside")


def test_event_parser_counts_only_completed_successful_ipython_calls() -> None:
    stream = "\n".join(
        [
            json.dumps({"type": "session", "id": "s"}),
            json.dumps(
                {
                    "type": "tool_execution_start",
                    "toolCallId": "ok",
                    "toolName": "ipython",
                    "args": {"code": "child = await rlm('inspect')"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "toolCallId": "ok",
                    "toolName": "ipython",
                    "result": {"text": "done"},
                    "isError": False,
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_start",
                    "toolCallId": "failed",
                    "toolName": "ipython",
                    "args": {"code": "await mcp.call_tool('zetsu', 'search', {})"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "toolCallId": "failed",
                    "toolName": "ipython",
                    "result": {"error": "boom"},
                    "isError": True,
                }
            ),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "provider": "laplace-local-p8",
                        "model": "laplace-quality-qwen38-mtp8",
                        "usage": {"input": 10, "output": 2, "totalTokens": 12},
                        "content": [{"type": "text", "text": "PRIME_OK"}],
                    },
                }
            ),
        ]
    )
    events, final, executions = parse_prime_events(stream)
    assert len(events) == 6
    assert final == "PRIME_OK"
    assert len(executions) == 2
    assert executions[0].ipython_source == "child = await rlm('inspect')"
    assert executions[1].ipython_source is None


def test_environment_is_secret_minimal_and_zetsu_token_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOULD_NOT_LEAK", "secret")
    monkeypatch.setenv("LAPLACE_ZETSU_TOKEN", "token-value")
    env = _prime_environment(
        {"PATH": "/usr/bin", "LANG": "C.UTF-8", "SHOULD_NOT_LEAK": "secret"},
        enable_zetsu_readonly=False,
        zetsu_token_env="LAPLACE_ZETSU_TOKEN",
    )
    assert env["PATH"] == "/usr/bin"
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "-p no:cacheprovider" in env["PYTEST_ADDOPTS"]
    assert "SHOULD_NOT_LEAK" not in env
    assert "LAPLACE_ZETSU_TOKEN" not in env
    with_token = _prime_environment(
        {"PATH": "/usr/bin"},
        enable_zetsu_readonly=True,
        zetsu_token_env="LAPLACE_ZETSU_TOKEN",
    )
    assert with_token["LAPLACE_ZETSU_TOKEN"] == "token-value"


def test_rlm_usage_requires_durable_child_attribution(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    usage = {
        "input": 7,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 7,
        "cost": {
            "input": 0.0,
            "output": 0.0,
            "cacheRead": 0.0,
            "cacheWrite": 0.0,
            "total": 0.0,
        },
    }

    entries = [
        {
            "type": "session",
            "version": 3,
            "id": "root",
            "timestamp": "2026-09-04T00:00:00.000Z",
            "cwd": "/workspace",
        },
        {
            "type": "child_usage_attributed",
            "id": "attr0001",
            "parentId": "a1",
            "timestamp": "2026-09-04T00:00:01.000Z",
            "targetId": "a1",
            "childUsage": usage,
            "aggregateUsage": usage,
        },
    ]

    (sessions / "root.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    assert _count_child_usage_attributions(sessions) == 1


def test_prime_executable_resolution_preserves_launcher_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "node" / "bin"
    package = tmp_path / "node" / "lib" / "prime-agent"
    bin_dir.mkdir(parents=True)
    package.mkdir(parents=True)

    target = package / "cli.js"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)

    launcher = bin_dir / "prime-agent"
    launcher.symlink_to(target)

    monkeypatch.setenv("PATH", str(bin_dir))
    assert resolve_prime_agent_executable() == launcher.absolute()


def test_bwrap_binds_authorized_paths_before_masking_original_parent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "state" / "worktree"
    prime_state = tmp_path / "state" / "prime"
    workspace.mkdir(parents=True)
    (workspace / ".git").write_text(
        "gitdir: /readonly/gitdir\n",
        encoding="utf-8",
    )
    prime_state.mkdir()
    parent = workspace.parent

    argv = _bwrap_argv(
        bwrap=Path("/usr/bin/bwrap"),
        workspace=workspace,
        state_root=prime_state,
        kernel_python=Path(sys.executable),
        command=["prime-agent"],
        hide_paths=[parent],
    )

    run_tmpfs = argv.index("/run")
    first_dir = argv.index("--dir")
    bind_index = argv.index(str(workspace))
    hide_index = argv.index(str(parent), bind_index + 1)

    assert argv[run_tmpfs - 1] == "--tmpfs"
    assert run_tmpfs < first_dir
    assert bind_index < hide_index

    assert "/run/laplace-prime/workspace" in argv
    assert "/run/laplace-prime/workspace/.git" in argv
    assert "/run/laplace-prime/state" in argv
    assert "/run/laplace-prime/kernel" in argv
    assert argv.count("--dir") >= 4

def _fake_prime(path: Path) -> None:
    script = f"""#!{sys.executable}
import json, sys
if '--version' in sys.argv:
    print('prime-agent 0.9.1')
    raise SystemExit(0)
print(json.dumps({{'type':'session','id':'fake'}}))
print(json.dumps({{
    'type':'tool_execution_start',
    'toolCallId':'t1',
    'toolName':'ipython',
    'args':{{'code':'x = 1'}},
}}))
print(json.dumps({{
    'type':'tool_execution_end',
    'toolCallId':'t1',
    'toolName':'ipython',
    'result':{{'text':'1'}},
    'isError':False,
}}))
print(json.dumps({{
    'type':'message_end',
    'message':{{
        'role':'assistant',
        'provider':'laplace-local-p8',
        'model':'laplace-quality-qwen38-mtp8',
        'usage':{{'input':3,'output':1,'totalTokens':4}},
        'content':[{{'type':'text','text':'DONE'}}],
    }},
}}))
raise SystemExit(0)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def test_runner_uses_pinned_prime_and_exact_model_identity_without_bwrap(tmp_path: Path) -> None:
    executable = tmp_path / "prime-agent"
    _fake_prime(executable)
    assert "0.9.1" in require_prime_agent_version(executable)
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / ".git").write_text("gitdir: /readonly/gitdir\n", encoding="utf-8")
    result = run_prime_agent(
        executable=executable,
        profile=_profile(),
        state_root=tmp_path / "prime-state",
        workspace=workspace,
        prompt="Do bounded work.",
        timeout_seconds=10,
        require_bwrap=False,
    )
    assert result.final_text == "DONE"
    assert result.observed_models == ("laplace-quality-qwen38-mtp8",)
    assert result.observed_providers == ("laplace-local-p8",)
    assert result.ipython_call_count == 1
    assert result.usage["input"] == 3
    assert result.bwrap_used is False


def test_kernel_runtime_resolution_is_explicit_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = tmp_path / "kernel-venv" / "bin" / "python"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    kernel.chmod(0o755)
    monkeypatch.setenv(PRIME_AGENT_KERNEL_PYTHON_ENV, str(kernel))
    assert resolve_prime_kernel_python() == kernel.resolve()
    with pytest.raises(PrimeAgentHarnessError, match="prime_agent_kernel_not_prepared"):
        resolve_prime_kernel_python(tmp_path / "missing")


def test_version_and_backend_contract_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_prime_version("prime-agent 0.9.1") == (0, 9, 1)
    assert PRIME_AGENT_DIR_ENV == "PRIME_AGENT_CODING_AGENT_DIR"
    monkeypatch.setenv("LAPLACE_ZETSU_AGENT_BACKEND", "prime")
    assert repository_agent_backend() == "prime"
    monkeypatch.setenv("LAPLACE_ZETSU_AGENT_BACKEND", "something-else")
    with pytest.raises(PrimeAgentHarnessError, match="prime_agent_backend_invalid"):
        repository_agent_backend()

def test_parse_prime_events_rejects_malformed_jsonl() -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        parse_prime_events,
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_event_json_invalid$",
    ):
        parse_prime_events(
            '{"type":"turn_start"}\n'
            'THIS IS NOT JSON\n'
            '{"type":"turn_end"}\n'
        )


def test_parse_prime_events_requires_json_objects() -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        parse_prime_events,
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_event_object_required$",
    ):
        parse_prime_events('"not-an-event"\n')


def test_atomic_json_uses_collision_safe_temporary_file(
    tmp_path: Path,
) -> None:
    import json
    import os

    from research_workspace.prime_agent_harness import _atomic_json

    target = tmp_path / "state.json"

    # Occupy the exact predictable name used by the former PID-based
    # implementation. A collision-safe implementation must still work.
    predictable = tmp_path / f".state.json.{os.getpid()}.tmp"
    predictable.write_text("occupied\n", encoding="utf-8")

    _atomic_json(target, {"generation": 1})
    _atomic_json(target, {"generation": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "generation": 2
    }
    assert predictable.read_text(encoding="utf-8") == "occupied\n"

    temporary_files = set(tmp_path.glob(".state.json.*.tmp"))
    assert temporary_files == {predictable}


def test_bwrap_masks_project_prime_configuration(
    tmp_path: Path,
) -> None:
    from research_workspace.prime_agent_harness import (
        _BWRAP_WORKSPACE,
        _bwrap_argv,
    )

    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    kernel = tmp_path / "kernel" / "bin" / "python"

    (workspace / ".git").mkdir(parents=True)
    (workspace / ".prime" / "agent").mkdir(parents=True)
    (workspace / ".prime" / "agent" / "settings.json").write_text(
        '{"defaultProvider":"UNTRUSTED"}\n',
        encoding="utf-8",
    )
    state.mkdir()
    kernel.parent.mkdir(parents=True)
    kernel.write_text("", encoding="utf-8")

    argv = _bwrap_argv(
        bwrap=Path("/usr/bin/bwrap"),
        workspace=workspace,
        state_root=state,
        kernel_python=kernel,
        command=["/bin/true"],
        hide_paths=(),
    )

    mask = [
        "--tmpfs",
        str(_BWRAP_WORKSPACE / ".prime"),
    ]

    assert any(
        argv[index : index + len(mask)] == mask
        for index in range(len(argv) - len(mask) + 1)
    )


def test_bwrap_rejects_symlinked_project_prime(
    tmp_path: Path,
) -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        _bwrap_argv,
    )

    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    kernel = tmp_path / "kernel" / "bin" / "python"
    external = tmp_path / "external-prime"

    (workspace / ".git").mkdir(parents=True)
    state.mkdir()
    kernel.parent.mkdir(parents=True)
    kernel.write_text("", encoding="utf-8")
    external.mkdir()
    (workspace / ".prime").symlink_to(
        external,
        target_is_directory=True,
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_project_configuration_unsafe$",
    ):
        _bwrap_argv(
            bwrap=Path("/usr/bin/bwrap"),
            workspace=workspace,
            state_root=state,
            kernel_python=kernel,
            command=["/bin/true"],
            hide_paths=(),
        )

def test_child_usage_attribution_counts_valid_session_jsonl(
    tmp_path: Path,
) -> None:
    from research_workspace.prime_agent_harness import (
        _count_child_usage_attributions,
    )

    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"session","version":3}\n'
        '{"type":"message","id":"a1"}\n'
        '{"type":"child_usage_attributed","targetId":"a1",'
        '"childUsage":{"input":10,"output":2},'
        '"aggregateUsage":{"input":20,"output":4}}\n'
        '{"type":"child_usage_attributed","targetId":"a1",'
        '"childUsage":{"input":5,"output":1},'
        '"aggregateUsage":{"input":25,"output":5}}\n',
        encoding="utf-8",
    )

    assert _count_child_usage_attributions(tmp_path) == 2


def test_child_usage_attribution_rejects_malformed_session_jsonl(
    tmp_path: Path,
) -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        _count_child_usage_attributions,
    )

    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"session","version":3}\n'
        'NOT JSON\n',
        encoding="utf-8",
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_session_json_invalid$",
    ):
        _count_child_usage_attributions(tmp_path)


def test_child_usage_attribution_rejects_non_object_session_entry(
    tmp_path: Path,
) -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        _count_child_usage_attributions,
    )

    session = tmp_path / "session.jsonl"
    session.write_text(
        '"not-an-entry"\n',
        encoding="utf-8",
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_session_entry_invalid$",
    ):
        _count_child_usage_attributions(tmp_path)


def test_child_usage_attribution_requires_session_entry_type(
    tmp_path: Path,
) -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        _count_child_usage_attributions,
    )

    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"id":"missing-type"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_session_entry_invalid$",
    ):
        _count_child_usage_attributions(tmp_path)


def test_child_usage_attribution_rejects_invalid_utf8(
    tmp_path: Path,
) -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        _count_child_usage_attributions,
    )

    session = tmp_path / "session.jsonl"
    session.write_bytes(
        b'{"type":"session","version":3}\n'
        b'\xff\xfe\n'
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_session_json_invalid$",
    ):
        _count_child_usage_attributions(tmp_path)

def test_child_usage_attribution_rejects_incomplete_attribution(
    tmp_path: Path,
) -> None:
    import pytest

    from research_workspace.prime_agent_harness import (
        PrimeAgentHarnessError,
        _count_child_usage_attributions,
    )

    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"session","version":3}\n'
        '{"type":"child_usage_attributed","targetId":"a1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        PrimeAgentHarnessError,
        match="^prime_agent_child_usage_attribution_invalid$",
    ):
        _count_child_usage_attributions(tmp_path)

