#!/usr/bin/env python3
"""Qualify Prime Agent as a harness over Laplace's selected local P8 model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess  # nosec B404 - argv is fixed and shell=False
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.zetsu_runtime import default_state_root, load_local_plus_token
from research_workspace.zetsu_sdk_stdio import ZetsuBackend

from research_workspace.prime_agent_harness import (
    DEFAULT_ZETSU_ENDPOINT,
    DEFAULT_ZETSU_TOKEN_ENV,
    PrimeAgentHarnessError,
    PrimeAgentRunResult,
    load_selected_p8_profile,
    probe_local_model,
    require_prime_agent_version,
    resolve_prime_agent_executable,
    resolve_prime_kernel_python,
    run_prime_agent,
)

JsonObject: TypeAlias = dict[str, object]
ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--prime-agent", default="prime-agent")
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--kernel-python", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--zetsu-endpoint", default=DEFAULT_ZETSU_ENDPOINT)
    parser.add_argument("--zetsu-token-env", default=DEFAULT_ZETSU_TOKEN_ENV)
    parser.add_argument("--laplace-state-root", type=Path, default=default_state_root())
    parser.add_argument("--skip-zetsu", action="store_true")
    parser.add_argument("--skip-agent-task", action="store_true")
    return parser


def _write_fixture(path: Path, files: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    commands = (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "prime-pilot@laplace.local"],
        ["git", "config", "user.name", "Laplace Prime Pilot"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "prime pilot fixture"],
    )
    for argv in commands:
        completed = subprocess.run(argv, cwd=path, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise PrimeAgentHarnessError(
                f"prime_pilot_git_fixture_failed:{argv[1]}:{completed.stderr[-1000:]}"
            )


def _run_pytest(workspace: Path) -> JsonObject:
    completed = subprocess.run(  # nosec B603
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8_000:],
        "stderr": completed.stderr[-8_000:],
    }


def _persist_run(root: Path, name: str, result: PrimeAgentRunResult) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "events.jsonl").write_text(result.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(result.stderr, encoding="utf-8")
    (directory / "final.txt").write_text(result.final_text + "\n", encoding="utf-8")


def _zetsu_call(
    endpoint: str, token: str, name: str, arguments: dict[str, object], *, timeout: float
) -> JsonObject:
    async def call() -> JsonObject:
        backend = ZetsuBackend(endpoint, token, timeout=timeout)
        return await backend.call_tool(name, arguments)

    return asyncio.run(call())


def _agent_task_gate(
    *,
    run_root: Path,
    laplace_state_root: Path,
    endpoint: str,
    token: str,
    profile_model: str,
    profile_provider: str,
    wait_timeout_seconds: int,
) -> JsonObject:
    repository = run_root / "agent-task-repository"
    _write_fixture(
        repository,
        {
            "offset.py": """from __future__ import annotations\n\n\ndef shifted(value: int) -> int:
    return value + 2\n""",
            "test_offset.py": (
                "from offset import shifted\n\n\n"
                "def test_shifted_adds_one() -> None:\n"
                "    assert shifted(41) == 42\n"
            ),
        },
    )
    repo_id = f"prime-p8-pilot-{run_root.name.lower()}"
    authorizations = RepositoryAuthorizationStore(
        laplace_state_root / "tiered_serving/repository_authorizations.sqlite3"
    )
    authorizations.register(repo_id, repository)
    authorizations.grant("plus-local", repo_id, base_revision="HEAD")
    result: JsonObject
    try:
        result = _zetsu_call(
            endpoint,
            token,
            "agent_task",
            {
                "repo_id": repo_id,
                "instruction": (
                    "Fix the off-by-one defect in offset.py. Keep the implementation minimal, "
                    "run the bound verification, and finish only when it passes."
                ),
                "lane": "quality",
                "agent_backend": "prime",
                "max_steps": 12,
                "max_chars": 12000,
                "verification_argv": ["pytest", "-q"],
                "apply_to_repository": True,
                "allow_mutation": True,
                "wait_timeout_seconds": wait_timeout_seconds,
                "telemetry": True,
            },
            timeout=float(wait_timeout_seconds + 30),
        )
    finally:
        try:
            authorizations.revoke("plus-local", repo_id)
        except Exception:
            pass

    (run_root / "agent-task-mcp-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    structured = result.get("structuredContent")
    payload = structured if isinstance(structured, dict) else {}
    prime = payload.get("prime_agent")
    prime_evidence = prime if isinstance(prime, dict) else {}
    promotion = payload.get("promotion")
    promotion_evidence = promotion if isinstance(promotion, dict) else {}
    independent = _run_pytest(repository)
    passed = (
        result.get("isError") is not True
        and payload.get("status") == "SUCCESS"
        and payload.get("repository_agent_backend") == "prime"
        and prime_evidence.get("observed_models") == [profile_model]
        and prime_evidence.get("observed_providers") == [profile_provider]
        and prime_evidence.get("bwrap_used") is True
        and promotion_evidence.get("applied") is True
        and independent.get("returncode") == 0
    )
    return {
        "name": "zetsu_agent_task_prime_backend",
        "status": "PASS" if passed else "FAIL",
        "repo_id": repo_id,
        "mcp_is_error": result.get("isError"),
        "agent_status": payload.get("status"),
        "repository_agent_backend": payload.get("repository_agent_backend"),
        "prime_agent": prime_evidence,
        "promotion": promotion_evidence,
        "independent_pytest": independent,
    }


def _gate(
    *,
    name: str,
    result: PrimeAgentRunResult,
    expected_marker: str,
    expected_provider: str,
    expected_model: str,
    pytest_result: JsonObject | None = None,
    require_rlm: bool = False,
    require_zetsu: bool = False,
) -> JsonObject:
    identity_pass = (
        result.observed_models == (expected_model,)
        and result.observed_providers == (expected_provider,)
    )
    passed = result.returncode == 0 and expected_marker in result.final_text and identity_pass
    if pytest_result is not None:
        passed = passed and pytest_result.get("returncode") == 0
    if require_rlm:
        passed = passed and result.used_rlm
    if require_zetsu:
        passed = passed and result.used_zetsu_mcp
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "prime_returncode": result.returncode,
        "expected_marker": expected_marker,
        "rlm_observed": result.used_rlm,
        "rlm_child_usage_count": result.rlm_child_usage_count,
        "zetsu_mcp_observed": result.used_zetsu_mcp,
        "event_count": len(result.events),
        "ipython_call_count": result.ipython_call_count,
        "successful_ipython_cell_count": len(result.successful_ipython_cells),
        "identity_pass": identity_pass,
        "observed_models": list(result.observed_models),
        "observed_providers": list(result.observed_providers),
        "usage": result.usage,
        "pytest": pytest_result,
        "final_text": result.final_text[-2_000:],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository_root.expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    state_root = (
        args.state_root.expanduser().resolve()
        if args.state_root is not None
        else repository / ".runtime" / "prime-agent-p8-pilot"
    )
    runtime_root = (repository / ".runtime").resolve()
    try:
        state_root.relative_to(runtime_root)
    except ValueError as exc:
        raise PrimeAgentHarnessError("prime_pilot_state_must_be_under_repository_runtime") from exc
    run_root = state_root / "qualification" / stamp
    run_root.mkdir(parents=True, exist_ok=False)

    executable = resolve_prime_agent_executable(args.prime_agent)
    version = require_prime_agent_version(executable)
    kernel_python = resolve_prime_kernel_python(args.kernel_python)
    profile = load_selected_p8_profile(repository)
    models_response = probe_local_model(profile)
    laplace_state_root = args.laplace_state_root.expanduser().resolve()
    zetsu_token = os.environ.get(args.zetsu_token_env)
    if not args.skip_zetsu or not args.skip_agent_task:
        if not zetsu_token:
            zetsu_token = load_local_plus_token(laplace_state_root)
            os.environ[args.zetsu_token_env] = zetsu_token

    gates: list[JsonObject] = []

    fix_workspace = run_root / "workspaces" / "local-fix"
    _write_fixture(
        fix_workspace,
        {
            "packet_window.py": """from __future__ import annotations


def admit(sequence: int, base: int, width: int) -> bool:
    \"\"\"Return True iff sequence is inside [base, base + width).\"\"\"
    if width <= 0:
        raise ValueError(\"width must be positive\")
    return base <= sequence <= base + width
""",
            "test_packet_window.py": """import pytest

from packet_window import admit


def test_lower_and_upper_boundary() -> None:
    assert admit(10, 10, 4)
    assert admit(13, 10, 4)
    assert not admit(14, 10, 4)


def test_outside_window() -> None:
    assert not admit(9, 10, 4)
    assert not admit(20, 10, 4)


def test_invalid_width() -> None:
    with pytest.raises(ValueError):
        admit(10, 10, 0)
""",
        },
    )
    fix_prompt = (
        "Work only in the current directory. This is a disposable qualification fixture. "
        "Inspect the code, fix the defect, and run `python -m pytest -q` through the persistent "
        "Python/IPython control environment. Do not use network access and do not inspect files "
        "outside the current directory. When the tests pass, reply exactly PRIME_LOCAL_FIX_OK."
    )
    fix_result = run_prime_agent(
        executable=executable,
        kernel_python=kernel_python,
        profile=profile,
        state_root=run_root / "prime-state" / "local-fix",
        workspace=fix_workspace,
        prompt=fix_prompt,
        timeout_seconds=args.timeout_seconds,
    )
    _persist_run(run_root, "local-fix", fix_result)
    fix_pytest = _run_pytest(fix_workspace)
    gates.append(
        _gate(
            name="local_p8_repository_repair",
            result=fix_result,
            expected_marker="PRIME_LOCAL_FIX_OK",
            expected_provider=profile.provider_id,
            expected_model=profile.model_id,
            pytest_result=fix_pytest,
        )
    )

    rlm_workspace = run_root / "workspaces" / "rlm-fix"
    _write_fixture(
        rlm_workspace,
        {
            "rolling_sum.py": """from __future__ import annotations


def rolling_sum(values: list[int], width: int) -> list[int]:
    if width <= 0:
        raise ValueError(\"width must be positive\")
    if width > len(values):
        return []
    return [
        sum(values[index : index + width])
        for index in range(len(values) - width)
    ]
""",
            "test_rolling_sum.py": """import pytest

from rolling_sum import rolling_sum


def test_windows_include_the_last_valid_window() -> None:
    assert rolling_sum([1, 2, 3, 4], 2) == [3, 5, 7]


def test_full_width() -> None:
    assert rolling_sum([2, 3, 5], 3) == [10]


def test_width_larger_than_input() -> None:
    assert rolling_sum([1, 2], 3) == []


def test_invalid_width() -> None:
    with pytest.raises(ValueError):
        rolling_sum([1], 0)
""",
        },
    )
    rlm_prompt = (
        "Work only in the current directory. Before editing, you MUST spawn exactly one recursive "
        "child with `rlm(...)` and ask it to inspect the defect and explicitly reply to the parent "
        "with its finding. Use `await rlm.list_subagents()` to confirm that child reaches "
        "completed "
        "status before you finish. Use the child finding, fix the implementation, and run "
        "`python -m pytest -q`. Do not use network access or files outside the current directory. "
        "When tests pass and the child completed, reply exactly PRIME_RLM_OK."
    )
    rlm_result = run_prime_agent(
        executable=executable,
        kernel_python=kernel_python,
        profile=profile,
        state_root=run_root / "prime-state" / "rlm-fix",
        workspace=rlm_workspace,
        prompt=rlm_prompt,
        timeout_seconds=args.timeout_seconds,
    )
    _persist_run(run_root, "rlm-fix", rlm_result)
    rlm_pytest = _run_pytest(rlm_workspace)
    gates.append(
        _gate(
            name="local_p8_recursive_subagent",
            result=rlm_result,
            expected_marker="PRIME_RLM_OK",
            expected_provider=profile.provider_id,
            expected_model=profile.model_id,
            pytest_result=rlm_pytest,
            require_rlm=True,
        )
    )

    zetsu_status = "SKIP"
    if not args.skip_zetsu:
        if not zetsu_token:
            raise PrimeAgentHarnessError(f"zetsu_token_missing:{args.zetsu_token_env}")
        zetsu_workspace = run_root / "workspaces" / "zetsu-readonly"
        _write_fixture(zetsu_workspace, {"README.md": "Prime/Zetsu read-only qualification.\n"})
        zetsu_prompt = (
            "Use the pre-imported Python `mcp` module, not shell commands, for this gate. "
            "First call "
            "`await mcp.list_tools('zetsu')`. Then call one enabled read-only Zetsu context tool "
            "through `await mcp.call_tool('zetsu', ...)` with a query about the selected P8 "
            "serving "
            "profile. Print the returned Python object so the event trace contains the evidence. "
            "Do not modify files. Reply exactly PRIME_ZETSU_OK only after the MCP call succeeds."
        )
        zetsu_result = run_prime_agent(
            executable=executable,
            kernel_python=kernel_python,
            profile=profile,
            state_root=run_root / "prime-state" / "zetsu-readonly",
            workspace=zetsu_workspace,
            prompt=zetsu_prompt,
            timeout_seconds=args.timeout_seconds,
            enable_zetsu_readonly=True,
            zetsu_endpoint=args.zetsu_endpoint,
            zetsu_token_env=args.zetsu_token_env,
        )
        _persist_run(run_root, "zetsu-readonly", zetsu_result)
        zetsu_gate = _gate(
            name="prime_to_zetsu_mcp",
            result=zetsu_result,
            expected_marker="PRIME_ZETSU_OK",
            expected_provider=profile.provider_id,
            expected_model=profile.model_id,
            require_zetsu=True,
        )
        gates.append(zetsu_gate)
        zetsu_status = str(zetsu_gate["status"])

    if not args.skip_agent_task:
        if not zetsu_token:
            raise PrimeAgentHarnessError(f"zetsu_token_missing:{args.zetsu_token_env}")
        gates.append(
            _agent_task_gate(
                run_root=run_root,
                laplace_state_root=laplace_state_root,
                endpoint=args.zetsu_endpoint,
                token=zetsu_token,
                profile_model=profile.model_id,
                profile_provider=profile.provider_id,
                wait_timeout_seconds=min(max(int(args.timeout_seconds), 1), 1800),
            )
        )

    status = "PASS" if all(gate.get("status") == "PASS" for gate in gates) else "FAIL"
    summary = {
        "schema_version": 1,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_root": str(repository),
        "state_root": str(state_root),
        "laplace_state_root": str(laplace_state_root),
        "run_root": str(run_root),
        "prime_agent_version": version,
        "prime_agent_executable": str(executable),
        "prime_agent_kernel_python": str(kernel_python),
        "upstream_reference": {
            "release": "v0.9.1",
            "release_commit": "81ae3cb34d27d38ee37f9e205a1e73694993b344",
        },
        "local_model": {
            "provider_id": profile.provider_id,
            "model_id": profile.model_id,
            "endpoint": profile.endpoint,
            "base_url": profile.base_url,
            "context_window": profile.context_window,
            "max_tokens": profile.max_tokens,
            "served_model_count": len(models_response.get("data", [])),
        },
        "zetsu_gate": zetsu_status,
        "gates": gates,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrimeAgentHarnessError as exc:
        print(f"prime-agent-p8 qualification: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
