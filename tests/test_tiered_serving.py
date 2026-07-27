from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import httpx
import pytest
import uvicorn

from research_workspace.agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentSessionBinding,
    AgentToolPolicy,
)
from research_workspace.operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.repository_authorization import (
    RepositoryAuthorizationError,
    RepositoryAuthorizationStore,
    validate_workspace_path,
)
from research_workspace.service_tiers import (
    LanePolicy,
    ModelLane,
    ModelRoute,
    PriorityAdmissionScheduler,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.serving_profiles import (
    InstalledServingCapabilities,
    ResolvedServingProfile,
    ServingProfileError,
    load_profiles,
    resolve_all,
)
from research_workspace.serving_profile_runtime import (
    OwnedProfileProcess,
    ServingProfileRuntime,
)
from research_workspace.serving_benchmark import (
    BenchmarkRequest,
    RequestMeasurement,
    context_probe_prompt,
    load_quality_manifest,
    run_concurrent,
    score_quality_output,
    summarize,
    tier_mix,
)
from research_workspace.user_capabilities import (
    CapabilityTier,
    UserCapabilityStore,
)

ROOT = Path(__file__).resolve().parents[1]


def _git_repository(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "fixture@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Fixture"],
        check=True,
    )
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _ChatBackend:
    def __init__(self, responses: Sequence[dict[str, object]] | None = None) -> None:
        self.responses: list[dict[str, object]] = (
            list(responses)
            if responses is not None
            else [{"content": "fixture", "finish_reason": "stop"}]
        )
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> dict[str, object]:
        with self._lock:
            self.calls.append(
                {
                    "messages": list(messages),
                    "route": route,
                    "tools": list(tools),
                    "request_id": request_id,
                }
            )
            index = min(len(self.calls) - 1, len(self.responses) - 1)
            return dict(self.responses[index])


class _AgentBackend:
    def __init__(self) -> None:
        self.calls: list[AgentSessionBinding] = []

    def run(
        self,
        *,
        binding: AgentSessionBinding,
        instruction: str,
        route: ModelRoute,
        request_id: str,
    ) -> dict[str, object]:
        del instruction, route, request_id
        self.calls.append(binding)
        return {
            "content": "bounded fixture agent result",
            "finish_reason": "stop",
            "verification_status": "PASSED",
        }


class _ConcurrentChatBackend(_ChatBackend):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.maximum_active = 0

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> dict[str, object]:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.05)
            return super().complete(
                messages=messages,
                route=route,
                tools=tools,
                request_id=request_id,
            )
        finally:
            with self._lock:
                self.active -= 1


def _policy() -> LanePolicy:
    return LanePolicy(
        routes={
            ModelLane.QUALITY: ModelRoute(
                ModelLane.QUALITY, "quality-model", "http://127.0.0.1:8200", 0
            ),
            ModelLane.STANDARD: ModelRoute(
                ModelLane.STANDARD, "standard-model", "http://127.0.0.1:8201", 10
            ),
            ModelLane.ECONOMY: ModelRoute(
                ModelLane.ECONOMY, "codev-model", "http://127.0.0.1:8202", 20
            ),
        }
    )


def _service(
    tmp_path: Path,
    *,
    chat: _ChatBackend | None = None,
    agent: _AgentBackend | None = None,
) -> tuple[
    TieredServingService,
    UserCapabilityStore,
    RepositoryAuthorizationStore,
    AgentSandboxManager,
]:
    users = UserCapabilityStore(tmp_path / "state/users.sqlite3")
    repositories = RepositoryAuthorizationStore(tmp_path / "state/repos.sqlite3")
    sandboxes = AgentSandboxManager(tmp_path / "state/worktrees", repositories)
    service = TieredServingService(
        users=users,
        sandboxes=sandboxes,
        lane_policy=_policy(),
        chat_backend=chat or _ChatBackend(),
        agent_backend=agent or _AgentBackend(),
        audit_log=TierAuditLog(tmp_path / "state/audit.jsonl"),
    )
    return service, users, repositories, sandboxes


def test_basic_chat_is_tool_free_and_lane_is_independent(tmp_path: Path) -> None:
    backend = _ChatBackend()
    service, users, _, _ = _service(tmp_path, chat=backend)
    users.set_user("basic-a", CapabilityTier.BASIC)

    result = service.chat(
        user_id="basic-a",
        lane=ModelLane.QUALITY,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result["capability_tier"] == "basic"
    assert result["requested_lane"] == "quality"
    assert result["model_id"] == "quality-model"
    assert backend.calls[0]["tools"] == []
    audit = json.loads((tmp_path / "state/audit.jsonl").read_text().splitlines()[0])
    assert audit["tool_schema_count"] == 0
    assert {
        "user_id",
        "capability_tier",
        "session_id",
        "mode",
        "repo_id",
        "sandbox_id",
        "requested_quality_lane",
        "effective_quality_lane",
        "route",
        "tool_policy_id",
        "network_policy_id",
        "queue_wait_seconds",
        "trace_id",
        "priority",
        "context_limit",
        "output_limit",
        "escalated",
        "escalation_reason",
        "validator_results",
    } <= set(audit)
    assert audit["mode"] == "chat"
    assert audit["repo_id"] is None
    assert audit["network_policy_id"] == "local-model-only-v1"


def test_standard_escalates_once_only_on_deterministic_gate(tmp_path: Path) -> None:
    backend = _ChatBackend(
        [
            {"content": "", "finish_reason": "stop"},
            {"content": "quality recovery", "finish_reason": "stop"},
            {"content": "must not be called", "finish_reason": "stop"},
        ]
    )
    service, users, _, _ = _service(tmp_path, chat=backend)
    users.set_user("plus-a", CapabilityTier.PLUS)

    result = service.chat(
        user_id="plus-a",
        lane=ModelLane.STANDARD,
        messages=[{"role": "user", "content": "validate"}],
    )

    assert len(backend.calls) == 2
    assert result["effective_lane"] == "quality"
    assert result["escalation"] == {
        "from_lane": "standard",
        "to_lane": "quality",
        "trigger_gate": "non_empty_content",
        "passed": True,
    }


def test_economy_schema_failure_escalates_once_and_never_silently_passes(
    tmp_path: Path,
) -> None:
    recovered_backend = _ChatBackend(
        [
            {"content": "not-json", "finish_reason": "stop"},
            {"content": '{"status":"PASS"}', "finish_reason": "stop"},
        ]
    )
    service, users, _, _ = _service(tmp_path, chat=recovered_backend)
    users.set_user("basic-json", CapabilityTier.BASIC)
    recovered = service.chat(
        user_id="basic-json",
        lane=ModelLane.ECONOMY,
        domain="json",
        messages=[{"role": "user", "content": "return JSON"}],
    )
    assert len(recovered_backend.calls) == 2
    assert recovered["effective_lane"] == "quality"

    failed_backend = _ChatBackend(
        [
            {"content": "not-json", "finish_reason": "stop"},
            {"content": "still-not-json", "finish_reason": "stop"},
            {"content": '{"must":"never run"}', "finish_reason": "stop"},
        ]
    )
    failed_service, failed_users, _, _ = _service(tmp_path / "failed", chat=failed_backend)
    failed_users.set_user("basic-json", CapabilityTier.BASIC)
    with pytest.raises(Exception, match="response_validation_failed"):
        failed_service.chat(
            user_id="basic-json",
            lane=ModelLane.ECONOMY,
            domain="json",
            messages=[{"role": "user", "content": "return JSON"}],
        )
    assert len(failed_backend.calls) == 2


def test_economy_codev_is_restricted_to_systemverilog(tmp_path: Path) -> None:
    backend = _ChatBackend()
    service, users, _, _ = _service(tmp_path, chat=backend)
    users.set_user("basic-a", CapabilityTier.BASIC)
    non_rtl = service.chat(
        user_id="basic-a",
        lane=ModelLane.ECONOMY,
        domain="python",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert non_rtl["model_id"] == "standard-model"
    rtl = service.chat(
        user_id="basic-a",
        lane=ModelLane.ECONOMY,
        domain="systemverilog",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert rtl["model_id"] == "codev-model"


def test_plus_sessions_are_isolated_and_revocation_is_immediate(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    commit = _git_repository(repository_root)
    service, users, repositories, sandboxes = _service(tmp_path)
    users.set_user("plus-a", CapabilityTier.PLUS)
    users.set_user("plus-b", CapabilityTier.PLUS)
    users.set_user("basic-a", CapabilityTier.BASIC)
    repositories.register("repo", repository_root)
    repositories.grant("plus-a", "repo", base_revision=commit)
    repositories.grant("plus-b", "repo", base_revision=commit)
    policy = AgentToolPolicy("bounded", ("apply_patch", "run_validation"))

    with pytest.raises(Exception, match="capability_denied"):
        service.create_agent_session(
            user_id="basic-a",
            repo_id="repo",
            session_id="basic-session",
            tool_policy=policy,
        )
    left = service.create_agent_session(
        user_id="plus-a", repo_id="repo", session_id="left", tool_policy=policy
    )
    right = service.create_agent_session(
        user_id="plus-b", repo_id="repo", session_id="right", tool_policy=policy
    )
    left_root = Path(str(left["binding"]["worktree_root"]))  # type: ignore[index]
    right_root = Path(str(right["binding"]["worktree_root"]))  # type: ignore[index]
    assert left_root != right_root
    assert left_root.is_dir() and right_root.is_dir()
    with pytest.raises(AgentSandboxError, match="unknown_agent_session"):
        sandboxes.require_active("left", user_id="plus-b")

    repositories.revoke("plus-a", "repo")
    with pytest.raises(AgentSandboxError, match="repository_authorization_revoked"):
        sandboxes.require_active("left", user_id="plus-a")
    assert right_root.is_dir()


def test_repository_path_escape_matrix(tmp_path: Path) -> None:
    root = tmp_path / "worktree"
    root.mkdir()
    (root / "safe").mkdir()
    (root / "safe/file.txt").write_text("safe", encoding="utf-8")
    assert validate_workspace_path(root, "safe/file.txt") == root / "safe/file.txt"

    with pytest.raises(RepositoryAuthorizationError, match="path_escape"):
        validate_workspace_path(root, "../outside")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(RepositoryAuthorizationError, match="symlink_escape"):
        validate_workspace_path(root, "link/secret")
    os.link(root / "safe/file.txt", root / "hardlink.txt")
    with pytest.raises(RepositoryAuthorizationError, match="hardlink_escape"):
        validate_workspace_path(root, "hardlink.txt")
    (root / "nested").mkdir()
    (root / "nested/.git").mkdir()
    with pytest.raises(RepositoryAuthorizationError, match="nested_repository_escape"):
        validate_workspace_path(root, "nested/file.py")
    (root / "mounted").mkdir()
    with pytest.raises(RepositoryAuthorizationError, match="bind_mount_escape"):
        validate_workspace_path(root, "mounted/file", mount_points=(root / "mounted",))


def test_submodule_path_is_rejected_even_when_not_checked_out(tmp_path: Path) -> None:
    root = tmp_path / "worktree"
    root.mkdir()
    (root / ".gitmodules").write_text(
        '[submodule "external"]\n\tpath = vendor/external\n\turl = ../external\n',
        encoding="utf-8",
    )
    with pytest.raises(RepositoryAuthorizationError, match="submodule_escape"):
        validate_workspace_path(root, "vendor/external/source.py")


def test_profiles_resolve_deterministically_and_fail_closed(tmp_path: Path) -> None:
    help_text = " ".join(
        {
            "--port",
            "--served-model-name",
            "--host",
            "--max-model-len",
            "--max-num-seqs",
            "--max-num-batched-tokens",
            "--kv-cache-dtype",
            "--gpu-memory-utilization",
            "--scheduling-policy",
            "--generation-config",
            "--enable-prefix-caching",
            "--no-enable-prefix-caching",
            "--prefix-caching-hash-algo",
            "--enable-chunked-prefill",
            "--no-enable-chunked-prefill",
            "--cpu-offload-gb",
            "--cpu-offload-params",
            "--offload-backend",
            "--offload-group-size",
            "--offload-num-in-group",
            "--offload-prefetch-step",
            "--kv-offloading-size",
            "--kv-offloading-backend",
            "--language-model-only",
        }
    )
    capabilities = InstalledServingCapabilities.from_help(
        version="0.25.0", help_text=help_text
    )
    profiles = load_profiles(ROOT / "configs/serving_profiles")
    first = resolve_all(
        profiles,
        capabilities,
        executable=Path("/opt/vllm/bin/vllm"),
        require_model=False,
    )
    second = resolve_all(
        profiles,
        capabilities,
        executable=Path("/opt/vllm/bin/vllm"),
        require_model=False,
    )
    assert [item.resolution_sha256 for item in first] == [
        item.resolution_sha256 for item in second
    ]
    p2 = next(item for item in first if item.profile.profile_id.startswith("P2_"))
    assert "--cpu-offload-params" in p2.command
    assert "experts" in p2.command

    unsupported = InstalledServingCapabilities.from_help(
        version="fixture",
        help_text="--port --max-model-len",
    )
    with pytest.raises(ServingProfileError, match="unsupported_profile"):
        resolve_all(
            profiles,
            unsupported,
            executable=Path("/opt/vllm/bin/vllm"),
            require_model=False,
        )


@pytest.mark.anyio
async def test_api_enforces_basic_chat_only_and_plus_repository_binding(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    _git_repository(repository_root)
    chat = _ChatBackend()
    agent = _AgentBackend()
    service, users, repositories, _ = _service(tmp_path, chat=chat, agent=agent)
    users.set_user("basic-api", CapabilityTier.BASIC)
    users.set_user("plus-api", CapabilityTier.PLUS)
    repositories.register("repo", repository_root)
    repositories.grant("plus-api", "repo")
    operator = OperatorService(ROOT, tmp_path / "operator")
    tokens = {
        "basic-token-00000000000000000000000": AuthCredential(
            "read", "basic-api", CapabilityTier.BASIC
        ),
        "plus-token-000000000000000000000000": AuthCredential(
            "read", "plus-api", CapabilityTier.PLUS
        ),
    }
    app = create_operator_app(
        operator,
        OperatorAuth(tokens),
        settings=OperatorApiSettings(fixture_mode=True),
        tiered=service,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        basic_headers = {
            "Authorization": "Bearer basic-token-00000000000000000000000"
        }
        basic_session = await client.post("/api/v1/session", headers=basic_headers)
        basic_csrf = basic_session.json()["csrf_token"]
        assert basic_session.json()["capability_tier"] == "basic"
        assert (await client.get("/api/v1/dashboard", headers=basic_headers)).status_code == 403
        chat_response = await client.post(
            "/api/v1/chat",
            headers={**basic_headers, "X-CSRF-Token": basic_csrf},
            json={
                "lane": "standard",
                "domain": "general",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert chat_response.status_code == 200
        denied_agent = await client.post(
            "/api/v1/agent/sessions",
            headers={**basic_headers, "X-CSRF-Token": basic_csrf},
            json={"repo_id": "repo", "session_id": "denied"},
        )
        assert denied_agent.status_code == 403

        plus_headers = {
            "Authorization": "Bearer plus-token-000000000000000000000000"
        }
        plus_session = await client.post("/api/v1/session", headers=plus_headers)
        plus_csrf = plus_session.json()["csrf_token"]
        created = await client.post(
            "/api/v1/agent/sessions",
            headers={**plus_headers, "X-CSRF-Token": plus_csrf},
            json={"repo_id": "repo", "session_id": "plus-session"},
        )
        assert created.status_code == 200
        assert created.json()["binding"]["repo_id"] == "repo"
        run = await client.post(
            "/api/v1/agent/sessions/plus-session/messages",
            headers={**plus_headers, "X-CSRF-Token": plus_csrf},
            json={
                "lane": "standard",
                "domain": "python",
                "instruction": "Make the bounded fixture change.",
            },
        )
        assert run.status_code == 200
        assert agent.calls[0].user_id == "plus-api"
        status = await client.get(
            "/api/v1/agent/sessions/plus-session/status",
            headers=plus_headers,
        )
        assert status.status_code == 200
        assert status.json()["status"] == "ACTIVE_CLEAN"
        assert status.json()["last_result"]["status"] == "SUCCESS"
        cancelled = await client.post(
            "/api/v1/agent/sessions/plus-session/cancel",
            headers={**plus_headers, "X-CSRF-Token": plus_csrf},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
        final_status = await client.get(
            "/api/v1/agent/sessions/plus-session/status",
            headers=plus_headers,
        )
        assert final_status.json()["status"] == "CANCELLED"

    audit_lines = [
        json.loads(line)
        for line in (tmp_path / "state/audit.jsonl").read_text().splitlines()
    ]
    denied = next(item for item in audit_lines if item.get("outcome") == "DENIED")
    assert denied["failure_category"] == "capability_denied"
    assert "hello" not in json.dumps(denied)


def test_real_http_api_dispatches_blocking_model_calls_concurrently(
    tmp_path: Path,
) -> None:
    backend = _ConcurrentChatBackend()
    service, users, _, _ = _service(tmp_path, chat=backend)
    tokens: dict[str, AuthCredential] = {}
    for index in range(4):
        user_id = f"parallel-{index}"
        users.set_user(user_id, CapabilityTier.BASIC)
        tokens[f"parallel-token-{index}-000000000000000000"] = AuthCredential(
            "read", user_id, CapabilityTier.BASIC
        )
    app = create_operator_app(
        OperatorService(ROOT, tmp_path / "operator"),
        OperatorAuth(tokens),
        settings=OperatorApiSettings(fixture_mode=False),
        tiered=service,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10)
        headers: list[dict[str, str]] = []
        for token in tokens:
            authorization = {"Authorization": f"Bearer {token}"}
            session = client.post("/api/v1/session", headers=authorization)
            headers.append(
                {
                    **authorization,
                    "X-CSRF-Token": session.json()["csrf_token"],
                }
            )
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(
                executor.map(
                    lambda current: client.post(
                        "/api/v1/chat",
                        headers=current,
                        json={
                            "lane": "standard",
                            "domain": "general",
                            "messages": [{"role": "user", "content": "parallel"}],
                        },
                    ),
                    headers,
                )
            )
        client.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert all(response.status_code == 200 for response in responses)
    assert backend.maximum_active >= 2


def test_benchmark_mix_context_quality_and_concurrency_are_deterministic(
    tmp_path: Path,
) -> None:
    lanes = tier_mix(10)
    assert lanes.count(ModelLane.QUALITY) == 2
    assert lanes.count(ModelLane.STANDARD) == 6
    assert lanes.count(ModelLane.ECONOMY) == 2
    prompt, markers = context_probe_prompt(2_048)
    assert all(marker in prompt for marker in markers)
    cases = load_quality_manifest(ROOT / "configs/serving_quality_manifest.json")
    strict_json = next(case for case in cases if case.case_id == "strict_json")
    passing = score_quality_output(
        "fixture",
        strict_json,
        '{"status":"PASS","count":3}',
    )
    assert passing.score == 1.0
    assert passing.passed_hard_gates
    failing = score_quality_output("fixture", strict_json, "```not json```")
    assert failing.score == 0.0
    assert not failing.passed_hard_gates

    requests = [
        BenchmarkRequest(
            request_id=f"request-{index}",
            capability_tier="basic",
            lane=lane,
            domain="systemverilog" if lane is ModelLane.ECONOMY else "general",
            prompt="fixture",
            context_tokens=0,
            max_output_tokens=8,
        )
        for index, lane in enumerate(lanes)
    ]

    def probe(
        request: BenchmarkRequest, *, profile_id: str, concurrency: int
    ) -> RequestMeasurement:
        time.sleep(0.001)
        return RequestMeasurement(
            profile_id,
            request.request_id,
            request.capability_tier,
            request.lane.value,
            request.domain,
            request.context_tokens,
            concurrency,
            "SUCCESS",
            0.0,
            1.0,
            2.0,
            0.1,
            4,
            2,
            20.0,
            None,
            None,
        )

    measured = run_concurrent(
        requests, profile_id="fixture", concurrency=4, probe=probe
    )
    summary = summarize("fixture", measured)
    assert len(measured) == 10
    assert summary.success_count == 10
    assert summary.aggregate_output_tokens_per_second > 0
    assert all(item.queue_time_ms is not None for item in measured)


def test_safe_shutdown_releases_only_recorded_process_group(tmp_path: Path) -> None:
    owned_process = subprocess.Popen(["sleep", "120"], start_new_session=True)
    unrelated = subprocess.Popen(["sleep", "120"], start_new_session=True)
    runtime = ServingProfileRuntime(tmp_path / "runtime")
    try:
        start_ticks = int(
            Path(f"/proc/{owned_process.pid}/stat")
            .read_text(encoding="utf-8")
            .split()[21]
        )
        record = OwnedProfileProcess(
            profile_id="P0_baseline",
            pid=owned_process.pid,
            process_group_id=os.getpgid(owned_process.pid),
            proc_start_ticks=start_ticks,
            command_sha256="a" * 64,
            resolution_sha256="b" * 64,
            log_path=str(tmp_path / "owned.log"),
            started_at_utc="2026-07-27T00:00:00+00:00",
        )
        runtime.ownership_path.write_text(
            json.dumps(asdict(record)), encoding="utf-8"
        )
        result = runtime.release_owned(timeout_seconds=2)
        owned_process.wait(timeout=5)
        assert result["status"] == "RELEASED_OWNED_PROFILE"
        assert unrelated.poll() is None
    finally:
        if owned_process.poll() is None:
            owned_process.terminate()
            owned_process.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_profile_startup_timeout_is_structured_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = replace(
        load_profiles(ROOT / "configs/serving_profiles")[0],
        startup_timeout=1,
    )
    resolved = ResolvedServingProfile(
        profile=profile,
        command=("/opt/vllm/bin/vllm", "serve", profile.model_path),
        installed_version="fixture",
        installed_help_sha256="a" * 64,
        resolution_sha256="b" * 64,
    )
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(
        "research_workspace.serving_profile_runtime.time.monotonic",
        lambda: next(clock),
    )
    runtime = ServingProfileRuntime(tmp_path / "timeout-runtime")
    with pytest.raises(Exception, match="profile_startup_timeout") as caught:
        runtime.wait_ready(resolved)
    assert getattr(caught.value, "evidence") == {
        "profile_id": profile.profile_id,
        "last_error": "not_probed",
    }


def test_quality_reservation_prevents_lower_lane_starvation() -> None:
    policy = LanePolicy(
        routes=_policy().routes,
        quality_reserved_slots=1,
        standard_capacity=2,
        economy_capacity=4,
    )
    scheduler = PriorityAdmissionScheduler(policy)
    release = threading.Event()
    entered: list[str] = []

    def hold(lane: ModelLane, name: str) -> None:
        with scheduler.admit(lane):
            entered.append(name)
            release.wait(5)

    initial = [
        threading.Thread(target=hold, args=(ModelLane.ECONOMY, f"economy-{index}"))
        for index in range(3)
    ]
    for thread in initial:
        thread.start()
    deadline = time.monotonic() + 2
    while len(entered) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    waiting_economy = threading.Thread(
        target=hold, args=(ModelLane.ECONOMY, "economy-waiting")
    )
    waiting_economy.start()
    deadline = time.monotonic() + 2
    while not scheduler.snapshot()["waiting"] and time.monotonic() < deadline:
        time.sleep(0.01)
    quality = threading.Thread(target=hold, args=(ModelLane.QUALITY, "quality"))
    quality.start()
    deadline = time.monotonic() + 2
    while "quality" not in entered and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert "quality" in entered
        assert "economy-waiting" not in entered
    finally:
        release.set()
        for thread in [*initial, waiting_economy, quality]:
            thread.join(timeout=5)
