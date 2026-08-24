#!/usr/bin/env python3
"""Certify tier isolation and mixed load through the real localhost HTTP API."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import socket
import subprocess  # nosec B404
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import httpx
import uvicorn

from research_workspace.agent_sandbox import AgentSandboxManager, AgentSessionBinding
from research_workspace.auth_registry import (
    RegisteredUser,
    RegisteredUserRegistry,
    hash_secret,
    write_registry,
)
from research_workspace.auth_sessions import (
    AuthAuditLog,
    RegisteredEmailAuth,
    SessionStore,
)
from research_workspace.conversations import ConversationStore
from research_workspace.operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.service_tiers import (
    LanePolicy,
    LocalOpenAIChatBackend,
    ModelLane,
    ModelRoute,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.serving_benchmark import (
    load_quality_manifest,
    score_quality_output,
    tier_mix,
)
from research_workspace.user_capabilities import CapabilityTier, UserCapabilityStore


class _BoundedFixtureAgent:
    """Non-generative agent result used only for HTTP lifecycle/isolation checks."""

    def run(
        self,
        *,
        binding: AgentSessionBinding,
        instruction: str,
        route: ModelRoute,
        request_id: str,
    ) -> dict[str, object]:
        del instruction, route, request_id
        return {
            "content": f"Bound worktree verified for {binding.repo_id}.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
        }


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--main-endpoint", default="http://127.0.0.1:8102")
    parser.add_argument("--main-model", default="laplace-qwen3.6-35b-a3b-w4a16")
    parser.add_argument("--quality-context-limit", type=int, default=32_768)
    parser.add_argument("--standard-context-limit", type=int, default=16_384)
    parser.add_argument("--codev-endpoint", default="http://127.0.0.1:8103")
    parser.add_argument("--codev-model", default="laplace-codev-r1-rl-qwen-7b-w4a16")
    parser.add_argument("--users", type=int, default=12)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--skip-gui", action="store_true")
    return parser


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _list_length(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _required_float(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{label}_missing")
    return float(value)


def _probe_model(endpoint: str, expected: str) -> dict[str, object]:
    request = urllib.request.Request(  # nosec B310 - fixed localhost inputs
        endpoint.rstrip("/") + "/v1/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # nosec B310
            raw: object = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"status": "UNAVAILABLE", "error": type(exc).__name__}
    data = raw.get("data") if isinstance(raw, dict) else None
    identities = (
        [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
        if isinstance(data, list)
        else []
    )
    return {
        "status": "HEALTHY_EXACT_MODEL" if expected in identities else "MISMATCH",
        "expected": expected,
        "served": identities,
    }


def _git_repository(path: Path, marker: str) -> str:
    path.mkdir(parents=True)
    (path / "owner.txt").write_text(marker + "\n", encoding="utf-8")
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "GIT_AUTHOR_NAME": "Laplace Certification",
        "GIT_AUTHOR_EMAIL": "laplace-certification@localhost",
        "GIT_COMMITTER_NAME": "Laplace Certification",
        "GIT_COMMITTER_EMAIL": "laplace-certification@localhost",
    }
    for command in (
        ["git", "init", "-q", str(path)],
        ["git", "-C", str(path), "add", "owner.txt"],
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"],
    ):
        completed = subprocess.run(  # nosec B603 B607 - fixed Git verbs
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"fixture_repository_failed:{command[1]}:{completed.stderr[-1000:]}")
    revision = subprocess.run(  # nosec B603 B607 - fixed Git query
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=environment,
    )
    return revision.stdout.strip()


def _session(client: httpx.Client, token: str) -> tuple[dict[str, str], dict[str, object]]:
    authorization = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/session", headers=authorization)
    response.raise_for_status()
    value: object = response.json()
    if not isinstance(value, dict) or not isinstance(value.get("csrf_token"), str):
        raise RuntimeError("session_response_invalid")
    headers = {**authorization, "X-CSRF-Token": value["csrf_token"]}
    return headers, value


def _gui_smoke(base_url: str, email: str, password: str) -> dict[str, object]:
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.goto(base_url, wait_until="networkidle")
        page.get_by_label("Email address").first.fill(email)
        page.locator("#login-password").fill(password)
        page.get_by_role("button", name="Sign in", exact=True).last.click()
        page.locator("#auth-dialog").wait_for(state="hidden")
        page.locator("#chat-form").wait_for()
        page.locator("#chat-message").fill("Return exactly the token PASS.")
        page.get_by_role("button", name="Send", exact=True).click()
        assistant = page.locator("#message-list .message-card.assistant").last
        assistant.get_by_text("PASS", exact=True).wait_for(timeout=120_000)
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        response_text = assistant.inner_text()
        browser.close()
    return {
        "status": (
            "PASS" if not console_errors and not overflow and "PASS" in response_text else "FAIL"
        ),
        "console_errors": console_errors,
        "horizontal_overflow": overflow,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    if arguments.users < 5 or arguments.requests < 5 or arguments.requests % 5:
        raise SystemExit("users must be >=5 and requests must be a positive multiple of five")

    main_model = arguments.main_model
    codev_model = arguments.codev_model
    endpoint_evidence = {
        "main": _probe_model(arguments.main_endpoint, main_model),
        "codev": _probe_model(arguments.codev_endpoint, codev_model),
    }
    if any(item.get("status") != "HEALTHY_EXACT_MODEL" for item in endpoint_evidence.values()):
        (output_root / "endpoint_preflight.json").write_text(
            json.dumps(endpoint_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 2

    state = output_root / "state"
    users = UserCapabilityStore(state / "users.sqlite3")
    authorizations = RepositoryAuthorizationStore(state / "repositories.sqlite3")
    sandboxes = AgentSandboxManager(state / "worktrees", authorizations)
    credentials: dict[str, AuthCredential] = {}
    token_by_user: dict[str, str] = {}
    repository_by_user: dict[str, str] = {}
    for index in range(arguments.users):
        user_id = f"live-user-{index:02d}"
        tier = CapabilityTier.BASIC if index % 2 == 0 else CapabilityTier.PLUS
        users.set_user(user_id, tier)
        token = f"laplace-live-{secrets.token_urlsafe(32)}"
        token_by_user[user_id] = token
        credentials[token] = AuthCredential("read", user_id, tier)
        if tier is CapabilityTier.PLUS:
            repo_id = f"repo-{index:02d}"
            repository = output_root / "fixture_repositories" / repo_id
            revision = _git_repository(repository, f"OWNER_{index:02d}")
            authorizations.register(repo_id, repository)
            authorizations.grant(user_id, repo_id, base_revision=revision)
            repository_by_user[user_id] = repo_id

    operator_token = f"laplace-live-operator-{secrets.token_urlsafe(32)}"
    credentials[operator_token] = AuthCredential("admin", "live-operator", CapabilityTier.OPERATOR)
    users.set_user("live-operator", CapabilityTier.OPERATOR)
    browser_email = "live-operator@localhost.invalid"
    browser_password = secrets.token_urlsafe(32)
    registry_path = state / "auth/registered_users.yaml"
    write_registry(
        registry_path,
        [
            RegisteredUser(
                email=browser_email,
                user_id="live-operator",
                display_name="Live Operator",
                enabled=True,
                capability_tier=CapabilityTier.OPERATOR,
                role="admin",
                default_lane="quality",
                authorized_repo_ids=(),
                password_hash=hash_secret(browser_password, password_policy=True),
                must_change_password=False,
            )
        ],
    )
    registered_auth = RegisteredEmailAuth(
        RegisteredUserRegistry(registry_path),
        SessionStore(state / "auth/sessions.sqlite3"),
        AuthAuditLog(state / "auth/authentication_audit.jsonl"),
    )
    policy = LanePolicy(
        routes={
            ModelLane.QUALITY: ModelRoute(
                ModelLane.QUALITY,
                main_model,
                arguments.main_endpoint,
                0,
                context_limit=arguments.quality_context_limit,
                output_limit=256,
            ),
            ModelLane.STANDARD: ModelRoute(
                ModelLane.STANDARD,
                main_model,
                arguments.main_endpoint,
                10,
                context_limit=arguments.standard_context_limit,
                output_limit=256,
            ),
            ModelLane.ECONOMY: ModelRoute(
                ModelLane.ECONOMY,
                codev_model,
                arguments.codev_endpoint,
                20,
                context_limit=8_192,
                output_limit=256,
            ),
        },
        quality_reserved_slots=1,
        standard_capacity=2,
        economy_capacity=4,
    )
    tiered = TieredServingService(
        users=users,
        sandboxes=sandboxes,
        lane_policy=policy,
        chat_backend=LocalOpenAIChatBackend(timeout_seconds=300),
        agent_backend=_BoundedFixtureAgent(),
        audit_log=TierAuditLog(state / "audit.jsonl"),
    )
    port = _free_port()
    app = create_operator_app(
        OperatorService(root, state / "operator"),
        OperatorAuth(credentials),
        settings=OperatorApiSettings(
            port=port,
            allowed_origins=(
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            ),
            bearer_api_enabled=True,
        ),
        tiered=tiered,
        registered_auth=registered_auth,
        conversation_store=ConversationStore(state / "conversations.sqlite3"),
    )
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
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("real_api_startup_failed")

    base_url = f"http://127.0.0.1:{port}"
    results: list[dict[str, object]] = []
    queue_snapshots: list[dict[str, object]] = []
    gui_result: dict[str, object] = {"status": "SKIPPED"}
    try:
        with httpx.Client(base_url=base_url, timeout=360) as client:
            headers_by_user = {
                user_id: _session(client, token)[0] for user_id, token in token_by_user.items()
            }
            user_ids = sorted(token_by_user)
            workload = list(tier_mix(arguments.requests))

            def invoke(index: int, lane: ModelLane) -> dict[str, object]:
                user_id = user_ids[index % len(user_ids)]
                marker = f"LIVE_USER_{index % len(user_ids):02d}_REQUEST_{index:03d}"
                domain = "systemverilog" if lane is ModelLane.ECONOMY else "general"
                prompt = (
                    f"Reply with only `valid && ready` and marker {marker}."
                    if lane is ModelLane.ECONOMY
                    else f"Reply with exactly `PASS {marker}`."
                )
                started = time.perf_counter()
                response = httpx.post(
                    base_url + "/api/v1/chat",
                    headers=headers_by_user[user_id],
                    json={
                        "lane": lane.value,
                        "domain": domain,
                        "session_id": f"mixed-{index:03d}",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=360,
                )
                elapsed_ms = (time.perf_counter() - started) * 1_000
                value: object = response.json()
                return {
                    "index": index,
                    "user_id": user_id,
                    "lane": lane.value,
                    "http_status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "marker": marker,
                    "response": value,
                }

            stop_poll = threading.Event()

            def poll_queue() -> None:
                first_user = user_ids[0]
                with httpx.Client(base_url=base_url, timeout=30) as poll_client:
                    while not stop_poll.is_set():
                        try:
                            response = poll_client.get(
                                "/api/v1/tier/capabilities",
                                headers={"Authorization": f"Bearer {token_by_user[first_user]}"},
                            )
                        except httpx.TransportError:
                            if stop_poll.wait(0.02):
                                break
                            continue
                        if response.status_code == 200:
                            value = response.json()
                            if isinstance(value, dict) and isinstance(value.get("queue"), dict):
                                queue_snapshots.append(value["queue"])
                        stop_poll.wait(0.02)

            poller = threading.Thread(target=poll_queue, daemon=True)
            poller.start()
            with ThreadPoolExecutor(max_workers=arguments.users) as executor:
                futures = [
                    executor.submit(invoke, index, lane) for index, lane in enumerate(workload)
                ]
                for future in as_completed(futures):
                    results.append(future.result())
            stop_poll.set()
            poller.join(timeout=5)

            lifecycle: list[dict[str, object]] = []
            plus_users = sorted(repository_by_user)
            for user_id in plus_users:
                session_id = f"agent-{user_id}"
                created = client.post(
                    "/api/v1/agent/sessions",
                    headers=headers_by_user[user_id],
                    json={
                        "repo_id": repository_by_user[user_id],
                        "session_id": session_id,
                    },
                )
                status = client.get(
                    f"/api/v1/agent/sessions/{session_id}/status",
                    headers={"Authorization": f"Bearer {token_by_user[user_id]}"},
                )
                lifecycle.append(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "create_status": created.status_code,
                        "status_status": status.status_code,
                        "repo_id": (
                            created.json().get("binding", {}).get("repo_id")
                            if isinstance(created.json(), dict)
                            else None
                        ),
                        "worktree_status": (
                            created.json().get("binding", {}).get("worktree_status")
                            if isinstance(created.json(), dict)
                            else None
                        ),
                    }
                )
            cross_user = client.get(
                f"/api/v1/agent/sessions/agent-{plus_users[0]}/status",
                headers={"Authorization": f"Bearer {token_by_user[plus_users[1]]}"},
            )
            for user_id in plus_users:
                session_id = f"agent-{user_id}"
                cancelled = client.post(
                    f"/api/v1/agent/sessions/{session_id}/cancel",
                    headers=headers_by_user[user_id],
                )
                final_status = client.get(
                    f"/api/v1/agent/sessions/{session_id}/status",
                    headers={"Authorization": f"Bearer {token_by_user[user_id]}"},
                )
                current = next(item for item in lifecycle if item["session_id"] == session_id)
                current["cancel_status"] = cancelled.status_code
                current["final_status"] = final_status.json().get("status")
            first_basic = next(
                user_id for user_id in user_ids if users.get(user_id).tier is CapabilityTier.BASIC
            )
            basic_agent = client.post(
                "/api/v1/agent/sessions",
                headers=headers_by_user[first_basic],
                json={"repo_id": "forbidden", "session_id": "basic-denied"},
            )

            lane_quality: dict[str, dict[str, object]] = {}
            cases = load_quality_manifest(root / "configs/serving_quality_manifest.json")
            for lane in ModelLane:
                scored = []
                for case in cases:
                    api_domain = (
                        case.domain
                        if case.domain in {"general", "python", "json", "systemverilog"}
                        else "general"
                    )
                    response = client.post(
                        "/api/v1/chat",
                        headers=headers_by_user[user_ids[0]],
                        json={
                            "lane": lane.value,
                            "domain": api_domain,
                            "session_id": f"quality-{lane.value}-{case.case_id}",
                            "messages": [{"role": "user", "content": case.prompt}],
                        },
                    )
                    payload: object = response.json()
                    response_body = payload.get("response") if isinstance(payload, dict) else None
                    content = (
                        response_body.get("content") if isinstance(response_body, dict) else ""
                    )
                    case_score = asdict(
                        score_quality_output(
                            lane.value,
                            case,
                            content if isinstance(content, str) else "",
                        )
                    )
                    case_score["http_status"] = response.status_code
                    case_score["api_domain"] = api_domain
                    case_score["api_success"] = (
                        response.status_code == 200
                        and isinstance(payload, dict)
                        and payload.get("status") == "SUCCESS"
                    )
                    scored.append(case_score)
                score = sum(float(item["score"]) for item in scored) / len(scored)
                hard = all(
                    bool(item["passed_hard_gates"]) and bool(item["api_success"]) for item in scored
                )
                lane_quality[lane.value] = {
                    "score": score,
                    "passed_hard_gates": hard,
                    "cases": scored,
                }

            if not arguments.skip_gui:
                gui_result = _gui_smoke(base_url, browser_email, browser_password)

        own_markers = {str(item["marker"]) for item in results}
        successes = [
            item
            for item in results
            if item["http_status"] == 200
            and isinstance(item["response"], dict)
            and item["response"].get("status") == "SUCCESS"
        ]
        leakage = []
        for item in successes:
            response_text = json.dumps(item["response"], sort_keys=True)
            foreign = sorted(
                marker
                for marker in own_markers
                if marker != item["marker"] and marker in response_text
            )
            if foreign:
                leakage.append({"index": item["index"], "foreign_markers": foreign})
        route_counts = Counter(
            str(item["response"].get("model_id"))
            for item in successes
            if isinstance(item["response"], dict)
        )
        observed_waiting = max(
            (_list_length(snapshot.get("waiting")) for snapshot in queue_snapshots),
            default=0,
        )
        quality_baseline = _required_float(
            lane_quality["quality"]["score"],
            label="quality_baseline",
        )
        floors = {
            "quality": quality_baseline * 0.99,
            "standard": quality_baseline * 0.95,
            "economy": quality_baseline * 0.85,
        }
        quality_floor_pass = {
            lane: _required_float(
                lane_quality[lane]["score"],
                label=f"{lane}_quality_score",
            )
            >= floor
            and bool(lane_quality[lane]["passed_hard_gates"])
            for lane, floor in floors.items()
        }
        worktree_repositories = [
            str(item["repo_id"]) for item in lifecycle if isinstance(item.get("repo_id"), str)
        ]
        audit_lines = [
            json.loads(line)
            for line in (state / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        required_audit_fields = {
            "user_id",
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
        }
        request_audits = [item for item in audit_lines if item.get("mode") == "chat"]
        pass_conditions = {
            "endpoint_identity": all(
                item.get("status") == "HEALTHY_EXACT_MODEL" for item in endpoint_evidence.values()
            ),
            "mixed_requests": len(successes) == arguments.requests,
            "exact_mix": Counter(item["lane"] for item in results)
            == {"quality": 12, "standard": 36, "economy": 12}
            if arguments.requests == 60
            else Counter(item["lane"] for item in results)
            == Counter(lane.value for lane in tier_mix(arguments.requests)),
            "quality_route_present": route_counts[main_model] > 0,
            "codev_systemverilog_route_present": route_counts[codev_model] > 0,
            "queue_visible": observed_waiting > 0,
            "no_cross_user_context_leakage": not leakage,
            "isolated_worktrees": (
                len(worktree_repositories) == len(set(worktree_repositories)) == len(lifecycle)
                and all(item.get("worktree_status") == "ACTIVE_ISOLATED" for item in lifecycle)
            ),
            "cross_user_session_denied": cross_user.status_code == 403,
            "basic_agent_denied": basic_agent.status_code == 403,
            "cancellation_and_final_visibility": all(
                item.get("cancel_status") == 200 and item.get("final_status") == "CANCELLED"
                for item in lifecycle
            ),
            "quality_floors": all(quality_floor_pass.values()),
            "audit_schema": bool(request_audits)
            and all(required_audit_fields <= set(item) for item in request_audits),
            "audit_has_no_raw_prompts": all(
                marker not in json.dumps(audit_lines) for marker in own_markers
            ),
            "gui_live_integration": gui_result.get("status") in {"PASS", "SKIPPED"},
        }
        report = {
            "status": "PASS" if all(pass_conditions.values()) else "FAIL",
            "endpoint_evidence": endpoint_evidence,
            "users": arguments.users,
            "requests": arguments.requests,
            "successes": len(successes),
            "lane_counts": dict(sorted(Counter(item["lane"] for item in results).items())),
            "route_counts": dict(sorted(route_counts.items())),
            "max_observed_waiting": observed_waiting,
            "queue_snapshots_observed": len(queue_snapshots),
            "cross_user_leakage": leakage,
            "agent_lifecycle": lifecycle,
            "quality_floors": floors,
            "quality_floor_pass": quality_floor_pass,
            "gui": gui_result,
            "pass_conditions": pass_conditions,
            "requests_sha256": hashlib.sha256(
                json.dumps(results, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        (output_root / "endpoint_preflight.json").write_text(
            json.dumps(endpoint_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_root / "lane_quality_results.json").write_text(
            json.dumps(lane_quality, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_root / "parallel_api_smoke.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_root / "request_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output_root": str(output_root), "status": report["status"]}))
        return 0 if report["status"] == "PASS" else 1
    finally:
        server.should_exit = True
        server_thread.join(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
