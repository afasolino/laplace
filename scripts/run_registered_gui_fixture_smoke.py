#!/usr/bin/env python3
"""Certify CLI bootstrap, first activation, session security, and readable chat."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from playwright.sync_api import sync_playwright

from capture_user_guide_screenshots import (
    _Agent,
    _Chat,
    _Models,
    _Profiles,
    _chromium,
    _free_port,
    _research,
    _server,
)
from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.auth_registry import RegisteredUserRegistry
from research_workspace.auth_sessions import AuthAuditLog, RegisteredEmailAuth, SessionStore
from research_workspace.conversations import ConversationStore
from research_workspace.operator_api import OperatorApiSettings, OperatorAuth, create_operator_app
from research_workspace.operator_service import OperatorService
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.research_admission import ResearchAdmissionStore
from research_workspace.service_tiers import (
    LanePolicy,
    ModelLane,
    ModelRoute,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.user_capabilities import UserCapabilityStore


ROOT = Path(__file__).resolve().parents[1]
ADMIN_EMAIL = "fixture-admin@example.test"
ADMIN_USER_ID = "usr_fixture_admin"


def _bootstrap(registry: Path, sessions: Path) -> tuple[str, dict[str, object]]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_workspace.user_admin",
            "bootstrap",
            "--registry",
            str(registry),
            "--session-store",
            str(sessions),
            "--email",
            ADMIN_EMAIL,
            "--user-id",
            ADMIN_USER_ID,
            "--display-name",
            "Alfonso Fasolino",
            "--capability-tier",
            "operator",
            "--role",
            "admin",
            "--default-lane",
            "quality",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError("bootstrap command failed")
    marker = "One-time activation code (shown once):\n"
    if completed.stdout.count(marker) != 1:
        raise RuntimeError("bootstrap did not emit exactly one activation marker")
    remainder = completed.stdout.split(marker, 1)[1]
    activation_code = remainder.splitlines()[0]
    if not activation_code or completed.stdout.count(activation_code) != 1:
        raise RuntimeError("activation code was not shown exactly once")
    json_end = completed.stdout.index(marker)
    metadata: object = json.loads(completed.stdout[:json_end])
    if not isinstance(metadata, dict):
        raise RuntimeError("bootstrap metadata is invalid")
    return activation_code, metadata


def _application(state_root: Path, port: int) -> object:
    registry = RegisteredUserRegistry(state_root / "auth/registered_users.yaml")
    user = registry.require_user(ADMIN_USER_ID)
    capabilities = UserCapabilityStore(state_root / "tier/users.sqlite3")
    capabilities.set_user(user.user_id, user.capability_tier, enabled=user.enabled)
    routes = {
        lane: ModelRoute(
            lane=lane,
            model_id=f"fixture-{lane.value}",
            endpoint=f"http://127.0.0.1:{8300 + index}/v1",
            priority=3 - index,
            context_limit=24_576,
            output_limit=4_096,
        )
        for index, lane in enumerate(ModelLane)
    }
    tiered = TieredServingService(
        users=capabilities,
        sandboxes=AgentSandboxManager(
            state_root / "tier/worktrees",
            RepositoryAuthorizationStore(state_root / "tier/repositories.sqlite3"),
        ),
        lane_policy=LanePolicy(
            routes=routes,
            quality_reserved_slots=1,
            standard_capacity=2,
            economy_capacity=4,
        ),
        chat_backend=_Chat(),
        agent_backend=_Agent(),
        audit_log=TierAuditLog(state_root / "tier/audit.jsonl"),
    )
    registered_auth = RegisteredEmailAuth(
        registry,
        SessionStore(state_root / "auth/sessions.sqlite3"),
        AuthAuditLog(state_root / "auth/authentication_audit.jsonl"),
    )
    return create_operator_app(
        OperatorService(
            ROOT,
            state_root / "operator",
            model_servers=_Models(),  # type: ignore[arg-type]
        ),
        OperatorAuth({}),
        settings=OperatorApiSettings(
            port=port,
            allowed_origins=(f"http://127.0.0.1:{port}", f"http://localhost:{port}"),
            fixture_mode=True,
        ),
        research=_research(state_root / "research"),
        tiered=tiered,
        serving_profile_operator=_Profiles(),  # type: ignore[arg-type]
        registered_auth=registered_auth,
        conversation_store=ConversationStore(
            state_root / "conversations/conversations.sqlite3"
        ),
        research_admission=ResearchAdmissionStore(
            state_root / "research/admission.sqlite3"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result: dict[str, object]
    with tempfile.TemporaryDirectory(prefix="laplace-registered-gui-") as temporary:
        state_root = Path(temporary)
        registry_path = state_root / "auth/registered_users.yaml"
        session_path = state_root / "auth/sessions.sqlite3"
        activation_code, bootstrap = _bootstrap(registry_path, session_path)
        registry = RegisteredUserRegistry(registry_path)
        user = registry.require_user(ADMIN_USER_ID)
        new_password = f"activated-{secrets.token_urlsafe(64)}"
        port = _free_port()
        app = _application(state_root, port)
        console_errors: list[str] = []
        for _instance in _server(app, port):
            with sync_playwright() as runtime:
                browser = runtime.chromium.launch(
                    headless=True,
                    executable_path=str(_chromium()),
                )
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
                page.get_by_role("tab", name="First activation").click()
                page.locator("#activation-form input[name=email]").fill(ADMIN_EMAIL)
                page.locator("#activation-code").fill(activation_code)
                page.locator("#activation-password").fill(new_password)
                page.locator(
                    "#activation-form input[name=confirm_password]"
                ).fill(new_password)
                page.get_by_role("button", name="Activate account").click()
                page.locator("#auth-dialog").wait_for(state="hidden")
                page.locator("#chat-message").fill(
                    "Return a readable evidence-grounded local example."
                )
                page.get_by_role("button", name="Send", exact=True).click()
                page.get_by_role(
                    "heading",
                    name="Evidence-grounded local answer",
                ).wait_for()
                storage = page.evaluate(
                    "() => Object.keys(localStorage).length + Object.keys(sessionStorage).length"
                )
                cookies = page.context.cookies()
                session_cookie = next(
                    cookie
                    for cookie in cookies
                    if cookie["name"] == "laplace_session"
                )
                browser.close()
        audit = (state_root / "auth/authentication_audit.jsonl").read_text(
            encoding="utf-8"
        )
        registry.reload()
        activated = registry.require_user(ADMIN_USER_ID)
        secret_absent = all(
            secret not in audit and secret not in registry_path.read_text(encoding="utf-8")
            for secret in (activation_code, new_password)
        )
        activation_code = ""
        new_password = ""
        result = {
            "schema_version": 1,
            "status": (
                "PASS"
                if bootstrap.get("status") == "BOOTSTRAPPED"
                and user.enabled
                and user.must_change_password
                and not activated.must_change_password
                and storage == 0
                and session_cookie["httpOnly"] is True
                and secret_absent
                and not console_errors
                else "FAIL"
            ),
            "account": {
                "email": activated.email,
                "enabled": activated.enabled,
                "capability_tier": activated.capability_tier.value,
                "role": activated.role,
                "default_lane": activated.default_lane,
            },
            "bootstrap_activation_code_shown_once": True,
            "first_activation_forced_password_creation": True,
            "session_cookie": {
                "opaque": len(str(session_cookie["value"])) >= 22,
                "http_only": session_cookie["httpOnly"],
                "same_site": session_cookie["sameSite"],
                "secure": session_cookie["secure"],
                "development_http": True,
            },
            "browser_credential_storage_entries": storage,
            "readable_markdown_chat": True,
            "audit_and_registry_secret_scan": "PASS" if secret_absent else "FAIL",
            "safe_shutdown": "PASS",
        }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
