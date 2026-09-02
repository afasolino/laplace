#!/usr/bin/env python3
"""Generate sanitized user-guide screenshots from a disposable local fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator, Sequence

import uvicorn

from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.auth_registry import (
    RegisteredUser,
    RegisteredUserRegistry,
    hash_secret,
    write_registry,
)
from research_workspace.auth_sessions import AuthAuditLog, RegisteredEmailAuth, SessionStore
from research_workspace.conversations import ConversationStore
from research_workspace.operator_api import OperatorApiSettings, OperatorAuth, create_operator_app
from research_workspace.operator_service import OperatorService
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.research_admission import ResearchAdmissionStore
from research_workspace.research_models import (
    ClaimAssertion,
    DiscoveredSource,
    FetchedSource,
    FixtureResearchAdapter,
)
from research_workspace.research_plane import DeepResearchService
from research_workspace.service_tiers import (
    LanePolicy,
    ModelLane,
    ModelRoute,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.user_capabilities import CapabilityTier, UserCapabilityStore


ROOT = Path(__file__).resolve().parents[1]
ADMIN_EMAIL = "afasolino@unisa.it"
PLUS_EMAIL = "researcher@example.test"
SCREENSHOTS = (
    "login.png",
    "first_activation.png",
    "chat.png",
    "chat_response_details.png",
    "conversation_history.png",
    "plus_agent.png",
    "agent_diff_and_tests.png",
    "deep_research.png",
    "operator_dashboard.png",
    "user_management.png",
    "model_and_gpu_status.png",
    "remote_access_status.png",
)


class _Models:
    def status(self) -> dict[str, object]:
        return {
            "status": "OBSERVED",
            "gpu_observation": {
                "status": "OBSERVED",
                "gpu": {
                    "name": "NVIDIA RTX A6000 (fixture)",
                    "memory_free_mib": 32_768,
                    "utilization_percent": 7,
                },
            },
            "servers": [
                {
                    "profile": "P8_qwen38_w4a16_mtp",
                    "port": 8207,
                    "expected_model_id": "Qwen3.8 27B MTP8 · quality/standard",
                    "endpoint_observation": {"status": "HEALTHY_EXACT_MODEL"},
                },
                {
                    "profile": "CodeV",
                    "port": 8103,
                    "expected_model_id": "CodeV specialist · economy RTL",
                    "endpoint_observation": {"status": "HEALTHY_EXACT_MODEL"},
                },
            ],
            "laplace_owned_processes": [
                {"pid": 41001, "profile": "P8_qwen38_w4a16_mtp"},
                {"pid": 41002, "profile": "CodeV"},
            ],
        }

    def start(self) -> dict[str, object]:
        return {"status": "STARTED_HEALTHY_SERVERS"}

    def release_owned(self) -> dict[str, object]:
        return {"status": "RELEASED_LAPLACE_OWNED_SERVERS"}


class _Profiles:
    def status(self) -> dict[str, object]:
        return {"status": "RUNNING", "profile_id": "P8_qwen38_w4a16_mtp"}


class _Chat:
    def complete(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": (
                "## Evidence-grounded local answer\n\n"
                "The indexed source supports a deterministic workflow with explicit "
                "file, page, section, and chunk provenance.\n\n"
                "```python\nfrom pathlib import Path\nresult = Path(\"report.md\").read_text()\n```\n\n"
                "The response remains private and tool-free."
            ),
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "usage": {"prompt_tokens": 1260, "completion_tokens": 184},
        }


class _Agent:
    def run(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": "Implemented the bounded fixture update and verified it locally.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "modified_paths": ["src/analysis.py", "tests/test_analysis.py"],
            "diff": (
                "--- a/src/analysis.py\n"
                "+++ b/src/analysis.py\n"
                "@@ -4,3 +4,7 @@\n"
                "+def summarize(values: list[float]) -> float:\n"
                "+    \"\"\"Return a deterministic arithmetic mean.\"\"\"\n"
                "+    return sum(values) / len(values)\n"
            ),
            "tests": [
                {"name": "pytest tests/test_analysis.py", "status": "PASSED"},
                {"name": "ruff check", "status": "PASSED"},
            ],
        }


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _users(admin_password: str, plus_password: str) -> list[RegisteredUser]:
    return [
        RegisteredUser(
            email=ADMIN_EMAIL,
            user_id="usr_afasolino",
            display_name="Alfonso Fasolino",
            enabled=True,
            capability_tier=CapabilityTier.OPERATOR,
            role="admin",
            default_lane="quality",
            authorized_repo_ids=(),
            password_hash=hash_secret(admin_password, password_policy=True),
            must_change_password=False,
        ),
        RegisteredUser(
            email=PLUS_EMAIL,
            user_id="usr_researcher",
            display_name="Local Researcher",
            enabled=True,
            capability_tier=CapabilityTier.PLUS,
            role="user",
            default_lane="standard",
            authorized_repo_ids=("fixture-project",),
            password_hash=hash_secret(plus_password, password_policy=True),
            must_change_password=False,
        ),
    ]


def _research(root: Path) -> DeepResearchService:
    source = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://example.org/method",
            title="Deterministic local research method",
            backend="local_governed_corpus",
            query="",
            source_type="official_documentation",
            license="CC-BY-4.0",
            publication="Fixture Engineering Manual",
            publication_date="2026-07-01",
        ),
        content=b"Deterministic evidence led workflows retain citations and conflicts.",
        assertions=(
            ClaimAssertion(
                normalized_claim="Deterministic workflows retain cited evidence.",
                claim_key="retained-evidence",
                confidence=0.92,
            ),
        ),
        retrieved_at="2026-07-27T12:00:00+00:00",
    )
    adapter = FixtureResearchAdapter([source])
    return DeepResearchService(
        root,
        {
            "local_governed_corpus": adapter,
            "local_uploaded_documents": adapter,
        },
        clock=lambda: "2026-07-27T12:00:00+00:00",
    )


def _git_repository(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "fixture@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Fixture User"], check=True)
    (path / "README.md").write_text("# Fixture project\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)


def _application(
    root: Path,
    port: int,
    admin_password: str,
    plus_password: str,
    *,
    external_url: str | None = None,
) -> object:
    users = _users(admin_password, plus_password)
    registry_path = root / "auth/registered_users.yaml"
    write_registry(registry_path, users)
    registered_auth = RegisteredEmailAuth(
        RegisteredUserRegistry(registry_path),
        SessionStore(root / "auth/sessions.sqlite3"),
        AuthAuditLog(root / "auth/authentication_audit.jsonl"),
    )
    capability_store = UserCapabilityStore(root / "tier/users.sqlite3")
    for user in users:
        capability_store.set_user(user.user_id, user.capability_tier)
    repositories = RepositoryAuthorizationStore(root / "tier/repositories.sqlite3")
    repository = root / "fixture-project"
    _git_repository(repository)
    repositories.register("fixture-project", repository)
    repositories.grant("usr_researcher", "fixture-project")
    routes = {
        lane: ModelRoute(
            lane=lane,
            model_id={
                ModelLane.QUALITY: "Qwen main · quality",
                ModelLane.STANDARD: "Qwen main · standard",
                ModelLane.ECONOMY: "CodeV / main eligible route",
            }[lane],
            endpoint=f"http://127.0.0.1:{8102 if lane is not ModelLane.ECONOMY else 8103}/v1",
            priority={ModelLane.QUALITY: 3, ModelLane.STANDARD: 2, ModelLane.ECONOMY: 1}[lane],
            context_limit={ModelLane.QUALITY: 24_576, ModelLane.STANDARD: 16_384, ModelLane.ECONOMY: 8_192}[lane],
            output_limit={ModelLane.QUALITY: 4_096, ModelLane.STANDARD: 2_048, ModelLane.ECONOMY: 1_024}[lane],
        )
        for lane in ModelLane
    }
    tiered = TieredServingService(
        users=capability_store,
        sandboxes=AgentSandboxManager(root / "tier/worktrees", repositories),
        lane_policy=LanePolicy(
            routes=routes,
            quality_reserved_slots=1,
            standard_capacity=2,
            economy_capacity=4,
        ),
        chat_backend=_Chat(),
        agent_backend=_Agent(),
        audit_log=TierAuditLog(root / "tier/audit.jsonl"),
    )
    return create_operator_app(
        OperatorService(ROOT, root / "operator", model_servers=_Models()),  # type: ignore[arg-type]
        OperatorAuth({}),
        settings=OperatorApiSettings(
            port=port,
            deployment_mode="reverse-proxy" if external_url else "local",
            allowed_origins=(
                (external_url,)
                if external_url
                else (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
            ),
            allowed_hosts=("localhost",) if external_url else ("127.0.0.1", "localhost"),
            external_url=external_url,
            fixture_mode=True,
        ),
        research=_research(root / "research"),
        tiered=tiered,
        serving_profile_operator=_Profiles(),  # type: ignore[arg-type]
        registered_auth=registered_auth,
        conversation_store=ConversationStore(root / "conversations/conversations.sqlite3"),
        research_admission=ResearchAdmissionStore(root / "research/admission.sqlite3"),
    )


def _server(app: object, port: int) -> Iterator[uvicorn.Server]:
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
            proxy_headers=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("screenshot fixture failed to start")
    try:
        yield server
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("screenshot fixture failed to stop cleanly")


def _chromium() -> Path:
    candidates = sorted(
        (Path.home() / ".cache/ms-playwright").glob(
            "chromium-*/chrome-linux64/chrome"
        ),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Playwright Chromium is unavailable; run `playwright install chromium`")
    return candidates[0]


def _login(page: object, email: str, password: str) -> None:
    page.locator("#login-form input[name=email]").fill(email)  # type: ignore[attr-defined]
    page.locator("#login-password").fill(password)  # type: ignore[attr-defined]
    page.get_by_role("button", name="Sign in", exact=True).last.click()  # type: ignore[attr-defined]
    page.locator("#auth-dialog").wait_for(state="hidden")  # type: ignore[attr-defined]


def _capture(page: object, output: Path, name: str) -> None:
    page.screenshot(path=str(output / name), full_page=True)  # type: ignore[attr-defined]


def _create_screenshots(
    base: str,
    output: Path,
    admin_password: str,
    plus_password: str,
) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, executable_path=str(_chromium()))
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
            reduced_motion="reduce",
        )
        console_errors: list[str] = []
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(base, wait_until="networkidle")
        _capture(page, output, "login.png")
        page.get_by_role("tab", name="First activation").click()
        page.locator("#activation-form input[name=email]").fill(ADMIN_EMAIL)
        _capture(page, output, "first_activation.png")
        page.get_by_role("tab", name="Sign in").click()
        _login(page, ADMIN_EMAIL, admin_password)

        page.locator("#chat-message").fill(
            "Show an evidence-grounded local example with Python."
        )
        page.get_by_role("button", name="Send", exact=True).click()
        page.get_by_role("heading", name="Evidence-grounded local answer").wait_for()
        _capture(page, output, "chat.png")
        page.get_by_text("Response details", exact=True).click()
        _capture(page, output, "chat_response_details.png")
        _capture(page, output, "conversation_history.png")

        page.get_by_role("button", name="Research", exact=True).click()
        page.get_by_label("Research question").fill(
            "How does a deterministic local workflow retain cited evidence?"
        )
        page.get_by_role("button", name="Create research job").click()
        page.get_by_role("button", name="Run admitted job").click()
        page.locator("#research-sources").get_by_text(
            "Deterministic local research method", exact=True
        ).wait_for()
        _capture(page, output, "deep_research.png")

        page.get_by_role("button", name="Operations", exact=True).click()
        page.locator("#operations-cards .info-card").first.wait_for()
        _capture(page, output, "operator_dashboard.png")
        page.get_by_role("button", name="Users", exact=True).click()
        page.get_by_role("cell", name=ADMIN_EMAIL, exact=True).wait_for()
        _capture(page, output, "user_management.png")
        page.get_by_role("button", name="Models & GPU", exact=True).click()
        page.get_by_text("NVIDIA RTX A6000 (fixture)", exact=True).wait_for()
        _capture(page, output, "model_and_gpu_status.png")
        page.get_by_role("button", name="System", exact=True).click()
        _capture(page, output, "remote_access_status.png")

        page.get_by_role("button", name="Sign out", exact=True).click()
        page.locator("#auth-dialog").wait_for(state="visible")
        _login(page, PLUS_EMAIL, plus_password)
        page.get_by_role("button", name="Agent", exact=True).click()
        _capture(page, output, "plus_agent.png")
        page.get_by_label("Authorized repository").select_option("fixture-project")
        page.get_by_label("Bounded task").fill(
            "Add a deterministic summary helper and its focused test."
        )
        page.get_by_role("button", name="Start isolated agent").click()
        page.get_by_text("pytest tests/test_analysis.py · PASSED", exact=True).wait_for()
        _capture(page, output, "agent_diff_and_tests.png")
        browser.close()
        if console_errors:
            raise RuntimeError(f"browser console errors: {console_errors}")


def _manifest(output: Path) -> dict[str, object]:
    files = []
    for name in SCREENSHOTS:
        path = output / name
        if not path.is_file() or path.stat().st_size < 1_000:
            raise RuntimeError(f"missing or empty screenshot: {name}")
        files.append(
            {
                "file": name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "synthetic_data": True,
            }
        )
    value = {
        "schema_version": 1,
        "viewport": {"width": 1440, "height": 1000, "scale": 1},
        "contains_live_secrets": False,
        "files": files,
    }
    (output / "screenshot_manifest.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/user_guide/assets",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    admin_password = f"fixture-admin-{secrets.token_urlsafe(32)}"
    plus_password = f"fixture-plus-{secrets.token_urlsafe(32)}"
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="laplace-screenshot-fixture-") as temporary:
        app = _application(Path(temporary), port, admin_password, plus_password)
        for _server_instance in _server(app, port):
            _create_screenshots(
                f"http://127.0.0.1:{port}",
                output,
                admin_password,
                plus_password,
            )
    manifest = _manifest(output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "screenshots": len(manifest["files"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
