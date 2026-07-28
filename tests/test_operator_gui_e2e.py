from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
import uvicorn

from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.auth_registry import RegisteredUser, RegisteredUserRegistry, hash_secret, write_registry
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
EMAIL = "afasolino@unisa.it"
PASSWORD = "fixture password with enough local entropy"


class _ModelServers:
    def status(self) -> dict[str, object]:
        return {
            "status": "OBSERVED",
            "gpu_observation": {
                "status": "OBSERVED",
                "gpu": {
                    "name": "Fixture A6000",
                    "memory_free_mib": 49_000,
                    "utilization_percent": 0,
                },
            },
            "servers": [
                {
                    "profile": "phase2_main",
                    "port": 8102,
                    "expected_model_id": "fixture-main",
                    "endpoint_observation": {"status": "HEALTHY_EXACT_MODEL"},
                }
            ],
            "laplace_owned_processes": [],
        }

    def start(self) -> dict[str, object]:
        return {"status": "STARTED_HEALTHY_SERVERS"}

    def release_owned(self) -> dict[str, object]:
        return {"status": "RELEASED_LAPLACE_OWNED_SERVERS"}


class _ServingProfiles:
    def status(self) -> dict[str, object]:
        return {"status": "RUNNING", "profile_id": "P1_fp8_kv"}


class _ChatBackend:
    def complete(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": (
                "## Evidence-led answer\n\n"
                "Laplace returned readable **Markdown**.\n\n"
                "```python\nresult = \"local\"\nprint(result)\n```"
            ),
            "finish_reason": "stop",
            "verification_status": "PASSED",
        }


class _AgentBackend:
    def run(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": "Fixture agent result.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "modified_paths": ["src/fixture.py"],
            "diff": "--- a/src/fixture.py\n+++ b/src/fixture.py\n+result = 'verified'\n",
            "tests": [{"name": "pytest", "status": "PASSED"}],
        }


def _research(state_root: Path) -> DeepResearchService:
    source = FetchedSource(
        discovered=DiscoveredSource(
            canonical_url="https://example.org/gui-fixture",
            title="GUI fixture evidence",
            backend="fixture",
            query="",
            source_type="official_documentation",
            license="CC-BY-4.0",
        ),
        content=b"Fixture evidence supports the responsive operator view.",
        assertions=(
            ClaimAssertion(
                normalized_claim="The fixture supports the responsive operator view.",
                claim_key="responsive-view",
            ),
        ),
        retrieved_at="2026-07-27T00:00:00+00:00",
    )
    return DeepResearchService(
        state_root,
        {
            "fixture": FixtureResearchAdapter([source]),
            "local_governed_corpus": FixtureResearchAdapter([source]),
            "local_uploaded_documents": FixtureResearchAdapter([source]),
        },
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _tiered(root: Path) -> TieredServingService:
    users = UserCapabilityStore(root / "tier/users.sqlite3")
    users.set_user("usr_afasolino", CapabilityTier.OPERATOR)
    routes = {
        lane: ModelRoute(
            lane=lane,
            model_id=f"fixture-{lane.value}",
            endpoint=f"http://127.0.0.1:{8102 + index}/v1",
            priority=3 - index,
            context_limit=24_576,
            output_limit=4_096,
        )
        for index, lane in enumerate(ModelLane)
    }
    return TieredServingService(
        users=users,
        sandboxes=AgentSandboxManager(
            root / "tier/worktrees",
            RepositoryAuthorizationStore(root / "tier/repositories.sqlite3"),
        ),
        lane_policy=LanePolicy(
            routes=routes,
            quality_reserved_slots=1,
            standard_capacity=2,
            economy_capacity=4,
        ),
        chat_backend=_ChatBackend(),
        agent_backend=_AgentBackend(),
        audit_log=TierAuditLog(root / "tier/audit.jsonl"),
    )


@pytest.fixture
def gui_server(tmp_path: Path) -> Iterator[str]:
    operator = OperatorService(
        ROOT,
        tmp_path / "operator",
        model_servers=_ModelServers(),  # type: ignore[arg-type]
    )
    account = RegisteredUser(
        email=EMAIL,
        user_id="usr_afasolino",
        display_name="Alfonso Fasolino",
        enabled=True,
        capability_tier=CapabilityTier.OPERATOR,
        role="admin",
        default_lane="quality",
        authorized_repo_ids=(),
        password_hash=hash_secret(PASSWORD, password_policy=True),
        must_change_password=False,
    )
    registry_path = tmp_path / "auth/registered_users.yaml"
    write_registry(registry_path, [account])
    registered_auth = RegisteredEmailAuth(
        RegisteredUserRegistry(registry_path),
        SessionStore(tmp_path / "auth/sessions.sqlite3"),
        AuthAuditLog(tmp_path / "auth/audit.jsonl"),
    )
    port = _free_port()
    app = create_operator_app(
        operator,
        OperatorAuth({}),
        settings=OperatorApiSettings(
            port=port,
            allowed_origins=(
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            ),
            fixture_mode=True,
        ),
        research=_research(tmp_path / "research"),
        tiered=_tiered(tmp_path),
        registered_auth=registered_auth,
        conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
        research_admission=ResearchAdmissionStore(tmp_path / "research_admission.sqlite3"),
        serving_profile_operator=_ServingProfiles(),  # type: ignore[arg-type]
    )
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
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        pytest.fail("fixture GUI server did not start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _authenticate(page: object, base_url: str) -> None:
    page.goto(base_url, wait_until="networkidle")  # type: ignore[attr-defined]
    page.get_by_label("Email address").first.fill(EMAIL)  # type: ignore[attr-defined]
    page.locator("#login-password").fill(PASSWORD)  # type: ignore[attr-defined]
    page.get_by_role("button", name="Sign in", exact=True).last.click()  # type: ignore[attr-defined]
    page.locator("#auth-dialog").wait_for(state="hidden")  # type: ignore[attr-defined]
    page.locator("#account-tier").get_by_text("operator · admin", exact=True).wait_for()  # type: ignore[attr-defined]


def test_registered_gui_chat_research_operator_accessibility_and_responsive(
    gui_server: str,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    console_errors: list[str] = []
    page_errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        chromium = next(
            (Path.home() / ".cache/ms-playwright").glob(
                "chromium-*/chrome-linux64/chrome"
            )
        )
        browser = runtime.chromium.launch(
            headless=True,
            executable_path=str(chromium),
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text)
                if message.type == "error"
                else None
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        _authenticate(page, gui_server)

        page.locator("#chat-message").fill("Show a readable local response.")
        page.get_by_role("button", name="Send", exact=True).click()  # type: ignore[attr-defined]
        page.get_by_role("heading", name="Evidence-led answer").wait_for()  # type: ignore[attr-defined]
        page.locator(".code-block").get_by_text('result = "local"', exact=False).wait_for()
        page.get_by_text("Response details", exact=True).click()
        assert page.locator(".metadata-grid").is_visible()

        page.get_by_role("button", name="Research", exact=True).click()
        page.get_by_label("Research question").fill("What supports the responsive operator view?")
        page.get_by_role("button", name="Create research job").click()
        page.get_by_role("button", name="Run admitted job").click()
        page.get_by_role("heading", name="Cited report").wait_for()
        page.locator("#research-sources").get_by_text("GUI fixture evidence", exact=True).wait_for()

        page.get_by_role("button", name="Operations", exact=True).click()
        page.get_by_role("heading", name="Operator dashboard").wait_for()
        page.locator("#operations-cards .info-card").first.wait_for()
        page.get_by_role("button", name="Users", exact=True).click()
        page.get_by_role("cell", name=EMAIL, exact=True).wait_for()
        page.get_by_role("button", name="Models & GPU", exact=True).click()
        page.get_by_text("Fixture A6000", exact=True).wait_for()

        inert_markdown = page.evaluate(
            """() => {
              const rendered = renderMarkdown(
                '<img src=x onerror=alert(1)> [unsafe](javascript:alert(1)) ' +
                '[safe](https://example.org/) <script>alert(1)</script>'
              );
              return {
                scripts: rendered.querySelectorAll('script').length,
                images: rendered.querySelectorAll('img').length,
                unsafeLinks: [...rendered.querySelectorAll('a')]
                  .filter((link) => !['http:', 'https:'].includes(new URL(link.href).protocol)).length,
                textIncludesLiteral: rendered.textContent.includes('<script>alert(1)</script>'),
                externalRel: rendered.querySelector('a')?.rel
              };
            }"""
        )
        assert inert_markdown == {
            "scripts": 0,
            "images": 0,
            "unsafeLinks": 0,
            "textIncludesLiteral": True,
            "externalRel": "noopener noreferrer",
        }

        accessibility = page.evaluate(
            """() => {
              const ids = [...document.querySelectorAll('[id]')].map((el) => el.id);
              const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
              const unlabeled = [...document.querySelectorAll('input,select,textarea')]
                .filter((el) => !el.labels?.length && !el.getAttribute('aria-label'));
              const unnamedButtons = [...document.querySelectorAll('button')]
                .filter((el) => !el.textContent.trim() && !el.getAttribute('aria-label'));
              return {
                duplicateIds,
                unlabeled: unlabeled.length,
                unnamedButtons: unnamedButtons.length,
                hasMain: Boolean(document.querySelector('main')),
                hasSkipLink: Boolean(document.querySelector('.skip-link')),
                hasLiveRegion: Boolean(document.querySelector('[aria-live]')),
                credentialStorage: Object.keys(localStorage).length + Object.keys(sessionStorage).length,
                focusRule: [...document.styleSheets].some((sheet) =>
                  [...sheet.cssRules].some((rule) => rule.selectorText?.includes(':focus-visible'))
                )
              };
            }"""
        )
        assert accessibility == {
            "duplicateIds": [],
            "unlabeled": 0,
            "unnamedButtons": 0,
            "hasMain": True,
            "hasSkipLink": True,
            "hasLiveRegion": True,
            "credentialStorage": 0,
            "focusRule": True,
        }

        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="Chat", exact=True).click()
        responsive = page.evaluate(
            """() => ({
              overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
              sidebarPosition: getComputedStyle(document.querySelector('.sidebar')).position,
              composerVisible: document.querySelector('.composer').getBoundingClientRect().width > 0
            })"""
        )
        assert responsive == {
            "overflow": False,
            "sidebarPosition": "fixed",
            "composerVisible": True,
        }
        assert console_errors == []
        browser.close()
