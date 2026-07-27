from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
import uvicorn

from research_workspace.operator_api import (
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.research_models import (
    ClaimAssertion,
    DiscoveredSource,
    FetchedSource,
    FixtureResearchAdapter,
)
from research_workspace.research_plane import DeepResearchService


ROOT = Path(__file__).resolve().parents[1]
ADMIN_TOKEN = "gui-admin-token-00000000000000000000"


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
                },
                {
                    "profile": "phase2_rtl_worker",
                    "port": 8103,
                    "expected_model_id": "fixture-codev",
                    "endpoint_observation": {"status": "HEALTHY_EXACT_MODEL"},
                },
            ],
            "laplace_owned_processes": [],
        }

    def start(self) -> dict[str, object]:
        return {"status": "STARTED_HEALTHY_SERVERS"}

    def release_owned(self) -> dict[str, object]:
        return {"status": "RELEASED_LAPLACE_OWNED_SERVERS"}


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
        {"fixture": FixtureResearchAdapter([source])},
        clock=lambda: "2026-07-27T00:00:00+00:00",
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def gui_server(tmp_path: Path) -> Iterator[str]:
    operator = OperatorService(
        ROOT,
        tmp_path,
        model_servers=_ModelServers(),  # type: ignore[arg-type]
    )
    operator.request_approval(
        "START_GPU_RUN",
        "fixture-run",
        {"configuration_sha256": "1" * 64},
        actor_role="operate",
    )
    port = _free_port()
    app = create_operator_app(
        operator,
        OperatorAuth({ADMIN_TOKEN: "admin"}),
        settings=OperatorApiSettings(
            port=port,
            allowed_origins=(
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            ),
            fixture_mode=True,
        ),
        research=_research(tmp_path),
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
    page.get_by_label("Access token").fill(ADMIN_TOKEN)  # type: ignore[attr-defined]
    page.get_by_role("button", name="Authenticate").click()  # type: ignore[attr-defined]
    page.get_by_text("Local API online").wait_for()  # type: ignore[attr-defined]


def test_responsive_gui_research_approval_and_accessibility_smoke(
    gui_server: str,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    console_errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text)
                if message.type == "error"
                else None
            ),
        )
        _authenticate(page, gui_server)
        page.get_by_role("heading", name="Operational overview").wait_for()
        assert page.locator("#dashboard-cards .metric").count() == 4

        page.get_by_role("link", name="Run builder").click()
        page.get_by_role("button", name="Prepare immutable run").click()
        page.locator("#run-preview").get_by_text("PREPARED", exact=False).wait_for()

        page.get_by_role("link", name="Research").click()
        page.get_by_label("Backend").fill("fixture")
        page.get_by_role("button", name="Create research job").click()
        page.get_by_role("button", name="Run stages").click()
        page.locator("#evidence-ledger").get_by_text(
            "The fixture supports the responsive operator view.", exact=True
        ).wait_for()
        assert page.locator("#research-stages li.done").count() == 13

        page.get_by_role("link", name="Approvals").click()
        page.get_by_text("START_GPU_RUN", exact=True).wait_for()
        page.get_by_role("button", name="Approve").click()
        page.locator("#approval-list").get_by_text(
            "fixture-run · APPROVED", exact=True
        ).wait_for()

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
                hasLiveRegion: Boolean(document.querySelector('[aria-live]'))
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
        }

        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("link", name="Overview").click()
        responsive = page.evaluate(
            """() => ({
              overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
              railPosition: getComputedStyle(document.querySelector('.rail')).position,
              visibleHero: document.querySelector('.hero').getBoundingClientRect().width > 0
            })"""
        )
        assert responsive == {
            "overflow": False,
            "railPosition": "fixed",
            "visibleHero": True,
        }
        assert console_errors == []
        browser.close()
