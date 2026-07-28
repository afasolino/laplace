#!/usr/bin/env python3
"""Capture the v6 GUI screenshot set from a disposable, synthetic fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
import time
from pathlib import Path
from typing import Sequence

from playwright.sync_api import Page, sync_playwright

from capture_user_guide_screenshots import (
    _Models,
    _Profiles,
    _chromium,
    _free_port,
    _git_repository,
    _research,
    _server,
)
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
from research_workspace.personal_corpus import PersonalCorpusPolicy, PersonalCorpusStore
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.research_admission import ResearchAdmissionStore
from research_workspace.service_tiers import (
    LanePolicy,
    ModelLane,
    ModelRoute,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.user_capabilities import (
    Capability,
    CapabilityTier,
    UserCapabilityStore,
)


ROOT = Path(__file__).resolve().parents[1]
ADMIN_EMAIL = "fixture-admin@example.test"
NO_REPOSITORY_EMAIL = "fixture-no-repository@example.test"
SCREENSHOTS = (
    "domain_selector.png",
    "agent_no_repository.png",
    "agent_new_worktree.png",
    "agent_worktree_history.png",
    "agent_progress.png",
    "personal_corpus_empty.png",
    "folder_upload_manifest.png",
    "personal_corpus_indexed.png",
    "retrieval_source_selector.png",
    "chat_processing_state.png",
    "markdown_table.png",
    "admin_capabilities.png",
)


class _Chat:
    def complete(self, **_kwargs: object) -> dict[str, object]:
        time.sleep(0.8)
        return {
            "content": (
                "## Synthetic evidence table\n\n"
                "| Source | Page | Chunk | State |\n"
                "| --- | ---: | --- | --- |\n"
                "| fixture-reference.md | 1 | chk_fixture_001 | verified |\n"
                "| synthetic-design.sv | 2 | chk_fixture_002 | indexed |\n\n"
                "The fixture response contains no private or live source."
            ),
            "finish_reason": "stop",
            "verification_status": "PASSED",
        }


class _Agent:
    def run(self, **_kwargs: object) -> dict[str, object]:
        time.sleep(0.8)
        return {
            "content": "Synthetic Agent task completed.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "modified_paths": ["src/synthetic_summary.py"],
            "command_count": 2,
            "diff": (
                "--- /dev/null\n"
                "+++ b/src/synthetic_summary.py\n"
                "@@ -0,0 +1,2 @@\n"
                "+def summary() -> str:\n"
                '+    return "fixture"\n'
            ),
            "tests": [{"name": "pytest synthetic fixture", "status": "PASSED"}],
        }


def _users(admin_password: str, no_repository_password: str) -> list[RegisteredUser]:
    combined = tuple(Capability)
    return [
        RegisteredUser(
            email=ADMIN_EMAIL,
            user_id="fixture_admin",
            display_name="Synthetic Administrator",
            enabled=True,
            capability_tier=CapabilityTier.OPERATOR,
            role="admin",
            default_lane="quality",
            authorized_repo_ids=("synthetic-project",),
            password_hash=hash_secret(admin_password, password_policy=True),
            must_change_password=False,
            capabilities=combined,
        ),
        RegisteredUser(
            email=NO_REPOSITORY_EMAIL,
            user_id="fixture_no_repository",
            display_name="Synthetic Researcher",
            enabled=True,
            capability_tier=CapabilityTier.PLUS,
            role="user",
            default_lane="standard",
            authorized_repo_ids=(),
            password_hash=hash_secret(no_repository_password, password_policy=True),
            must_change_password=False,
            capabilities=(
                Capability.CHAT,
                Capability.AGENT,
                Capability.PERSONAL_CORPUS,
            ),
        ),
    ]


def _application(
    state_root: Path,
    port: int,
    admin_password: str,
    no_repository_password: str,
) -> object:
    users = _users(admin_password, no_repository_password)
    registry_path = state_root / "auth/registered_users.yaml"
    write_registry(registry_path, users)
    registered_auth = RegisteredEmailAuth(
        RegisteredUserRegistry(registry_path),
        SessionStore(state_root / "auth/sessions.sqlite3"),
        AuthAuditLog(state_root / "auth/authentication_audit.jsonl"),
    )
    capability_store = UserCapabilityStore(state_root / "tier/users.sqlite3")
    for user in users:
        capability_store.set_user(
            user.user_id,
            user.capability_tier,
            capabilities=user.effective_capabilities,
        )
    authorizations = RepositoryAuthorizationStore(
        state_root / "tier/repository_authorizations.sqlite3"
    )
    repository = state_root / "synthetic-project"
    _git_repository(repository)
    authorizations.register("synthetic-project", repository)
    authorizations.grant(
        "fixture_admin",
        "synthetic-project",
        base_revision="HEAD",
    )
    routes = {
        lane: ModelRoute(
            lane=lane,
            model_id=f"Synthetic local {lane.value}",
            endpoint=f"http://127.0.0.1:{8102 + index}/v1",
            priority=3 - index,
            context_limit=8_192,
            output_limit=1_024,
        )
        for index, lane in enumerate(ModelLane)
    }
    tiered = TieredServingService(
        users=capability_store,
        sandboxes=AgentSandboxManager(
            state_root / "tier/worktrees",
            authorizations,
            retention_days=30,
        ),
        lane_policy=LanePolicy(routes=routes),
        chat_backend=_Chat(),
        agent_backend=_Agent(),
        audit_log=TierAuditLog(state_root / "tier/audit.jsonl"),
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
            allowed_origins=(
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            ),
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
        personal_corpora=PersonalCorpusStore(
            state_root,
            policy=PersonalCorpusPolicy(min_free_disk_bytes=1),
        ),
    )


def _login(page: Page, email: str, password: str) -> None:
    page.locator("#login-form input[name=email]").fill(email)
    page.locator("#login-password").fill(password)
    page.get_by_role("button", name="Sign in", exact=True).last.click()
    page.locator("#auth-dialog").wait_for(state="hidden")


def _capture(page: Page, output: Path, name: str) -> None:
    page.screenshot(path=str(output / name), full_page=True)


def _screenshots(
    base_url: str,
    output: Path,
    selected_folder: Path,
    admin_password: str,
    no_repository_password: str,
) -> None:
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True,
            executable_path=str(_chromium()),
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 960},
            device_scale_factor=1,
            reduced_motion="reduce",
        )
        console_errors: list[str] = []
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text)
                if message.type == "error"
                else None
            ),
        )
        page.goto(base_url, wait_until="networkidle")
        _login(page, ADMIN_EMAIL, admin_password)

        _capture(page, output, "domain_selector.png")
        page.get_by_label("Sources", exact=True).select_option("both")
        _capture(page, output, "retrieval_source_selector.png")

        page.locator("#chat-message").fill("Render the synthetic evidence table.")
        page.get_by_role("button", name="Send", exact=True).click()
        page.locator("#chat-state").get_by_text("VALIDATING", exact=False).wait_for()
        _capture(page, output, "chat_processing_state.png")
        page.get_by_role("heading", name="Synthetic evidence table").wait_for()
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator(".composer").evaluate("(element) => { element.style.display = 'none'; }")
        page.locator(".sidebar").evaluate("(element) => { element.style.display = 'none'; }")
        page.locator(".message-card.assistant").last.screenshot(
            path=str(output / "markdown_table.png")
        )
        page.locator(".composer").evaluate("(element) => { element.style.display = ''; }")
        page.locator(".sidebar").evaluate("(element) => { element.style.display = ''; }")
        page.set_viewport_size({"width": 1440, "height": 960})

        page.get_by_role("button", name="Knowledge", exact=True).click()
        page.get_by_role("heading", name="Your personal corpus is empty").wait_for()
        _capture(page, output, "personal_corpus_empty.png")
        page.get_by_label("Corpus name").fill("Synthetic references")
        page.get_by_role("button", name="Create corpus", exact=True).click()
        page.get_by_text("Synthetic references", exact=True).wait_for()
        page.locator("#corpus-folder-input").set_input_files(str(selected_folder))
        page.get_by_role("button", name="Preview selected folder", exact=True).click()
        page.get_by_role("cell", name="ACCEPTED", exact=True).wait_for()
        page.get_by_role("cell", name="REJECTED", exact=True).wait_for()
        _capture(page, output, "folder_upload_manifest.png")
        page.get_by_role("button", name="Index accepted files", exact=True).click()
        page.get_by_text("Indexed 1 source(s)", exact=False).wait_for()
        _capture(page, output, "personal_corpus_indexed.png")

        page.get_by_role("button", name="Agent", exact=True).click()
        page.get_by_label("Authorized repository").select_option("synthetic-project")
        page.get_by_label("Bounded task").fill(
            "Add a deterministic synthetic summary helper and focused fixture test."
        )
        _capture(page, output, "agent_new_worktree.png")
        page.get_by_role("button", name="Start isolated agent").click()
        page.get_by_text("Running bounded task", exact=True).wait_for()
        _capture(page, output, "agent_progress.png")
        page.get_by_text("pytest synthetic fixture · PASSED", exact=True).wait_for()
        page.locator("#agent-worktree-history").get_by_text(
            "Add a deterministic synthetic summary helper", exact=False
        ).wait_for()
        _capture(page, output, "agent_worktree_history.png")

        page.get_by_role("button", name="Users", exact=True).click()
        page.get_by_role("cell", name=ADMIN_EMAIL, exact=True).wait_for()
        page.locator("#user-table details").first.locator("summary").click()
        _capture(page, output, "admin_capabilities.png")

        page.get_by_role("button", name="Sign out", exact=True).click()
        page.locator("#auth-dialog").wait_for(state="visible")
        _login(page, NO_REPOSITORY_EMAIL, no_repository_password)
        page.get_by_role("button", name="Agent", exact=True).click()
        page.get_by_text(
            "No repository is authorized for this account.", exact=True
        ).wait_for()
        _capture(page, output, "agent_no_repository.png")
        browser.close()
        if console_errors:
            raise RuntimeError(f"browser console errors: {console_errors}")


def _manifest(output: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for name in SCREENSHOTS:
        path = output / name
        if not path.is_file() or path.stat().st_size < 1_000:
            raise RuntimeError(f"missing or empty screenshot: {name}")
        files.append(
            {
                "file": name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fixture_data_only": True,
            }
        )
    manifest = {
        "schema_version": 1,
        "viewport": {
            "desktop": {"width": 1440, "height": 960, "scale": 1},
            "mobile": {"width": 390, "height": 844, "scale": 1},
        },
        "contains_live_secrets": False,
        "contains_private_documents": False,
        "files": files,
    }
    (output / "agent_personal_corpus_gui_v6_screenshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


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
    no_repository_password = f"fixture-user-{secrets.token_urlsafe(32)}"
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="laplace-v6-screenshot-fixture-") as temporary:
        fixture_root = Path(temporary)
        selected_folder = fixture_root / "selected-synthetic-folder"
        selected_folder.mkdir()
        (selected_folder / "fixture-reference.md").write_text(
            "# Synthetic reference\nFixture-only retrieval marker.\n",
            encoding="utf-8",
        )
        (selected_folder / "unsafe-fixture.exe").write_bytes(b"MZfixture")
        app = _application(
            fixture_root / "state",
            port,
            admin_password,
            no_repository_password,
        )
        for _server_instance in _server(app, port):
            _screenshots(
                f"http://127.0.0.1:{port}",
                output,
                selected_folder,
                admin_password,
                no_repository_password,
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
