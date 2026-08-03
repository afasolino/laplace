#!/usr/bin/env python3
"""Run registered GUI authentication and a real local CodeV browser smoke."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess  # nosec B404 - fixed local commands
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from playwright.sync_api import sync_playwright

from research_workspace.repository_authorization import RepositoryAuthorizationStore


ROOT = Path(__file__).resolve().parents[1]
ADMIN_EMAIL = "afasolino@unisa.it"
PLUS_EMAIL = "laplace-plus-smoke@localhost"
REPO_ID = "live-systemverilog-fixture"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _chromium() -> Path:
    candidates = sorted(
        (Path.home() / ".cache/ms-playwright").glob(
            "chromium-*/chrome-linux64/chrome"
        )
    )
    if not candidates:
        raise RuntimeError("Playwright Chromium is not installed")
    return candidates[-1]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def _admin(
    state_root: Path,
    *arguments: str,
    expect_activation: bool = False,
) -> tuple[dict[str, object], str]:
    completed = subprocess.run(  # nosec B603 - fixed local module and arguments
        [
            sys.executable,
            "-m",
            "research_workspace.user_admin",
            *arguments,
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    if completed.returncode != 0:
        raise RuntimeError("local user administration command failed")
    marker = "One-time activation code (shown once):\n"
    if expect_activation:
        if completed.stdout.count(marker) != 1:
            raise RuntimeError("activation code was not emitted exactly once")
        metadata_text, remainder = completed.stdout.split(marker, 1)
        activation = remainder.splitlines()[0]
        if not activation or completed.stdout.count(activation) != 1:
            raise RuntimeError("activation code was not emitted exactly once")
    else:
        metadata_text = completed.stdout
        activation = ""
    metadata: object = json.loads(metadata_text)
    if not isinstance(metadata, dict):
        raise RuntimeError("local user administration output is malformed")
    return metadata, activation


def _git_fixture(path: Path) -> str:
    (path / "rtl").mkdir(parents=True)
    (path / "python").mkdir(parents=True)
    (path / "rtl/example.sv").write_text(
        "module example(input logic a, output logic y);\n"
        "  assign y = a;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    (path / "rtl/tb_example.sv").write_text(
        "module tb_example;\n"
        "  logic a;\n"
        "  logic y;\n"
        "  example dut(.a(a), .y(y));\n"
        "  initial begin\n"
        "    a = 1'b0; #1; if (y !== 1'b1) $fatal(1, \"invert zero\");\n"
        "    a = 1'b1; #1; if (y !== 1'b0) $fatal(1, \"invert one\");\n"
        "    $display(\"SYSTEMVERILOG_VERIFY_PASS\");\n"
        "    $finish;\n"
        "  end\n"
        "endmodule\n",
        encoding="utf-8",
    )
    (path / "python/value.py").write_text(
        "def value() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (path / "python/test_value.py").write_text(
        "from python.value import value\n\n\n"
        "def test_value() -> None:\n"
        "    assert value() == 2\n",
        encoding="utf-8",
    )
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "GIT_AUTHOR_NAME": "Laplace Live Smoke",
        "GIT_AUTHOR_EMAIL": "laplace-live-smoke@localhost",
        "GIT_COMMITTER_NAME": "Laplace Live Smoke",
        "GIT_COMMITTER_EMAIL": "laplace-live-smoke@localhost",
    }
    for command in (
        ["git", "init", "-q", str(path)],
        ["git", "-C", str(path), "add", "."],
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"],
    ):
        completed = subprocess.run(  # nosec B603 B607 - fixed Git verbs
            command,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError("disposable repository creation failed")
    revision = subprocess.run(  # nosec B603 B607 - fixed Git query
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return revision.stdout.strip()


def _wait_for_server(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Operator server exited during startup")
        try:
            with urllib.request.urlopen(  # nosec B310 - fixed loopback URL
                base_url + "/api/v1/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise RuntimeError("Operator server readiness timed out")


def _activate(page: object, email: str, code: str, password: str) -> None:
    page.get_by_role("tab", name="First activation").click()  # type: ignore[attr-defined]
    page.locator("#activation-form input[name=email]").fill(email)  # type: ignore[attr-defined]
    page.locator("#activation-code").fill(code)  # type: ignore[attr-defined]
    page.locator("#activation-password").fill(password)  # type: ignore[attr-defined]
    page.locator("#activation-form input[name=confirm_password]").fill(password)  # type: ignore[attr-defined]
    page.get_by_role("button", name="Activate account").click()  # type: ignore[attr-defined]
    page.locator("#auth-dialog").wait_for(state="hidden", timeout=30_000)  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codev-endpoint", default="http://127.0.0.1:8103")
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    endpoint = arguments.codev_endpoint.rstrip("/")
    try:
        with urllib.request.urlopen(endpoint + "/v1/models", timeout=10) as response:  # nosec B310
            models: object = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError("CodeV endpoint is unavailable") from exc
    served = [
        item.get("id")
        for item in models.get("data", [])
        if isinstance(item, dict)
    ] if isinstance(models, dict) and isinstance(models.get("data"), list) else []
    expected_model = "laplace-codev-r1-rl-qwen-7b-w4a16"
    if expected_model not in served:
        raise RuntimeError("CodeV endpoint serves an unexpected model")

    result: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="laplace-live-registered-") as temporary:
        temporary_root = Path(temporary)
        state_root = temporary_root / "external-state"
        registry = state_root / "auth/registered_users.yaml"
        sessions = state_root / "auth/sessions.sqlite3"
        repository = temporary_root / "authorized-repository"
        revision = _git_fixture(repository)
        bootstrap, admin_code = _admin(
            state_root,
            "bootstrap",
            "--registry",
            str(registry),
            "--session-store",
            str(sessions),
            "--email",
            ADMIN_EMAIL,
            "--user-id",
            "usr_afasolino",
            "--display-name",
            "Alfonso Fasolino",
            "--capability-tier",
            "operator",
            "--role",
            "admin",
            "--default-lane",
            "quality",
            expect_activation=True,
        )
        added, plus_code = _admin(
            state_root,
            "add",
            "--registry",
            str(registry),
            "--session-store",
            str(sessions),
            "--email",
            PLUS_EMAIL,
            "--user-id",
            "usr_live_plus",
            "--display-name",
            "Live Plus Fixture",
            "--capability-tier",
            "plus",
            "--role",
            "user",
            "--default-lane",
            "economy",
            expect_activation=True,
        )
        _admin(
            state_root,
            "authorize-repo",
            "--registry",
            str(registry),
            "--email",
            PLUS_EMAIL,
            "--repo-id",
            REPO_ID,
        )
        authorizations = RepositoryAuthorizationStore(
            state_root / "tiered_serving/repository_authorizations.sqlite3"
        )
        authorizations.register(REPO_ID, repository)
        authorizations.grant("usr_live_plus", REPO_ID, base_revision=revision)

        admin_password = f"live-admin-{secrets.token_urlsafe(64)}"
        plus_password = f"live-plus-{secrets.token_urlsafe(64)}"
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(  # nosec B603 - fixed local module
            [
                sys.executable,
                "-m",
                "research_workspace.operator_server",
                "--repository-root",
                str(ROOT),
                "--state-root",
                str(state_root),
                "--user-registry",
                str(registry),
                "--session-store",
                str(sessions),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--allowed-origin",
                base_url,
                "--allowed-origin",
                f"http://localhost:{port}",
                "--allowed-host",
                "127.0.0.1",
                "--allowed-host",
                "localhost",
            ],
            cwd=ROOT,
            env=_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        console_errors: list[str] = []
        page_errors: list[str] = []
        operator_stopped = False
        try:
            _wait_for_server(base_url, process)
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
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.goto(base_url, wait_until="networkidle")
                _activate(page, ADMIN_EMAIL, admin_code, admin_password)
                page.locator("#account-tier").get_by_text(
                    "operator · admin", exact=True
                ).wait_for()
                page.locator("#chat-lane").select_option("economy")
                page.locator("#chat-domain").select_option("systemverilog")
                page.locator("#chat-message").fill(
                    "Reply in concise Markdown with heading 'Live CodeV', one bullet, "
                    "and a fenced SystemVerilog code block containing a module named pass."
                )
                page.get_by_role("button", name="Send", exact=True).click()
                page.locator(".message-card.assistant").wait_for(timeout=180_000)
                assistant_text = page.locator(".message-card.assistant").inner_text()
                page.get_by_text("Response details", exact=True).click()
                details_text = page.locator(".metadata-grid").inner_text()
                code_blocks = page.locator(".message-card.assistant .code-block").count()
                page.get_by_role("button", name="Sign out", exact=True).click()
                page.locator("#auth-dialog").wait_for(state="visible")

                _activate(page, PLUS_EMAIL, plus_code, plus_password)
                page.get_by_role("button", name="Agent", exact=True).click()
                page.locator("#agent-repository").select_option(REPO_ID)
                page.locator("#agent-form select[name=lane]").select_option("economy")
                page.locator("#agent-domain").select_option("systemverilog")
                page.locator("#agent-form textarea[name=instruction]").fill(
                    "The file rtl/example.sv contains `assign y = a;`. Modify only that "
                    "line to `assign y = ~a;`. Return the requested strict JSON edit "
                    "object with path rtl/example.sv, exact old text, and exact "
                    "replacement text."
                )
                page.get_by_role("button", name="Start isolated agent").click()
                page.locator("#agent-state", has_text="Complete").wait_for(
                    timeout=180_000
                )
                diff_text = page.locator("#agent-diff").inner_text()
                tests_text = page.locator("#agent-tests").inner_text()
                storage_entries = page.evaluate(
                    "() => Object.keys(localStorage).length + "
                    "Object.keys(sessionStorage).length"
                )
                session_cookie = next(
                    cookie
                    for cookie in page.context.cookies()
                    if cookie["name"] == "laplace_session"
                )
                browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            operator_stopped = process.poll() is not None

        registry_text = registry.read_text(encoding="utf-8")
        audit_text = (state_root / "auth/authentication_audit.jsonl").read_text(
            encoding="utf-8"
        )
        secrets_absent = all(
            secret not in registry_text and secret not in audit_text
            for secret in (admin_code, plus_code, admin_password, plus_password)
        )
        admin_code = plus_code = admin_password = plus_password = ""
        checks = {
            "first_account_exact": bootstrap.get("email") == ADMIN_EMAIL
            and bootstrap.get("capability_tier") == "operator"
            and bootstrap.get("role") == "admin"
            and bootstrap.get("default_lane") == "quality",
            "activation_codes_shown_once": True,
            "live_codev_identity": expected_model in served,
            "live_chat_readable": bool(assistant_text.strip()),
            "live_chat_code_block": code_blocks > 0,
            "live_chat_response_details": expected_model in details_text,
            "plus_account_registered": added.get("capability_tier") == "plus",
            "plus_repository_isolated": REPO_ID in diff_text
            or "rtl/example.sv" in diff_text,
            "plus_diff_readable": "assign y = -a" not in diff_text
            and "assign y = ~a" in diff_text,
            "plus_verification_readable": "PASSED" in tests_text,
            "opaque_http_only_session": session_cookie["httpOnly"] is True
            and len(str(session_cookie["value"])) >= 22,
            "browser_credential_storage_empty": storage_entries == 0,
            "credentials_absent_from_registry_and_audit": secrets_absent,
            "operator_server_safe_shutdown": operator_stopped,
            "no_browser_errors": not console_errors and not page_errors,
        }
        result = {
            "schema_version": 1,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "served_model": expected_model,
            "operator_bind": "127.0.0.1",
            "registered_account": {
                "email": ADMIN_EMAIL,
                "capability_tier": "operator",
                "role": "admin",
                "default_lane": "quality",
            },
            "temporary_plus_account": {
                "email": PLUS_EMAIL,
                "capability_tier": "plus",
                "authorized_repository_id": REPO_ID,
            },
            "console_error_count": len(console_errors),
            "page_error_count": len(page_errors),
        }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
