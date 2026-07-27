#!/usr/bin/env python3
"""Display and certify real run/research evidence through the localhost GUI."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import threading
import time
from pathlib import Path

import uvicorn

from research_workspace.operator_api import (
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.research_plane import DeepResearchService


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--second-run-id", required=True)
    parser.add_argument("--research-job-id", required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--admin-token", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository_root.resolve()
    state_root = arguments.state_root.resolve()
    bundle = arguments.bundle_path.resolve()
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    operator = OperatorService(repository, state_root)
    research = DeepResearchService(state_root, {})
    run_record = operator.get_run(arguments.run_id, actor_role="read")
    terminal = run_record.get("terminal_result")
    if not isinstance(terminal, dict):
        raise RuntimeError("live run has no terminal evidence")
    corpus_sha256 = str(terminal.get("corpus_snapshot_sha256", ""))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    app = create_operator_app(
        operator,
        OperatorAuth({arguments.admin_token: "admin"}),
        settings=OperatorApiSettings(
            port=port,
            allowed_origins=(base_url, f"http://localhost:{port}"),
        ),
        research=research,
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
        raise RuntimeError("live GUI server did not start")

    console_errors: list[str] = []
    acceptance: dict[str, bool] = {}
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as runtime:
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
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Access token").fill(arguments.admin_token)
            page.get_by_role("button", name="Authenticate").click()
            page.get_by_text("Local API online").wait_for()
            page.get_by_role("heading", name="Operational overview").wait_for()
            acceptance["dashboard_loads"] = True
            acceptance["real_run_in_dashboard"] = page.get_by_text(
                arguments.run_id, exact=True
            ).count() > 0
            acceptance["research_visible_independently"] = page.get_by_text(
                "What official and repository evidence supports local append-only "
                "OpenTelemetry-compatible execution traces?",
                exact=True,
            ).count() > 0
            page.get_by_role("link", name="Live run").click()
            page.get_by_label("Run ID").fill(arguments.run_id)
            page.get_by_role("button", name="Inspect run").click()
            page.locator("#gate-matrix .mini-card").first.wait_for()
            gate_text = page.locator("#gate-matrix").inner_text()
            run_text = page.locator("#run-record").inner_text()
            acceptance["verilator_simulation_gate_pass"] = (
                "verilator_simulation" in gate_text
                and "PASS" in gate_text
                and "executed yes" in gate_text
            )
            acceptance["reviewer_approval_displayed"] = (
                '"reviewer_approved": true' in run_text
                and '"verdict": "approve"' in run_text
            )
            acceptance["frozen_corpus_hash_displayed"] = (
                len(corpus_sha256) == 64 and corpus_sha256 in run_text
            )
            time.sleep(2)
            acceptance["live_stage_updates_arrive"] = (
                page.locator("#event-timeline li").count() > 0
            )
            page.screenshot(path=output / "live_run_desktop.png", full_page=True)

            page.get_by_role("link", name="Hardware").click()
            page.locator("#model-servers .stack-item").first.wait_for()
            hardware_text = page.locator("#model-servers").inner_text()
            acceptance["gpu_and_endpoint_health_displayed"] = (
                "HEALTHY_EXACT_MODEL" in hardware_text
                and "phase2_main" in hardware_text
                and "phase2_rtl_worker" in hardware_text
            )

            relative_bundle = bundle.relative_to(repository).as_posix()
            response = page.request.get(
                f"{base_url}/api/v1/artifacts/download?path={relative_bundle}",
                headers={"Authorization": f"Bearer {arguments.admin_token}"},
            )
            downloaded = response.body()
            acceptance["certification_bundle_downloadable"] = (
                response.ok
                and hashlib.sha256(downloaded).hexdigest()
                == hashlib.sha256(bundle.read_bytes()).hexdigest()
            )
            comparison = page.request.get(
                f"{base_url}/api/v1/runs/compare/"
                f"{arguments.run_id}/{arguments.second_run_id}",
                headers={"Authorization": f"Bearer {arguments.admin_token}"},
            )
            acceptance["real_run_comparison_available"] = comparison.ok
            manifest = page.request.get(f"{base_url}/manifest.webmanifest")
            acceptance["pwa_manifest_available"] = manifest.ok

            accessibility = page.evaluate(
                """() => ({
                  duplicateIds: [...document.querySelectorAll('[id]')]
                    .map((el) => el.id)
                    .filter((id, index, ids) => ids.indexOf(id) !== index),
                  unlabeled: [...document.querySelectorAll('input,select,textarea')]
                    .filter((el) => !el.labels?.length && !el.getAttribute('aria-label')).length,
                  unnamedButtons: [...document.querySelectorAll('button')]
                    .filter((el) => !el.textContent.trim() && !el.getAttribute('aria-label')).length,
                  hasMain: Boolean(document.querySelector('main')),
                  hasSkipLink: Boolean(document.querySelector('.skip-link')),
                  hasLiveRegion: Boolean(document.querySelector('[aria-live]'))
                })"""
            )
            acceptance["accessibility_smoke"] = accessibility == {
                "duplicateIds": [],
                "unlabeled": 0,
                "unnamedButtons": 0,
                "hasMain": True,
                "hasSkipLink": True,
                "hasLiveRegion": True,
            }
            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_role("link", name="Live run").click()
            responsive = page.evaluate(
                """() => ({
                  overflow: document.documentElement.scrollWidth >
                    document.documentElement.clientWidth,
                  railPosition: getComputedStyle(document.querySelector('.rail')).position,
                  gateWidth: document.querySelector('#gate-matrix').getBoundingClientRect().width
                })"""
            )
            acceptance["mobile_viewport_usable"] = (
                responsive["overflow"] is False
                and responsive["railPosition"] == "fixed"
                and responsive["gateWidth"] > 0
            )
            page.screenshot(path=output / "live_run_mobile.png", full_page=True)
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    acceptance["no_console_errors"] = not console_errors
    evidence = {
        "schema_version": 1,
        "status": "PASS" if all(acceptance.values()) else "FAILED",
        "run_id": arguments.run_id,
        "second_run_id": arguments.second_run_id,
        "research_job_id": arguments.research_job_id,
        "bundle_path": str(bundle),
        "acceptance": acceptance,
        "console_errors": console_errors,
        "screenshots": [
            str(output / "live_run_desktop.png"),
            str(output / "live_run_mobile.png"),
        ],
    }
    (output / "live_gui_smoke.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
