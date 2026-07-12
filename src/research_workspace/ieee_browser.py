from __future__ import annotations

import json
import importlib.util
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .projects import load_project


PROFILE = Path.home() / "AppData" / "Local" / "FormalScienceBrowser"


def browser_profile() -> Path:
    configured = (
        Path(os.getenv("FORMALSCIENCE_BROWSER_PROFILE", str(PROFILE))).expanduser().resolve()
    )
    if str(configured).startswith(str(Path.cwd().resolve())):
        raise ValueError("Browser profile must be outside the application repository")
    if "OneDrive" in str(configured):
        raise ValueError("Browser profile must be outside OneDrive")
    return configured


def browser_init() -> dict[str, Any]:
    profile = browser_profile()
    profile.mkdir(parents=True, exist_ok=True)
    manifest = profile / "formalscience-browser.json"
    manifest.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "purpose": "visible IEEE manual-authentication session",
                "credentials_persisted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "READY",
        "profile": str(profile),
        "playwright_installed": _playwright_available(),
    }


def _playwright_available() -> bool:
    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except ModuleNotFoundError:
        return False


def login() -> dict[str, Any]:
    info = browser_init()
    if not info["playwright_installed"]:
        return {
            "status": "PLAYWRIGHT_REQUIRED",
            "profile": info["profile"],
            "instruction": "Install the optional Playwright dependency, then run this command in a visible session. The user must complete IEEE/institutional login manually; automation never enters credentials.",
        }
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(info["profile"], headless=False)
            page = context.pages[0] if context.pages else context.new_page()
            page.goto("https://ieeexplore.ieee.org", wait_until="domcontentloaded", timeout=60_000)
            input(
                "Complete IEEE/institutional login manually in the visible browser, then press Enter here. Never enter credentials into automation. "
            )
            context.close()
        return {
            "status": "MANUAL_LOGIN_CONFIRMED",
            "profile": info["profile"],
            "credentials_persisted": False,
        }
    except Exception as exc:
        return {"status": "BROWSER_ERROR", "profile": info["profile"], "error": str(exc)}


def _load_candidate(
    project: str, candidate_id: int, *, root: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    path = queue_path(project, root=root)
    queue = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if candidate_id < 0 or candidate_id >= len(queue):
        raise IndexError("candidate id not in queue")
    return path, queue[candidate_id]


def open_candidate(project: str, candidate_id: int, *, root: Path | None = None) -> dict[str, Any]:
    _, candidate = _load_candidate(project, candidate_id, root=root)
    url = str(candidate.get("canonical_url") or "")
    if not url.startswith("https://ieeexplore.ieee.org/"):
        raise ValueError("IEEE open requires a trusted ieeexplore.ieee.org article URL")
    info = browser_init()
    return {
        "status": "MANUAL_BROWSER_REQUIRED"
        if info["playwright_installed"]
        else "PLAYWRIGHT_REQUIRED",
        "candidate_id": candidate_id,
        "url": url,
        "title": candidate.get("title"),
        "doi": candidate.get("doi"),
        "ieee_article_number": candidate.get("ieee_article_number"),
        "profile": info["profile"],
        "instruction": "Confirm the title/DOI/article number in the visible headed browser; automation must not enter credentials.",
    }


def download_candidate(
    project: str, candidate_id: int, *, root: Path | None = None, batch_size: int = 1
) -> dict[str, Any]:
    _, candidate = _load_candidate(project, candidate_id, root=root)
    if candidate.get("approval_state") != "USER_APPROVED":
        raise PermissionError(
            "Explicit user approval is required before an IEEE subscribed-content download"
        )
    if batch_size != 1:
        raise ValueError("The default IEEE browser batch is one item")
    info = browser_init()
    return {
        "status": "PLAYWRIGHT_REQUIRED"
        if not info["playwright_installed"]
        else "MANUAL_VISIBLE_DOWNLOAD_REQUIRED",
        "candidate_id": candidate_id,
        "profile": info["profile"],
        "destination": "Data/Downloads/IEEE/Pending",
        "instruction": "Use only the unambiguous official IEEE PDF download control. Stop on CAPTCHA, 401/403/429, access warnings, expired login, or unexpected navigation.",
    }


def ieee_status(project: str | None = None, *, root: Path | None = None) -> dict[str, Any]:
    profile = browser_profile()
    result: dict[str, Any] = {
        "profile": str(profile),
        "profile_exists": profile.is_dir(),
        "playwright_installed": _playwright_available(),
        "credentials_persisted": False,
    }
    if project:
        result["queue"] = list_queue(project, root=root)
    return result


def queue_path(project: str, *, root: Path | None = None) -> Path:
    paths, _ = load_project(project, root=root)
    path = paths.data / "Downloads" / "IEEE" / "Pending" / "candidate_queue.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def queue_candidate(
    project: str, candidate: dict[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    path = queue_path(project, root=root)
    queue = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    candidate = {
        **candidate,
        "queued_at": datetime.now(UTC).isoformat(),
        "approval_state": "NOT_APPROVED",
        "download_state": "PENDING",
    }
    queue.append(candidate)
    path.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "QUEUED",
        "queue": str(path),
        "candidate_id": len(queue) - 1,
        "batch_default": 1,
    }


def list_queue(project: str, *, root: Path | None = None) -> dict[str, Any]:
    path = queue_path(project, root=root)
    return {
        "queue": str(path),
        "items": json.loads(path.read_text(encoding="utf-8")) if path.exists() else [],
    }


def approve_download(
    project: str, candidate_id: int, *, root: Path | None = None, batch_size: int = 1
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 3:
        raise ValueError("IEEE batch size must be 1..3")
    path = queue_path(project, root=root)
    queue = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if candidate_id < 0 or candidate_id >= len(queue):
        raise IndexError("candidate id not in queue")
    queue[candidate_id]["approval_state"] = "USER_APPROVED"
    queue[candidate_id]["approved_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "APPROVED",
        "candidate_id": candidate_id,
        "batch_size": batch_size,
        "next_action": "Run ieee download only in a visible manually authenticated browser.",
    }
