"""User-facing ``laplace`` command for local FormalScience projects.

The command is intentionally a small orchestration layer.  The existing
``research-workspace`` command remains available for low-level/reproducible
operations; this module adds project discovery, lifecycle state, and safe
delegation from any working directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .acquisition import download_open_access, promote_download
from .chat import ChatEngine, ChatProject, ConversationStore
from .engineering import (
    AgentTaskStore,
    Domain,
    EngineeringError,
    LocalToolRunner,
    ReferenceLibrary,
    normalize_task_spec,
    retrieve_engineering_evidence,
)
from .extraction import Provenance, extract_metrics, write_record
from .inference import Engine, ServingCandidate, benchmark_local_candidate
from .ieee_browser import (
    approve_download,
    browser_init,
    download_candidate,
    ieee_status,
    list_queue,
    login,
    open_candidate,
    queue_candidate,
)
from .library import ingest_library
from .online import fetch_public_webpage, search_general_web, search_scholarly
from .projects import COLLECTIONS, ProjectError, ProjectPaths, validate_project_name
from .real_benchmark import ollama_tags
from .retrieval import evidence_packet, search as local_search
from .probe import collect_probe
from .paired_benchmark import run_paired_quality_benchmark, run_valid_paired_benchmark
from .team_runner import LocalTeamRunner


APP_HOME = Path(os.getenv("LAPLACE_HOME", str(Path.home() / ".laplace"))).expanduser()
REGISTRY_PATH = APP_HOME / "projects.json"
CONFIG_PATH = APP_HOME / "config.yaml"
MODEL = "qwen3:4b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
ENDPOINT = "http://127.0.0.1:11434"
APPLICATION_ROOT = Path(__file__).resolve().parents[2]


class LaplaceError(RuntimeError):
    """A recoverable user-facing Laplace error."""


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_home() -> None:
    try:
        APP_HOME.mkdir(parents=True, exist_ok=True)
        (APP_HOME / "logs").mkdir(parents=True, exist_ok=True)
        if not CONFIG_PATH.exists():
            CONFIG_PATH.write_text(
                yaml.safe_dump(
                    {
                        "bind_host": "127.0.0.1",
                        "port": 8000,
                        "main_model": MODEL,
                        "embedding_model": EMBEDDING_MODEL,
                        "ollama_endpoint": ENDPOINT,
                        "local_only": True,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        if not REGISTRY_PATH.exists():
            REGISTRY_PATH.write_text("[]\n", encoding="utf-8")
    except PermissionError as exc:
        raise LaplaceError(
            f"Cannot write the global Laplace directory {APP_HOME}; set LAPLACE_HOME to a writable local directory or grant user-profile access"
        ) from exc


def _registry() -> list[dict[str, Any]]:
    _ensure_home()
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaplaceError(f"Invalid Laplace registry: {exc}") from exc
    if not isinstance(value, list):
        raise LaplaceError("Laplace registry must contain a JSON list")
    return [item for item in value if isinstance(item, dict)]


def _write_registry(items: list[dict[str, Any]]) -> None:
    _ensure_home()
    REGISTRY_PATH.write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _register(project: Path, *, name: str | None = None) -> dict[str, Any]:
    project = project.resolve()
    project_name = name or project.name
    validate_project_name(project_name)
    items = _registry()
    for item in items:
        same_name = item.get("name") == project_name
        same_path = Path(str(item.get("path", ""))).resolve() == project
        if same_name and not same_path:
            raise LaplaceError(
                f"Project name '{project_name}' is already registered at {item.get('path')}"
            )
        if same_path and not same_name:
            raise LaplaceError(f"Project path is already registered as '{item.get('name')}'")
    record = next((item for item in items if item.get("name") == project_name), None)
    if record is None:
        record = {
            "name": project_name,
            "path": str(project),
            "created": _now(),
            "last_seen": _now(),
            "validation": "VALID",
            "project_id": f"{project_name}:{project}",
        }
        items.append(record)
    else:
        record["last_seen"] = _now()
        record["validation"] = "VALID"
    _write_registry(items)
    return record


def _unregister(name: str) -> dict[str, Any]:
    items = _registry()
    kept = [item for item in items if item.get("name") != name]
    _write_registry(kept)
    return {"status": "UNREGISTERED" if len(kept) != len(items) else "NOT_FOUND", "name": name}


def _project_from_dir(path: Path) -> tuple[ProjectPaths, dict[str, Any]]:
    root = path.expanduser().resolve()
    config_path = root / ".laplace" / "project.yaml"
    if not config_path.is_file():
        raise LaplaceError(f"Laplace project configuration not found: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LaplaceError(f"Cannot read Laplace project: {exc}") from exc
    if not isinstance(config, dict) or config.get("project", {}).get("name") != root.name:
        raise LaplaceError(".laplace/project.yaml name does not match the project directory")
    paths = ProjectPaths(root.name, root, config_path, root / "Data", root / "Outputs")
    return paths, config


def detect_project(explicit: Path | None = None) -> tuple[ProjectPaths, dict[str, Any]]:
    if explicit is not None:
        return _project_from_dir(explicit)
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".laplace" / "project.yaml").is_file():
            paths, config = _project_from_dir(candidate)
            _register(candidate)
            return paths, config
    raise LaplaceError(
        "No Laplace project was found in the current directory or its parents. "
        "Run `laplace --init NAME` here, or pass `--project PATH`."
    )


def _tree(root: Path) -> dict[str, Path]:
    data = root / "Data"
    downloads = data / "Downloads"
    return {
        "laplace": root / ".laplace",
        "config": root / "Config",
        "parsed": data / "Parsed",
        "metadata": data / "Metadata",
        "vector_store": data / "VectorStore",
        "cache": data / "Cache",
        "logs": data / "Logs",
        "quarantine": data / "Quarantine",
        "downloads": downloads,
        "open_access": downloads / "OpenAccess",
        "ieee_pending": downloads / "IEEE" / "Pending",
        "ieee_downloaded": downloads / "IEEE" / "Downloaded",
        "ieee_failed": downloads / "IEEE" / "Failed",
        "drafts": root / "Outputs" / "Drafts",
        "evidence": root / "Outputs" / "EvidencePackets",
        "extractions": root / "Outputs" / "Extractions",
        "comparisons": root / "Outputs" / "Comparisons",
        "reports": root / "Outputs" / "Reports",
    }


def _project_yaml(name: str, root: Path) -> dict[str, Any]:
    library = Path(os.getenv("FORMALSCIENCE_ROOT", str(Path.home() / "FormalScience"))) / "Library"
    return {
        "project": {
            "name": name,
            "project_id": f"{name}:{root}",
            "created_at": _now(),
            "local_only": True,
        },
        "library": {
            "root": str(library),
            "collections": list(COLLECTIONS),
            "selected": ["MyWorks"],
        },
        "paths": {"data": str(root / "Data"), "outputs": str(root / "Outputs")},
        "models": {
            "main_text": MODEL,
            "embedding": EMBEDDING_MODEL,
            "endpoint": ENDPOINT,
            "context_tokens": 8192,
        },
        "retrieval": {"mode": "hybrid", "final_evidence_chunks": 6, "require_page_grounding": True},
        "writing": {"style_profile": "formal IEEE-style English"},
        "providers": {"online_search": True, "ieee_api": False, "browser": False},
        "security": {
            "localhost_only": True,
            "no_credential_storage": True,
            "source_pdfs_immutable": True,
        },
    }


def init_laplace(name: str, *, cwd: Path | None = None, force: bool = False) -> dict[str, Any]:
    base = (cwd or Path.cwd()).resolve()
    target = base if name == "." else (base / name).resolve()
    if name != ".":
        validate_project_name(name)
    project_name = target.name
    validate_project_name(project_name)
    if target.exists() and any(target.iterdir()) and not force:
        raise LaplaceError(
            f"Refusing to initialize a non-empty directory: {target}; use --force only when intentional"
        )
    if "library" in {part.lower() for part in target.parts}:
        raise LaplaceError("Projects cannot be initialized inside a Library directory")
    # A bare ancestor `.git` is not proof that this is the Laplace application
    # repository. Sandboxed and CI `/tmp` mounts commonly contain one, while a
    # project below them remains a valid user workspace. Python/application
    # markers retain the intended refusal without that false positive.
    repo_markers = {"pyproject.toml", "AGENTS.md", "CODEX_PROMPT.md"}
    ancestors = (target, *target.parents)
    inside_application_repo = any(
        any((ancestor / marker).exists() for marker in repo_markers) for ancestor in ancestors
    )
    if not force and inside_application_repo:
        raise LaplaceError(
            "Refusing to initialize inside an application repository without --force"
        )
    for directory in _tree(target).values():
        directory.mkdir(parents=True, exist_ok=True)
    config_path = target / ".laplace" / "project.yaml"
    if config_path.exists() and not force:
        raise LaplaceError(f"Laplace project already exists: {target}")
    config_path.write_text(
        yaml.safe_dump(_project_yaml(project_name, target), sort_keys=False), encoding="utf-8"
    )
    state = {
        "project": project_name,
        "project_id": f"{project_name}:{target}",
        "created_at": _now(),
        "server": {"running": False},
    }
    (target / ".laplace" / "state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    config_dir = target / "Config"
    (config_dir / "instructions.md").write_text(
        "# Project instructions\n\nKeep claims grounded in local evidence.\n", encoding="utf-8"
    )
    (config_dir / "writing_style.yaml").write_text(
        "style: formal IEEE-style English\n", encoding="utf-8"
    )
    (config_dir / "retrieval.yaml").write_text(
        "mode: hybrid\nfinal_evidence_chunks: 6\n", encoding="utf-8"
    )
    (config_dir / "providers.yaml").write_text(
        "online_search: true\nieee_api: false\n", encoding="utf-8"
    )
    record = _register(target)
    return {"status": "CREATED", "project": str(target), "registry": record}


def _state(paths: ProjectPaths) -> dict[str, Any]:
    path = paths.root / ".laplace" / "state.json"
    if not path.exists():
        return {"project": paths.name, "server": {"running": False}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"project": paths.name, "server": {"running": False}, "state_error": "invalid JSON"}
    return (
        value if isinstance(value, dict) else {"project": paths.name, "server": {"running": False}}
    )


def _write_state(paths: ProjectPaths, state: dict[str, Any]) -> None:
    (paths.root / ".laplace" / "state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def _database(paths: ProjectPaths) -> Path:
    return paths.data / "Metadata" / "workspace.db"


def _status(paths: ProjectPaths, config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "project": paths.name,
        "root": str(paths.root),
        "validation": {},
        "server": _state(paths).get("server", {}),
    }
    missing = [
        name for name, path in _tree(paths.root).items() if not path.is_dir() and name != "laplace"
    ]
    result["validation"] = {"valid": not missing, "missing_directories": missing}
    db = _database(paths)
    if db.exists():
        import sqlite3

        with sqlite3.connect(db) as conn:
            result["documents"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            result["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    else:
        result["documents"] = 0
        result["chunks"] = 0
    result["downloads"] = len(list((paths.data / "Downloads").rglob("*.pdf")))
    result["queued_candidates"] = len(list_queue(paths.name, root=paths.root).get("items", []))
    result["drafts"] = len(list((paths.outputs / "Drafts").glob("*")))
    result["configured_models"] = config.get("models", {})
    return result


def _doctor() -> dict[str, Any]:
    probe = collect_probe()
    runtime: dict[str, Any] = {
        "endpoint": ENDPOINT,
        "loopback_only": True,
        "models": [],
        "status": "UNAVAILABLE",
    }
    try:
        tags = ollama_tags(ENDPOINT)
        runtime["models"] = [
            str(item.get("name")) for item in tags.get("models", []) if isinstance(item, dict)
        ]
        runtime["status"] = "AVAILABLE"
    except Exception as exc:
        runtime["error"] = str(exc)
    executable = shutil.which("ollama") or (
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    )
    try:
        executable_exists = bool(executable and Path(executable).exists())
    except PermissionError:
        executable_exists = False
        runtime["executable_status"] = "permission_blocked"
    runtime["executable"] = str(executable) if executable_exists or executable else None
    runtime["main_model_installed"] = MODEL in runtime["models"]
    runtime["embedding_model_installed"] = EMBEDDING_MODEL in runtime["models"]
    return {
        "application": {"version": __version__, "local_only": True},
        "runtime": runtime,
        "probe": probe,
        "localhost": True,
        "npu_optional": True,
    }


def _library_root(config: dict[str, Any]) -> Path:
    configured = config.get("library", {}).get("root")
    return (
        Path(
            str(
                configured
                or (
                    Path(os.getenv("FORMALSCIENCE_ROOT", str(Path.home() / "FormalScience")))
                    / "Library"
                )
            )
        )
        .expanduser()
        .resolve()
    )


def _ingest(
    paths: ProjectPaths, config: dict[str, Any], collection: str, dry_run: bool
) -> dict[str, Any]:
    relative = Path(collection)
    if relative.is_absolute() or ".." in relative.parts:
        raise LaplaceError("Collection path must be relative and cannot contain '..'")
    if not relative.parts or relative.parts[0] not in COLLECTIONS:
        raise LaplaceError(f"Collection must begin with one of: {', '.join(COLLECTIONS)}")
    source = (_library_root(config) / relative).resolve()
    library_root = _library_root(config)
    if library_root not in source.parents and source != library_root:
        raise LaplaceError("Collection path escapes the configured Library root")
    files = sorted(source.rglob("*.pdf")) if source.is_dir() else []
    if dry_run:
        return {
            "status": "DRY_RUN",
            "collection": collection,
            "source_directory": str(source),
            "pdf_count": len(files),
            "files": [str(item) for item in files],
        }
    return ingest_library(
        paths.name,
        collection=relative.parts[0],
        root=paths.root,
        source_dir=source,
        project_paths_override=paths,
    )


def _chat_context(paths: ProjectPaths) -> ChatProject:
    config: dict[str, Any] = {}
    if paths.config.is_file():
        value = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            config = value
    project = config.get("project", {})
    models = config.get("models", {})
    return ChatProject(
        root=paths.root,
        database=paths.data / "Metadata" / "laplace.db",
        project_id=str(project.get("project_id") or f"{paths.name}:{paths.root}"),
        name=paths.name,
        model=str(models.get("main_text") or MODEL),
        endpoint=str(models.get("endpoint") or ENDPOINT),
        context_tokens=min(8192, int(models.get("context_tokens", 8192))),
    )


def _search(paths: ProjectPaths, query: str, limit: int = 6) -> dict[str, Any]:
    evidence = local_search(_database(paths), query, limit=limit)
    packet = evidence_packet(query, evidence)
    target = (
        paths.outputs / "Reports" / f"search_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
    packet["report_path"] = str(target)
    return packet


def _ask(paths: ProjectPaths, query: str) -> dict[str, Any]:
    project = _chat_context(paths)
    store = ConversationStore(project)
    conversation = store.create(title=query[:80] or "New chat", collections=["MyWorks"], mode="ASK")
    response = ChatEngine(project, store).answer(conversation.conversation_id, query)
    target = paths.outputs / "Conversations" / conversation.conversation_id / "response.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    detail = store.detail(conversation.conversation_id)
    rejected = next(
        (item for item in detail.messages if item.get("state") == "CITATION_REJECTED"), None
    )
    rejected_path: str | None = None
    if rejected is not None:
        draft_target = target.parent / "rejected_draft.json"
        draft_target.write_text(
            json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rejected_path = str(draft_target)
    return {
        "response": response.model_dump(),
        "rejected_response": rejected,
        "status": response.status,
        "answer_path": str(target),
        "rejected_path": rejected_path,
        "conversation_id": conversation.conversation_id,
        "revisions": detail.messages[-2:] if rejected is not None else [response.model_dump()],
    }


def _print_chat(
    result: dict[str, Any],
    *,
    as_json: bool = False,
    verbose: bool = False,
    show_rejected_draft: bool = False,
) -> None:
    if as_json:
        _json(result)
        return
    response = result.get("response", result)
    print(str(response.get("content", "")).strip())
    rejected = result.get("rejected_response")
    if rejected:
        print("\nModel draft: citation rejected")
        print("Grounded fallback: generated")
        if show_rejected_draft:
            print("\nRejected draft:\n" + str(rejected.get("content", "")).strip())
        print(f"Rejected draft saved: {result.get('rejected_path')}")
    citations = response.get("citations", [])
    if citations:
        print("\nSources:")
        for item in citations:
            print(
                f"[{item.get('citation_id')}] {item.get('title') or item.get('filename')}, p. {item.get('page') or '—'}"
            )
    print(f"\nGrounding: {response.get('status', 'UNKNOWN')}")
    print(f"Model: {response.get('model', MODEL)}")
    print(f"Saved: {result.get('answer_path', 'project conversation store')}")
    if verbose:
        _json(result)


def _print_search(result: dict[str, Any], *, as_json: bool = False, verbose: bool = False) -> None:
    if as_json or verbose:
        _json(result)
        return
    evidence = result.get("evidence", [])
    if not evidence:
        print("No indexed evidence found.")
        return
    print(f"Retrieved {len(evidence)} passages for: {result.get('query', '')}\n")
    for index, item in enumerate(evidence, 1):
        print(
            f"[{index}] {item.get('title') or item.get('filename')} — p. {item.get('page') or '—'}"
        )
        print(f"    {str(item.get('text', '')).strip()[:420]}")
    print(f"\nGrounding: VALID\nSaved: {result.get('report_path', 'project report')}")


def _write(
    paths: ProjectPaths, mode: str, instruction: str, input_path: Path | None
) -> dict[str, Any]:
    if input_path:
        instruction = (
            instruction + "\nInput:\n" + input_path.read_text(encoding="utf-8", errors="replace")
        )
    result = _ask(paths, instruction)
    target = paths.outputs / "Drafts" / f"{mode}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.md"
    response = result.get("response", result)
    answer = str(response.get("content", "")) if isinstance(response, dict) else ""
    target.write_text(f"# {mode}\n\n{answer}\n\nStatus: {result.get('status')}\n", encoding="utf-8")
    result["draft_path"] = str(target)
    return result


def _backup(paths: ProjectPaths) -> dict[str, Any]:
    target = paths.root / "Backup" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for relative in (
        Path(".laplace/project.yaml"),
        Path(".laplace/state.json"),
        Path("Config"),
        Path("Data/Metadata"),
        Path("Data/Parsed"),
        Path("Outputs"),
    ):
        source = paths.root / relative
        if not source.exists():
            continue
        destination = target / relative
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied.append(str(relative))
    return {"status": "BACKED_UP", "path": str(target), "copied": copied}


def _start(
    paths: ProjectPaths,
    background: bool,
    *,
    no_browser: bool = False,
    model: str = MODEL,
    embedding_model: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    tags = ollama_tags(ENDPOINT)
    installed = {str(item.get("name")) for item in tags.get("models", []) if isinstance(item, dict)}
    missing = [item for item in (model, embedding_model) if item not in installed]
    if missing:
        raise LaplaceError(f"Required local Ollama models are missing: {', '.join(missing)}")
    state = _state(paths)
    existing_pid = state.get("server", {}).get("pid")
    if existing_pid:
        try:
            os.kill(int(existing_pid), 0)
            info = {
                "status": "ALREADY_RUNNING",
                "pid": existing_pid,
                "bind": "127.0.0.1",
                "chat": "http://127.0.0.1:8000/chat",
                "dashboard": "http://127.0.0.1:8000/dashboard",
            }
            if not no_browser:
                webbrowser.open(str(info["chat"]))
            return info
        except (OSError, ValueError):
            pass
    log_path = paths.data / "Logs" / "laplace-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "research_workspace.laplace_server:create_project_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    environment = os.environ.copy()
    environment["FORMALSCIENCE_ACTIVE_PROJECT"] = str(paths.root)
    environment["RW_MODEL"] = model
    environment["RW_EMBEDDING_MODEL"] = embedding_model
    if background:
        stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=APPLICATION_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    else:
        process = subprocess.Popen(command, cwd=APPLICATION_ROOT, env=environment)
    state["server"] = {
        "running": True,
        "pid": process.pid,
        "bind": "127.0.0.1",
        "port": 8000,
        "started_at": _now(),
    }
    _write_state(paths, state)
    chat_url = "http://127.0.0.1:8000/chat"
    for _ in range(30):
        try:
            with urllib.request.urlopen(chat_url, timeout=0.5):
                break
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    info = {
        "status": "STARTED",
        "pid": process.pid,
        "bind": "127.0.0.1",
        "port": 8000,
        "log": str(log_path),
        "chat": chat_url,
        "dashboard": "http://127.0.0.1:8000/dashboard",
        "model": model,
        "embedding_model": embedding_model,
    }
    if not no_browser:
        webbrowser.open(chat_url)
    if not background:
        try:
            process.wait()
        finally:
            state["server"] = {"running": False, "stopped_at": _now(), "previous_pid": process.pid}
            _write_state(paths, state)
    return info


def _stop(paths: ProjectPaths) -> dict[str, Any]:
    state = _state(paths)
    server = state.get("server", {})
    pid = server.get("pid")
    if not pid:
        return {"status": "NOT_RUNNING"}
    try:
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True, check=False
        )
    except (OSError, ValueError):
        pass
    state["server"] = {"running": False, "stopped_at": _now(), "previous_pid": pid}
    _write_state(paths, state)
    return {"status": "STOPPED", "pid": pid}


def _extract(paths: ProjectPaths, kind: str, source: Path) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8", errors="replace")
    record = extract_metrics(
        text,
        Provenance(filename=source.name, page=None, section=None, chunk_id=f"{source.name}:c0"),
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = paths.outputs / "Extractions" / f"{kind}_{stamp}.json"
    csv_path = paths.outputs / "Extractions" / f"{kind}_{stamp}.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_record(record, json_path, csv_path)
    return {
        "status": "EXTRACTED",
        "type": kind,
        "json": str(json_path),
        "csv": str(csv_path),
        "metrics": len(record.metrics),
    }


def _engineering_domain(value: str) -> Domain:
    if value == "python":
        return "python"
    if value == "systemverilog":
        return "systemverilog"
    raise LaplaceError("Engineering domain must be python or systemverilog")


def _reference_operation(paths: ProjectPaths, values: list[str]) -> dict[str, Any]:
    if len(values) < 2:
        raise LaplaceError(
            "Reference usage: init|status|sync|verify|select|ingest DOMAIN [VALUE ...]"
        )
    action, domain_value, *rest = values
    domain = _engineering_domain(domain_value)
    library = ReferenceLibrary(paths.root, domain)
    if action == "init":
        catalog = APPLICATION_ROOT / "codex_a6000" / "reference_sources" / f"{domain}_sources.yaml"
        return library.initialize(catalog)
    if action == "status":
        return library.status()
    if action == "sync":
        return library.synchronize()
    if action == "verify":
        return library.verify(rest[0] if rest else None)
    if action == "select":
        return library.select(rest)
    if action == "ingest":
        return library.ingest(_database(paths))
    raise LaplaceError("Reference usage: init|status|sync|verify|select|ingest DOMAIN [VALUE ...]")


def _agent_task_operation(
    paths: ProjectPaths, specification_path: Path, domain_value: str
) -> dict[str, Any]:
    domain = _engineering_domain(domain_value)
    try:
        raw = json.loads(specification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaplaceError(f"Cannot read task specification: {exc}") from exc
    if not isinstance(raw, dict):
        raise LaplaceError("Task specification must be a JSON object")
    normalized = normalize_task_spec(APPLICATION_ROOT, domain, raw)
    store = AgentTaskStore(paths.root)
    task = store.create(domain, normalized)
    store.transition(
        task.task_id, "requirements", role="supervisor", note="Schema validated by CLI"
    )
    return {"status": "TASK_CREATED", "task": store.load(task.task_id).to_json()}


def _agent_research(paths: ProjectPaths, task_id: str, query: str) -> dict[str, Any]:
    store = AgentTaskStore(paths.root)
    task = store.load(task_id)
    evidence = retrieve_engineering_evidence(APPLICATION_ROOT, paths.root, task, query=query)
    artifact = store.write_artifact(
        task_id, role="researcher", name="evidence_packet", payload=evidence
    )
    evidence["artifact_path"] = str(artifact)
    return evidence


def _candidate_engine(value: object) -> Engine:
    if value == "vllm":
        return "vllm"
    if value == "sglang":
        return "sglang"
    raise LaplaceError("Serving candidate engine must be vllm or sglang")


def _candidate_text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise LaplaceError(f"Serving candidate {key} must be a non-empty string")
    return value


def _load_serving_candidate(path: Path) -> ServingCandidate:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaplaceError(f"Cannot read serving candidate: {exc}") from exc
    if not isinstance(raw, dict):
        raise LaplaceError("Serving candidate must be an object")
    candidate_raw: dict[str, object] = raw
    prefix_caching = candidate_raw.get("prefix_caching")
    chunked_prefill = candidate_raw.get("chunked_prefill")
    if not isinstance(prefix_caching, bool) or not isinstance(chunked_prefill, bool):
        raise LaplaceError("Serving candidate prefix_caching and chunked_prefill must be booleans")
    return ServingCandidate(
        engine=_candidate_engine(candidate_raw.get("engine")),
        endpoint=_candidate_text(candidate_raw, "endpoint"),
        model=_candidate_text(candidate_raw, "model"),
        revision=_candidate_text(candidate_raw, "revision"),
        quantization=_candidate_text(candidate_raw, "quantization"),
        kernel=_candidate_text(candidate_raw, "kernel"),
        prefix_caching=prefix_caching,
        chunked_prefill=chunked_prefill,
        cuda_graph_mode=_candidate_text(candidate_raw, "cuda_graph_mode"),
        scheduler=_candidate_text(candidate_raw, "scheduler"),
    )


def _agent_candidate(path: Path, prompt: str) -> dict[str, Any]:
    return benchmark_local_candidate(APPLICATION_ROOT, _load_serving_candidate(path), prompt=prompt)


def _agent_run(
    paths: ProjectPaths, task_id: str, candidate_path: Path, query: str
) -> dict[str, Any]:
    return LocalTeamRunner(
        APPLICATION_ROOT, paths.root, _load_serving_candidate(candidate_path)
    ).run(task_id, query=query)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laplace", description="Local-only FormalScience research workspace"
    )
    parser.add_argument("--project", type=Path, help="Explicit Laplace project directory")
    parser.add_argument(
        "--force", action="store_true", help="Explicitly confirm a safe, non-destructive override"
    )
    parser.add_argument("--yes", action="store_true", help="Confirm cache cleanup")
    parser.add_argument(
        "--background", action="store_true", help="Run --start as a detached localhost process"
    )
    parser.add_argument(
        "--foreground", action="store_true", help="Keep --start attached to the current terminal"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit complete machine-readable output")
    parser.add_argument(
        "--show-rejected-draft",
        action="store_true",
        help="Print the preserved draft rejected by citation validation",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local chat page")
    parser.add_argument("--init", metavar="NAME")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--unregister", metavar="NAME")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--chat", action="store_true", help="Open the current project chat page")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--clean-cache", action="store_true")
    parser.add_argument("--edit", choices=("instructions", "style", "retrieval", "providers"))
    parser.add_argument("--ingest", metavar="COLLECTION")
    parser.add_argument("--search", metavar="QUERY")
    parser.add_argument("--ask", metavar="QUERY")
    parser.add_argument("--write", nargs=2, metavar=("MODE", "INSTRUCTION"))
    parser.add_argument("--instructions", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--research", metavar="QUERY")
    parser.add_argument("--web", nargs="+", metavar="WEB")
    parser.add_argument("--queue", nargs="*", metavar="QUEUE")
    parser.add_argument("--download", type=int, metavar="CANDIDATE_ID")
    parser.add_argument("--ieee", nargs="*", metavar="IEEE")
    parser.add_argument("--promote", nargs=2, metavar=("DOCUMENT", "DESTINATION"))
    parser.add_argument("--extract", nargs=2, metavar=("TYPE", "SOURCE"))
    parser.add_argument("--compare", nargs="+", metavar="SOURCE")
    parser.add_argument(
        "--references",
        nargs="+",
        metavar="REFERENCE",
        help="Governed references: init|status|sync|verify|select|ingest DOMAIN [VALUE ...]",
    )
    parser.add_argument(
        "--agent-task-spec",
        type=Path,
        metavar="JSON",
        help="Normalize and persist an engineering task",
    )
    parser.add_argument("--agent-domain", choices=("python", "systemverilog"))
    parser.add_argument("--agent-research", nargs=2, metavar=("TASK_ID", "QUERY"))
    parser.add_argument(
        "--python-quality", action="store_true", help="Run allowlisted Python quality gates"
    )
    parser.add_argument(
        "--eda-flow",
        nargs="+",
        metavar="RTL",
        help="Run allowlisted RTL lint, simulation and synthesis checks",
    )
    parser.add_argument("--eda-top", metavar="MODULE")
    parser.add_argument("--eda-testbench", metavar="RTL")
    parser.add_argument("--agent-model-benchmark", type=Path, metavar="CANDIDATE_JSON")
    parser.add_argument(
        "--agent-benchmark-prompt", default="Return a concise local engineering summary."
    )
    parser.add_argument("--agent-run", metavar="TASK_ID")
    parser.add_argument("--agent-candidate", type=Path, metavar="CANDIDATE_JSON")
    parser.add_argument("--agent-query", metavar="QUERY")
    parser.add_argument("--paired-quality-benchmark", action="store_true")
    parser.add_argument(
        "--paired-candidate",
        type=Path,
        metavar="CANDIDATE_JSON",
        help="Run the real paired benchmark using this local A6000 serving candidate",
    )
    parser.add_argument(
        "--paired-base-commit",
        metavar="COMMIT",
        help="Exact clean checkpoint commit shared by both paired benchmark lanes",
    )
    parser.add_argument(
        "--paired-timeout-seconds",
        type=int,
        default=900,
        help="Per-lane paired benchmark timeout (default: 900)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.version:
            print(f"laplace {__version__}")
            return 0
        if args.init is not None:
            _json(init_laplace(args.init, force=args.force))
            return 0
        if args.list:
            _json({"projects": _registry()})
            return 0
        if args.unregister:
            _json(_unregister(args.unregister))
            return 0
        if args.doctor:
            _json(_doctor())
            return 0
        if args.paired_quality_benchmark:
            if args.paired_candidate is None:
                result = run_paired_quality_benchmark(APPLICATION_ROOT)
            else:
                if args.paired_timeout_seconds <= 0:
                    raise LaplaceError("--paired-timeout-seconds must be positive")
                base_commit = args.paired_base_commit
                if base_commit is None:
                    completed = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=APPLICATION_ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    if completed.returncode != 0:
                        raise LaplaceError(
                            f"Cannot resolve paired benchmark base commit: {completed.stderr.strip()}"
                        )
                    base_commit = completed.stdout.strip()
                result = run_valid_paired_benchmark(
                    APPLICATION_ROOT,
                    _load_serving_candidate(args.paired_candidate),
                    base_commit=base_commit,
                    timeout_seconds=args.paired_timeout_seconds,
                )
            _json(result)
            return 0 if result.get("status") == "MEASURED" else 2
        if args.agent_model_benchmark:
            _json(_agent_candidate(args.agent_model_benchmark, args.agent_benchmark_prompt))
            return 0
        paths, config = detect_project(args.project)
        if args.references:
            _json(_reference_operation(paths, args.references))
            return 0
        if args.agent_task_spec:
            if not args.agent_domain:
                raise LaplaceError("--agent-task-spec requires --agent-domain")
            _json(_agent_task_operation(paths, args.agent_task_spec, args.agent_domain))
            return 0
        if args.agent_research:
            _json(_agent_research(paths, args.agent_research[0], args.agent_research[1]))
            return 0
        if args.python_quality:
            _json(
                LocalToolRunner(
                    APPLICATION_ROOT, paths.outputs / "AgentTeam" / "tool_logs"
                ).run_python_quality_gates()
            )
            return 0
        if args.eda_flow:
            _json(
                LocalToolRunner(
                    APPLICATION_ROOT, paths.outputs / "AgentTeam" / "tool_logs"
                ).run_eda_flow(args.eda_flow, top_module=args.eda_top, testbench=args.eda_testbench)
            )
            return 0
        if args.agent_run:
            if args.agent_candidate is None or not args.agent_query:
                raise LaplaceError("--agent-run requires --agent-candidate and --agent-query")
            result = _agent_run(paths, args.agent_run, args.agent_candidate, args.agent_query)
            _json(result)
            return 0 if result.get("status") == "COMPLETE" else 2
        if args.validate:
            missing = [
                str(path)
                for name, path in _tree(paths.root).items()
                if name != "laplace" and not path.is_dir()
            ]
            _json(
                {"valid": not missing, "project": str(paths.root), "missing_directories": missing}
            )
            return 0 if not missing else 2
        if args.status:
            _json(_status(paths, config))
            return 0
        if args.config:
            _json(config)
            return 0
        if args.start:
            info = _start(paths, args.background or not args.foreground, no_browser=args.no_browser)
            if args.json:
                _json(info)
            else:
                print(f"Laplace project: {paths.name}")
                print(f"Model: {info.get('model', MODEL)}")
                print(f"Embeddings: {info.get('embedding_model', EMBEDDING_MODEL)}")
                print(f"Dashboard: {info.get('dashboard', 'http://127.0.0.1:8000/dashboard')}")
                print(f"Chat: {info.get('chat', 'http://127.0.0.1:8000/chat')}")
            return 0
        if args.chat:
            info = _start(paths, True, no_browser=False)
            if args.json:
                _json(info)
            else:
                print(info.get("chat", "http://127.0.0.1:8000/chat"))
            return 0
        if args.stop:
            _json(_stop(paths))
            return 0
        if args.backup:
            _json(_backup(paths))
            return 0
        if args.clean_cache:
            if not args.yes:
                raise LaplaceError(
                    "Cache cleanup removes rebuildable files; repeat with --clean-cache --yes"
                )
            cache = paths.data / "Cache"
            shutil.rmtree(cache, ignore_errors=True)
            cache.mkdir(parents=True, exist_ok=True)
            _json({"status": "CACHE_CLEANED", "path": str(cache)})
            return 0
        if args.edit:
            files = {
                "instructions": "instructions.md",
                "style": "writing_style.yaml",
                "retrieval": "retrieval.yaml",
                "providers": "providers.yaml",
            }
            target = paths.root / "Config" / files[args.edit]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text("", encoding="utf-8")
            _json(
                {
                    "status": "READY_TO_EDIT",
                    "path": str(target),
                    "note": "Open this local file in your editor; Laplace never stores credentials.",
                }
            )
            return 0
        if args.ingest:
            _json(_ingest(paths, config, args.ingest, args.dry_run))
            return 0
        if args.search:
            _print_search(_search(paths, args.search), as_json=args.json, verbose=args.verbose)
            return 0
        if args.ask:
            _print_chat(
                _ask(paths, args.ask),
                as_json=args.json,
                verbose=args.verbose,
                show_rejected_draft=args.show_rejected_draft,
            )
            return 0
        if args.write:
            mode, instruction = args.write
            if args.instructions:
                instruction = args.instructions.read_text(encoding="utf-8", errors="replace")
            _print_chat(
                _write(paths, mode, instruction, args.input),
                as_json=args.json,
                verbose=args.verbose,
                show_rejected_draft=args.show_rejected_draft,
            )
            return 0
        if args.research:
            result = search_scholarly(args.research, limit=10)
            target = (
                paths.outputs
                / "Reports"
                / f"research_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["report_path"] = str(target)
            _json(result)
            return 0 if result.get("status") in {"AVAILABLE", "PARTIAL"} else 2
        if args.web:
            action, *values = args.web
            if action == "fetch" and len(values) == 1:
                _json(fetch_public_webpage(values[0]))
                return 0
            if action == "search" and values:
                web_result = search_general_web(" ".join(values), limit=10)
                _json(
                    {
                        "provider": web_result.provider,
                        "status": web_result.status,
                        "results": [item.__dict__ for item in web_result.results],
                        "error": web_result.error,
                    }
                )
                return 0 if web_result.status == "AVAILABLE" else 2
            raise LaplaceError("Usage: --web fetch URL or --web search QUERY")
        if args.queue is not None:
            action = args.queue[0] if args.queue else "list"
            if action == "list":
                _json(list_queue(paths.name, root=paths.root))
            elif action == "add" and len(args.queue) == 2:
                candidate_path = Path(args.queue[1]).resolve()
                _json(
                    queue_candidate(
                        paths.name,
                        json.loads(candidate_path.read_text(encoding="utf-8-sig")),
                        root=paths.root,
                    )
                )
            elif action == "clear" and args.force:
                queue_path = paths.data / "Downloads" / "IEEE" / "Pending" / "candidate_queue.json"
                queue_path.write_text("[]\n", encoding="utf-8")
                _json({"status": "CLEARED", "path": str(queue_path)})
            elif action == "remove" and len(args.queue) == 2 and args.force:
                queue_path = paths.data / "Downloads" / "IEEE" / "Pending" / "candidate_queue.json"
                queue = (
                    json.loads(queue_path.read_text(encoding="utf-8"))
                    if queue_path.exists()
                    else []
                )
                candidate_id = int(args.queue[1])
                if candidate_id < 0 or candidate_id >= len(queue):
                    raise LaplaceError("candidate id not in queue")
                queue.pop(candidate_id)
                queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
                _json({"status": "REMOVED", "candidate_id": candidate_id})
            else:
                raise LaplaceError(
                    "Queue mutations require --force and use add PATH, remove ID, or clear"
                )
            return 0
        if args.download is not None:
            queued = list_queue(paths.name, root=paths.root).get("items", [])
            if not isinstance(queued, list) or args.download < 0 or args.download >= len(queued):
                raise LaplaceError("candidate id not in queue")
            candidate = queued[args.download]
            if isinstance(candidate, dict) and candidate.get("open_access") is True:
                _json(download_open_access(paths.name, candidate, root=paths.root))
            else:
                _json(download_candidate(paths.name, args.download, root=paths.root))
            return 0
        if args.ieee is not None:
            action = args.ieee[0] if args.ieee else "status"
            if action == "status":
                _json(ieee_status(paths.name, root=paths.root))
            elif action == "login":
                _json(login())
            elif action == "browser-init":
                _json(browser_init())
            elif action == "open" and len(args.ieee) == 2:
                _json(open_candidate(paths.name, int(args.ieee[1]), root=paths.root))
            elif action == "approve" and len(args.ieee) == 2 and args.force:
                _json(approve_download(paths.name, int(args.ieee[1]), root=paths.root))
            elif action == "download" and len(args.ieee) == 2:
                _json(download_candidate(paths.name, int(args.ieee[1]), root=paths.root))
            else:
                raise LaplaceError(
                    "IEEE usage: --ieee status|login|browser-init|open ID|approve ID|download ID"
                )
            return 0
        if args.promote:
            document, destination = args.promote
            if "." not in Path(document).name:
                import sqlite3

                with sqlite3.connect(_database(paths)) as conn:
                    row = conn.execute(
                        "SELECT filename FROM documents WHERE id=?", (document,)
                    ).fetchone()
                if row:
                    document = str(row[0])
            parts = destination.split("/", 1)
            if parts[0] not in {"MyTopics", "OtherTopics", "Documentations"}:
                raise LaplaceError(
                    "Promotion destination must be MyTopics, OtherTopics, or Documentations"
                )
            _json(
                promote_download(
                    paths.name,
                    document,
                    parts[0],
                    topic=parts[1] if len(parts) == 2 else None,
                    confirm=args.force,
                    root=paths.root,
                )
            )
            return 0
        if args.extract:
            _json(_extract(paths, args.extract[0], Path(args.extract[1]).resolve()))
            return 0
        if args.compare:
            records: list[dict[str, Any]] = []
            for item in args.compare:
                source = Path(item).resolve()
                if not source.is_file():
                    raise LaplaceError(f"Comparison source not found: {source}")
                if source.suffix.lower() == ".csv":
                    with source.open(newline="", encoding="utf-8-sig") as stream:
                        records.append(
                            {"source": str(source), "rows": list(csv.DictReader(stream))}
                        )
                elif source.suffix.lower() == ".json":
                    value = json.loads(source.read_text(encoding="utf-8-sig"))
                    records.append({"source": str(source), "value": value})
                else:
                    raise LaplaceError("Comparison inputs must be CSV or JSON")
            target = (
                paths.outputs
                / "Comparisons"
                / f"comparison_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {"sources": records, "deterministic": True}, indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            _json({"status": "COMPARISON_WRITTEN", "path": str(target), "sources": len(records)})
            return 0
        _parser().print_help()
        return 0
    except (
        LaplaceError,
        EngineeringError,
        ProjectError,
        OSError,
        ValueError,
        PermissionError,
        IndexError,
        json.JSONDecodeError,
    ) as exc:
        _json({"status": "ERROR", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
