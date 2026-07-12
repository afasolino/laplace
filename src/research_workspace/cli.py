from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze_file
from .core import configure_logging, ensure_layout, load_settings, write_manifest
from .documents import ingest
from .draft_workflow import grounded_related_work
from .acquisition import download_open_access, promote_download
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
from .library import ingest_downloads, ingest_library
from .online import fetch_public_webpage, search_ieee, search_scholarly
from .probe import collect_probe, write_probe
from .projects import ProjectError, init_project, list_projects, project_summary, validate_project
from .retrieval import evidence_packet, search as local_search
from .real_benchmark import BenchmarkError, run_real_benchmark


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="research-workspace")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("config")
    sub.add_parser("probe")
    bench = sub.add_parser("benchmark-model")
    bench.add_argument("--model")
    add = sub.add_parser("ingest")
    add.add_argument("path", type=Path)
    add.add_argument("--class", dest="document_class", default="project_document")
    log = sub.add_parser("analyze-log")
    log.add_argument("path", type=Path)
    log.add_argument("--profile", default="generic")
    project = sub.add_parser("project")
    project_sub = project.add_subparsers(dest="project_command")
    project_init = project_sub.add_parser("init")
    project_init.add_argument("name")
    project_init.add_argument("--root", type=Path)
    project_init.add_argument("--update", action="store_true")
    project_list = project_sub.add_parser("list")
    project_list.add_argument("--root", type=Path)
    project_show = project_sub.add_parser("show")
    project_show.add_argument("name")
    project_show.add_argument("--root", type=Path)
    project_validate = project_sub.add_parser("validate")
    project_validate.add_argument("name")
    project_validate.add_argument("--root", type=Path)
    lib = sub.add_parser("library-ingest")
    lib.add_argument("project")
    lib.add_argument("--collection", default="MyWorks")
    lib.add_argument("--root", type=Path)
    downloads = sub.add_parser("ingest-downloads")
    downloads.add_argument("project")
    downloads.add_argument("--root", type=Path)
    scholarly = sub.add_parser("search")
    scholarly.add_argument("query")
    scholarly.add_argument("--providers", default="crossref,openalex,arxiv")
    scholarly.add_argument("--limit", type=int, default=10)
    scholarly.add_argument("--offline", action="store_true")
    scholarly.add_argument("--project")
    scholarly.add_argument("--root", type=Path)
    ieee_search = sub.add_parser("search-ieee")
    ieee_search.add_argument("query")
    ieee_search.add_argument("--limit", type=int, default=10)
    ieee_search.add_argument("--project")
    ieee_search.add_argument("--root", type=Path)
    fetch = sub.add_parser("fetch-webpage")
    fetch.add_argument("url")
    local = sub.add_parser("local-search")
    local.add_argument("query")
    local.add_argument("--project", required=True)
    local.add_argument("--limit", type=int, default=6)
    local.add_argument("--class", dest="document_class")
    local.add_argument("--collection")
    local.add_argument("--author")
    local.add_argument("--year", type=int)
    local.add_argument("--doi")
    local.add_argument("--availability")
    local.add_argument("--source-kind")
    local.add_argument("--root", type=Path)
    draft = sub.add_parser("draft-related-work")
    draft.add_argument("project")
    draft.add_argument("query")
    draft.add_argument("--root", type=Path)
    candidate = sub.add_parser("candidate-queue")
    candidate.add_argument("project")
    candidate.add_argument("candidate_json", type=Path)
    candidate.add_argument("--root", type=Path)
    oa = sub.add_parser("download-open-access")
    oa.add_argument("project")
    oa.add_argument("candidate_json", type=Path)
    oa.add_argument("--root", type=Path)
    promote = sub.add_parser("promote-download")
    promote.add_argument("project")
    promote.add_argument("filename")
    promote.add_argument("--collection", required=True)
    promote.add_argument("--topic")
    promote.add_argument("--confirm", action="store_true")
    promote.add_argument("--root", type=Path)
    ieee = sub.add_parser("ieee")
    ieee_sub = ieee.add_subparsers(dest="ieee_command")
    ieee_sub.add_parser("browser-init")
    ieee_sub.add_parser("login")
    ieee_open = ieee_sub.add_parser("open")
    ieee_open.add_argument("project")
    ieee_open.add_argument("candidate_id", type=int)
    ieee_open.add_argument("--root", type=Path)
    ieee_download = ieee_sub.add_parser("download")
    ieee_download.add_argument("project")
    ieee_download.add_argument("candidate_id", type=int)
    ieee_download.add_argument("--root", type=Path)
    ieee_queue = ieee_sub.add_parser("queue")
    ieee_queue.add_argument("project")
    ieee_queue.add_argument("--root", type=Path)
    ieee_approve = ieee_sub.add_parser("approve")
    ieee_approve.add_argument("project")
    ieee_approve.add_argument("candidate_id", type=int)
    ieee_approve.add_argument("--batch-size", type=int, default=1)
    ieee_approve.add_argument("--root", type=Path)
    ieee_status_parser = ieee_sub.add_parser("status")
    ieee_status_parser.add_argument("--project")
    ieee_status_parser.add_argument("--root", type=Path)
    ieee_ingest = ieee_sub.add_parser("ingest-downloads")
    ieee_ingest.add_argument("project")
    ieee_ingest.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    configure_logging()
    settings = load_settings()
    ensure_layout(settings)
    if args.command == "project":
        try:
            if args.project_command == "init":
                print(
                    json.dumps(
                        {
                            "status": "CREATED",
                            "project": str(
                                init_project(args.name, root=args.root, update=args.update).root
                            ),
                        },
                        indent=2,
                    )
                )
                return 0
            if args.project_command == "list":
                print(json.dumps({"projects": list_projects(root=args.root)}, indent=2))
                return 0
            if args.project_command == "show":
                print(
                    json.dumps(
                        project_summary(args.name, root=args.root), indent=2, ensure_ascii=False
                    )
                )
                return 0
            if args.project_command == "validate":
                print(json.dumps(validate_project(args.name, root=args.root), indent=2))
                return 0
        except ProjectError as exc:
            print(json.dumps({"status": "PROJECT_ERROR", "error": str(exc)}))
            return 2
        parser.error("project requires init, list, show, or validate")
    if args.command == "library-ingest":
        print(
            json.dumps(
                ingest_library(args.project, collection=args.collection, root=args.root),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "ingest-downloads":
        print(
            json.dumps(ingest_downloads(args.project, root=args.root), indent=2, ensure_ascii=False)
        )
        return 0
    if args.command == "search":
        result = search_scholarly(
            args.query,
            providers=[item.strip() for item in args.providers.split(",") if item.strip()],
            limit=args.limit,
            offline=args.offline,
        )
        if args.project:
            paths, _ = __import__(
                "research_workspace.projects", fromlist=["load_project"]
            ).load_project(args.project, root=args.root)
            target = paths.outputs / "Reports" / "online_search.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["report_path"] = str(target)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "search-ieee":
        ieee_result = search_ieee(args.query, limit=args.limit)
        print(
            json.dumps(
                {
                    "provider": ieee_result.provider,
                    "status": ieee_result.status,
                    "query": ieee_result.query,
                    "results": [item.__dict__ for item in ieee_result.results],
                    "error": ieee_result.error,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if ieee_result.status == "AVAILABLE" else 2
    if args.command == "fetch-webpage":
        print(json.dumps(fetch_public_webpage(args.url), indent=2, ensure_ascii=False))
        return 0
    if args.command == "local-search":
        from .projects import load_project

        paths, _ = load_project(args.project, root=args.root)
        evidence = local_search(
            paths.data / "Metadata" / "workspace.db",
            args.query,
            limit=args.limit,
            document_class=args.document_class,
            collection=args.collection,
            author=args.author,
            year=args.year,
            doi=args.doi,
            availability=args.availability,
            source_kind=args.source_kind,
        )
        packet = evidence_packet(args.query, evidence)
        report_path = paths.outputs / "Reports" / "local_search.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        packet["report_path"] = str(report_path)
        print(json.dumps(packet, indent=2, ensure_ascii=False))
        return 0
    if args.command == "draft-related-work":
        print(
            json.dumps(
                grounded_related_work(
                    args.project,
                    args.query,
                    root=args.root,
                    endpoint=settings.model_endpoint,
                    model=settings.model or "qwen3:4b",
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "candidate-queue":
        print(
            json.dumps(
                queue_candidate(
                    args.project,
                    json.loads(args.candidate_json.read_text(encoding="utf-8-sig")),
                    root=args.root,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "download-open-access":
        print(
            json.dumps(
                download_open_access(
                    args.project,
                    json.loads(args.candidate_json.read_text(encoding="utf-8-sig")),
                    root=args.root,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "promote-download":
        print(
            json.dumps(
                promote_download(
                    args.project,
                    args.filename,
                    args.collection,
                    topic=args.topic,
                    confirm=args.confirm,
                    root=args.root,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "ieee":
        if args.ieee_command == "browser-init":
            print(json.dumps(browser_init(), indent=2))
            return 0
        if args.ieee_command == "login":
            print(json.dumps(login(), indent=2))
            return 0
        if args.ieee_command == "open":
            print(
                json.dumps(
                    open_candidate(args.project, args.candidate_id, root=args.root), indent=2
                )
            )
            return 0
        if args.ieee_command == "download":
            print(
                json.dumps(
                    download_candidate(args.project, args.candidate_id, root=args.root), indent=2
                )
            )
            return 0
        if args.ieee_command == "queue":
            print(json.dumps(list_queue(args.project, root=args.root), indent=2))
            return 0
        if args.ieee_command == "approve":
            print(
                json.dumps(
                    approve_download(
                        args.project, args.candidate_id, root=args.root, batch_size=args.batch_size
                    ),
                    indent=2,
                )
            )
            return 0
        if args.ieee_command == "status":
            print(json.dumps(ieee_status(args.project, root=args.root), indent=2))
            return 0
        if args.ieee_command == "ingest-downloads":
            print(
                json.dumps(
                    ingest_downloads(args.project, root=args.root), indent=2, ensure_ascii=False
                )
            )
            return 0
        parser.error("ieee requires browser-init, login, queue, or approve")
    if args.command == "config":
        print(
            json.dumps(
                {
                    "bind_host": settings.bind_host,
                    "context_tokens": settings.context_tokens,
                    "concurrency": settings.concurrency,
                    "model": settings.model,
                    "embedding_model": settings.embedding_model,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "probe":
        paths = write_probe(settings.root, collect_probe())
        write_manifest(settings, "probe", {}, list(paths))
        print(paths[0])
        return 0
    if args.command == "ingest":
        print(
            json.dumps(
                ingest(args.path, settings.root, settings.database, args.document_class), indent=2
            )
        )
        return 0
    if args.command == "analyze-log":
        print(json.dumps(analyze_file(args.path, args.profile), indent=2))
        return 0
    if args.command == "benchmark-model":
        model = args.model or settings.model
        if not model:
            print(
                json.dumps(
                    {
                        "status": "MODEL_REQUIRED",
                        "message": "Set RW_MODEL after recording licence and installing a local model.",
                    }
                )
            )
            return 2
        if model != settings.model:
            settings = settings.__class__(**{**settings.__dict__, "model": model})
        try:
            result = run_real_benchmark(settings)
            result_path = settings.root / "outputs" / "model_benchmark.json"
            result_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            write_manifest(
                settings,
                "benchmark-model",
                {"model": model, "embedding_model": settings.embedding_model},
                [result_path],
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        except BenchmarkError as exc:
            result = {
                "status": "MODEL_RUNTIME_ERROR",
                "model": model,
                "embedding_model": settings.embedding_model,
                "error": str(exc),
            }
            (settings.root / "outputs" / "model_benchmark.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            return 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
