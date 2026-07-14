"""Minimal localhost-safe MCP stdio bridge for Laplace engineering tools.

It intentionally implements only JSON-RPC messages needed for tool discovery
and calls.  There is no generic shell operation and every path remains under
the project or repository configured at process start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from .engineering import (
    AgentTaskStore,
    Domain,
    EngineeringError,
    JsonObject,
    LocalToolRunner,
    ReferenceLibrary,
    normalize_task_spec,
    resolve_shared_reference_root,
    retrieve_engineering_evidence,
)
from .inference import Engine, ServingCandidate, benchmark_local_candidate
from .team_runner import LocalTeamRunner


def _object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EngineeringError(f"{label} must be an object")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EngineeringError(f"{label} must be a non-empty string")
    return value


def _domain(value: object) -> Domain:
    if value not in {"python", "systemverilog"}:
        raise EngineeringError("domain must be python or systemverilog")
    if value == "python":
        return "python"
    return "systemverilog"


def _engine(value: object) -> Engine:
    if value == "vllm":
        return "vllm"
    if value == "sglang":
        return "sglang"
    raise EngineeringError("candidate.engine must be vllm or sglang")


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise EngineeringError(f"{label} must be a boolean")
    return value


def _candidate(value: object) -> ServingCandidate:
    candidate_value = _object(value, label="candidate")
    return ServingCandidate(
        engine=_engine(candidate_value.get("engine")),
        endpoint=_text(candidate_value.get("endpoint"), label="candidate.endpoint"),
        model=_text(candidate_value.get("model"), label="candidate.model"),
        revision=_text(candidate_value.get("revision"), label="candidate.revision"),
        quantization=_text(candidate_value.get("quantization"), label="candidate.quantization"),
        kernel=_text(candidate_value.get("kernel"), label="candidate.kernel"),
        prefix_caching=_boolean(
            candidate_value.get("prefix_caching"), label="candidate.prefix_caching"
        ),
        chunked_prefill=_boolean(
            candidate_value.get("chunked_prefill"), label="candidate.chunked_prefill"
        ),
        cuda_graph_mode=_text(
            candidate_value.get("cuda_graph_mode"), label="candidate.cuda_graph_mode"
        ),
        scheduler=_text(candidate_value.get("scheduler"), label="candidate.scheduler"),
    )


def tool_definitions() -> list[JsonObject]:
    names = [
        "normalize_software_task",
        "normalize_python_task",
        "normalize_systemverilog_task",
        "research_task",
        "implement_task",
        "verify_patch",
        "review_patch",
        "run_tests",
        "run_python_quality_gates",
        "run_eda_flow",
        "reference_status",
        "benchmark_local_models",
        "run_paired_quality_benchmark",
    ]
    return [
        {
            "name": name,
            "description": "Local Laplace governed engineering operation.",
            "inputSchema": {"type": "object", "additionalProperties": True},
        }
        for name in names
    ]


class McpService:
    def __init__(self, repository_root: Path, project_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.project_root = project_root.resolve()
        self.runner = LocalToolRunner(
            self.repository_root, self.project_root / "Outputs" / "AgentTeam" / "tool_logs"
        )

    def _reference(self, domain: Domain) -> ReferenceLibrary:
        shared = resolve_shared_reference_root()
        return (
            ReferenceLibrary(shared, domain, shared=True)
            if shared is not None
            else ReferenceLibrary(self.project_root, domain)
        )

    def call(self, name: str, arguments: JsonObject) -> JsonObject:
        if name in {
            "normalize_software_task",
            "normalize_python_task",
            "normalize_systemverilog_task",
        }:
            default_domain = (
                "python"
                if name == "normalize_python_task"
                else "systemverilog"
                if name == "normalize_systemverilog_task"
                else None
            )
            domain = _domain(arguments.get("domain", default_domain))
            specification = _object(arguments.get("specification"), label="specification")
            normalized = normalize_task_spec(self.repository_root, domain, specification)
            return {"status": "NORMALIZED", "specification": normalized}
        if name == "reference_status":
            domain = _domain(arguments.get("domain"))
            library = self._reference(domain)
            action = str(arguments.get("action", "status"))
            if action == "status":
                return library.status()
            if action == "sync":
                return library.synchronize()
            if action == "initialize":
                catalog = _text(arguments.get("catalog_path"), label="catalog_path")
                return library.initialize(self.repository_root / catalog)
            if action == "verify":
                reference_id = arguments.get("reference_id")
                if reference_id is not None and not isinstance(reference_id, str):
                    raise EngineeringError("reference_id must be a string")
                return library.verify(reference_id)
            if action == "select":
                topics = arguments.get("topics", [])
                if not isinstance(topics, list) or not all(
                    isinstance(item, str) for item in topics
                ):
                    raise EngineeringError("topics must be a list of strings")
                return library.select(list(topics))
            if action == "ingest":
                return library.ingest(
                    None
                    if library.shared
                    else self.project_root / "Data" / "Metadata" / "workspace.db"
                )
            raise EngineeringError("Unsupported reference_status action")
        if name == "research_task":
            task_id = _text(arguments.get("task_id"), label="task_id")
            query = _text(arguments.get("query"), label="query")
            task = AgentTaskStore(self.project_root).load(task_id)
            return retrieve_engineering_evidence(
                self.repository_root,
                self.project_root,
                task,
                query=query,
                shared_reference_root=resolve_shared_reference_root(),
            )
        if name == "implement_task":
            task_id = _text(arguments.get("task_id"), label="task_id")
            task = AgentTaskStore(self.project_root).load(task_id)
            candidate_value = arguments.get("candidate")
            query_value = arguments.get("query")
            if candidate_value is not None and isinstance(query_value, str) and query_value:
                return LocalTeamRunner(
                    self.repository_root, self.project_root, _candidate(candidate_value)
                ).run(task.task_id, query=query_value)
            return {
                "status": "MODEL_REQUIRED",
                "task_id": task.task_id,
                "state": task.state,
                "reason": "Implementation is delegated only to the configured local CUDA model in an isolated worktree; this tool never substitutes CPU generation.",
            }
        if name in {"verify_patch", "run_tests", "run_python_quality_gates"}:
            paths = arguments.get("paths", ["src", "tests"])
            if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
                raise EngineeringError("paths must be a list of strings")
            return self.runner.run_python_quality_gates(list(paths))
        if name == "run_eda_flow":
            files = arguments.get("source_files")
            if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
                raise EngineeringError("source_files must be a list of strings")
            top_module = arguments.get("top_module")
            testbench = arguments.get("testbench")
            if top_module is not None and not isinstance(top_module, str):
                raise EngineeringError("top_module must be a string")
            if testbench is not None and not isinstance(testbench, str):
                raise EngineeringError("testbench must be a string")
            return self.runner.run_eda_flow(list(files), top_module=top_module, testbench=testbench)
        if name == "review_patch":
            task_id = _text(arguments.get("task_id"), label="task_id")
            task = AgentTaskStore(self.project_root).load(task_id)
            return {
                "status": "REVIEW_REQUIRED",
                "task_id": task.task_id,
                "state": task.state,
                "reason": "A reviewer receives requirements, diff, evidence and verifier logs; it cannot merge or edit source.",
            }
        if name == "benchmark_local_models":
            return benchmark_local_candidate(
                self.repository_root,
                _candidate(arguments.get("candidate")),
                prompt=_text(arguments.get("prompt"), label="prompt"),
            )
        if name == "run_paired_quality_benchmark":
            from .paired_benchmark import run_paired_quality_benchmark

            return run_paired_quality_benchmark(self.repository_root)
        raise EngineeringError(f"Unknown MCP tool: {name}")


def _reply(
    request_id: object, result: JsonObject | None = None, error: str | None = None
) -> JsonObject:
    response: JsonObject = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        response["result"] = result or {}
    else:
        response["error"] = {"code": -32000, "message": error}
    return response


def run_stdio(
    service: McpService, input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout
) -> int:
    for raw in input_stream:
        try:
            request_value: object = json.loads(str(raw))
            request = _object(request_value, label="JSON-RPC request")
            request_id = request.get("id")
            method = _text(request.get("method"), label="method")
            if method == "initialize":
                response = _reply(
                    request_id,
                    {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {"name": "laplace-engineering", "version": "0.1.0"},
                        "capabilities": {"tools": {}},
                    },
                )
            elif method == "tools/list":
                response = _reply(request_id, {"tools": tool_definitions()})
            elif method == "tools/call":
                params = _object(request.get("params"), label="params")
                response = _reply(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    service.call(
                                        _text(params.get("name"), label="tool name"),
                                        _object(params.get("arguments", {}), label="arguments"),
                                    ),
                                    ensure_ascii=False,
                                ),
                            }
                        ]
                    },
                )
            else:
                response = _reply(request_id, error=f"Unsupported MCP method: {method}")
        except (EngineeringError, json.JSONDecodeError, ValueError) as exc:
            response = _reply(None, error=str(exc))
        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-engineering-mcp")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_stdio(McpService(args.repository_root, args.project_root))
    except EngineeringError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
