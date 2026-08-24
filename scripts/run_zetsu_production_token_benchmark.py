#!/usr/bin/env python3
"""Run the frozen three-pair production Codex/Laplace benchmark exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess  # nosec B404 - fixed local executables and frozen task argv
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_workspace.personal_corpus import PersonalCorpusPolicy, PersonalCorpusStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_CONFIG = ROOT / "configs/benchmarks/zetsu_production_token_tasks_v3.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/qwen38_certification/20260824-primary-e6b4/production-token-pairs-v3"
)
PROFILE_PATH = ROOT / "configs/serving_profile_candidates/P7_qwen38_w4a16_mtp.json"
VLLM = ROOT / ".venv-vllm-cu129/bin/vllm"
FFMPEG = ROOT / ".runtime/ffmpeg7/lib"
BASE_REVISION = "833a308e218eeb05b24a7c7d70d9fd7c1b9bceff"
OWNER = "codex-production-token-benchmark-v2"
MODEL = "gpt-5.6-terra"
REASONING = "high"
PORT = 8878
MCP_TOOLS = (
    "search",
    "get_evidence",
    "project_context",
    "experiment_context",
    "delegate",
    "agent_task",
    "rtl_task",
    "verify",
)
MCP_SCHEMA_UTF8_BYTES = 4_647
CONDITIONS = ("baseline", "zetsu")


class _UnusedAgent:
    def run(self, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("legacy_agent_backend_not_used")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 120,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=dict(env) if env is not None else None,
    )


def _git(worktree: Path, *argv: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return _run(("git", "-C", str(worktree), *argv), timeout=timeout)


def _git_value(worktree: Path, *argv: str) -> str:
    result = _git(worktree, *argv)
    if result.returncode != 0:
        raise RuntimeError(f"git_failed:{' '.join(argv)}:{result.stderr[-500:]}")
    return result.stdout.strip()


def _repository_state(worktree: Path, base_revision: str) -> str:
    commit = _git_value(worktree, "rev-parse", "--verify", f"{base_revision}^{{commit}}")
    tree = _git_value(worktree, "rev-parse", "--verify", f"{base_revision}^{{tree}}")
    return _sha256_bytes(f"commit={commit}\ntree={tree}\n".encode("utf-8"))


def _capabilities() -> Any:
    from research_workspace.serving_profiles import InstalledServingCapabilities

    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(FFMPEG)
    version = _run((str(VLLM), "--version"), timeout=120, env=environment)
    help_result = _run((str(VLLM), "serve", "--help=all"), timeout=120, env=environment)
    if version.returncode != 0 or help_result.returncode != 0:
        raise RuntimeError("vllm_capability_probe_failed")
    return InstalledServingCapabilities.from_help(
        version=version.stdout,
        help_text=help_result.stdout,
    )


def _load_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise RuntimeError("invalid_task_config")
    tasks = value["tasks"]
    if len(tasks) != 3 or any(not isinstance(item, dict) for item in tasks):
        raise RuntimeError("benchmark_requires_exactly_three_tasks")
    ids = [item.get("id") for item in tasks]
    if len(set(ids)) != 3 or any(not isinstance(item, str) for item in ids):
        raise RuntimeError("benchmark_task_ids_invalid")
    return value


def _prepare_worktrees(
    output: Path, tasks: Sequence[Mapping[str, object]]
) -> dict[tuple[str, str], Path]:
    worktrees: dict[tuple[str, str], Path] = {}
    worktree_root = output / "worktrees"
    worktree_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_id = str(task["id"])
        for condition in CONDITIONS:
            target = worktree_root / f"{task_id}-{condition}"
            result = _git(
                ROOT,
                "worktree",
                "add",
                "--detach",
                str(target),
                BASE_REVISION,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"benchmark_worktree_creation_failed:{target}:{result.stderr[-1000:]}"
                )
            if _git_value(target, "status", "--porcelain=v1", "--untracked-files=all"):
                raise RuntimeError(f"benchmark_worktree_not_clean:{target}")
            worktrees[(task_id, condition)] = target
    return worktrees


def _events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: object = json.loads(line)
        if isinstance(value, dict):
            events.append(value)
    return events


def _mcp_events(events: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        item
        for event in events
        if event.get("type") == "item.completed"
        and isinstance((item := event.get("item")), Mapping)
        and item.get("type") == "mcp_tool_call"
        and item.get("server") == "zetsu"
    ]


def _structured_content(event: Mapping[str, object]) -> Mapping[str, object] | None:
    result = event.get("result")
    structured = result.get("structured_content") if isinstance(result, Mapping) else None
    return structured if isinstance(structured, Mapping) else None


def _qwen_usage(
    mcp_events: Sequence[Mapping[str, object]],
) -> tuple[int, int, int, bool, str | None]:
    qwen_input = 0
    qwen_output = 0
    qwen_calls = 0
    exact = True
    checkpoint_path: str | None = None
    for event in mcp_events:
        structured = _structured_content(event)
        if not isinstance(structured, Mapping):
            continue
        checkpoint = structured.get("checkpoint_path")
        if isinstance(checkpoint, str):
            checkpoint_path = checkpoint
        telemetry = structured.get("telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        input_value = telemetry.get("qwen_input_tokens")
        output_value = telemetry.get("qwen_output_tokens")
        call_value = telemetry.get("qwen_calls")
        if (
            isinstance(input_value, int)
            and not isinstance(input_value, bool)
            and isinstance(output_value, int)
            and not isinstance(output_value, bool)
            and isinstance(call_value, int)
            and not isinstance(call_value, bool)
        ):
            qwen_input += input_value
            qwen_output += output_value
            qwen_calls += call_value
            exact = exact and telemetry.get("qwen_token_usage_complete") is True
        else:
            exact = False
    return qwen_input, qwen_output, qwen_calls, exact, checkpoint_path


def _turn_usage(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        dict(usage)
        for event in events
        if event.get("type") == "turn.completed"
        and isinstance((usage := event.get("usage")), Mapping)
    ]


def _gpu_values(samples: Sequence[Mapping[str, object]], field: str) -> list[int]:
    values: list[int] = []
    for sample in samples:
        value = sample.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError("invalid_gpu_sample")
        values.append(value)
    return values


def _route_from_events(events: Sequence[Mapping[str, object]]) -> str:
    tools = [str(item.get("tool")) for item in _mcp_events(events)]
    return "local_codex" if not tools else "+".join(tools)


def _developer_instruction(task: Mapping[str, object], condition: str, repo_id: str) -> str:
    common = (
        "This is one frozen production-efficiency benchmark execution. Use no network and inspect "
        "no unrelated files. The user prompt must remain the sole task definition. Deterministic "
        "repository assertions, not a model finish claim, decide correctness. "
    )
    if condition == "baseline":
        return common + (
            "This is standalone Codex: no Laplace or Zetsu MCP is available. Complete the request "
            "locally and keep any mutation within the paths named by the prompt."
        )
    task_id = str(task["id"])
    if task_id == "local_control":
        route = (
            "The production routing policy classifies this as a trivial current-checkout operation. "
            "Complete it locally and make zero MCP calls."
        )
    elif task_id == "production_context":
        route = (
            "The production routing policy classifies this as indexed historical project context. "
            "Call mcp__zetsu__project_context exactly once, with the exact user prompt as query, "
            "max_results=4, max_chars=6000, and telemetry=true. Use its compact evidence to answer; "
            "do not make a redundant retrieval/delegation call or scan broad unrelated paths."
        )
    elif task_id == "usage_inspection_feature":
        verifier = task.get("verification_argv")
        route = (
            "The production routing policy classifies this coherent implementation as suitable for "
            "agent_task. Before delegation bind this caller-selected verifier exactly: "
            f"{json.dumps(verifier, separators=(',', ':'))}. Call mcp__zetsu__agent_task exactly once "
            f"with repo_id={repo_id!r}, the exact user prompt as instruction, lane='quality', "
            "max_steps=16, max_chars=16000, that verification_argv, and telemetry=true. The agent's "
            "worktree is isolated: consume its exact inline handoff.patch and apply the same verified "
            "change to the current worktree without rediscovering or independently reimplementing it. "
            "Run the same direct pytest verifier locally after the latest mutation. Do not make a "
            "second Zetsu call; Qwen finish text is not correctness evidence."
        )
    else:
        raise RuntimeError(f"unknown_task:{task_id}")
    return common + "Follow the checked-in production Zetsu skill. " + route


def _codex_argv(
    task: Mapping[str, object],
    condition: str,
    worktree: Path,
    result_dir: Path,
    *,
    server_url: str | None,
    repo_id: str,
) -> list[str]:
    implementation = task.get("kind") == "implementation"
    argv = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--strict-config",
        "--json",
        "--color",
        "never",
        "-m",
        MODEL,
        "-c",
        f'model_reasoning_effort="{REASONING}"',
        "-c",
        f"developer_instructions={json.dumps(_developer_instruction(task, condition, repo_id))}",
        "-s",
        "workspace-write" if implementation else "read-only",
        "-C",
        str(worktree),
        "-o",
        str(result_dir / "final_answer.txt"),
    ]
    if condition == "zetsu":
        if server_url is None:
            raise RuntimeError("production_server_url_missing")
        argv.extend(
            (
                "-c",
                f'mcp_servers.zetsu.url="{server_url}"',
                "-c",
                'mcp_servers.zetsu.bearer_token_env_var="LAPLACE_ZETSU_TOKEN"',
                "-c",
                "mcp_servers.zetsu.enabled=true",
                "-c",
                "mcp_servers.zetsu.required=true",
                "-c",
                f"mcp_servers.zetsu.enabled_tools={json.dumps(MCP_TOOLS)}",
                "-c",
                # This one-shot non-interactive process cannot surface an approval
                # prompt. The user already authorized this exact localhost mutation.
                'mcp_servers.zetsu.default_tools_approval_mode="approve"',
                "-c",
                "mcp_servers.zetsu.startup_timeout_sec=15",
                "-c",
                "mcp_servers.zetsu.tool_timeout_sec=1900",
            )
        )
    argv.append("-")
    return argv


def _run_codex(
    output: Path,
    config_sha: str,
    task: Mapping[str, object],
    condition: str,
    worktree: Path,
    *,
    server_url: str | None,
    token: str | None,
) -> dict[str, object]:
    task_id = str(task["id"])
    result_dir = output / task_id / condition
    result_dir.mkdir(parents=True, exist_ok=False)
    repo_id = f"production-v2-{task_id}"
    environment = dict(os.environ)
    if condition == "zetsu":
        if token is None:
            raise RuntimeError("production_token_missing")
        environment["LAPLACE_ZETSU_TOKEN"] = token
    else:
        environment.pop("LAPLACE_ZETSU_TOKEN", None)
    argv = _codex_argv(
        task,
        condition,
        worktree,
        result_dir,
        server_url=server_url,
        repo_id=repo_id,
    )
    started = time.perf_counter()
    with (result_dir / "codex.jsonl").open("w", encoding="utf-8") as output_handle:
        completed = subprocess.run(  # nosec B603
            argv,
            input=str(task["prompt"]),
            stdout=output_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=2300,
            env=environment,
        )
    elapsed = time.perf_counter() - started
    (result_dir / "codex.stderr.log").write_text(completed.stderr, encoding="utf-8")
    answer = result_dir / "final_answer.txt"
    if task.get("kind") == "read_only" and answer.is_file():
        value = answer.read_text(encoding="utf-8")
        if not value.endswith("\n"):
            answer.write_text(value + "\n", encoding="utf-8")
    events = _events(result_dir / "codex.jsonl")
    thread_ids = [
        str(event["thread_id"])
        for event in events
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str)
    ]
    mcp = _mcp_events(events)
    qwen_input, qwen_output, qwen_calls, qwen_exact, checkpoint = _qwen_usage(mcp)
    result_bytes = sum(
        len(
            json.dumps(
                item.get("result"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        for item in mcp
    )
    metadata = {
        "task_config_sha256": config_sha,
        "prompt_sha256": _sha256_bytes(str(task["prompt"]).encode("utf-8")),
        "repository_state_sha256": _repository_state(worktree, BASE_REVISION),
        "base_revision": BASE_REVISION,
        "codex_model": MODEL,
        "codex_reasoning": REASONING,
        "session_id": thread_ids[-1] if thread_ids else "missing-session-id",
        "worktree_path": str(worktree),
        "fresh_session": len(thread_ids) == 1,
        "fresh_worktree": True,
        "run_index": 1,
        "condition_label": (
            "standalone_codex" if condition == "baseline" else "production_codex_laplace"
        ),
        "zetsu_tool_calls": len(mcp),
        "actual_production_route": _route_from_events(events),
        "active_mcp_schema_utf8_bytes": MCP_SCHEMA_UTF8_BYTES if condition == "zetsu" else 0,
        "active_mcp_tools": list(MCP_TOOLS) if condition == "zetsu" else [],
        "zetsu_evidence_context_bytes": result_bytes,
        "zetsu_evidence_context_tokens_approx": (result_bytes + 3) // 4 if result_bytes else 0,
        "zetsu_evidence_context_token_method": "utf8_compact_mcp_result_bytes_div4",
        "local_qwen_input_tokens": qwen_input,
        "local_qwen_output_tokens": qwen_output,
        "local_qwen_usage_exact": qwen_exact,
        "local_qwen_calls": qwen_calls,
        "qwen_checkpoint_path": checkpoint,
        "codex_turn_usage": _turn_usage(events),
        "actual_codex_credits": None,
        "elapsed_seconds": elapsed,
        "elapsed_method": "supervisor_monotonic_clock",
        "codex_returncode": completed.returncode,
        "prompt_transport": "exact_utf8_config_string_via_stdin_without_added_newline",
        "transport_timeout_seconds": 2300,
        "mcp_timeout_seconds": 1900 if condition == "zetsu" else None,
        "local_agent_wall_budget_seconds": 1800 if condition == "zetsu" else None,
    }
    (result_dir / "result.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark_run_complete": True,
                "condition": condition,
                "elapsed_seconds": round(elapsed, 3),
                "returncode": completed.returncode,
                "route": metadata["actual_production_route"],
                "task": task_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metadata


def _ingest_context(corpus: PersonalCorpusStore, worktree: Path) -> dict[str, object]:
    facts = (
        "checkpoint_manifest=configs/model_manifests/qwen38_27b_a6000.json",
        "migration_runbook=docs/QWEN38_PRODUCTION_MIGRATION.md",
        "zetsu_contract=docs/ZETSU.md",
        "serving_runtime=src/research_workspace/serving_profile_runtime.py",
    )
    paths = [item.split("=", 1)[1] for item in facts]
    sections: list[str] = [
        "# Frozen production architecture context",
        *facts,
        "",
        "The exact source hashes below bind these compact facts to the frozen repository state.",
    ]
    provenance: list[dict[str, object]] = []
    for relative in paths:
        source = worktree / relative
        data = source.read_bytes()
        sections.append(f"source_sha256[{relative}]={_sha256_bytes(data)}")
        provenance.append(
            {"path": relative, "sha256": _sha256_bytes(data), "size_bytes": len(data)}
        )
    content = "\n".join(sections).encode("utf-8")
    corpus_record = corpus.create_corpus(OWNER, "Frozen Qwen3.8 production architecture v2")
    corpus_id = str(corpus_record["corpus_id"])
    upload = corpus.create_upload(
        OWNER, corpus_id, idempotency_key="production-token-benchmark-v2-upload"
    )
    upload_id = str(upload["upload_id"])
    staged = corpus.stage_file(
        OWNER,
        upload_id,
        logical_path="benchmark/qwen38-zetsu-production-context.md",
        content=content,
        client_mime="text/markdown",
    )
    indexed = corpus.index_upload(
        OWNER, upload_id, idempotency_key="production-token-benchmark-v2-index"
    )
    return {
        "corpus_id": corpus_id,
        "content_sha256": _sha256_bytes(content),
        "provenance": provenance,
        "staged": staged,
        "indexed": indexed,
    }


def _preserve_and_cleanup_worktrees(
    output: Path, worktrees: Mapping[tuple[str, str], Path]
) -> dict[str, object]:
    evidence = output / "preserved-evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []
    failures: list[dict[str, str]] = []
    for (task_id, condition), worktree in worktrees.items():
        label = f"{task_id}-{condition}"
        if worktree.is_dir():
            status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
            patch = _git(worktree, "diff", "--no-ext-diff", "--binary", "HEAD", "--")
            (evidence / f"{label}.status").write_text(status.stdout, encoding="utf-8")
            (evidence / f"{label}.patch").write_text(patch.stdout, encoding="utf-8")
            (evidence / f"{label}.head").write_text(
                _git_value(worktree, "rev-parse", "HEAD") + "\n", encoding="utf-8"
            )
            result = _git(ROOT, "worktree", "remove", "--force", str(worktree))
            if result.returncode == 0:
                removed.append(str(worktree))
            else:
                failures.append({"path": str(worktree), "stderr": result.stderr[-1000:]})
    return {"removed": removed, "failures": failures}


def _remove_agent_worktrees(output: Path) -> dict[str, object]:
    roots = output / "service-state/agent-worktrees" / OWNER
    removed: list[str] = []
    failures: list[dict[str, str]] = []
    if roots.is_dir():
        for worktree in sorted(path for path in roots.iterdir() if path.is_dir()):
            status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
            patch = _git(worktree, "diff", "--no-ext-diff", "--binary", "HEAD", "--")
            evidence = output / "preserved-evidence"
            (evidence / f"agent-{worktree.name}.status").write_text(status.stdout, encoding="utf-8")
            (evidence / f"agent-{worktree.name}.patch").write_text(patch.stdout, encoding="utf-8")
            result = _git(ROOT, "worktree", "remove", "--force", str(worktree))
            if result.returncode == 0:
                removed.append(str(worktree))
            else:
                failures.append({"path": str(worktree), "stderr": result.stderr[-1000:]})
    return {"removed": removed, "failures": failures}


def main() -> int:
    args = _parser().parse_args()
    task_config = args.task_config.resolve()
    output = args.output_root.resolve()
    if output == Path("/tmp") or Path("/tmp") in output.parents:
        raise SystemExit("benchmark output under /tmp is forbidden")
    if output.exists():
        raise SystemExit(f"output root already exists; benchmark is single-run only: {output}")
    if _git_value(ROOT, "rev-parse", "HEAD") != BASE_REVISION:
        raise SystemExit("benchmark base revision mismatch")
    config = _load_config(task_config)
    tasks = list(config["tasks"])
    config_sha = _sha256(task_config)
    output.mkdir(parents=True)
    freeze = {
        "task_config": str(task_config),
        "task_config_sha256": config_sha,
        "base_revision": BASE_REVISION,
        "base_tree": _git_value(ROOT, "rev-parse", f"{BASE_REVISION}^{{tree}}"),
        "repository_state_sha256": _repository_state(ROOT, BASE_REVISION),
        "codex_model": MODEL,
        "codex_reasoning": REASONING,
        "runs_per_condition": 1,
        "conditions": ["standalone_codex", "production_codex_laplace"],
        "mcp_tools": list(MCP_TOOLS),
        "mcp_schema_utf8_bytes": MCP_SCHEMA_UTF8_BYTES,
        "profile_path": str(PROFILE_PATH),
        "profile_sha256": _sha256(PROFILE_PATH),
    }
    (output / "frozen-definition.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    worktrees: dict[tuple[str, str], Path] = {}
    run_records: list[dict[str, object]] = []
    report: dict[str, object] = {"freeze": freeze, "runs": run_records}
    runtime: Any = None
    server: Any = None
    server_thread: threading.Thread | None = None
    stop_gpu = threading.Event()
    sampler: threading.Thread | None = None
    gpu_samples: list[dict[str, object]] = []
    exit_code = 1
    try:
        worktrees = _prepare_worktrees(output, tasks)
        state = output / "service-state"
        state.mkdir()
        token = secrets.token_urlsafe(48)
        corpus = PersonalCorpusStore(
            state / "corpus",
            policy=PersonalCorpusPolicy(
                min_free_disk_bytes=1,
                # The spawn worker inherits the test/supervisor interpreter's loaded
                # virtual mappings. This raises only its address-space ceiling; the
                # sole controlled packet remains ~44 KiB and all upload validation stays active.
                max_file_bytes=256 * 1024 * 1024,
            ),
        )
        report["corpus"] = _ingest_context(corpus, worktrees[("production_context", "zetsu")])

        # Import the serving stack only after the spawn-isolated corpus parser has
        # finished, so its inherited address-space limit is not consumed by vLLM/API
        # orchestration modules before it parses the small frozen context packet.
        from dataclasses import asdict

        import uvicorn

        from research_workspace.agent_sandbox import AgentSandboxManager
        from research_workspace.operator_api import (
            AuthCredential,
            OperatorApiSettings,
            OperatorAuth,
            create_operator_app,
        )
        from research_workspace.operator_service import OperatorService
        from research_workspace.repository_authorization import RepositoryAuthorizationStore
        from research_workspace.service_tiers import (
            LanePolicy,
            LocalOpenAIChatBackend,
            ModelLane,
            ModelRoute,
            TierAuditLog,
            TieredServingService,
        )
        from research_workspace.serving_profile_runtime import (
            ServingProfileRuntime,
            observe_gpu,
        )
        from research_workspace.serving_profiles import ServingProfile, resolve_profile
        from research_workspace.user_capabilities import CapabilityTier, UserCapabilityStore
        from research_workspace.zetsu_mcp import tool_definitions

        definitions = tool_definitions()
        actual_tools = tuple(str(item["name"]) for item in definitions)
        actual_schema_bytes = len(
            json.dumps(definitions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if actual_tools != MCP_TOOLS or actual_schema_bytes != MCP_SCHEMA_UTF8_BYTES:
            raise RuntimeError("frozen_mcp_schema_drift")
        users = UserCapabilityStore(state / "users.sqlite3")
        users.set_user(OWNER, CapabilityTier.PLUS)
        authorizations = RepositoryAuthorizationStore(state / "repositories.sqlite3")
        for task in tasks:
            task_id = str(task["id"])
            repository = worktrees[(task_id, "zetsu")]
            repo_id = f"production-v2-{task_id}"
            authorizations.register(repo_id, repository)
            authorizations.grant(OWNER, repo_id, base_revision=BASE_REVISION)
        sandboxes = AgentSandboxManager(state / "agent-worktrees", authorizations)
        profile = ServingProfile.from_mapping(json.loads(PROFILE_PATH.read_text(encoding="utf-8")))
        resolved = resolve_profile(profile, _capabilities(), executable=VLLM, require_model=True)
        runtime = ServingProfileRuntime(state / "model-runtime", ffmpeg_library_path=FFMPEG)
        gpu_samples.append(asdict(observe_gpu()))

        def sample_gpu() -> None:
            while not stop_gpu.wait(1.0):
                try:
                    gpu_samples.append(asdict(observe_gpu()))
                except Exception:
                    pass

        sampler = threading.Thread(target=sample_gpu, daemon=True)
        sampler.start()
        owned = runtime.start(resolved)
        report["qwen_runtime"] = {
            "pid": owned.pid,
            "readiness": runtime.wait_ready(resolved),
            "profile_id": profile.profile_id,
            "profile_sha256": _sha256(PROFILE_PATH),
        }
        quality = ModelRoute(
            ModelLane.QUALITY,
            profile.served_model_name,
            f"http://127.0.0.1:{profile.port}",
            0,
            131072,
            4096,
        )
        standard = ModelRoute(
            ModelLane.STANDARD,
            profile.served_model_name,
            f"http://127.0.0.1:{profile.port}",
            10,
            131072,
            2048,
        )
        economy = ModelRoute(
            ModelLane.ECONOMY,
            "laplace-codev-r1-rl-qwen-7b-w4a16",
            "http://127.0.0.1:8103",
            20,
            8192,
            2048,
        )
        tiered = TieredServingService(
            users=users,
            sandboxes=sandboxes,
            lane_policy=LanePolicy(
                {
                    ModelLane.QUALITY: quality,
                    ModelLane.STANDARD: standard,
                    ModelLane.ECONOMY: economy,
                },
                quality_reserved_slots=1,
                standard_capacity=2,
                economy_capacity=4,
            ),
            chat_backend=LocalOpenAIChatBackend(timeout_seconds=1900),
            agent_backend=_UnusedAgent(),
            audit_log=TierAuditLog(state / "tier-audit.jsonl"),
        )
        app = create_operator_app(
            OperatorService(ROOT, state / "operator"),
            OperatorAuth({token: AuthCredential("read", OWNER, CapabilityTier.PLUS)}),
            settings=OperatorApiSettings(
                port=PORT,
                bearer_api_enabled=True,
                allowed_origins=(f"http://127.0.0.1:{PORT}",),
            ),
            tiered=tiered,
            personal_corpora=corpus,
        )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=PORT,
                log_level="warning",
                access_log=False,
                lifespan="off",
            )
        )
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError("operator_start_failed")
        server_url = f"http://127.0.0.1:{PORT}/mcp"
        print(
            json.dumps(
                {
                    "production_service_ready": True,
                    "mcp_timeout_seconds": 1900,
                    "model": profile.served_model_name,
                    "profile": profile.profile_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for task in tasks:
            result = _run_codex(
                output,
                config_sha,
                task,
                "baseline",
                worktrees[(str(task["id"]), "baseline")],
                server_url=None,
                token=None,
            )
            run_records.append(result)
        for task in tasks:
            result = _run_codex(
                output,
                config_sha,
                task,
                "zetsu",
                worktrees[(str(task["id"]), "zetsu")],
                server_url=server_url,
                token=token,
            )
            run_records.append(result)
        server.should_exit = True
        server_thread.join(timeout=30)
        if server_thread.is_alive():
            raise RuntimeError("operator_stop_timeout")
        server = None
        reporter = _run(
            (
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/benchmark_zetsu_token_efficiency.py"),
                str(output),
                "--task-config",
                str(task_config),
                "--output",
                str(output / "report.json"),
            ),
            cwd=ROOT,
            timeout=600,
        )
        (output / "report.stdout.json").write_text(reporter.stdout, encoding="utf-8")
        (output / "report.stderr.log").write_text(reporter.stderr, encoding="utf-8")
        report["reporter_returncode"] = reporter.returncode
        exit_code = 0 if reporter.returncode == 0 else 1
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=30)
        if runtime is not None:
            try:
                report["qwen_release"] = runtime.release_owned(timeout_seconds=90)
            except Exception as exc:
                report["qwen_release"] = {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}:{exc}",
                }
        stop_gpu.set()
        if sampler is not None:
            sampler.join(timeout=5)
        if gpu_samples:
            used_values = _gpu_values(gpu_samples, "used_mib")
            free_values = _gpu_values(gpu_samples, "free_mib")
            report["gpu"] = {
                "sample_count": len(gpu_samples),
                "peak_used_mib": max(used_values),
                "minimum_free_mib": min(free_values),
                "final": gpu_samples[-1],
            }
        if worktrees:
            report["benchmark_worktree_cleanup"] = _preserve_and_cleanup_worktrees(
                output, worktrees
            )
            report["agent_worktree_cleanup"] = _remove_agent_worktrees(output)
        (output / "supervisor-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
