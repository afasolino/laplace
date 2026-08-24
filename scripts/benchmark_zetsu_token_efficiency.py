#!/usr/bin/env python3
"""Validate and summarize the frozen three-pair Codex/Zetsu token check."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404 - only frozen argv from repository config is executed
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = ROOT / "configs/benchmarks/zetsu_token_tasks.json"
CONDITIONS = ("baseline", "zetsu")


def _load_tasks(path: Path = TASK_CONFIG) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise RuntimeError("invalid_benchmark_task_config")
    return value


def _task_ids(config: Mapping[str, object]) -> tuple[str, ...]:
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("invalid_benchmark_task_config")
    result: list[str] = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise RuntimeError("invalid_benchmark_task_config")
        result.append(str(task["id"]))
    if len(result) != 3 or len(set(result)) != 3:
        raise RuntimeError("benchmark_requires_exactly_three_unique_tasks")
    return tuple(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task-config", type=Path, default=TASK_CONFIG)
    parser.add_argument("--print-task-ids", action="store_true")
    parser.add_argument(
        "--print-repository-state-sha256",
        type=Path,
        default=None,
        metavar="WORKTREE",
        help="Print the deterministic HEAD+tree digest used by benchmark run metadata.",
    )
    parser.add_argument(
        "--inspect-codex-usage",
        type=Path,
        default=None,
        metavar="JSONL",
        help="Print one compact normalized usage object for a Codex JSONL stream.",
    )
    return parser


def _objects_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: object = json.loads(line)
        if isinstance(value, dict):
            yield value


def _normalized_usage(value: object) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }
    if not isinstance(value, Mapping):
        return result
    aliases = {
        "input_tokens": ("input_tokens",),
        "cached_input_tokens": ("cached_input_tokens",),
        "output_tokens": ("output_tokens",),
        "reasoning_tokens": ("reasoning_output_tokens", "reasoning_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for target, names in aliases.items():
        for name in names:
            raw = value.get(name)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                result[target] = raw
                break
    return result


def _with_uncached_input(value: dict[str, object]) -> dict[str, object]:
    input_tokens = value.get("input_tokens")
    cached_tokens = value.get("cached_input_tokens")
    value["uncached_input_tokens"] = (
        input_tokens - cached_tokens
        if isinstance(input_tokens, int)
        and isinstance(cached_tokens, int)
        and input_tokens >= cached_tokens
        else None
    )
    return value


def _usage_from_codex_jsonl(path: Path) -> dict[str, object]:
    """Parse Codex usage without recursively summing cumulative events."""

    final_total: Mapping[str, object] | None = None
    turn_usages: list[Mapping[str, object]] = []
    for root in _objects_from_jsonl(path):
        if root.get("type") == "event_msg":
            payload = root.get("payload")
            if isinstance(payload, Mapping) and payload.get("type") == "token_count":
                info = payload.get("info")
                if isinstance(info, Mapping):
                    total = info.get("total_token_usage")
                    if isinstance(total, Mapping):
                        final_total = total
        if root.get("type") == "turn.completed":
            usage = root.get("usage")
            if isinstance(usage, Mapping):
                turn_usages.append(usage)

    if final_total is not None:
        normalized_total = _normalized_usage(final_total)
        comparison_total = normalized_total["total_tokens"]
        comparison_source = "model_reported_total_tokens"
        if comparison_total is None:
            input_tokens = normalized_total["input_tokens"]
            output_tokens = normalized_total["output_tokens"]
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                comparison_total = input_tokens + output_tokens
                comparison_source = "exact_input_plus_output"
        return _with_uncached_input(
            {
                **normalized_total,
                "comparison_total_tokens": comparison_total,
                "comparison_total_source": comparison_source
                if comparison_total is not None
                else None,
                "source": "event_msg.token_count.info.total_token_usage:last",
                "exact_model_reported": True,
                "turn_records": len(turn_usages),
            }
        )

    if turn_usages:
        normalized_turns = [_normalized_usage(item) for item in turn_usages]
        summed: dict[str, int | None] = {}
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            values = [item[field] for item in normalized_turns]
            summed[field] = (
                sum(value for value in values if isinstance(value, int))
                if all(isinstance(value, int) for value in values)
                else None
            )
        comparison_total = summed["total_tokens"]
        comparison_source = "model_reported_total_tokens"
        if comparison_total is None:
            input_tokens = summed["input_tokens"]
            output_tokens = summed["output_tokens"]
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                comparison_total = input_tokens + output_tokens
                comparison_source = "exact_input_plus_output"
        return _with_uncached_input(
            {
                **summed,
                "comparison_total_tokens": comparison_total,
                "comparison_total_source": comparison_source
                if comparison_total is not None
                else None,
                "source": "turn.completed.usage:per_turn_sum",
                "exact_model_reported": True,
                "turn_records": len(turn_usages),
            }
        )

    return _with_uncached_input(
        {
            **_normalized_usage(None),
            "comparison_total_tokens": None,
            "comparison_total_source": None,
            "source": None,
            "exact_model_reported": False,
            "turn_records": 0,
        }
    )


def _mcp_calls_from_codex_jsonl(path: Path) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for root in _objects_from_jsonl(path):
        if root.get("type") != "item.completed":
            continue
        item = root.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "mcp_tool_call":
            continue
        result = item.get("result")
        result_bytes = len(
            json.dumps(result, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        )
        calls.append(
            {
                "server": item.get("server"),
                "tool": item.get("tool"),
                "status": item.get("status"),
                "result_utf8_bytes": result_bytes,
            }
        )
    return calls


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(worktree: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _git_value(worktree: Path, *argv: str) -> str | None:
    result = _git(worktree, *argv)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _repository_state_sha256(worktree: Path, base_revision: str) -> str | None:
    commit = _git_value(worktree, "rev-parse", "--verify", f"{base_revision}^{{commit}}")
    tree = _git_value(worktree, "rev-parse", "--verify", f"{base_revision}^{{tree}}")
    if commit is None or tree is None:
        return None
    return hashlib.sha256(f"commit={commit}\ntree={tree}\n".encode("utf-8")).hexdigest()


def _worktree_head(worktree: Path) -> str | None:
    return _git_value(worktree, "rev-parse", "--verify", "HEAD")


def _worktree_status(worktree: Path) -> str | None:
    result = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
    return result.stdout if result.returncode == 0 else None


def _changed_paths(worktree: Path) -> set[str] | None:
    status = _worktree_status(worktree)
    if status is None:
        return None
    changed: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.add(path.strip('"'))
    return changed


def _metadata(folder: Path) -> dict[str, Any]:
    path = folder / "result.json"
    if not path.is_file():
        return {}
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _safe_worktree(metadata: Mapping[str, object]) -> Path | None:
    value = metadata.get("worktree_path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).resolve()
    return path if path.is_dir() and (path / ".git").exists() else None


def _check_read_only(
    folder: Path, task: Mapping[str, object], metadata: Mapping[str, object]
) -> tuple[str, str]:
    correctness = task.get("correctness")
    if not isinstance(correctness, Mapping):
        return "FAIL", "missing_correctness_spec"
    artifact = correctness.get("artifact")
    required = correctness.get("required_paths")
    if not isinstance(artifact, str) or not isinstance(required, list):
        return "FAIL", "invalid_read_only_correctness_spec"
    answer_path = folder / artifact
    if not answer_path.is_file():
        return "FAIL", "answer_artifact_missing"
    answer = answer_path.read_text(encoding="utf-8")
    expected_answer = correctness.get("answer_exact")
    if isinstance(expected_answer, str) and answer != expected_answer:
        return "FAIL", "answer_exact_mismatch"
    worktree = _safe_worktree(metadata)
    if worktree is None:
        return "FAIL", "worktree_path_missing_or_invalid"
    for raw in required:
        if not isinstance(raw, str) or not (worktree / raw).exists():
            return "FAIL", f"repository_fact_missing:{raw}"
        if raw not in answer:
            return "FAIL", f"answer_missing_required_path:{raw}"
    base_revision = metadata.get("base_revision")
    if not isinstance(base_revision, str) or _worktree_head(worktree) != base_revision:
        return "FAIL", "read_only_head_changed"
    status = _worktree_status(worktree)
    if status is None:
        return "FAIL", "read_only_git_status_unavailable"
    if status:
        return "FAIL", "read_only_worktree_changed"
    return "PASS", "deterministic_path_assertions_passed_and_worktree_clean"


def _check_commands(task: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[str, str]:
    correctness = task.get("correctness")
    commands = correctness.get("commands") if isinstance(correctness, Mapping) else None
    worktree = _safe_worktree(metadata)
    if worktree is None or not isinstance(commands, list) or not isinstance(correctness, Mapping):
        return "FAIL", "implementation_check_inputs_missing"
    base_revision = metadata.get("base_revision")
    if not isinstance(base_revision, str) or _worktree_head(worktree) != base_revision:
        return "FAIL", "implementation_head_changed"
    allowed_raw = correctness.get("allowed_changed_paths")
    required_raw = correctness.get("required_changed_paths")
    if not isinstance(allowed_raw, list) or not isinstance(required_raw, list):
        return "FAIL", "implementation_path_spec_missing"
    allowed = {item for item in allowed_raw if isinstance(item, str) and item}
    required = {item for item in required_raw if isinstance(item, str) and item}
    if (
        len(allowed) != len(allowed_raw)
        or len(required) != len(required_raw)
        or not required <= allowed
    ):
        return "FAIL", "implementation_path_spec_invalid"
    changed = _changed_paths(worktree)
    if changed is None:
        return "FAIL", "implementation_git_status_unavailable"
    if not required <= changed:
        return "FAIL", "implementation_required_paths_unchanged"
    if not changed <= allowed:
        return "FAIL", "implementation_unexpected_changed_paths"
    for index, raw in enumerate(commands):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("argv"), list):
            return "FAIL", f"invalid_command_spec:{index}"
        argv = raw["argv"]
        if not argv or not all(isinstance(item, str) and item for item in argv):
            return "FAIL", f"invalid_command_argv:{index}"
        before_head = _worktree_head(worktree)
        before_status = _worktree_status(worktree)
        if before_head is None or before_status is None:
            return "FAIL", f"command_pre_state_unavailable:{index}"
        completed = subprocess.run(  # nosec B603
            list(argv), cwd=worktree, capture_output=True, text=True, check=False, timeout=120
        )
        after_head = _worktree_head(worktree)
        after_status = _worktree_status(worktree)
        if after_head != before_head or after_status != before_status:
            return "FAIL", f"correctness_command_mutated_worktree:{index}"
        expected_rc = raw.get("returncode", 0)
        if completed.returncode != expected_rc:
            return "FAIL", f"command_failed:{index}:rc={completed.returncode}"
        exact = raw.get("stdout_exact")
        if isinstance(exact, str) and completed.stdout != exact:
            return "FAIL", f"stdout_mismatch:{index}"
    final_changed = _changed_paths(worktree)
    if final_changed is None or not required <= final_changed or not final_changed <= allowed:
        return "FAIL", "implementation_change_scope_drifted_during_checks"
    return "PASS", "frozen_commands_and_change_scope_passed"


def _deterministic_correctness(
    folder: Path, task: Mapping[str, object], metadata: Mapping[str, object]
) -> tuple[str, str]:
    kind = task.get("kind")
    if kind == "read_only":
        return _check_read_only(folder, task, metadata)
    if kind == "implementation":
        return _check_commands(task, metadata)
    return "FAIL", "unknown_task_kind"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _methodology_metadata(
    task: Mapping[str, object],
    metadata: Mapping[str, object],
    config_sha: str,
    condition: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    prompt = task.get("prompt")
    if metadata.get("task_config_sha256") != config_sha:
        reasons.append("task_config_hash_mismatch")
    if not isinstance(prompt, str) or metadata.get("prompt_sha256") != _text_sha256(prompt):
        reasons.append("prompt_hash_mismatch")
    for field in (
        "repository_state_sha256",
        "base_revision",
        "codex_model",
        "codex_reasoning",
        "session_id",
        "worktree_path",
    ):
        if not isinstance(metadata.get(field), str) or not str(metadata[field]).strip():
            reasons.append(f"missing_{field}")
    worktree = _safe_worktree(metadata)
    base_revision = metadata.get("base_revision")
    if worktree is None:
        reasons.append("worktree_path_invalid")
    elif isinstance(base_revision, str):
        resolved_base = _git_value(worktree, "rev-parse", "--verify", f"{base_revision}^{{commit}}")
        head = _worktree_head(worktree)
        expected_state = _repository_state_sha256(worktree, base_revision)
        if resolved_base is None or head != resolved_base:
            reasons.append("worktree_head_not_frozen_base")
        if expected_state is None or metadata.get("repository_state_sha256") != expected_state:
            reasons.append("repository_state_hash_mismatch")
    calls = metadata.get("zetsu_tool_calls")
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
        reasons.append("missing_zetsu_tool_calls")
    elif condition == "baseline" and calls != 0:
        reasons.append("baseline_used_zetsu")
    elif condition == "zetsu":
        route = task.get("production_route")
        minimum = route.get("min_zetsu_calls", 1) if isinstance(route, Mapping) else 1
        maximum = route.get("max_zetsu_calls") if isinstance(route, Mapping) else None
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            reasons.append("invalid_min_zetsu_calls")
        elif calls < minimum:
            reasons.append("zetsu_condition_below_route_call_minimum")
        if isinstance(maximum, int) and not isinstance(maximum, bool) and calls > maximum:
            reasons.append("zetsu_condition_above_route_call_maximum")
    if metadata.get("fresh_session") is not True:
        reasons.append("fresh_session_not_attested")
    if metadata.get("fresh_worktree") is not True:
        reasons.append("fresh_worktree_not_attested")
    if metadata.get("run_index") != 1:
        reasons.append("run_index_not_one")
    return not reasons, reasons


def _pair_methodology(
    baseline: Mapping[str, object], zetsu: Mapping[str, object]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    baseline_meta = baseline.get("methodology")
    zetsu_meta = zetsu.get("methodology")
    if not isinstance(baseline_meta, Mapping) or not isinstance(zetsu_meta, Mapping):
        return False, ["methodology_metadata_missing"]
    for field in (
        "repository_state_sha256",
        "base_revision",
        "codex_model",
        "codex_reasoning",
        "prompt_sha256",
        "task_config_sha256",
    ):
        if baseline_meta.get(field) != zetsu_meta.get(field):
            reasons.append(f"pair_mismatch_{field}")
    if baseline_meta.get("session_id") == zetsu_meta.get("session_id"):
        reasons.append("sessions_not_distinct")
    if baseline_meta.get("worktree_path") == zetsu_meta.get("worktree_path"):
        reasons.append("worktrees_not_distinct")
    return not reasons, reasons


def _run_record(
    root: Path, task: Mapping[str, object], condition: str, config_sha: str
) -> dict[str, Any]:
    task_id = str(task["id"])
    folder = root / task_id / condition
    metadata = _metadata(folder)
    jsonl_path = folder / "codex.jsonl"
    usage = (
        _usage_from_codex_jsonl(folder / "codex.jsonl")
        if (folder / "codex.jsonl").is_file()
        else _usage_from_codex_jsonl_empty()
    )
    methodology_valid, methodology_reasons = _methodology_metadata(
        task, metadata, config_sha, condition
    )
    correctness, reason = (
        _deterministic_correctness(folder, task, metadata)
        if methodology_valid
        else ("FAIL", "methodology_invalid:" + ",".join(methodology_reasons))
    )
    methodology = {
        key: metadata.get(key)
        for key in (
            "task_config_sha256",
            "prompt_sha256",
            "repository_state_sha256",
            "base_revision",
            "codex_model",
            "codex_reasoning",
            "session_id",
            "worktree_path",
            "fresh_session",
            "fresh_worktree",
            "run_index",
            "zetsu_tool_calls",
        )
    }
    return {
        "task": task_id,
        "condition": condition,
        "usage": usage,
        "correctness": correctness,
        "correctness_reason": reason,
        "methodology_valid": methodology_valid,
        "methodology_reasons": methodology_reasons,
        "methodology": methodology,
        "zetsu_evidence_context_tokens_approx": metadata.get(
            "zetsu_evidence_context_tokens_approx"
        ),
        "zetsu_evidence_context_token_method": metadata.get("zetsu_evidence_context_token_method"),
        "local_qwen_input_tokens": metadata.get("local_qwen_input_tokens"),
        "local_qwen_output_tokens": metadata.get("local_qwen_output_tokens"),
        "local_qwen_usage_exact": metadata.get("local_qwen_usage_exact"),
        "local_qwen_calls": metadata.get("local_qwen_calls"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "zetsu_tool_calls": metadata.get("zetsu_tool_calls"),
        "actual_production_route": metadata.get("actual_production_route"),
        "mcp_calls": _mcp_calls_from_codex_jsonl(jsonl_path) if jsonl_path.is_file() else [],
        "active_mcp_schema_utf8_bytes": metadata.get("active_mcp_schema_utf8_bytes"),
        "actual_codex_credits": metadata.get("actual_codex_credits"),
    }


def _usage_from_codex_jsonl_empty() -> dict[str, object]:
    return _with_uncached_input(
        {
            **_normalized_usage(None),
            "comparison_total_tokens": None,
            "comparison_total_source": None,
            "source": None,
            "exact_model_reported": False,
            "turn_records": 0,
        }
    )


def _primary_tokens(record: Mapping[str, object]) -> int | None:
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return None
    if usage.get("exact_model_reported") is not True:
        return None
    total = usage.get("comparison_total_tokens")
    return total if isinstance(total, int) and total >= 0 else None


def main() -> int:
    args = _parser().parse_args()
    if args.inspect_codex_usage is not None:
        print(
            json.dumps(
                _usage_from_codex_jsonl(args.inspect_codex_usage.resolve()),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    config_path = args.task_config.resolve()
    config = _load_tasks(config_path)
    task_ids = _task_ids(config)
    if args.print_task_ids:
        print("\n".join(task_ids))
        return 0
    if args.print_repository_state_sha256 is not None:
        worktree = args.print_repository_state_sha256.resolve()
        head = _worktree_head(worktree)
        if head is None:
            raise SystemExit("WORKTREE is not a readable Git worktree")
        digest = _repository_state_sha256(worktree, head)
        if digest is None:
            raise SystemExit("unable to derive repository-state SHA-256")
        print(digest)
        return 0
    if args.result_root is None:
        raise SystemExit("result_root is required unless a print-only option is used")

    config_sha = _sha256(config_path)
    root = args.result_root.resolve()
    task_by_id = {str(item["id"]): item for item in config["tasks"] if isinstance(item, dict)}
    pairs: list[dict[str, Any]] = []
    baseline_sum = 0
    zetsu_sum = 0
    valid_count = 0
    for task_id in task_ids:
        task = task_by_id[task_id]
        baseline = _run_record(root, task, "baseline", config_sha)
        zetsu = _run_record(root, task, "zetsu", config_sha)
        baseline_tokens = _primary_tokens(baseline)
        zetsu_tokens = _primary_tokens(zetsu)
        pass_pair = baseline["correctness"] == "PASS" and zetsu["correctness"] == "PASS"
        pair_methodology_valid, pair_methodology_reasons = _pair_methodology(baseline, zetsu)
        saving: float | None = None
        if (
            pass_pair
            and pair_methodology_valid
            and isinstance(baseline_tokens, int)
            and isinstance(zetsu_tokens, int)
            and baseline_tokens > 0
        ):
            saving = 1.0 - (zetsu_tokens / baseline_tokens)
            baseline_sum += baseline_tokens
            zetsu_sum += zetsu_tokens
            valid_count += 1
        pairs.append(
            {
                "task": task_id,
                "baseline": baseline,
                "zetsu": zetsu,
                "baseline_codex_tokens": baseline_tokens,
                "zetsu_codex_tokens": zetsu_tokens,
                "saving": saving,
                "pass_pair": pass_pair,
                "pair_methodology_valid": pair_methodology_valid,
                "pair_methodology_reasons": pair_methodology_reasons,
                "included_in_aggregate": saving is not None,
            }
        )
    aggregate = 1.0 - (zetsu_sum / baseline_sum) if valid_count and baseline_sum > 0 else None
    report = {
        "schema_version": 3,
        "method": "three_frozen_single-run_paired_engineering_tasks",
        "task_config_path": str(config_path),
        "task_config_sha256": config_sha,
        "pairs": pairs,
        "aggregate_pair_count": valid_count,
        "aggregate_baseline_codex_tokens": baseline_sum if valid_count else None,
        "aggregate_zetsu_codex_tokens": zetsu_sum if valid_count else None,
        "aggregate_codex_saving": aggregate,
        "statistical_claim": False,
        "codex_usage_policy": (
            "prefer the final model-reported cumulative total; otherwise sum each distinct "
            "turn.completed usage record once; unavailable fields remain null"
        ),
        "qwen_tokens_excluded_from_codex_metric": True,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
