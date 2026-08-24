from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_zetsu_token_efficiency", ROOT / "scripts/benchmark_zetsu_token_efficiency.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *argv],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark")
    (repo / "facts.txt").write_text("fact\n", encoding="utf-8")
    _git(repo, "add", "facts.txt")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_codex_usage_parser_uses_last_cumulative_record_once(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "codex.jsonl"
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 4,
                        "total_tokens": 110,
                    }
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 180,
                        "cached_input_tokens": 70,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 7,
                        "total_tokens": 200,
                    }
                },
            },
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")
    usage = module._usage_from_codex_jsonl(path)
    assert usage["input_tokens"] == 180
    assert usage["output_tokens"] == 20
    assert usage["reasoning_tokens"] == 7
    assert usage["total_tokens"] == 200
    assert usage["source"].endswith(":last")


def test_codex_usage_parser_sums_distinct_completed_turns_once(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "codex.jsonl"
    events = [
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 40,
                "cached_input_tokens": 10,
                "output_tokens": 5,
                "reasoning_output_tokens": 2,
                "total_tokens": 45,
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 60,
                "cached_input_tokens": 20,
                "output_tokens": 8,
                "reasoning_output_tokens": 3,
                "total_tokens": 68,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")
    usage = module._usage_from_codex_jsonl(path)
    assert usage["input_tokens"] == 100
    assert usage["cached_input_tokens"] == 30
    assert usage["output_tokens"] == 13
    assert usage["reasoning_tokens"] == 5
    assert usage["total_tokens"] == 113
    assert usage["turn_records"] == 2
    assert usage["source"] == "turn.completed.usage:per_turn_sum"


def test_codex_usage_parser_keeps_unavailable_fields_null(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "codex.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 40, "cached_input_tokens": 12, "output_tokens": 5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    usage = module._usage_from_codex_jsonl(path)
    assert usage["input_tokens"] == 40
    assert usage["reasoning_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["comparison_total_tokens"] == 45
    assert usage["comparison_total_source"] == "exact_input_plus_output"
    assert usage["uncached_input_tokens"] == 28


def test_inspect_codex_usage_is_compact_and_does_not_require_result_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    path = tmp_path / "codex.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 40,
                    "cached_input_tokens": 12,
                    "output_tokens": 5,
                    "reasoning_output_tokens": 2,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv", ["benchmark_zetsu_token_efficiency.py", "--inspect-codex-usage", str(path)]
    )
    assert module.main() == 0
    assert capsys.readouterr().out == (
        '{"cached_input_tokens":12,"comparison_total_source":"exact_input_plus_output",'
        '"comparison_total_tokens":45,"exact_model_reported":true,"input_tokens":40,'
        '"output_tokens":5,"reasoning_tokens":2,"source":"turn.completed.usage:per_turn_sum",'
        '"total_tokens":null,"turn_records":1,"uncached_input_tokens":28}\n'
    )


def test_codex_usage_parser_derives_comparison_total_from_exact_installed_schema(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "codex.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 71_419,
                    "cached_input_tokens": 49_152,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 710,
                    "reasoning_output_tokens": 284,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    usage = module._usage_from_codex_jsonl(path)

    assert usage["total_tokens"] is None
    assert usage["comparison_total_tokens"] == 72_129
    assert usage["comparison_total_source"] == "exact_input_plus_output"


def test_methodology_recomputes_frozen_repository_state_and_zetsu_use(tmp_path: Path) -> None:
    module = _module()
    repo, base = _repo(tmp_path)
    task = {"prompt": "same prompt"}
    common = {
        "task_config_sha256": "a" * 64,
        "prompt_sha256": module._text_sha256("same prompt"),
        "repository_state_sha256": module._repository_state_sha256(repo, base),
        "base_revision": base,
        "codex_model": "codex-model",
        "codex_reasoning": "high",
        "fresh_session": True,
        "fresh_worktree": True,
        "run_index": 1,
        "worktree_path": str(repo),
    }
    baseline_meta = {**common, "session_id": "baseline-session", "zetsu_tool_calls": 0}
    valid, reasons = module._methodology_metadata(task, baseline_meta, "a" * 64, "baseline")
    assert valid is True and reasons == []
    invalid, reasons = module._methodology_metadata(
        task,
        {**baseline_meta, "repository_state_sha256": "d" * 64},
        "a" * 64,
        "baseline",
    )
    assert invalid is False
    assert "repository_state_hash_mismatch" in reasons
    invalid, reasons = module._methodology_metadata(
        task, {**baseline_meta, "zetsu_tool_calls": 1}, "a" * 64, "baseline"
    )
    assert invalid is False and "baseline_used_zetsu" in reasons
    zetsu_meta = {**common, "session_id": "zetsu-session", "zetsu_tool_calls": 1}
    valid, reasons = module._methodology_metadata(task, zetsu_meta, "a" * 64, "zetsu")
    assert valid is True and reasons == []


def test_production_local_route_permits_zero_zetsu_calls(tmp_path: Path) -> None:
    module = _module()
    repo, base = _repo(tmp_path)
    task = {
        "prompt": "same prompt",
        "production_route": {"min_zetsu_calls": 0, "max_zetsu_calls": 0},
    }
    metadata = {
        "task_config_sha256": "a" * 64,
        "prompt_sha256": module._text_sha256("same prompt"),
        "repository_state_sha256": module._repository_state_sha256(repo, base),
        "base_revision": base,
        "codex_model": "codex-model",
        "codex_reasoning": "high",
        "session_id": "production-local",
        "worktree_path": str(repo),
        "fresh_session": True,
        "fresh_worktree": True,
        "run_index": 1,
        "zetsu_tool_calls": 0,
    }
    assert module._methodology_metadata(task, metadata, "a" * 64, "zetsu") == (
        True,
        [],
    )


def test_pair_methodology_requires_same_frozen_inputs_and_distinct_runs() -> None:
    module = _module()
    common = {
        "task_config_sha256": "a" * 64,
        "prompt_sha256": "p" * 64,
        "repository_state_sha256": "b" * 64,
        "base_revision": "c" * 40,
        "codex_model": "codex-model",
        "codex_reasoning": "high",
    }
    baseline_meta = {**common, "session_id": "baseline-session", "worktree_path": "/x/base"}
    zetsu_meta = {**common, "session_id": "zetsu-session", "worktree_path": "/x/zetsu"}
    pair_valid, pair_reasons = module._pair_methodology(
        {"methodology": baseline_meta}, {"methodology": zetsu_meta}
    )
    assert pair_valid is True and pair_reasons == []
    invalid, reasons = module._pair_methodology(
        {"methodology": baseline_meta},
        {"methodology": {**zetsu_meta, "repository_state_sha256": "d" * 64}},
    )
    assert invalid is False
    assert "pair_mismatch_repository_state_sha256" in reasons


def test_read_only_correctness_rejects_modified_worktree(tmp_path: Path) -> None:
    module = _module()
    repo, base = _repo(tmp_path)
    result_folder = tmp_path / "result"
    result_folder.mkdir()
    (result_folder / "final_answer.txt").write_text("fact=facts.txt\n", encoding="utf-8")
    task = {
        "correctness": {
            "artifact": "final_answer.txt",
            "required_paths": ["facts.txt"],
            "answer_exact": "fact=facts.txt\n",
        }
    }
    metadata = {"worktree_path": str(repo), "base_revision": base}
    assert module._check_read_only(result_folder, task, metadata)[0] == "PASS"
    (repo / "facts.txt").write_text("changed\n", encoding="utf-8")
    assert module._check_read_only(result_folder, task, metadata) == (
        "FAIL",
        "read_only_worktree_changed",
    )


def test_task_config_is_exactly_three_frozen_tasks() -> None:
    module = _module()
    config = module._load_tasks()
    assert module._task_ids(config) == ("routing", "zetsu_architecture", "tiny_agent_change")
    assert config["runs_per_condition"] == 1
    assert config["schema_version"] == 3
    third = config["tasks"][2]
    assert third["prompt"].startswith("Implement this exact small change:")
    assert third["correctness"]["allowed_changed_paths"] == [
        "scripts/benchmark_zetsu_token_efficiency.py",
        "tests/test_zetsu_token_benchmark.py",
    ]
    assert (
        third["correctness"]["required_changed_paths"]
        == third["correctness"]["allowed_changed_paths"]
    )
