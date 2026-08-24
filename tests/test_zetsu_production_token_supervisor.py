from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from research_workspace.personal_corpus import PersonalCorpusPolicy, PersonalCorpusStore


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_zetsu_production_token_benchmark",
        ROOT / "scripts/run_zetsu_production_token_benchmark.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *argv],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_synthetic_snapshot_includes_audited_changes_without_moving_branch(tmp_path: Path) -> None:
    module = _module()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark")
    (repo / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "base")
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
    config = repo / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    output = repo / "output"
    output.mkdir()
    module.ROOT = repo
    module.COMMITTED_BASE_REVISION = parent

    revision, tree = module._snapshot_revision(output, config)

    assert _git(repo, "rev-parse", "HEAD") == parent
    assert _git(repo, "show", f"{revision}:tracked.txt") == "after"
    assert _git(repo, "show", f"{revision}:config.json") == "{}"
    assert _git(repo, "rev-parse", f"{revision}^{{tree}}") == tree
    assert not (output / "synthetic-snapshot.index").exists()


def test_frozen_production_routes_are_minimal_and_verifier_is_direct() -> None:
    module = _module()
    config = module._load_config(module.DEFAULT_TASK_CONFIG)
    tasks = {task["id"]: task for task in config["tasks"]}
    assert tuple(tasks) == (
        "local_control",
        "production_context",
        "usage_comparison_feature",
    )
    local = module._developer_instruction(tasks["local_control"], "production-v4-local_control")
    retrieval = module._developer_instruction(
        tasks["production_context"], "production-v4-production_context"
    )
    implementation = module._developer_instruction(
        tasks["usage_comparison_feature"],
        "production-v4-usage_comparison_feature",
    )
    assert "make zero MCP calls" in local
    assert "call it exactly once" in retrieval
    assert "agent_task exactly once" in implementation
    assert '["pytest","tests/test_zetsu_token_benchmark.py","-q"]' in implementation
    assert "python -m pytest" not in implementation
    assert "apply_to_repository=true" in implementation
    assert "treat the handoff as authoritative" in implementation
    assert "do not reread" in implementation


def test_condition_argv_has_no_mcp_baseline_and_only_zetsu_in_production(
    tmp_path: Path,
) -> None:
    module = _module()
    task = module._load_config(module.DEFAULT_TASK_CONFIG)["tasks"][0]
    baseline = module._codex_argv(
        task,
        "baseline",
        tmp_path,
        tmp_path,
        server_url=None,
        repo_id="repo",
    )
    production = module._codex_argv(
        task,
        "zetsu",
        tmp_path,
        tmp_path,
        server_url="http://127.0.0.1:8878/mcp",
        repo_id="repo",
    )
    assert not any("mcp_servers" in item for item in baseline)
    assert not any("mcp_servers" in item and "zetsu" not in item for item in production)
    assert "mcp_servers.zetsu.tool_timeout_sec=1900" in production
    assert 'mcp_servers.zetsu.default_tools_approval_mode="approve"' in production
    assert "--ignore-user-config" in baseline and "--ignore-user-config" in production
    assert "--approve-for-me" not in baseline and "--approve-for-me" not in production
    baseline_policy = next(
        value.split("=", 1)[1]
        for index, value in enumerate(baseline)
        if baseline[index - 1] == "-c" and value.startswith("developer_instructions=")
    )
    production_policy = next(
        value.split("=", 1)[1]
        for index, value in enumerate(production)
        if production[index - 1] == "-c" and value.startswith("developer_instructions=")
    )
    assert baseline_policy == production_policy


def test_frozen_context_packet_indexes_before_any_codex_run(tmp_path: Path) -> None:
    module = _module()
    corpus = PersonalCorpusStore(
        tmp_path / "state",
        policy=PersonalCorpusPolicy(min_free_disk_bytes=1, max_file_bytes=256 * 1024 * 1024),
    )
    task = next(
        item
        for item in module._load_config(module.DEFAULT_TASK_CONFIG)["tasks"]
        if item["id"] == "production_context"
    )
    result = module._ingest_context(corpus, ROOT, task)
    assert result["indexed"]["status"] == "INDEXED"
    found = corpus.search(
        module.OWNER,
        "source repository p7 mtp quality economy rollback",
        limit=6,
    )
    assert found["retrieval_used"] is True
    assert "source_repository=barrydeen/Qwen3.8-27B-AWQ-4bit" in str(found["results"])
