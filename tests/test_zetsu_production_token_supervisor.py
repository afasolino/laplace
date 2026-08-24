from __future__ import annotations

import importlib.util
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


def test_frozen_production_routes_are_minimal_and_verifier_is_direct() -> None:
    module = _module()
    config = module._load_config(module.DEFAULT_TASK_CONFIG)
    tasks = {task["id"]: task for task in config["tasks"]}
    assert tuple(tasks) == (
        "local_control",
        "production_context",
        "usage_inspection_feature",
    )
    local = module._developer_instruction(
        tasks["local_control"], "zetsu", "production-v2-local_control"
    )
    retrieval = module._developer_instruction(
        tasks["production_context"], "zetsu", "production-v2-production_context"
    )
    implementation = module._developer_instruction(
        tasks["usage_inspection_feature"],
        "zetsu",
        "production-v2-usage_inspection_feature",
    )
    assert "make zero MCP calls" in local
    assert "project_context exactly once" in retrieval
    assert "agent_task exactly once" in implementation
    assert '["pytest","tests/test_zetsu_token_benchmark.py","-q"]' in implementation
    assert "python -m pytest" not in implementation
    assert "inline handoff.patch" in implementation


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
    assert module.MCP_SCHEMA_UTF8_BYTES == 4_647


def test_frozen_context_packet_indexes_before_any_codex_run(tmp_path: Path) -> None:
    module = _module()
    corpus = PersonalCorpusStore(
        tmp_path / "state",
        policy=PersonalCorpusPolicy(min_free_disk_bytes=1, max_file_bytes=256 * 1024 * 1024),
    )
    result = module._ingest_context(corpus, ROOT)
    assert result["indexed"]["status"] == "INDEXED"
    found = corpus.search(
        module.OWNER,
        "checkpoint manifest migration runbook zetsu contract serving runtime",
        limit=4,
    )
    assert found["retrieval_used"] is True
    assert "checkpoint_manifest=configs/model_manifests/qwen38_27b_a6000.json" in str(
        found["results"]
    )
