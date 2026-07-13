# Engineering-agent operations

Initialize project-local reference libraries without network access:

```bash
laplace --project /path/to/project --references init python
laplace --project /path/to/project --references init systemverilog
laplace --project /path/to/project --references status python
laplace --project /path/to/project --references sync systemverilog
```

`sync` verifies local immutable snapshots only. It deliberately never clones a
catalog entry. Before a source can be registered, network access and licence
review must be approved; registration records its exact 40-hex commit, licence
text hash, selected-file hashes, attribution, permitted use and read-only
state. Reciprocal or uncertain material stays `reference_only_no_copy`.

Create a validated task and retrieve its ordered evidence packet:

```bash
laplace --project /path/to/project --agent-domain python --agent-task-spec task.json
laplace --project /path/to/project --agent-research TASK_ID "safe SQLite transaction"
laplace --project /path/to/project --python-quality
laplace --project /path/to/project --eda-flow benchmarks/a6000_agent_team/rtl/rv_skid_buffer.sv --eda-top rv_skid_buffer
```

For Codex stdio integration, keep configuration project-scoped:

```toml
[mcp_servers.laplace_engineering]
command = "/path/to/laplace/.venv/bin/python"
args = ["-m", "research_workspace.agent_mcp", "--repository-root", "/path/to/laplace", "--project-root", "/path/to/project"]
```

The harmless discovery check is a JSON-RPC `tools/list` call. The server binds
no TCP listener; existing Laplace FastAPI endpoints remain loopback-only.

Run the final comparison only on a host that proves the configured RTX A6000:

```bash
laplace --paired-quality-benchmark
```

When CUDA is unavailable, this writes the required JSON, CSV and Markdown
reports with `INVALID_BLOCKED_GPU`. That is evidence of an invalid comparison,
not a score, winner, or model-performance result.
