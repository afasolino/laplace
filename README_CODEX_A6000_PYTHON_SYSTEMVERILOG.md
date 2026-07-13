# Laplace A6000 Python/SystemVerilog software-agent overlay

This ZIP is an overlay for the current `afasolino/laplace` repository. Unzip it into the repository root. It does not replace the existing Laplace `README.md`, `AGENTS.md`, `CODEX_PROMPT.md`, or `PROJECT_CONFIG.yaml`.

The target is a Linux workstation with one RTX A6000. Python and SystemVerilog are first-class implementation domains. The final Codex phase runs a concurrent paired quality comparison between hosted Codex and the local Laplace team.

## A. Start from a clean Ubuntu workstation

The commands below target Ubuntu 22.04/24.04 with a working NVIDIA driver. Do not continue until the GPU is visible:

```bash
nvidia-smi
```

Install base tools:

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl unzip jq ripgrep build-essential pkg-config sqlite3 \
  verilator iverilog yosys
```

Install `uv` and expose it in the current shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Install Codex CLI and authenticate once:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
codex --version
codex
```

Complete the ChatGPT sign-in, then exit the initial Codex session. Verify non-interactive support:

```bash
codex exec --help
```

## B. Clone Laplace and apply the overlay

```bash
mkdir -p "$HOME/work"
cd "$HOME/work"
git clone https://github.com/afasolino/laplace.git
cd laplace
git switch -c feature/a6000-python-systemverilog-agents
```

Record the exact starting revision:

```bash
git rev-parse HEAD
git remote -v
git status --short --branch
```

Copy the ZIP to the workstation, then unzip it from the Laplace root:

```bash
unzip /absolute/path/local_agent_team_a6000_codex_laplace_python_systemverilog_v3.zip
```

Expected overlay files include:

```text
CODEX_LAPLACE_A6000_PYTHON_SYSTEMVERILOG_PROMPT.md
README_CODEX_A6000_PYTHON_SYSTEMVERILOG.md
codex_a6000/
  AGENT_RULES.md
  PROJECT_CONFIG.json
  reference_sources/
    python_sources.yaml
    systemverilog_sources.yaml
  templates/
    python_task_spec.schema.json
    systemverilog_task_spec.schema.json
    paired_benchmark_manifest.schema.json
  benchmarks/
    paired_task_catalog.yaml
  scripts/
    bootstrap_control_env_ubuntu.sh
    preflight_laplace_a6000.py
    preflight_laplace_a6000.sh
    preflight_laplace_a6000.ps1
```

## C. Create the CUDA-ready Laplace control environment

From the Laplace root:

```bash
./codex_a6000/scripts/bootstrap_control_env_ubuntu.sh
source .venv/bin/activate
```

The script creates `.venv` with Python 3.11, installs Laplace in editable mode with development tools, adds pytest coverage/property/security tools, installs a CUDA 12.4 PyTorch wheel, and verifies the A6000.

Manual equivalent:

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -e '.[dev]'
uv pip install pytest-cov hypothesis bandit
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CUDA verification:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3)
assert torch.cuda.is_available()
assert "A6000" in torch.cuda.get_device_name(0)
PY
```

Keep vLLM and SGLang out of `.venv`. Codex will create `.venv-vllm` and `.venv-sglang` only after inspecting compatibility.

## D. Optional: preserve the current Ollama fallback

Laplace currently uses an Ollama fallback. Install it only when you want the existing fallback path available:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

Check:

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags | jq .
```

## E. Run preflight and baseline checks

```bash
source .venv/bin/activate
./codex_a6000/scripts/preflight_laplace_a6000.sh
```

The report is written to:

```text
codex_a6000/runtime/preflight.json
```

Run the current baseline quality gates:

```bash
python -m pytest
ruff format --check src tests
ruff check src tests
python -m mypy src
```

Record any baseline failure without hiding it. Codex must distinguish existing failures from regressions.

Create a checkpoint commit containing only the overlay when desired:

```bash
git add CODEX_LAPLACE_A6000_PYTHON_SYSTEMVERILOG_PROMPT.md \
        README_CODEX_A6000_PYTHON_SYSTEMVERILOG.md \
        codex_a6000
git commit -m "Add A6000 Python and SystemVerilog agent-team specification"
```

## F. Start the implementation run

```bash
source .venv/bin/activate
codex
```

Paste exactly:

```text
Read CODEX_LAPLACE_A6000_PYTHON_SYSTEMVERILOG_PROMPT.md and execute it completely. Treat the current Laplace clone as the authoritative implementation baseline. Preserve its CLI, project lifecycle, RAG, citation/provenance validation, local server, data safety, tests, and Ollama fallback. Make high-quality Python and SystemVerilog first-class requirements, create the governed reference libraries, use the existing CUDA-ready .venv and RTX A6000, create isolated vLLM/SGLang environments only when needed, and proceed automatically through all repository-local phases. The final phase must run the concurrent paired Codex-direct versus Laplace-team benchmark and report relative code and implementation quality. Ask only for approvals explicitly required by the prompt.
```

Approve external network access when Codex needs to clone the curated reference repositories or download model checkpoints. Review licence findings before allowing reusable-code ingestion. Host package installation, privilege escalation, access outside approved roots, deletion, Git merge, and credentials remain approval-gated.

## G. Expected final evidence

The final run is incomplete until these exist:

```text
outputs/a6000_agent_team/comparison/codex_vs_laplace_results.json
outputs/a6000_agent_team/comparison/codex_vs_laplace_results.csv
outputs/a6000_agent_team/comparison/codex_vs_laplace_summary.md
```

The comparison must use at least four Python and two SystemVerilog tasks, identical clean base commits, separate worktrees, equal limits, held-out tests, actual concurrent execution, objective scoring, and explicit invalid-run reporting.
