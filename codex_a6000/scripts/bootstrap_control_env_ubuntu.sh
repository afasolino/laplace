#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "Run this script from inside the cloned Laplace repository." >&2
  exit 2
fi
cd "$ROOT"

if [[ ! -f pyproject.toml || ! -d src/research_workspace ]]; then
  echo "This does not look like the current Laplace repository." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Install/repair the NVIDIA driver before continuing." >&2
  exit 3
fi
nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is unavailable. Install it first with the command documented in README_CODEX_A6000_PYTHON_SYSTEMVERILOG.md." >&2
  exit 4
fi

uv python install 3.11
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -e '.[dev]'
uv pip install pytest-cov hypothesis bandit
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
from __future__ import annotations
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in .venv")
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print("VRAM GiB:", props.total_memory / 1024**3)
if "A6000" not in props.name or props.total_memory / 1024**3 < 45:
    raise SystemExit("Expected an RTX A6000-class 48 GiB GPU")
PY

echo "Control environment ready: $ROOT/.venv"
